"""
data_prep.py — Harmonized analysis panel for the UK heatwave-mortality thesis.

Builds a single tidy panel from:
  1. The 8 mortality CSVs (2014-2019, 2021-2022), schema-harmonized.
  2. HadCET v2.1.1.0 daily mean temperature (clean-break replacement of jittered Temp).
  3. Region-level covariates (H1 green space, H2 density, population).
  4. Recomputed Met Office heat windows (the supplied Heat.period was discarded).

Also exposes the three spatial weights matrices (Queen/Rook/kNN) built in
spatial_weights2.py.

Design decisions (see data_audit.md, step1_closure.md, sources.md):
  - Wales region code W99999999 -> W92000004.
  - Phantom 29-Feb rows (null Date.Temp) dropped.
  - Temp = HadCET daily national mean, broadcast to all region x sex rows.
  - Heat windows recomputed: >= N consecutive days with daily mean CET >= threshold.
  - Population denominators merged for offset(log(pop)) and per-100k rates.
  - 2020 absent from mortality data (COVID); HadCET covers it but it is not in the panel.

No geopandas/libpysal (unavailable in sandbox); pure-Python tooling used throughout.

Run: python data_prep.py  ->  writes panel_harmonized.parquet + prints head()/summary.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cet_integration import parse_hadcet  # noqa: E402

# ---- Paths (adjust to environment) ----
UPLOADS = Path("./data/uploads")
DATA_DIR = Path("./data/Data")
COVAR = Path("./covariates")
POLY = Path("./polygons")
CET_FILE = UPLOADS / "meantemp_daily_totals.txt"

ANALYSIS_YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2021, 2022]
BASELINE_START = 2009  # for rolling baseline buffer

REGION_ORDER = [
    "North East", "North West", "Yorkshire and The Humber", "East Midlands",
    "West Midlands", "East of England", "London", "South East", "South West", "Wales",
]
REGION_IDX = {n: i + 1 for i, n in enumerate(REGION_ORDER)}
WALES_FIX = {"W99999999": "W92000004"}


def load_mortality_year(year: int) -> pd.DataFrame:
    """Load one mortality CSV, harmonizing canonical-YYYY vs xYYYY schema."""
    canonical = DATA_DIR / f"{year}.csv"
    xfile = DATA_DIR / f"x{year}.csv"
    if canonical.exists():
        df = pd.read_csv(canonical, na_values=["NA", ""])
        df = df.rename(columns={df.columns[0]: "row_id", f"X{year}": "deaths"})
        date_col = "Date.Temp"
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    else:
        # xYYYY schema (used for 2022): space-separated names, death col == year,
        # Date.Temp holds "YYYY-MM-DD   T.T"
        df = pd.read_csv(xfile, na_values=["NA", ""])
        df = df.rename(columns={
            "Region code": "Region.code", "Region name": "Region.name",
            "Heat day": "Heat.day", "Heat period": "Heat.period",
            "Month day": "Month.day", str(year): "deaths",
        })
        df["date"] = pd.to_datetime(
            df["Date.Temp"].astype(str).str.split().str[0], errors="coerce"
        )
    # Drop phantom 29-Feb rows (null date)
    df = df.dropna(subset=["date"]).copy()
    # Region code fix
    df["Region.code"] = df["Region.code"].replace(WALES_FIX)
    # Discard the legacy jittered Temp and the discarded Heat.period (kept only as legacy ref)
    df = df.rename(columns={"Heat.period": "heat_period_legacy"})
    df["year"] = year
    keep = ["date", "year", "Sex", "Region.code", "Region.name", "Region",
            "deaths", "heat_period_legacy"]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def load_all_mortality() -> pd.DataFrame:
    frames = [load_mortality_year(y) for y in ANALYSIS_YEARS]
    panel = pd.concat(frames, ignore_index=True)
    # Canonical region index (override numeric Region with thesis-standard mapping by name)
    panel["region_idx"] = panel["Region.name"].map(REGION_IDX)
    assert panel["region_idx"].notna().all(), "unmapped region name"
    return panel


def attach_cet(panel: pd.DataFrame) -> pd.DataFrame:
    cet = parse_hadcet(CET_FILE, BASELINE_START, 2022)  # date, temp_cet
    panel = panel.merge(cet.rename(columns={"date": "date", "temp_cet": "temp"}),
                        on="date", how="left")
    assert panel["temp"].notna().all(), "CET coverage gap"
    return panel


def heat_windows(temp_by_date: pd.Series, threshold: float, min_consec: int = 3):
    """Met Office operational heatwave on a date-indexed daily-mean CET series.
    >= min_consec consecutive days with temp >= threshold. Returns is_heat (bool)
    and heat_period_id (int, NaN outside). Computed per calendar year."""
    s = temp_by_date.sort_index()
    is_heat = pd.Series(False, index=s.index)
    pid = pd.Series(np.nan, index=s.index)
    for yr, grp in s.groupby(s.index.year):
        above = (grp >= threshold).values
        n = len(above); period = 0; i = 0
        ih = np.zeros(n, bool); pp = np.full(n, np.nan)
        while i < n:
            if above[i]:
                j = i
                while j < n and above[j]:
                    j += 1
                if j - i >= min_consec:
                    period += 1; ih[i:j] = True; pp[i:j] = period
                i = j
            else:
                i += 1
        is_heat.loc[grp.index] = ih
        pid.loc[grp.index] = pp
    return is_heat, pid


def add_heat_flags(panel: pd.DataFrame, threshold: float = 20.0,
                   min_consec: int = 3) -> pd.DataFrame:
    """Add is_heat_day + heat_period_id at the primary threshold (default 20C, 3 days)."""
    daily = panel[["date", "temp"]].drop_duplicates().set_index("date")["temp"]
    is_heat, pid = heat_windows(daily, threshold, min_consec)
    panel = panel.merge(
        pd.DataFrame({"date": is_heat.index, "is_heat_day": is_heat.values,
                      "heat_period_id": pid.values}),
        on="date", how="left",
    )
    return panel


def attach_covariates(panel: pd.DataFrame) -> pd.DataFrame:
    cov = pd.read_csv(COVAR / "region_covariates_master.csv")
    cov = cov.rename(columns={"Region": "region_idx"})
    # population by region x sex x year (long)
    pop = pd.read_csv(COVAR / "population_region_sex_2014_2022.csv")
    pop_long = pop.melt(
        id_vars=["RGN22CD", "RGN22NM", "Sex", "Region"],
        value_vars=[str(y) for y in range(2014, 2023)],
        var_name="year", value_name="population",
    )
    pop_long["year"] = pop_long["year"].astype(int)
    pop_long = pop_long.rename(columns={"Region": "region_idx"})[
        ["region_idx", "Sex", "year", "population"]
    ]
    panel = panel.merge(pop_long, on=["region_idx", "Sex", "year"], how="left")
    cov_cols = ["region_idx", "greenspace_m2_per_capita_fit",
                "greenspace_m2_per_capita_all", "avg_dist_park_m",
                "pct_builtup", "dwellings_per_km2", "pop_density_per_km2",
                "land_area_km2"]
    panel = panel.merge(cov[cov_cols], on="region_idx", how="left")
    assert panel["population"].notna().all(), "population merge gap"
    assert panel["pct_builtup"].notna().all(), "covariate merge gap"
    return panel


def finalize(panel: pd.DataFrame) -> pd.DataFrame:
    panel["doy"] = panel["date"].dt.dayofyear
    panel["month"] = panel["date"].dt.month
    panel["dow"] = panel["date"].dt.dayofweek
    panel["sex_label"] = panel["Sex"].map({1: "Male", 2: "Female"})
    panel["deaths_per_100k"] = panel["deaths"] / panel["population"] * 1e5
    cols = ["date", "year", "month", "doy", "dow",
            "region_idx", "Region.name", "Region.code", "Sex", "sex_label",
            "deaths", "population", "deaths_per_100k", "temp",
            "is_heat_day", "heat_period_id", "heat_period_legacy",
            "greenspace_m2_per_capita_fit", "greenspace_m2_per_capita_all",
            "avg_dist_park_m", "pct_builtup", "dwellings_per_km2",
            "pop_density_per_km2", "land_area_km2"]
    cols = [c for c in cols if c in panel.columns]
    return panel[cols].sort_values(["region_idx", "Sex", "date"]).reset_index(drop=True)


def load_weights() -> dict:
    return json.load(open(POLY / "spatial_weights.json"))


def build_panel() -> pd.DataFrame:
    panel = load_all_mortality()
    panel = attach_cet(panel)
    panel = add_heat_flags(panel, threshold=20.0, min_consec=3)
    panel = attach_covariates(panel)
    panel = finalize(panel)
    return panel


if __name__ == "__main__":
    panel = build_panel()
    out = Path("./panel_harmonized.csv")
    panel.to_csv(out, index=False)
    print(f"Panel: {len(panel):,} rows x {panel.shape[1]} cols -> {out}")
    print(f"Years: {sorted(panel['year'].unique())}")
    print(f"Regions: {panel['region_idx'].nunique()} | Sexes: {panel['Sex'].nunique()}")
    print(f"Date range: {panel['date'].min().date()} to {panel['date'].max().date()}")
    print(f"Total deaths: {panel['deaths'].sum():,}")
    print(f"Heat days (>=20C,3d) region-sex-rows: {int(panel['is_heat_day'].sum()):,}")
    print("\nHead:")
    with pd.option_context("display.width", 240, "display.max_columns", 30):
        print(panel.head(8).to_string(index=False))
    W = load_weights()
    print("\nWeights matrices loaded:", list(W.keys()))
