"""
Step 1: Data audit for UK heatwave-mortality thesis CSVs.

Verifies the §3 schema against the 7 canonical CSVs, profiles missingness,
range-checks every numeric column, and surfaces anomalies for review BEFORE
any modeling decisions get locked in.

Outputs a JSON object on stdout that the markdown summary is built from.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("./data/Data")
CANONICAL_YEARS = [2014, 2015, 2016, 2017, 2018, 2019, 2021]

EXPECTED_REGIONS = {
    "E12000001": ("North East", 1),
    "E12000002": ("North West", 2),
    "E12000003": ("Yorkshire and The Humber", 3),
    "E12000004": ("East Midlands", 4),
    "E12000005": ("West Midlands", 5),
    "E12000006": ("East of England", 6),
    "E12000007": ("London", 7),
    "E12000008": ("South East", 8),
    "E12000009": ("South West", 9),
    "W92000004": ("Wales", 10),  # ONS code for Wales — may also be E12000010 or similar
}


def load_year(year: int) -> pd.DataFrame:
    p = DATA_DIR / f"{year}.csv"
    df = pd.read_csv(p, na_values=["NA", ""], keep_default_na=True)
    df = df.rename(columns={df.columns[0]: "row_id"})
    # Normalize the death-count column to a stable name
    death_col = f"X{year}"
    if death_col in df.columns:
        df = df.rename(columns={death_col: "deaths"})
    # Normalize Heat.day(s) variant to a stable name
    if "Heat.days" in df.columns:
        df = df.rename(columns={"Heat.days": "heat_temp"})
    elif "Heat.day" in df.columns:
        df = df.rename(columns={"Heat.day": "heat_temp"})
    df["year"] = year
    return df


def profile_year(year: int) -> dict:
    raw = pd.read_csv(DATA_DIR / f"{year}.csv", na_values=["NA", ""])
    df = load_year(year)

    n_rows = len(df)
    expected_rows = (366 if year % 4 == 0 else 365) * 10 * 2  # days × 10 regions × 2 sexes

    # Region check
    regions_in_data = (
        df[["Region.code", "Region.name", "Region"]].drop_duplicates().sort_values("Region")
    )

    # Days-per-(region,sex) check
    grp = df.groupby(["Region.code", "Sex"]).size()
    days_per_unit = grp.describe()[["min", "max", "mean"]].to_dict()
    days_per_unit_min = int(grp.min())
    days_per_unit_max = int(grp.max())

    # Missingness
    miss = df.isna().sum().to_dict()
    miss_pct = (df.isna().mean() * 100).round(2).to_dict()

    # Numeric range checks
    temp_stats = df["Temp"].describe().to_dict()
    deaths_stats = df["deaths"].describe().to_dict()

    # Heat period coverage
    heat_rows = df["Heat.period"].notna().sum()
    heat_unique_periods = df["Heat.period"].dropna().unique().tolist()
    heat_periods_sorted = sorted([int(p) for p in heat_unique_periods]) if heat_unique_periods else []

    # Heat-temp consistency: where Heat.period is non-NaN, heat_temp should also be non-NaN
    heat_period_notna = df["Heat.period"].notna()
    heat_temp_notna = df["heat_temp"].notna()
    heat_period_xor_temp = int((heat_period_notna != heat_temp_notna).sum())

    # Operational heatwave check: Temp >= 20 °C
    temp_ge20 = int((df["Temp"] >= 20).sum())
    # Days (unique by region) where temp >= 20
    region_day_ge20 = (
        df[df["Temp"] >= 20]
        .drop_duplicates(subset=["Region.code", "Month", "Day"])
        .groupby("Region.code")
        .size()
        .to_dict()
    )

    # Sex distribution
    sex_counts = df["Sex"].value_counts().to_dict()

    # Inspect the undocumented x and x.1 columns
    # Hypothesis: x = "MM-DD    <climatology temperature>"; x.1 = "MM-DD"
    x_sample = df["x"].dropna().unique()[:5].tolist()
    x1_sample = df["x.1"].dropna().unique()[:5].tolist()

    # Is `x` identical across years for the same MM-DD?  (sanity probe — will compare cross-year later)
    # Number of unique values of x: should equal number of calendar days
    n_unique_x = df["x"].nunique()
    n_unique_x1 = df["x.1"].nunique()

    # Date coverage — reconstruct date from Date.Temp and check completeness
    dates = pd.to_datetime(df["Date.Temp"], errors="coerce")
    n_bad_dates = int(dates.isna().sum())
    date_min = str(dates.min()) if dates.notna().any() else None
    date_max = str(dates.max()) if dates.notna().any() else None
    unique_dates = dates.dt.date.nunique()

    return {
        "year": year,
        "file": f"{year}.csv",
        "raw_columns": list(raw.columns),
        "n_rows": n_rows,
        "n_rows_expected": expected_rows,
        "n_rows_match_expected": n_rows == expected_rows,
        "n_columns": len(df.columns),
        "regions": {
            "n_unique": int(df["Region.code"].nunique()),
            "codes_present": sorted(df["Region.code"].unique().tolist()),
            "names_present": sorted(df["Region.name"].unique().tolist()),
            "numeric_ids_present": sorted(df["Region"].unique().tolist()),
        },
        "days_per_region_sex": {
            "min": days_per_unit_min,
            "max": days_per_unit_max,
            "expected": 366 if year % 4 == 0 else 365,
        },
        "missingness_pct": miss_pct,
        "missingness_count": {k: int(v) for k, v in miss.items()},
        "temp_stats": {k: (float(v) if pd.notna(v) else None) for k, v in temp_stats.items()},
        "deaths_stats": {k: (float(v) if pd.notna(v) else None) for k, v in deaths_stats.items()},
        "heat_period": {
            "rows_with_period": int(heat_rows),
            "periods_present": heat_periods_sorted,
            "n_distinct_periods": len(heat_periods_sorted),
            "heat_period_heat_temp_inconsistency": heat_period_xor_temp,
        },
        "temp_ge_20_rows": temp_ge20,
        "region_days_ge_20": region_day_ge20,
        "sex_distribution": {int(k): int(v) for k, v in sex_counts.items()},
        "undocumented_cols": {
            "x_sample": x_sample,
            "x.1_sample": x1_sample,
            "x_n_unique": int(n_unique_x),
            "x.1_n_unique": int(n_unique_x1),
        },
        "dates": {
            "n_bad_dates": n_bad_dates,
            "date_min": date_min,
            "date_max": date_max,
            "n_unique_dates": int(unique_dates),
            "expected_unique_dates": 366 if year % 4 == 0 else 365,
        },
    }


def cross_year_checks() -> dict:
    """Checks that span all years — column stability, x-column identity across years."""
    schemas = {}
    x_per_year = {}
    for y in CANONICAL_YEARS:
        df = load_year(y)
        # Original-name reload to capture true header variant
        raw_cols = list(pd.read_csv(DATA_DIR / f"{y}.csv", nrows=0).columns)
        schemas[y] = raw_cols
        x_per_year[y] = set(df["x"].dropna().unique().tolist())

    # Are x values identical across years? Take intersection vs union.
    union = set().union(*x_per_year.values())
    intersection = set.intersection(*x_per_year.values()) if x_per_year else set()
    identity_share = len(intersection) / len(union) if union else 0

    # Sample to display
    sample_x_2014 = sorted(x_per_year[2014])[:3]
    sample_x_2019 = sorted(x_per_year[2019])[:3]

    return {
        "header_variants": schemas,
        "x_column_cross_year_identity": {
            "union_size": len(union),
            "intersection_size": len(intersection),
            "identity_fraction": round(identity_share, 4),
            "sample_2014": sample_x_2014,
            "sample_2019": sample_x_2019,
        },
    }


def main() -> int:
    out = {"by_year": {}, "cross_year": cross_year_checks()}
    for y in CANONICAL_YEARS:
        out["by_year"][y] = profile_year(y)
    # Totals
    total_rows = sum(out["by_year"][y]["n_rows"] for y in CANONICAL_YEARS)
    out["totals"] = {
        "n_canonical_files": len(CANONICAL_YEARS),
        "total_rows": total_rows,
        "expected_total_per_prompt": "approx 51240 (prompt §3)",
    }
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
