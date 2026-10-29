"""
Hadley CET v2.1.1.0 daily-mean integration module.

This module:
1. Parses the Met Office HadCET daily-mean text file.
2. Subsets to a specified date range (default 2009-01-01 to 2022-12-31, covering
   the §7.2 rolling baseline buffer + analysis years).
3. Provides a function that, given a panel CSV from ./data/Data,
   replaces the jittered `Temp` column with HadCET daily mean broadcast across
   all 20 (region × sex) rows for that calendar day.

Status: parser and integration logic complete. Awaiting upload of the full
HadCET file at ./data/meantemp_daily_totals.txt.

Source: https://www.metoffice.gov.uk/hadobs/hadcet/data/meantemp_daily_totals.txt
Version: v2.1.1.0 + Provisional (current as of February 2026)
Citation: Met Office (2025). HadCET v2.0.1.0 (current operational v2.1.1.0).
  NERC EDS Centre for Environmental Data Analysis.
  https://doi.org/10.5285/4b541f71524244c5a19f7d8321b5563d
Adjustment: Since 1974, mean-temperature series adjusted by -0.2 °C for urban
  warming bias (Parker et al., 1992; Legg et al., 2025).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

# Default location once the user uploads the file
DEFAULT_CET_PATH = Path("./data/meantemp_daily_totals.txt")


def parse_hadcet(
    path: Path = DEFAULT_CET_PATH,
    start_year: int = 2009,
    end_year: int = 2022,
) -> pd.DataFrame:
    """
    Parse the Met Office HadCET daily-mean text file.

    File format (verified by web_fetch of the live URL):
        Date        Value
        1772-01-01    3.2
        1772-01-02    2.0
        ...
        2026-XX-XX   YY.Y   <- provisional values

    Parameters
    ----------
    path : Path
        Location of meantemp_daily_totals.txt.
    start_year, end_year : int
        Inclusive bounds.  Default 2009-2022 supplies the §7.2 rolling-baseline
        buffer (5 yr) plus all 8 analysis years.

    Returns
    -------
    pd.DataFrame with columns:
        date       (pd.Timestamp, dtype datetime64[ns])
        temp_cet   (float, °C — daily mean Central England Temperature)
    """
    if not path.exists():
        raise FileNotFoundError(
            f"HadCET file not found at {path}. "
            "Upload meantemp_daily_totals.txt from "
            "https://www.metoffice.gov.uk/hadobs/hadcet/data/meantemp_daily_totals.txt"
        )

    rows = []
    line_re = re.compile(r"^(\d{4}-\d{2}-\d{2})\s+(-?\d+\.\d+)\s*$")
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("Date"):  # header
                continue
            m = line_re.match(line)
            if not m:
                # Skip silently — handles any trailing notes/blank lines
                continue
            date_s, val_s = m.groups()
            rows.append((date_s, float(val_s)))

    df = pd.DataFrame(rows, columns=["date_str", "temp_cet"])
    df["date"] = pd.to_datetime(df["date_str"])
    df = df.drop(columns=["date_str"])
    df = df[(df["date"].dt.year >= start_year) & (df["date"].dt.year <= end_year)]
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "temp_cet"]]


def validate_hadcet(
    cet: pd.DataFrame,
    start_year: int = 2009,
    end_year: int = 2022,
) -> dict:
    """
    Sanity-check the parsed CET frame:
    - Date completeness (no gaps).
    - Plausible range (-15 to +35 °C).
    - Known landmark days (2022-07-19 daily mean should be > 20 °C — the UK record day).
    """
    expected_days = sum(366 if pd.Timestamp(f"{y}-01-01").is_leap_year else 365
                        for y in range(start_year, end_year + 1))
    actual_days = len(cet)
    gaps = (
        pd.date_range(
            start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq="D"
        ).difference(cet["date"])
    )
    out_of_range = cet[(cet["temp_cet"] < -15) | (cet["temp_cet"] > 35)]

    # Landmark: 2022-07-19 was the day Coningsby hit 40.3 °C; the daily CET mean
    # should be unusually warm but not 40 °C (CET is a daily mean across the
    # tri-Manchester-London-Bristol box). Expect something in 22-28 °C range.
    landmark = cet[cet["date"] == pd.Timestamp("2022-07-19")]
    landmark_temp = float(landmark["temp_cet"].iloc[0]) if len(landmark) else None

    return {
        "expected_days": expected_days,
        "actual_days": actual_days,
        "n_gaps": len(gaps),
        "gaps_sample": [str(g.date()) for g in gaps[:5]],
        "n_out_of_range": len(out_of_range),
        "out_of_range_sample": out_of_range.head(5).to_dict(orient="records"),
        "landmark_2022_07_19_temp": landmark_temp,
        "year_range": (
            (cet["date"].min().date().isoformat(), cet["date"].max().date().isoformat())
            if len(cet) else (None, None)
        ),
    }


def replace_temp_with_cet(
    panel: pd.DataFrame,
    cet: pd.DataFrame,
    panel_date_col: str = "Date.Temp",
) -> pd.DataFrame:
    """
    Replace the jittered `Temp` column in a YYYY-CSV panel with HadCET daily
    mean, broadcast to every (region × sex) row of that calendar day.

    This is the "clean-break" remediation chosen for Option (c) — the original
    jittered values are NOT preserved (per user decision on 2026-05-27).

    The function preserves all other columns. It DOES handle the phantom-29-Feb
    rows: rows where the panel date is null are dropped, since they were
    template artifacts (see §B5 of data_audit.md).

    Parameters
    ----------
    panel : pd.DataFrame
        A single-year CSV already loaded with the standard YYYY schema:
        must contain at minimum [`Sex`, `Region.code`, `Month`, `Day`, `Temp`].
    cet : pd.DataFrame
        Output of parse_hadcet().
    panel_date_col : str
        Column holding the ISO date string in the panel. Default "Date.Temp"
        (matches the canonical YYYY files). For xYYYY files the same column
        holds "YYYY-MM-DD   T.T" — caller should split that before passing.

    Returns
    -------
    pd.DataFrame
        Panel with `Temp` replaced. Phantom-Feb-29 rows dropped.
    """
    panel = panel.copy()
    panel[panel_date_col] = pd.to_datetime(panel[panel_date_col], errors="coerce")
    # Drop phantom Feb-29 rows (per §B5 audit decision)
    n_before = len(panel)
    panel = panel.dropna(subset=[panel_date_col]).copy()
    n_dropped = n_before - len(panel)

    # Drop the existing jittered Temp column
    panel = panel.drop(columns=["Temp"])

    # Merge in canonical CET
    panel = panel.merge(
        cet.rename(columns={"date": panel_date_col, "temp_cet": "Temp"}),
        on=panel_date_col,
        how="left",
        validate="many_to_one",
    )

    # Sanity check: every remaining row should have a CET value
    n_missing = panel["Temp"].isna().sum()
    if n_missing:
        raise ValueError(
            f"After CET merge, {n_missing} rows have null Temp. "
            "Likely cause: CET file does not cover the panel's date range. "
            f"Panel range: {panel[panel_date_col].min()} to {panel[panel_date_col].max()}."
        )

    return panel, {"phantom_rows_dropped": n_dropped, "rows_remaining": len(panel)}


# ---- Self-test stub ----

if __name__ == "__main__":
    # When invoked directly, validate against whichever file is available.
    print("HadCET integration module — self-test")
    print(f"Looking for HadCET file at: {DEFAULT_CET_PATH}")
    if DEFAULT_CET_PATH.exists():
        cet = parse_hadcet(DEFAULT_CET_PATH)
        diag = validate_hadcet(cet)
        print("Parsed CET:")
        for k, v in diag.items():
            print(f"  {k}: {v}")
    else:
        print(
            f"  NOT FOUND. Upload the file to {DEFAULT_CET_PATH} and re-run.\n"
            "  Direct URL: "
            "https://www.metoffice.gov.uk/hadobs/hadcet/data/meantemp_daily_totals.txt"
        )
