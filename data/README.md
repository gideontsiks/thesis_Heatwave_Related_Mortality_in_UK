# Raw inputs

The analysis panel in the repository root (panel_harmonized.csv) is the starting
point for the temporal and spatial frames, so most users do not need the raw
inputs. The harmonisation scripts (data_prep.py, audit.py) are provided for
transparency and for rebuilding the panel from scratch.

To rebuild from raw inputs, place the following here and run data_prep.py:

- Data/2014.csv ... Data/2022.csv : the per-year mortality files (one row per
  region, sex and day), as supplied for the study. These originate from Office
  for National Statistics death registrations for England and Wales.
- meantemp_daily_totals.txt : the HadCET v2.1.0.0 daily-mean Central England
  Temperature series (Met Office Hadley Centre, 2024).

These raw files are not redistributed in this repository. The mortality files in
particular should only be shared if their licence permits it.
