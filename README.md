# Heat-related mortality in England and Wales: analysis code and GIS data

Replication code and derived spatial data for the spatiotemporal analysis of
heat-related mortality in England and Wales, 2014 to 2022 (nine English regions
and Wales; 2020 excluded). The repository contains the calculation code, the
spatial layers and the harmonised analysis panel needed to reproduce the
temporal and spatial results and to extend the study.

## What is here

Calculation code:
- `glmtools.py`: natural-spline basis and a quasi-Poisson IRLS fitter used by the models.
- `spatial_stats.py`: Moran's I, local Moran's I (LISA), Getis-Ord Gi*, and bivariate Moran's I, with permutation inference.
- `cet_integration.py`: parser for the HadCET daily-mean series.
- `data_prep.py`: harmonises the raw per-year mortality files into the analysis panel.
- `audit.py`: the pre-analysis data audit (year-by-year structural checks).
- `analysis.py`: the temporal frame, a distributed-lag non-linear model (DLNM) and supporting GLMs; writes `analysis_results.json`.
- `gis_analysis.py`: the spatial frame (global and local Moran's I, Getis-Ord Gi*, bivariate Moran's I, OLS and spatial-lag / spatial-error regression); writes `gis_results.json`.
- `render_thesis_figures.py`: regenerates the figures and maps.

Data:
- `panel_harmonized.csv`: the harmonised region-by-sex-by-day panel (58,420 rows).
- `analysis_results.json`, `gis_results.json`: the computed results, for checking.
- `covariates/`: regional covariates, the per-year excess table, and population.
- `polygons/`: the NUTS-1 / ITL-1 region boundaries and the spatial-weights matrix.
- `data/`: where the raw inputs go (not redistributed here; see `data/README.md`).

## Requirements

Python 3.10 or newer. Install the dependencies:

```
pip install -r requirements.txt
```

## Reproduce the results

The harmonised panel is provided, so the temporal and spatial frames run directly:

```
python analysis.py
python gis_analysis.py
python render_thesis_figures.py
```

`analysis.py` prints the minimum-mortality temperature (18.3 C), the cumulative
heat relative risk (1.045, 95% CI 1.031 to 1.058) and the cold relative risk
(1.385), and writes `analysis_results.json`. `gis_analysis.py` prints the global
and annual Moran's I, the LISA classification (West Midlands High-High cluster,
p = 0.040), the Getis-Ord hot spots and the bivariate H1 and H2 tests, and writes
`gis_results.json`. `render_thesis_figures.py` writes the figures and maps to
`thesis_figures/`.

To rebuild the panel from the raw inputs, place the per-year mortality files in
`data/Data/` and the HadCET series in `data/meantemp_daily_totals.txt`, then run
`python data_prep.py` (the audit is run with `python audit.py`). The raw inputs
are not included here; see `data/README.md`.

## Annual spatial statistics

Excess deaths on each heat day are observed deaths minus an expected value equal
to the mean count in the same region, sex and week of the year across the panel
years, summed over heat days and divided by mid-year population. A heat day is a
day within a run of at least three consecutive days with a daily-mean temperature
at or above 20 C. The annual Moran's I is defined only for the six years that
contain at least one heat day (2014, 2017, 2018, 2019, 2021 and 2022); 2015 and
2016 have no heat days under this definition, and 2020 is excluded from the panel.

## Data provenance and licences

- Temperature: HadCET v2.1.0.0 daily-mean Central England Temperature (Met Office Hadley Centre, 2024), Open Government Licence v3.0.
- Public green space: OS Open Greenspace (Ordnance Survey, 2024), Open Government Licence v3.0.
- Built-up land: OS Open Built Up Areas (Ordnance Survey, 2024), Open Government Licence v3.0.
- Population: ONS mid-year population estimates, Open Government Licence v3.0.
- Region boundaries: ONS December 2022 generalised clipped boundaries, Open Government Licence v3.0.
- Mortality: daily death counts by region and sex for England and Wales, originating from Office for National Statistics death registrations. Confirm the exact access route and licence before relying on the included `panel_harmonized.csv`; if redistribution is not permitted, remove the panel and rebuild it from the raw files with `data_prep.py`.

## Related outputs

- Thesis: Spatiotemporal analysis of heatwave-related mortality in the United Kingdom (England and Wales), HSE University, Moscow, 2024 (revised 2025). https://doi.org/10.5281/zenodo.23736575
- A journal article derived from this work is in preparation.

## Licence

The code in this repository is released under the MIT Licence (see `LICENSE`).
The data files carry the licences of their original sources, as listed above.
