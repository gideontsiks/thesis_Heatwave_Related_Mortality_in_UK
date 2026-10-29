"""
gis_analysis.py — Spatial statistics for the UK heatwave-mortality thesis (Part C).

Runs the §6 spatial-statistics suite on the N=10 NUTS-1 panel using numpy/scipy
(esda/spreg unavailable in sandbox; implementations in spatial_stats.py validated
against published formulae). Weights matrices from polygons/spatial_weights.json.

⚠ N=10 CAVEAT: every result below is illustrative, not inferential at conventional
power. Permutation pseudo-p-values have limited resolution with 10 areal units and
spatial-regression coefficients have wide sampling variability. Stated in every
reported result per project brief §6.7.

Methods:
  6.1 Global Moran's I — annual + pooled excess heat-mortality, GSI, density.
  6.2 LISA (Local Moran) — pooled excess; HH/LL/HL/LH classification.
  6.3 Getis-Ord G* — hot/cold spots.
  6.4 Bivariate Moran's I — GSI×mortality, density×mortality (spatial H1/H2 tests).
  6.5 Spatial regression — OLS + LM diagnostics -> SLM or SEM.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from spatial_stats import (  # noqa: E402
    morans_i, local_morans, getis_ord_gstar, bivariate_morans,
    lm_tests, ml_spatial_lag, ml_spatial_error, ols,
)

POLY = Path("./polygons/spatial_weights.json")
VARS = Path("./covariates/spatial_analysis_vars.csv")
YEARS = Path("./covariates/spatial_excess_by_year.csv")

REGION_NAMES = ["North East", "North West", "Yorkshire and The Humber", "East Midlands",
                "West Midlands", "East of England", "London", "South East", "South West", "Wales"]


def load_W(key="queen"):
    d = json.load(open(POLY))
    return np.array(d[key]["W_rowstd"]), d[key]["idx"]


def main():
    Wq, idx = load_W("queen")
    Wr, _ = load_W("rook")
    Wk, _ = load_W("knn3")
    d = pd.read_csv(VARS).sort_values("region_idx").reset_index(drop=True)
    yr = pd.read_csv(YEARS).sort_values("region_idx").reset_index(drop=True)

    excess = d["excess_per_100k"].values
    gsi = d["greenspace_m2_per_capita_fit"].values
    dist = d["avg_dist_park_m"].values
    builtup = d["pct_builtup"].values

    results = {"_caveat": "N=10 NUTS-1 regions; spatial statistics illustrative, not inferential at conventional power."}

    # ---- 6.1 Global Moran's I ----
    print("=" * 70); print("6.1 GLOBAL MORAN'S I (Queen weights, 999 perms)"); print("=" * 70)
    g = {}
    for name, v in [("pooled_excess", excess), ("greenspace", gsi),
                    ("dist_park", dist), ("pct_builtup", builtup)]:
        r = morans_i(v, Wq)
        g[name] = r
        print(f"  {name:16s}: I={r['I']:+.3f}  E[I]={r['EI']:+.3f}  z={r['z_sim']:+.2f}  p={r['p_sim']:.3f}")
    # annual excess
    print("  -- annual excess heat-mortality --")
    g_annual = {}
    for col in [c for c in yr.columns if c != "region_idx"]:
        r = morans_i(yr[col].values, Wq)
        g_annual[col] = r
        print(f"    {col}: I={r['I']:+.3f}  z={r['z_sim']:+.2f}  p={r['p_sim']:.3f}")
    results["moran_global"] = g
    results["moran_annual"] = g_annual

    # robustness across weights
    print("  -- pooled excess across weights matrices --")
    rob = {}
    for wn, W in [("queen", Wq), ("rook", Wr), ("knn3", Wk)]:
        r = morans_i(excess, W)
        rob[wn] = r
        print(f"    {wn:6s}: I={r['I']:+.3f}  p={r['p_sim']:.3f}")
    results["moran_robustness"] = rob

    # ---- 6.2 LISA ----
    print("\n" + "=" * 70); print("6.2 LOCAL MORAN'S I (LISA) — pooled excess"); print("=" * 70)
    lisa = local_morans(excess, Wq)
    lisa_out = []
    for i, nm in enumerate(REGION_NAMES):
        sig = lisa["p_sim"][i] < 0.05
        print(f"  {nm:26s}: Ii={lisa['Ii'][i]:+.3f}  quad={lisa['quadrant'][i]}  "
              f"p={lisa['p_sim'][i]:.3f}  {'*' if sig else ''}")
        lisa_out.append({"region": nm, "Ii": float(lisa["Ii"][i]),
                         "quadrant": lisa["quadrant"][i], "p": float(lisa["p_sim"][i]),
                         "significant": bool(sig)})
    results["lisa"] = lisa_out

    # ---- 6.3 Getis-Ord G* ----
    print("\n" + "=" * 70); print("6.3 GETIS-ORD G* — pooled excess"); print("=" * 70)
    go = getis_ord_gstar(excess, Wq)
    go_out = []
    for i, nm in enumerate(REGION_NAMES):
        z = go["z_gstar"][i]
        lvl = ("hot 99%" if z > 2.58 else "hot 95%" if z > 1.96 else "hot 90%" if z > 1.65
               else "cold 99%" if z < -2.58 else "cold 95%" if z < -1.96 else "cold 90%" if z < -1.65
               else "ns")
        print(f"  {nm:26s}: G*z={z:+.2f}  {lvl}")
        go_out.append({"region": nm, "z_gstar": float(z), "level": lvl})
    results["getis_ord"] = go_out

    # ---- 6.4 Bivariate Moran's I (spatial H1 / H2) ----
    print("\n" + "=" * 70); print("6.4 BIVARIATE MORAN'S I (spatial H1/H2 tests)"); print("=" * 70)
    bv = {}
    # H1: does LOW green space cluster with HIGH mortality? -> negative I expected
    bv["greenspace_x_excess"] = bivariate_morans(gsi, excess, Wq)
    bv["distpark_x_excess"] = bivariate_morans(dist, excess, Wq)
    # H2: does HIGH density cluster with HIGH mortality? -> positive I expected
    bv["builtup_x_excess"] = bivariate_morans(builtup, excess, Wq)
    for k, r in bv.items():
        print(f"  {k:22s}: I_bv={r['I_bv']:+.3f}  z={r['z_sim']:+.2f}  p={r['p_sim']:.3f}")
    print("  H1 (green space): negative I_bv => low green space clusters with high mortality.")
    print("  H2 (% built-up):  positive I_bv => high density clusters with high mortality.")
    results["bivariate_moran"] = bv

    # ---- 6.5 Spatial regression ----
    print("\n" + "=" * 70); print("6.5 SPATIAL REGRESSION (excess ~ greenspace + builtup)"); print("=" * 70)
    # Standardize predictors for comparability
    def z(v): return (v - v.mean()) / v.std(ddof=0)
    X = np.column_stack([np.ones(10), z(gsi), z(builtup)])
    y = z(excess)
    b_ols, e_ols, cov_ols, s2 = ols(X, y)
    se_ols = np.sqrt(np.diag(cov_ols))
    print(f"  OLS:  intercept={b_ols[0]:+.3f}  GSI={b_ols[1]:+.3f} (se {se_ols[1]:.3f})  "
          f"builtup={b_ols[2]:+.3f} (se {se_ols[2]:.3f})")
    r2_ols = 1 - (e_ols @ e_ols) / ((y - y.mean()) @ (y - y.mean()))
    print(f"        R2={r2_ols:.3f}")
    lm = lm_tests(X, y, Wq)
    print(f"  LM-lag={lm['LM_lag']:.3f} (p={lm['p_LM_lag']:.3f})  "
          f"LM-error={lm['LM_error']:.3f} (p={lm['p_LM_error']:.3f})")
    slm = ml_spatial_lag(X, y, Wq)
    sem = ml_spatial_error(X, y, Wq)
    print(f"  SLM:  rho={slm['rho']:+.3f}  GSI={slm['beta'][1]:+.3f}  builtup={slm['beta'][2]:+.3f}  AIC={slm['aic']:.2f}")
    print(f"  SEM:  lambda={sem['lambda']:+.3f}  GSI={sem['beta'][1]:+.3f}  builtup={sem['beta'][2]:+.3f}  AIC={sem['aic']:.2f}")
    results["spatial_regression"] = {
        "ols_beta": b_ols.tolist(), "ols_se": se_ols.tolist(), "ols_r2": float(r2_ols),
        "lm_tests": lm, "slm": slm, "sem": sem,
    }

    json.dump(results, open("./gis_results.json", "w"), indent=2, default=float)
    print("\nSaved gis_results.json")
    print("\n⚠ ALL RESULTS: N=10 regions — illustrative, not inferential at conventional power.")
    return results


if __name__ == "__main__":
    main()
