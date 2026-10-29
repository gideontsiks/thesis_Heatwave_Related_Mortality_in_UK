"""Spatial statistics in numpy/scipy (no esda/spreg in sandbox).
Implements global Moran's I, Local Moran (LISA), Getis-Ord G*, bivariate Moran's I,
and ML spatial lag / error regression with LM diagnostics. Row-standardized W.
Validated against published formulae (Anselin 1995; Getis & Ord 1992; Anselin 1988).
N=10: every result is illustrative, not inferential at conventional power."""
import numpy as np
from scipy import stats, optimize, linalg

RNG = np.random.default_rng(42)


def _z(x):
    return (x - x.mean()) / x.std(ddof=0)


def morans_i(x, W, n_perm=999, seed=42):
    """Global Moran's I with permutation inference. W row-standardized (n x n)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); n = len(x)
    z = x - x.mean()
    S0 = W.sum()
    I = (n / S0) * (z @ W @ z) / (z @ z)
    EI = -1.0 / (n - 1)
    # permutation
    perms = np.empty(n_perm)
    for i in range(n_perm):
        zp = rng.permutation(z)
        perms[i] = (n / S0) * (zp @ W @ zp) / (zp @ zp)
    p_perm = (np.sum(np.abs(perms - EI) >= np.abs(I - EI)) + 1) / (n_perm + 1)
    z_score = (I - perms.mean()) / perms.std(ddof=1)
    return {"I": float(I), "EI": float(EI), "z_sim": float(z_score),
            "p_sim": float(p_perm), "perm_mean": float(perms.mean()),
            "perm_std": float(perms.std(ddof=1))}


def local_morans(x, W, n_perm=999, seed=42):
    """Local Moran's I (Anselin 1995 LISA) + HH/LL/HL/LH classification."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); n = len(x)
    z = x - x.mean()
    m2 = (z @ z) / n
    Wz = W @ z
    Ii = (z / m2) * Wz
    # conditional permutation
    p = np.empty(n); sims = np.empty((n, n_perm))
    for i in range(n):
        others = np.delete(np.arange(n), i)
        wi = W[i, others]
        for k in range(n_perm):
            zp = z[rng.permutation(others)]
            sims[i, k] = (z[i] / m2) * (wi @ zp)
    for i in range(n):
        ge = np.sum(sims[i] >= Ii[i]); le = np.sum(sims[i] <= Ii[i])
        p[i] = (min(ge, le) + 1) / (n_perm + 1)
    zz = _z(x); lag_z = W @ zz
    quad = np.empty(n, dtype=object)
    for i in range(n):
        if zz[i] > 0 and lag_z[i] > 0: quad[i] = "HH"
        elif zz[i] < 0 and lag_z[i] < 0: quad[i] = "LL"
        elif zz[i] > 0 and lag_z[i] < 0: quad[i] = "HL"
        else: quad[i] = "LH"
    return {"Ii": Ii, "p_sim": p, "quadrant": quad, "z": zz, "lag_z": lag_z}


def getis_ord_gstar(x, W, seed=42):
    """Getis-Ord G* (includes self). Returns z-scores (analytic)."""
    x = np.asarray(x, float); n = len(x)
    # G* uses binary/weighted W including diagonal = 1
    Wstar = W.copy()
    np.fill_diagonal(Wstar, 1.0)  # include self
    xbar = x.mean(); S = np.sqrt((x @ x) / n - xbar ** 2)
    Z = np.empty(n)
    for i in range(n):
        wi = Wstar[i]
        Wi = wi.sum()
        num = (wi @ x) - xbar * Wi
        den = S * np.sqrt((n * (wi @ wi) - Wi ** 2) / (n - 1))
        Z[i] = num / den if den > 0 else 0.0
    p = 2 * stats.norm.sf(np.abs(Z))
    return {"z_gstar": Z, "p": p}


def bivariate_morans(x, y, W, n_perm=999, seed=42):
    """Bivariate Moran's I: corr of x with spatial lag of y. I_xy = z_x' W z_y / (z_x'z_x)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float); y = np.asarray(y, float); n = len(x)
    zx = _z(x); zy = _z(y)
    S0 = W.sum()
    I = (n / S0) * (zx @ W @ zy) / (zx @ zx)
    perms = np.empty(n_perm)
    for k in range(n_perm):
        zyp = rng.permutation(zy)
        perms[k] = (n / S0) * (zx @ W @ zyp) / (zx @ zx)
    p = (np.sum(np.abs(perms) >= np.abs(I)) + 1) / (n_perm + 1)
    zscore = (I - perms.mean()) / perms.std(ddof=1)
    return {"I_bv": float(I), "z_sim": float(zscore), "p_sim": float(p)}


# ---------- Spatial regression ----------
def ols(X, y):
    beta, *_ = linalg.lstsq(X, y)
    resid = y - X @ beta
    n, p = X.shape
    s2 = resid @ resid / (n - p)
    cov = s2 * linalg.inv(X.T @ X)
    return beta, resid, cov, s2


def lm_tests(X, y, W):
    """Lagrange Multiplier tests for spatial lag vs error (Anselin 1988/1996)."""
    n = len(y)
    beta, e, cov, s2 = ols(X, y)
    # LM-error
    We = W @ e
    T = np.trace((W.T + W) @ W)
    LM_err = ((e @ We) / s2) ** 2 / T
    # LM-lag
    Wy = W @ y
    # numerator
    num = (e @ Wy) / s2
    XtXinv = linalg.inv(X.T @ X)
    M = np.eye(n) - X @ XtXinv @ X.T
    WXb = W @ (X @ beta)
    J = (WXb @ M @ WXb) / s2 + T
    LM_lag = num ** 2 / J
    return {
        "LM_error": float(LM_err), "p_LM_error": float(stats.chi2.sf(LM_err, 1)),
        "LM_lag": float(LM_lag), "p_LM_lag": float(stats.chi2.sf(LM_lag, 1)),
    }


def ml_spatial_lag(X, y, W):
    """ML spatial lag (SAR): y = rho W y + X beta + e. Concentrated likelihood over rho."""
    n = len(y)
    Wy = W @ y
    b_y, e_y, *_ = ols(X, y)
    b_wy, e_wy, *_ = ols(X, Wy)

    def negll(rho):
        e = e_y - rho * e_wy
        s2 = (e @ e) / n
        sign, logdet = np.linalg.slogdet(np.eye(n) - rho * W)
        return -(-(n / 2) * np.log(2 * np.pi * s2) - (e @ e) / (2 * s2) + logdet)

    res = optimize.minimize_scalar(negll, bounds=(-0.99, 0.99), method="bounded")
    rho = res.x
    beta = b_y - rho * b_wy
    e = y - rho * Wy - X @ beta
    s2 = (e @ e) / n
    ll = -res.fun
    k = X.shape[1] + 2
    aic = 2 * k - 2 * ll
    return {"rho": float(rho), "beta": beta.tolist(), "sigma2": float(s2),
            "loglik": float(ll), "aic": float(aic)}


def ml_spatial_error(X, y, W):
    """ML spatial error (SEM): y = X beta + u, u = lambda W u + e."""
    n = len(y)

    def negll(lam):
        A = np.eye(n) - lam * W
        Xs = A @ X; ys = A @ y
        b, *_ = linalg.lstsq(Xs, ys)
        e = ys - Xs @ b
        s2 = (e @ e) / n
        sign, logdet = np.linalg.slogdet(A)
        return -(-(n / 2) * np.log(2 * np.pi * s2) - (e @ e) / (2 * s2) + logdet)

    res = optimize.minimize_scalar(negll, bounds=(-0.99, 0.99), method="bounded")
    lam = res.x
    A = np.eye(n) - lam * W
    Xs = A @ X; ys = A @ y
    beta, *_ = linalg.lstsq(Xs, ys)
    e = ys - Xs @ beta
    s2 = (e @ e) / n
    ll = -res.fun
    k = X.shape[1] + 2
    aic = 2 * k - 2 * ll
    return {"lambda": float(lam), "beta": beta.tolist(), "sigma2": float(s2),
            "loglik": float(ll), "aic": float(aic)}


if __name__ == "__main__":
    # Validate Moran's I against a tiny known case: perfectly clustered pattern.
    W = np.array([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]], float)
    W = W / W.sum(1, keepdims=True)
    x = np.array([1.0, 1.0, 5.0, 5.0])  # spatially clustered
    r = morans_i(x, W, n_perm=999)
    print("Clustered x Moran's I (should be positive):", round(r["I"], 3), "p=", round(r["p_sim"], 3))
    x2 = np.array([1.0, 5.0, 1.0, 5.0])  # checkerboard
    r2 = morans_i(x2, W, n_perm=999)
    print("Checkerboard x Moran's I (should be negative):", round(r2["I"], 3))
