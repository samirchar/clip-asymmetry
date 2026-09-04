"""Regress average zero-shot performance on encoder parameter counts (CC12M grid).

Compares five functional forms relating average zero-shot accuracy to the vision
and text encoder parameter counts -- from a single linear-log term in the total
parameters up to the saturating-ceiling model with a capacity-mismatch penalty
(Eq. ref:ceiling) -- reporting each model's R^2, adjusted R^2 and leave-one-out
R^2, the Spearman correlation of total params vs. average zero-shot, and the
featured ceiling-model coefficients. Companion to zs_perf_regression_geometry.py,
which regresses on embedding geometry instead of parameter counts.

Prints the results as JSON and persists them to --out. No LaTeX, no functional-form
strings.

Usage:
    python bin/zs_perf_regression_params.py [--out PATH]
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit
import statsmodels.api as sm

from src.analysis import load_data, _base_query, loocv_r2, json_num, dump_json, ANALYSIS_DIR


def _results(data):
    """Scaling model comparison + Spearman + ceiling-model coefficients."""
    q_data = data["benchmarks"].query(_base_query(data, "cc12m", "zeroshot"))
    mp = (q_data.groupby(["model_fullname", "vision_params", "text_params", "total_params"])
          .agg(avg_perf=("preferred_metric_value", "mean")).reset_index())

    r, p = scipy_stats.spearmanr(mp["total_params"], mp["avg_perf"])

    # ── Linear-log OLS models  ──
    m1 = sm.OLS(mp["avg_perf"], sm.add_constant(np.log10(mp["total_params"]))).fit()

    X2 = pd.DataFrame(sm.add_constant(np.column_stack([
        np.log10(mp["vision_params"]), np.log10(mp["text_params"])])),
        columns=["const", "log_vision", "log_text"])
    m2 = sm.OLS(mp["avg_perf"], X2).fit()
    
    # Mean-centre the logs so the main effects stay interpretable; adj R^2 and the
    # interaction coefficient are unchanged by centring.
    lv = np.log10(mp["vision_params"]); lt = np.log10(mp["text_params"])
    lv_c = lv - lv.mean(); lt_c = lt - lt.mean()
    X3 = pd.DataFrame(sm.add_constant(np.column_stack([lv_c, lt_c, lv_c * lt_c])),
                      columns=["const", "log_vision", "log_text", "log_vision_x_text"])
    m3 = sm.OLS(mp["avg_perf"], X3).fit()

    # ── Nonlinear saturating-ceiling models (params in millions), with LOOCV ──
    nv_m = mp["vision_params"].to_numpy() / 1e6
    nt_m = mp["text_params"].to_numpy() / 1e6
    yv = mp["avg_perf"].to_numpy()
    n = len(yv)

    def m4_fn(X, a, b, kt):  # vision log-linear ceiling x saturating text
        return (a + b * np.log10(X[0])) * (1 - np.exp(-kt * X[1]))

    def m5_fn(X, a, b, kt, c):  # + Gaussian capacity-mismatch penalty (Eq. ceiling)
        dv = np.log10(X[0]) - np.log10(X[1])
        return (a + b * np.log10(X[0])) * (1 - np.exp(-kt * X[1])) * np.exp(-c * dv ** 2)

    def _nls_quality(fn, p0, k):
        """Return (coeffs, R^2, adj R^2, LOOCV R^2) for a nonlinear-least-squares fit."""
        pars = curve_fit(fn, (nv_m, nt_m), yv, p0=p0, maxfev=200000)[0]
        r2 = 1 - ((yv - fn((nv_m, nt_m), *pars)) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        adj = 1 - (1 - r2) * (n - 1) / (n - k)
        pred = np.empty(n)
        for i in range(n):
            mask = np.ones(n, bool); mask[i] = False
            pi = curve_fit(fn, (nv_m[mask], nt_m[mask]), yv[mask], p0=p0, maxfev=200000)[0]
            pred[i] = fn((nv_m[i:i + 1], nt_m[i:i + 1]), *pi)[0]
        loocv = 1 - ((yv - pred) ** 2).sum() / ((yv - yv.mean()) ** 2).sum()
        return pars, r2, adj, loocv

    _, r2_4, adj4, loocv4 = _nls_quality(m4_fn, [0.2, 0.04, 0.5], 3)
    p5, r2_5, adj5, loocv5 = _nls_quality(m5_fn, [0.2, 0.04, 0.5, 0.05], 4)

    def _row(name, k, r2, adj, loocv):
        return {"model": name, "n_params": k, "r2": json_num(r2),
                "adj_r2": json_num(adj), "loocv_r2": json_num(loocv)}

    return {
        "spearman_total_params": {"r": json_num(r), "p": json_num(p)},
        "model_comparison": [
            _row("linear-log: total params", 2, m1.rsquared, m1.rsquared_adj, loocv_r2(m1)),
            _row("linear-log: vision + text", 3, m2.rsquared, m2.rsquared_adj, loocv_r2(m2)),
            _row("linear-log: vision + text + interaction", 4, m3.rsquared, m3.rsquared_adj, loocv_r2(m3)),
            _row("saturating ceiling", 3, r2_4, adj4, loocv4),
            _row("saturating ceiling with mismatch penalty", 4, r2_5, adj5, loocv5),
        ],
        "ceiling_model_coefficients": {
            "beta0": json_num(p5[0]), "beta1": json_num(p5[1]),
            "k_text": json_num(p5[2]), "c_mismatch": json_num(p5[3]),
        },
    }


def main():
    ap = argparse.ArgumentParser(description="Regress zero-shot performance on encoder params.")
    ap.add_argument("--out", default=str(ANALYSIS_DIR / "zs_perf_regression_params.json"),
                    help="Path to persist the JSON results (also printed to stdout).")
    args = ap.parse_args()
    dump_json(_results(load_data()), args.out)


if __name__ == "__main__":
    main()
