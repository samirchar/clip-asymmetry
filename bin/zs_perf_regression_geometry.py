#!/usr/bin/env python
"""Is zero-shot performance predictable from embedding geometry?

Fits simple linear models of average zero-shot accuracy on embedding geometry
(alignment, image uniformity, text uniformity; lower is better for all) for the
CC12M models -- the architecture grid PLUS the modality-specific weight-decay
(degraded/fixed) runs -- at the final-epoch checkpoint. The geometry probe set is
selectable (MSCOCO or Flickr30k) via --probe.

Zero-shot is well predicted by a simple 2-term model, ZS ~ image_uniformity +
alignment; adding text uniformity does not help. The weight-decay runs fall on the
same relation as the grid. This is a predictive correlation, not a causal claim.

The script prints a per-predictor-set fit summary (adjusted R^2 and leave-one-out
R^2) as JSON and writes the tidy per-model table (CSV) the geometry figures are
drawn from.

Usage:
    python bin/analyze_geometry_prediction.py [--probe mscoco_captions|flickr30k] [--out-dir data/analysis] [--out PATH]
"""
import argparse
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.analysis import load_data, ANALYSIS_DIR, loocv_r2, _base_query, json_num, dump_json

METRICS = ["alignment", "image_uniformity", "text_uniformity"]

# Candidate predictor sets compared in the console fit summary.
PREDICTOR_SETS = [
    ("Image uniformity", ["image_uniformity"]),
    ("Alignment", ["alignment"]),
    ("Text uniformity", ["text_uniformity"]),
    ("Image uniformity + alignment", ["image_uniformity", "alignment"]),
    ("All three", ["image_uniformity", "alignment", "text_uniformity"]),
]


def _cols(probe):
    return [f"{m}_{probe}" for m in METRICS]


@lru_cache(maxsize=None)
def _load_models(probe):
    """CC12M models (grid + weight-decay) with avg zero-shot and this probe's geometry.

    Geometry (grid and weight-decay alike) is pre-merged into the benchmarks table by
    ``load_data``; here we just aggregate per run and label grid vs weight-decay runs
    via the architecture-grid query.
    """
    data = load_data()
    b, ct = data["benchmarks"], data["CHECKPOINT_TYPE"]
    geo = _cols(probe)
    zs = b[(b.super_task == "zeroshot") & (b.pretrain_dataset == "cc12m")
           & (b.checkpoint_type == ct)]
    grp = zs.groupby("run_name")
    df = grp[["vision_name", "text_name", "text_params", "vision_params", *geo]].first()
    df["zs"] = grp["preferred_metric_value"].mean() * 100
    df = df.dropna(subset=geo)
    grid_runs = b.query(_base_query(data, "cc12m", "zeroshot"))["run_name"].unique()
    df["kind"] = np.where(df.index.isin(grid_runs), "grid", "weight_decay")
    return df.reset_index()


def _fit(df, cols):
    """(adj R^2, LOOCV R^2) for avg zero-shot ~ `cols` via OLS."""
    model = sm.OLS(df["zs"].to_numpy(), sm.add_constant(df[cols].to_numpy())).fit()
    return model.rsquared_adj, loocv_r2(model)


def _lovo_predictions(df, probe):
    """Leave-one-vision-out predictions of avg zero-shot for the grid models.

    For each vision encoder, fit the 2-predictor geometry model (image uniformity
    + alignment) on all OTHER visions (grid plus their weight-decay runs), then
    predict the held-out vision's grid curve. Returns the grid rows with an added
    ``zs_pred_lovo`` column. This is the data behind the leave-one-vision-out
    figure; the figure itself is drawn in bin/generate_figures.py.
    """
    VIS = ["Atto", "Tiny", "Base", "Giant", "Colossal"]
    cols = [f"image_uniformity_{probe}", f"alignment_{probe}"]  # chosen 2-var model
    d = df.dropna(subset=cols + ["zs", "vision_name"]).set_index("run_name").copy()
    d["zs_pred_lovo"] = np.nan
    for v in VIS:
        train = d[d.vision_name != v]
        test = d[(d.vision_name == v) & (d.kind == "grid")]
        if not len(test):
            continue
        model = sm.OLS(train["zs"].to_numpy(),
                       sm.add_constant(train[cols].to_numpy())).fit()
        d.loc[test.index, "zs_pred_lovo"] = model.predict(
            sm.add_constant(test[cols].to_numpy(), has_constant="add"))
    return d[d.kind == "grid"].reset_index()


def _export_figure_data(out_dir):
    """Write the tidy per-model table that the geometry figures are drawn from.

    One row per (probe, model): encoder names/params, kind (grid vs weight_decay),
    image uniformity, alignment, observed avg zero-shot, and the leave-one-vision-out
    prediction (grid models only). Consumed by bin/generate_figures.py.
    """
    frames = []
    for probe in ("mscoco_captions", "flickr30k"):
        df = _load_models(probe)
        lovo = _lovo_predictions(df, probe)[["run_name", "zs_pred_lovo"]]
        out = df.merge(lovo, on="run_name", how="left").rename(columns={
            f"image_uniformity_{probe}": "image_uniformity",
            f"alignment_{probe}": "alignment",
            "zs": "zs_obs",
        })
        out["probe"] = probe
        frames.append(out[["probe", "run_name", "vision_name", "text_name",
                           "text_params", "vision_params", "kind", "image_uniformity",
                           "alignment", "zs_obs", "zs_pred_lovo"]])
    path = Path(out_dir) / "geometry_prediction_figure_data.csv"
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", default="mscoco_captions",
                    choices=["mscoco_captions", "flickr30k"],
                    help="Probe set the fit summary is computed for.")
    ap.add_argument("--out-dir", default=str(ANALYSIS_DIR),
                    help="Directory for the figure-data CSV.")
    ap.add_argument("--out", default=str(ANALYSIS_DIR / "geometry_prediction_fits.json"),
                    help="Path to persist the JSON fit summary (also printed to stdout).")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    probe = args.probe

    df = _load_models(probe)
    results = {
        "probe": probe,
        "n_grid": int((df.kind == "grid").sum()),
        "n_weight_decay": int((df.kind == "weight_decay").sum()),
        "fits": {
            label: dict(zip(("adj_r2", "loocv_r2"),
                            (json_num(v) for v in _fit(df, [f"{c}_{probe}" for c in cols]))))
            for label, cols in PREDICTOR_SETS
        },
    }
    dump_json(results, args.out)

    csv_path = _export_figure_data(out)
    print(f"Wrote figure data: {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
