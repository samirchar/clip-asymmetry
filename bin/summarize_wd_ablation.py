#!/usr/bin/env python
"""Report the WD-ablation disclosure numbers (Base--Giant and Giant--Giant).

Reports the average zero-shot performance of every weight-decay ablation run --
all three arms (text-only, vision-only, joint) at both raised levels, plus the
shared baseline -- for both the Base--Giant and Giant--Giant configurations.
This is the full-numbers disclosure behind the two WD-ablation figures (the
Giant appendix figure omits the joint arm for clarity; its numbers live here).

Avg zero-shot is computed identically to the figures: mean over the zero-shot
preferred-metric rows (retrieval counted as two rows), final-epoch checkpoint.

Prints the results as JSON: for each configuration, one record per arm with the
per-encoder weight decays, average zero-shot, and the run count.

Usage:
    python bin/make_wd_disclosure_table.py [--out PATH]
"""
import argparse
import re

from src.analysis import (load_data, _load_run_configs, _wd_effective, _wd_classify,
                          json_num, dump_json, ANALYSIS_DIR)

_CONFIGS = [("Base--Giant", "cc12m_wd_ablation"),
            ("Giant--Giant", "cc12m_wd_ablation_giant")]
# (scenario, level, text_wd, vision_wd); base wd = 0.5.
_ARMS = [
    ("baseline", 0.5, 0.5, 0.5),
    ("text", 0.75, 0.75, 0.5),
    ("text", 1.0, 1.0, 0.5),
    ("vision", 0.75, 0.5, 0.75),
    ("vision", 1.0, 0.5, 1.0),
    ("joint", 0.75, 0.75, 0.75),
    ("joint", 1.0, 1.0, 1.0),
]


def _avg_zs_by_cell(data, tag, pretrain_dataset="cc12m"):
    """Return {(scenario, level): (avg_zs, n_rows)} for a WD-ablation run family."""
    bench = data["benchmarks"]
    ct = data["CHECKPOINT_TYPE"]
    tag_mask = bench["wandb_tags"].str.contains(
        rf"(?:^|\|){re.escape(tag)}(?:\||$)", regex=True, na=False)
    abl = bench[tag_mask].query(
        f"pretrain_dataset == '{pretrain_dataset}' and super_task == 'zeroshot' "
        f"and checkpoint_type == '{ct}'")
    abl = abl[~abl["run_name"].str.contains("_1772044155", na=False)]
    avg_zs = abl.groupby("run_name")["preferred_metric_value"].mean().mul(100)
    n_rows = abl.groupby("run_name").size()
    configs = _load_run_configs()
    out = {}
    for run_name, zs in avg_zs.items():
        eff_t, eff_v = _wd_effective(run_name, configs.get(run_name))
        scenario, level = _wd_classify(eff_t, eff_v)
        if scenario is None:
            continue
        out[(scenario, round(float(level), 2))] = (zs, int(n_rows[run_name]))
    return out


def _results(data):
    out = {}
    for label, tag in _CONFIGS:
        cells = _avg_zs_by_cell(data, tag)
        rows = []
        for scen, lvl, twd, vwd in _ARMS:
            zs, n = cells.get((scen, round(lvl, 2)), (None, None))
            rows.append({"arm": scen, "lambda_t": twd, "lambda_v": vwd,
                         "avg_zs": json_num(zs), "n": n})
        out[label] = rows
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ANALYSIS_DIR / "wd_ablation.json"),
                    help="Path to persist the JSON results (also printed to stdout).")
    args = ap.parse_args()
    dump_json(_results(load_data()), args.out)


if __name__ == "__main__":
    main()
