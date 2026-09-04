#!/usr/bin/env python
"""Emit the degraded->fixed disclosure table (modality-specific weight decay).

For each vision encoder paired with the oversized Giant text encoder, reports the
average zero-shot performance of the degraded configuration (text/vision weight
decay 0.5/0.5) and its fixed counterpart (text weight decay raised to 1.0, vision
kept at 0.5), the absolute improvement, the number of datasets improved, and a
two-sided sign test over tasks. These are the numbers behind
Figure~\\ref{fig:degraded_then_fixed} and back the generalization paragraph in
Section "CLIP demands modality-specific encoder capacity".

It also reports the "reach the frontier" comparisons that back the main-text
claims -- each fixed model vs. the best architecture-grid model for its vision
encoder, and Giant--Giant fixed vs. the overall-best grid model (Colossal--Giant).
These are computed against the architecture grid under the same checkpoint policy
as the figures.

Avg zero-shot is computed identically to the figures (load_data => latest
checkpoint, random-chance ZS datasets removed, retrieval counted as two rows).

Note on recipes: the degraded and fixed runs use the modality-specific weight
decay training recipe, whereas the architecture-grid reference models use the
original recipe. The frontier comparison is therefore cross-recipe; it is
conservative, since the weight-decay recipe's baseline sits ~1 point below the
grid baseline.

Prints the results as JSON: a ``degraded_fixed`` record per vision encoder and a
``frontier`` list of the fixed-vs-grid comparisons.

Usage:
    python bin/make_degraded_fixed_table.py [--out PATH]
"""
import argparse

from src.analysis import (load_data, run_task_vec, model_task_vec, sign_test, json_num,
                          dump_json, ANALYSIS_DIR, ARCH_TAG, TEXT_GRID,
                          VISION_MODEL_PREFIX, _DEGRADED_FIXED_VISIONS,
                          _DEGRADED_FIXED_PAIRS)

# Ordered small -> large; keys match _DEGRADED_FIXED_PAIRS.
_VISIONS = _DEGRADED_FIXED_VISIONS  # ["Atto", "Tiny", "Base", "Giant"]


def _degraded_fixed_rows(data):
    """Per-vision degraded/fixed avg + improvement sign test."""
    bench, ct = data["benchmarks"], data["CHECKPOINT_TYPE"]
    rows = []
    for vis in _VISIONS:
        pair = _DEGRADED_FIXED_PAIRS[vis]
        deg = run_task_vec(bench, ct, pair["degraded"])
        fix = run_task_vec(bench, ct, pair["fixed"])
        delta, w, n, p = sign_test(fix, deg)
        rows.append({
            "vision": vis,
            "degraded": deg.mean() * 100.0,
            "fixed": fix.mean() * 100.0,
            "delta": delta, "wins": w, "n": n, "p": p,
        })
    return rows


def _frontier_rows(data):
    """Fixed model vs. its best-per-vision grid model (+ Giant vs overall best)."""
    bench, ct = data["benchmarks"], data["CHECKPOINT_TYPE"]
    # Architecture-grid averages (clean, tag-filtered).
    grid = bench[(bench["super_task"] == "zeroshot")
                 & (bench["pretrain_dataset"] == "cc12m")
                 & (bench["checkpoint_type"] == ct)
                 & (bench["wandb_tags"].str.contains(ARCH_TAG, na=False))]
    gavg = grid.groupby("model")["preferred_metric_value"].mean() * 100.0

    def best_for(vis):
        cands = [f"{VISION_MODEL_PREFIX[vis]}--{t}" for t in TEXT_GRID]
        cands = [m for m in cands if m in gavg.index]
        best = max(cands, key=lambda m: gavg[m])
        return best, gavg[best]

    overall_best = gavg.idxmax()
    rows = []
    for vis in _VISIONS:
        fix = run_task_vec(bench, ct, _DEGRADED_FIXED_PAIRS[vis]["fixed"])
        ref_model, ref_avg = best_for(vis)
        delta, w, n, p = sign_test(fix, model_task_vec(bench, ct, ref_model))
        rows.append({"fixed_vision": vis, "fixed_avg": fix.mean() * 100.0,
                     "ref": ref_model, "ref_avg": ref_avg, "kind": "best-per-vision",
                     "delta": delta, "wins": w, "n": n, "p": p})
        # Giant--Giant fixed also vs. the overall-best grid model.
        if vis == "Giant" and overall_best not in (ref_model,):
            delta, w, n, p = sign_test(fix, model_task_vec(bench, ct, overall_best))
            rows.append({"fixed_vision": vis, "fixed_avg": fix.mean() * 100.0,
                         "ref": overall_best, "ref_avg": gavg[overall_best],
                         "kind": "overall-best", "delta": delta, "wins": w, "n": n, "p": p})
    return rows


def _results(data):
    """Degraded->fixed per-vision records + the frontier comparisons, for JSON."""
    df = [{
        "vision": r["vision"],
        "degraded": json_num(r["degraded"]),
        "fixed": json_num(r["fixed"]),
        "delta": json_num(r["delta"]),
        "wins": r["wins"], "n": r["n"], "sign_p": json_num(r["p"]),
    } for r in _degraded_fixed_rows(data)]

    frontier = []
    for r in _frontier_rows(data):
        verdict = ("surpasses" if (r["p"] < 0.05 and r["delta"] > 0)
                   else "matches" if r["p"] >= 0.05 else "below")
        frontier.append({
            "fixed_vision": r["fixed_vision"], "fixed_avg": json_num(r["fixed_avg"]),
            "ref": r["ref"], "ref_avg": json_num(r["ref_avg"]), "kind": r["kind"],
            "delta": json_num(r["delta"]), "wins": r["wins"], "n": r["n"],
            "sign_p": json_num(r["p"]), "verdict": verdict,
        })
    return {"degraded_fixed": df, "frontier": frontier}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ANALYSIS_DIR / "degraded_fixed.json"),
                    help="Path to persist the JSON results (also printed to stdout).")
    args = ap.parse_args()
    dump_json(_results(load_data()), args.out)


if __name__ == "__main__":
    main()
