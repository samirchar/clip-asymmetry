#!/usr/bin/env python
"""Emit the degradation disclosure table (peak text encoder vs. oversized Giant text).

For each vision encoder, compares the best-performing (peak) text encoder in the
architecture grid against the oversized Giant text encoder (145M) -- i.e. the
core "bigger text encoders can hurt" claim. Reports the two averages, the drop,
the number of datasets on which the peak beats Giant, a two-sided sign test, and
a task-level paired bootstrap 95% CI on the mean drop.

Methodology is IDENTICAL to bin/make_degraded_fixed_table.py: numbers come from
load_data() (latest checkpoint, random-chance ZS datasets removed, retrieval
counted as two recall@1 rows), architecture-grid runs are tag-filtered to
``cc12m_architecture_joint``, and the sign test is two-sided over tasks -- the
same test used for the weight-decay and degraded->fixed tables.

The peak text encoder is selected empirically per vision row; because selection
and testing use the same task scores, the peak-vs-Giant p-values are optimistic
(a within-row selection effect that Holm across rows does not correct). We
therefore also report a PRE-SPECIFIED contrast (a fixed small text encoder vs.
Giant), which carries no selection bias.

Task-level bootstrap vs. sign test:
  - The sign test asks "is the drop directionally consistent across tasks?"
    (counts wins/losses; scale-invariant; ignores magnitude; yields a p-value).
  - The bootstrap yields a confidence INTERVAL on the mean drop in benchmark
    points (magnitude + uncertainty), which the sign test does not provide.
  Both quantify benchmark-composition (task-sampling) uncertainty for a single
  checkpoint; NEITHER captures training-seed variability.

Prints the results as JSON: one record per vision encoder (peak text, peak/Giant
averages, drop, sign test, bootstrap CI, the pre-specified contrast, and the text
ladder). This is a superset of the console/LaTeX numbers the old table reported.

Usage:
    python bin/make_degradation_table.py [--reference-text Tiny] [--out PATH]
"""
import argparse

from src.analysis import (load_data, model_task_vec, sign_test, bootstrap_ci, json_num,
                          dump_json, ANALYSIS_DIR, TEXT_GRID, VISION_MODEL_PREFIX)

# Vision encoders smallest -> largest (display names, also the VISION_MODEL_PREFIX keys).
_VISIONS = ["Atto", "Tiny", "Base", "Giant", "Colossal"]
_OVERSIZED = "Giant"  # oversized reference text encoder


def _rows(data, reference_text):
    bench, ct = data["benchmarks"], data["CHECKPOINT_TYPE"]
    rows = []
    for v in _VISIONS:
        giant = f"{VISION_MODEL_PREFIX[v]}--{_OVERSIZED}"
        gvec = model_task_vec(bench, ct, giant, pretrain_dataset="cc12m")
        # per-text averages for this vision row
        avgs = {}
        vecs = {}
        for t in TEXT_GRID:
            vec = model_task_vec(bench, ct, f"{VISION_MODEL_PREFIX[v]}--{t}",
                                 pretrain_dataset="cc12m")
            if len(vec):
                vecs[t] = vec
                avgs[t] = vec.mean() * 100.0
        peak_t = max(avgs, key=avgs.get)
        # PRIMARY: empirical peak vs Giant
        d, w, n, p = sign_test(vecs[peak_t], gvec)
        lo, hi, se = bootstrap_ci(vecs[peak_t], gvec)
        # PRE-SPECIFIED: fixed reference text vs Giant (no selection bias)
        pre = None
        if reference_text in vecs and reference_text != _OVERSIZED:
            dr, wr, nr, pr = sign_test(vecs[reference_text], gvec)
            lor, hir, _ = bootstrap_ci(vecs[reference_text], gvec)
            pre = dict(text=reference_text, delta=dr, wins=wr, n=nr, p=pr,
                       ci_lo=lor, ci_hi=hir)
        rows.append(dict(
            vision=v, peak_text=peak_t,
            peak_avg=avgs[peak_t], giant_avg=gvec.mean() * 100.0,
            delta=d, wins=w, n=n, p=p, ci_lo=lo, ci_hi=hi, se=se,
            ladder={t: avgs[t] for t in TEXT_GRID if t in avgs}, pre=pre))
    return rows


def _results(data, reference_text):
    """Per-vision degradation records (superset of the old console + LaTeX numbers)."""
    out = []
    for r in _rows(data, reference_text):
        pre = r["pre"]
        out.append({
            "vision": r["vision"],
            "peak_text": r["peak_text"],
            "peak_avg": json_num(r["peak_avg"]),
            "giant_avg": json_num(r["giant_avg"]),
            "delta": json_num(r["delta"]),          # peak - Giant (>0 => Giant degrades)
            "wins": r["wins"], "n": r["n"], "sign_p": json_num(r["p"]),
            "ci_lo": json_num(r["ci_lo"]), "ci_hi": json_num(r["ci_hi"]),
            "se": json_num(r["se"]),
            "pre_specified": None if pre is None else {
                "text": pre["text"], "delta": json_num(pre["delta"]),
                "wins": pre["wins"], "n": pre["n"], "p": json_num(pre["p"]),
                "ci_lo": json_num(pre["ci_lo"]), "ci_hi": json_num(pre["ci_hi"]),
            },
            "ladder": {t: json_num(r["ladder"][t]) for t in TEXT_GRID if t in r["ladder"]},
        })
    return {"reference_text": reference_text, "rows": out}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference-text", default="Tiny", choices=TEXT_GRID,
                    help="Pre-specified small text encoder for the selection-free "
                         "contrast (default: Tiny).")
    ap.add_argument("--out", default=str(ANALYSIS_DIR / "degradation.json"),
                    help="Path to persist the JSON results (also printed to stdout).")
    args = ap.parse_args()
    dump_json(_results(load_data(), args.reference_text), args.out)


if __name__ == "__main__":
    main()
