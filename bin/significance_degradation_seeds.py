#!/usr/bin/env python
"""Training-seed variance test for the zero-shot degradation claim.

Reviewers asked whether the zero-shot degradation ("bigger text encoders can
hurt") is just training-seed noise. We reran the Base vision encoder's
near-peak anchor (Base--Tiny) and degraded anchor (Base--Giant) at extra seeds
(tagged ``cc12m_seeds`` in wandb) and test the pre-specified directional
hypothesis that the near-peak config beats the degraded one.

Design (fixed before seeing the replication seeds):
  * Unit = training seed. Per seed we average the canonical zero-shot basket as
    the paper does; ``load_data`` already removes the random-chance datasets.
  * Seeds are matched across the two configs, so the pre-specified test is a
    ONE-SIDED PAIRED t-test with, for D = Tiny - Giant, H0: E[D] <= 0 vs
    H1: E[D] > 0. We report the mean difference, its one-sided 95%% CI lower
    bound, and Cohen's d_z.

The Tiny/Giant contrast was chosen from the original grid, so the original grid
run is *discovery* and the new seeds are prospective *replication*; with only a
few seeds the result is suggestive, not definitive. Prints the results as JSON
and persists them to --out.

Usage:
    python bin/analyze_seed_variance.py                 # Base, Tiny vs Giant, latest
    python bin/analyze_seed_variance.py --checkpoint best
"""
import argparse
import re

from scipy import stats as st

from src.analysis import load_data, json_num, dump_json, ANALYSIS_DIR


def _seed(run_name):
    m = re.search(r"seed_(\d+)", run_name)
    return m.group(1) if m else "orig"


def _paired_test(a, b, conf=0.95):
    """One-sided paired t-test (H1: a > b) with Cohen's d_z and the one-sided CI.

    The one-sided ("greater") CI is [low, +inf), so only the finite lower bound is
    reported (the +inf upper bound is constant and not JSON-representable).
    """
    d = a - b
    sd = d.std(ddof=1)
    res = st.ttest_rel(a, b, alternative="greater")
    return {
        "n_seeds": len(d),
        "mean_diff": d.mean(),
        "sd_diff": sd,
        "p_one_sided": float(res.pvalue),
        "ci_one_sided_low": float(res.confidence_interval(conf).low),
        "cohen_dz": d.mean() / sd,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vision", default="Base")
    ap.add_argument("--peak-text", default="Tiny")
    ap.add_argument("--degraded-text", default="Giant")
    ap.add_argument("--checkpoint", default="latest", choices=["latest", "best"])
    ap.add_argument("--tag", default="cc12m_seeds", help="wandb tag identifying the seed runs.")
    ap.add_argument("--out", default=str(ANALYSIS_DIR / "seed_variance.json"),
                    help="Path to persist the JSON results (also printed to stdout).")
    args = ap.parse_args()
    peak, degraded = args.peak_text, args.degraded_text

    # The seed replication runs are tagged only `cc12m_seeds` (not
    # `<dataset>_architecture_joint`), so this cannot reuse `_base_query`.
    b = load_data()["benchmarks"]
    sub = b[(b["super_task"] == "zeroshot") & (b["checkpoint_type"] == args.checkpoint)
            & (b["vision_name"] == args.vision) & (b["text_name"].isin([peak, degraded]))
            & (b["wandb_tags"].str.contains(args.tag, na=False))]
    # Per-seed basket mean (random-chance datasets already dropped by load_data).
    per_seed = (sub.assign(seed=sub["run_name"].map(_seed))
                .groupby(["text_name", "seed"])["preferred_metric_value"].mean().mul(100))
    piv = per_seed.unstack("text_name").dropna()
    if peak not in piv or degraded not in piv or len(piv) < 2:
        raise SystemExit(f"Need >=2 seeds present for both {peak} and {degraded}.")

    out = {
        "vision": args.vision, "peak_text": peak, "degraded_text": degraded,
        "checkpoint": args.checkpoint,
        "per_seed": {f"{args.vision}--{t}": {s: json_num(v) for s, v in piv[t].to_dict().items()}
                     for t in [peak, degraded]},
        "per_config": {f"{args.vision}--{t}": {"mean": json_num(piv[t].mean()),
                                               "sd": json_num(piv[t].std(ddof=1))}
                       for t in [peak, degraded]},
        "paired_test": {k: (json_num(v) if isinstance(v, float) else v)
                        for k, v in _paired_test(piv[peak].to_numpy(),
                                                 piv[degraded].to_numpy()).items()},
    }
    dump_json(out, args.out)


if __name__ == "__main__":
    main()
