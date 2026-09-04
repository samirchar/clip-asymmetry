"""Task-level significance of the weight-decay ablation asymmetry.

Per-dataset two-sided sign tests of the modality-specific weight-decay ablation:
raising the text encoder's weight decay improves zero-shot while raising the
vision encoder's degrades it. For both the Base--Giant and Giant--Giant
configurations, at the strong (effective WD 1.0) and mild (0.75) levels:
  * Test 1 (asymmetry):     text-only WD  >  vision-only WD
  * Test 2 (vs. symmetric): text-only WD  >  joint WD
each over the 28 zero-shot tasks, plus the relevant encoder parameter counts.
These back the sign-test claims in the paper's "modality-specific weight decay"
section (e.g. text-only > vision-only on 23/28 datasets, sign test p<0.001).

Prints the results as JSON and persists them to --out.

Usage:
    python bin/significance_wd_ablation_tasks.py [--out PATH]
"""
import argparse
import re

from scipy import stats

from src.analysis import (load_data, _load_run_configs, _wd_effective, _wd_classify,
                          json_num, dump_json, ANALYSIS_DIR)


def _wd_ablation_significance(data, pretrain_dataset="cc12m", tag=None, include_joint=True):
    """Per-dataset two-sided sign tests backing the WD-ablation claims.

    Two pre-specified tests at the strongest intervention (effective WD = 1.0),
    with the milder level (0.75) as robustness:

        Test 1 (asymmetry)     : text-only WD  >  vision-only WD
        Test 2 (vs. symmetric) : text-only WD  >  joint WD

    The sign test is used because the per-dataset differences are heterogeneous
    in scale and asymmetric, violating the paired t-test / Wilcoxon assumptions.
    ``tag`` selects the run family (default Base--Giant ``{dataset}_wd_ablation``);
    pass ``{dataset}_wd_ablation_giant`` for the Giant--Giant replication.
    """
    tag = tag or f"{pretrain_dataset}_wd_ablation"
    bench = data["benchmarks"]
    ct = data["CHECKPOINT_TYPE"]
    # Exact-token tag match: 'cc12m_wd_ablation' must not also catch the
    # 'cc12m_wd_ablation_giant' family (wandb_tags is a '|'-delimited string).
    tag_mask = bench["wandb_tags"].str.contains(
        rf"(?:^|\|){re.escape(tag)}(?:\||$)", regex=True, na=False)
    abl = bench[tag_mask].query(
        f"pretrain_dataset == '{pretrain_dataset}' and super_task == 'zeroshot' "
        f"and checkpoint_type == '{ct}'"
    )
    abl = abl[~abl["run_name"].str.contains("_1772044155", na=False)]

    configs = _load_run_configs()
    run_of = {}
    for run_name in abl["run_name"].unique():
        eff_t, eff_v = _wd_effective(run_name, configs.get(run_name))
        scenario, level = _wd_classify(eff_t, eff_v)
        if scenario is not None:
            run_of[(scenario, round(level, 2))] = run_name

    def per_task(run_name):
        # One value per zero-shot task, keyed by ``dataset|metric`` (retrieval
        # datasets contribute their two recall directions as *separate* tasks),
        # matching the task granularity used everywhere else (src.analysis
        # run_task_vec / model_task_vec, bootstrap_significance). Values in %.
        rows = abl[abl["run_name"] == run_name]
        key = rows["dataset"].astype(str) + "|" + rows["preferred_metric"].astype(str)
        return rows.assign(key=key).groupby("key")["preferred_metric_value"].mean() * 100

    results = []

    def test(label, high, low, level):
        if (high, level) not in run_of or (low, level) not in run_of:
            return
        hv = per_task(run_of[(high, level)])
        lv = per_task(run_of[(low, level)])
        idx = hv.index.intersection(lv.index)
        diff = (hv.loc[idx] - lv.loc[idx]).values
        wins = int((diff > 0).sum())
        n_eff = len(diff) - int((diff == 0).sum())
        p = stats.binomtest(wins, n_eff, 0.5, alternative="two-sided").pvalue
        results.append({"test": label, "high_wd": high, "low_wd": low, "level": level,
                        "wins": wins, "n": n_eff,
                        "mean_delta": json_num(diff.mean()), "p": json_num(p)})

    test("Test 1  text > vision (asymmetry)", "text", "vision", 1.0)
    if include_joint:
        test("Test 2  text > joint  (vs. symmetric)", "text", "joint", 1.0)
    test("Test 1  text > vision", "text", "vision", 0.75)
    if include_joint:
        test("Test 2  text > joint", "text", "joint", 0.75)
    return results


def _results(data):
    """WD-ablation per-dataset sign tests + the relevant encoder param counts."""
    sign_tests = {
        "Base--Giant": _wd_ablation_significance(data, "cc12m", tag="cc12m_wd_ablation"),
        "Giant--Giant": _wd_ablation_significance(
            data, "cc12m", tag="cc12m_wd_ablation_giant", include_joint=False),
    }
    arch = data["architectures"].drop_duplicates("model").set_index("model")
    encoder_params = {}
    for m in ["ViT-B-16--Giant", "ViT-Giant-16--Giant"]:
        if m in arch.index:
            vp = arch.loc[m, "vision_params"] / 1e6
            tp = arch.loc[m, "text_params"] / 1e6
            encoder_params[m] = {"vision_params_m": json_num(vp),
                                 "text_params_m": json_num(tp),
                                 "vision_over_text": json_num(vp / tp)}
    return {"sign_tests": sign_tests, "encoder_params": encoder_params}


def main():
    ap = argparse.ArgumentParser(description="Task-level WD-ablation sign tests.")
    ap.add_argument("--out", default=str(ANALYSIS_DIR / "significance_wd_ablation_tasks.json"),
                    help="Path to persist the JSON results (also printed to stdout).")
    args = ap.parse_args()
    dump_json(_results(load_data()), args.out)


if __name__ == "__main__":
    main()
