"""Shared analysis layer: benchmark data loading and the statistics helpers
used by the paper's tables and figures.

This module is the single home for everything the analysis scripts in ``bin/``
need (``load_data``, the weight-decay run classification, the sign-test/bootstrap
helpers, ...), so they no longer have to import the plotting script. Palette
construction still lives in :mod:`src.figures`; :func:`load_data` borrows it.
"""
import json
import re
import sys
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from src.benchmark_utils import process_benchmark
from src.figures import THEME, _build_palette_and_labels

# ── Paths ──────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"
WANDB_CACHE_DIR = DATA_DIR / "wandb_cache"
TRAIN_LOG_DIR = Path("/mnt/blob/openclip_logs")  # open_clip console logs (out.log), one dir per run
BENCHMARKS_DIR = ROOT / "benchmarks"
_WD_BASE = 0.5           # default (baseline) weight decay for both encoders

_DEGRADED_FIXED_VISIONS = ["Atto", "Tiny", "Base", "Giant"]  # small → large vision


# Explicit degraded/fixed run pairs; see docs/degraded_then_fixed_figure_plan.md.
# "fixed" = same architecture as "degraded" but with text weight decay raised
# (scale 2). All fixed runs and the Giant/Base degraded runs use the local-loss
# recipe (local_loss=True, gather_with_grad=True). ⚠️ The Atto & Tiny *degraded*
# runs are still the OLD-recipe (local_loss=False) architecture-grid runs; swap
# them to the local-loss wd_joint_scale0.5 versions when trained (only the run
# names below change — no plotting-code change needed).
_DEGRADED_FIXED_PAIRS = {
    "Atto":  {"degraded": "cc12m_architecture_ViT-Atto-16--Giant_wd_joint_scale0.5_1779556615_attempt_1",
              "fixed":    "cc12m_architecture_ViT-Atto-16--Giant_wd_vision_scale1_wd_text_scale2_1779556615"},
    "Tiny":  {"degraded": "cc12m_architecture_ViT-Tiny-16--Giant_wd_joint_scale0.5_1779556615_attempt_1",
              "fixed":    "cc12m_architecture_ViT-Tiny-16--Giant_wd_vision_scale1_wd_text_scale2_1779556615"},
    "Base":  {"degraded": "cc12m_architecture_ViT-B-16--Giant_wd_joint_scale0.5_1779556615",
              "fixed":    "cc12m_architecture_ViT-B-16--Giant_wdscale2_1779556615"},
    "Giant": {"degraded": "cc12m_architecture_ViT-Giant-16--Giant_wd_joint_scale0.5_1779556615_attempt_1",
              "fixed":    "cc12m_architecture_ViT-Giant-16--Giant_wd_vision_scale1_wd_text_scale2_1779556615"},
}


# ── Architecture-grid constants ────────────────────────
# Architecture-grid wandb tag; the grid's text encoders, smallest → largest; and
# the per-vision model-name prefix (keyed by the display name used in the tables).
ARCH_TAG = "cc12m_architecture_joint"
TEXT_GRID = ["Femto", "Atto", "Nano", "Tiny", "Base", "Giant"]
VISION_MODEL_PREFIX = {"Atto": "ViT-Atto-16", "Tiny": "ViT-Tiny-16", "Base": "ViT-B-16",
                       "Giant": "ViT-Giant-16", "Colossal": "ViT-Colossal-16"}

# Task-level paired bootstrap settings (module-global seeded RNG for reproducibility).
N_BOOT = 10_000
_BOOT_RNG = np.random.default_rng(0)


VISION_NAME_MAP = {
    "ViT-Atto-16": "Atto",
    "ViT-Tiny-16": "Tiny",
    "ViT-B-16": "Base",
    "ViT-Giant-16": "Giant",
    "ViT-Colossal-16": "Colossal",
}


def get_wandb_training_metrics(
    project,
    job_name_pattern="*",
    tags=None,
    entity=None,
    use_scan=True,
    samples=100_000,
    finished_only=True,
    cache=True,
):
    """Fetch per-step training metrics from wandb and cache one parquet per run.

    Ported from the explore_results notebook so this script can (re)generate the
    ``wandb_cache/`` used by load_wandb_meta / the OGR figure without the notebook.
    Each matching run's full history is written to ``WANDB_CACHE_DIR/<run_id>.parquet``.
    """
    from fnmatch import fnmatch
    import wandb

    if cache:
        WANDB_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    all_runs = api.runs(f"{entity}/{project}" if entity else project)

    matching = []
    for r in all_runs:
        if fnmatch(r.name, job_name_pattern) and (any(t in r.tags for t in tags) if tags else True):
            if finished_only and r.state != "finished":
                continue
            matching.append(r)
    print(f"Matched {len(matching)}/{len(all_runs)} runs with pattern '{job_name_pattern}'")

    dfs = []
    for run in matching:
        cache_path = WANDB_CACHE_DIR / f"{run.id}.parquet"
        if cache and cache_path.exists():
            hist = pd.read_parquet(cache_path)
        else:
            if use_scan:
                hist = pd.DataFrame(list(run.scan_history()))
                if "_step" in hist.columns:
                    non_step = [c for c in hist.columns if c != "_step"]
                    hist = hist.groupby("_step", as_index=False)[non_step].first()
            else:
                hist = run.history(samples=samples)
            # scan_history() can truncate long/resumed runs, dropping the final
            # epochs; append the last logged point from run.summary so the
            # final-epoch train loss (used by the OGR / degraded-fixed figures) is
            # not lost when a run's epoch labels stop before its final step.
            _fin = {k: run.summary.get(k) for k in ("_step", "epoch", "train/loss")}
            if _fin["_step"] is not None and (
                    "_step" not in hist.columns or hist.empty
                    or _fin["_step"] > hist["_step"].max()):
                hist = pd.concat([hist, pd.DataFrame([_fin])], ignore_index=True)
            hist["run_name"] = run.name
            hist["run_id"] = run.id
            hist["wandb_tags"] = ["|".join(list(run.tags))] * len(hist)
            hist["model"] = run.config.get("model")
            hist["checkpoint_path"] = run.config.get("checkpoint_path")
            hist["pretrain_dataset"] = run.config.get("train_data", run.config.get("dataset"))
            hist["config"] = [run.config] * len(hist)
            if cache:
                hist.to_parquet(cache_path, index=False)
        dfs.append(hist)

    if not dfs:
        print("No matching runs found.")
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def refresh_wandb_cache(tags=None):
    """Populate WANDB_CACHE_DIR from wandb using .env credentials/config.

    Uses WANDB_ENTITY / WANDB_PROJECT from the environment (loaded from .env),
    fetching the architecture-joint and WD-ablation runs used by the figures.
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()
    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT", "clip_ablation")
    if tags is None:
        tags = ["cc12m_architecture_joint",
                "cc3m_architecture_joint",
                "cc12m_wd_ablation",
                "cc12m_degraded_then_fixed",
                "cc12m_seeds"]
    print(f"Refreshing wandb cache: entity={entity} project={project} tags={tags}")
    get_wandb_training_metrics(project=project, entity=entity, tags=tags, use_scan=True, cache=True)


def load_wandb_meta():
    """Load wandb training metrics from cache and build wandb_meta.

    Mirrors notebook cells 1-2: loads cached parquet files from wandb_cache/,
    computes row_type, and aggregates val/train stats per run.
    """
    cache_dir = DATA_DIR / "wandb_cache"
    cache_files = list(cache_dir.glob("*.parquet"))
    if not cache_files:
        raise FileNotFoundError(f"No cached wandb data found in {cache_dir}")

    combined = pd.concat([pd.read_parquet(f) for f in cache_files], ignore_index=True)

    # Tag rows as train or val based on populated columns
    train_cols = [c for c in combined.columns if c.startswith("train/")]
    val_cols = [c for c in combined.columns if c.startswith("val/")]
    has_train = combined[train_cols].notna().any(axis=1) if train_cols else pd.Series(False, index=combined.index)
    has_val = combined[val_cols].notna().any(axis=1) if val_cols else pd.Series(False, index=combined.index)
    combined["row_type"] = "unknown"
    combined.loc[has_train, "row_type"] = "train"
    combined.loc[has_val, "row_type"] = "val"

    val_stats = combined.query("row_type == 'val'").groupby(
        ["run_name", "run_id", "checkpoint_path", "wandb_tags", "pretrain_dataset", "model"]
    ).agg(final_val_imgnet=("val/imagenet-zeroshot-val-top1", "last"))

    train_stats = combined.query("row_type == 'train'").groupby(
        ["run_name", "run_id", "checkpoint_path", "wandb_tags", "pretrain_dataset", "model"]
    ).agg(avg_samples_per_sec_per_gpu=("train/samples_per_second_per_gpu", "mean"))

    return val_stats.join(train_stats).reset_index()


def load_data():
    CHECKPOINT_TYPE = "latest"

    # Architecture details
    architectures = pd.read_csv(DATA_DIR / "architecture_details.csv").rename(
        columns={"architecture": "model"}
    )
    architectures[["vision_name", "text_name"]] = architectures["model"].str.split(
        "--", expand=True
    )

    # Apply consistent naming to vision encoders
    architectures["vision_name"] = architectures["vision_name"].map(VISION_NAME_MAP).fillna(architectures["vision_name"])

    # Build palettes and labels dynamically from architectures
    # Exclude W512H8L12 (duplicate of Base)
    arch_clean = architectures.query("text_name != 'W512H8L12'")
    vision_palette, vision_labels = _build_palette_and_labels(
        arch_clean, "vision_name", "vision_params", THEME.vision_cmap
    )
    text_palette, text_labels = _build_palette_and_labels(
        arch_clean, "text_name", "text_params", THEME.text_cmap
    )

    # Geometry metrics
    rankme = pd.concat([
        pd.read_json(ANALYSIS_DIR / "rankme_results_cc12m.json"),
        pd.read_json(ANALYSIS_DIR / "rankme_results_cc3m.json"),
    ])

    # Uniformity / alignment geometry. The weight-decay (degraded/fixed) runs live
    # in a separate file but share the same schema and the epoch_latest checkpoint,
    # so they merge on `pretrained` (the latest checkpoint) exactly like the grid.
    uni_align = pd.concat([
        pd.read_json(ANALYSIS_DIR / "uni_align_cc12m.json"),
        pd.read_json(ANALYSIS_DIR / "uni_align_cc3m.json"),
        pd.read_json(ANALYSIS_DIR / "uni_align_degraded_fixed_v2.json"),
    ])

    modality_gap_path = ANALYSIS_DIR / "modality_gap_results.json"
    if modality_gap_path.exists():
        modality_gap = pd.read_json(modality_gap_path)
        modality_gap["pretrained"] = modality_gap["model_path"].str.replace("blob", "default")
        modality_gap = modality_gap.pivot_table(
            index="pretrained", columns="dataset", values="final_modality_gap")
        modality_gap.columns = [f"final_modality_gap_{d}" for d in modality_gap.columns]
        modality_gap = modality_gap.reset_index()
    else:
        modality_gap = pd.DataFrame()

    for d in [rankme, uni_align]:
        d["pretrained"] = d["model_path"].str.replace("blob", "default")

    uni_align = uni_align.pivot_table(
        index=["architecture", "model_path", "max_samples", "pretrained"],
        columns="dataset",
        values=["image_uniformity", "text_uniformity", "alignment"],
    )
    uni_align.columns = [f"{m}_{d}" for m, d in uni_align.columns]
    uni_align = uni_align.reset_index()

    # Benchmarks
    bench_files = glob(str(BENCHMARKS_DIR / "*.csv"))
    benchmarks = pd.concat([pd.read_csv(f) for f in bench_files], ignore_index=True)
    benchmarks = benchmarks.drop_duplicates()
    benchmarks = process_benchmark(benchmarks, tasklist_path=str(ROOT / "configs/eval/tasklist.yml"))
    benchmarks = benchmarks.merge(architectures, on="model", how="left")

    # Add tags columns
    benchmarks = pd.concat(
        [benchmarks, benchmarks["tags"].str.get_dummies().astype(bool)], axis=1
    )
    benchmarks["super_task"] = benchmarks["task"].apply(
        lambda t: "linear_probe" if t.startswith("linear_probe") else (
            "zeroshot" if t.startswith("zeroshot") else np.nan
        )
    )
    benchmarks["granularity"] = benchmarks["tags"].apply(
        lambda t: "fine_grained" if "fine_grained" in t else (
            "coarse_grained" if "coarse_grained" in t else np.nan
        )
    )
    benchmarks["vtab_group"] = benchmarks["tags"].apply(
        lambda t: "natural" if "vtab_natural" in t else (
            "structured" if "vtab_structured" in t else (
                "specialized" if "vtab_specialized" in t else np.nan
            )
        )
    )

    # Merge geometry
    benchmarks = benchmarks.merge(
        rankme[["pretrained", "image_rankme", "text_rankme"]],
        on="pretrained", how="left",
    )
    benchmarks = benchmarks.merge(
        uni_align[[
            "pretrained", "alignment_flickr30k", "alignment_mscoco_captions",
            "image_uniformity_flickr30k", "image_uniformity_mscoco_captions",
            "text_uniformity_flickr30k", "text_uniformity_mscoco_captions",
        ]],
        on="pretrained", how="left",
    )
    if not modality_gap.empty:
        gap_cols = [c for c in modality_gap.columns if c.startswith("final_modality_gap_")]
        benchmarks = benchmarks.merge(
            modality_gap[["pretrained"] + gap_cols],
            on="pretrained", how="left",
        )

    # Merge wandb metadata 
    wandb_meta = load_wandb_meta()
    benchmarks = benchmarks.merge(
        wandb_meta[["run_name", "checkpoint_path", "wandb_tags",
                     "final_val_imgnet", "avg_samples_per_sec_per_gpu"]],
        how="left", on=["checkpoint_path"],
    ).dropna(subset=["wandb_tags"])

    # Remove random ZS datasets
    random_zs = [
        "vtab/clevr_closest_object_distance", "vtab/clevr_count_all",
        "vtab/kitti_closest_vehicle_distance", "wilds/camelyon17", "vtab/pcam",
    ]
    benchmarks = benchmarks.query(
        "not(super_task == 'zeroshot' and dataset in @random_zs)"
    )

    return {
        "benchmarks": benchmarks,
        "architectures": architectures,
        "CHECKPOINT_TYPE": CHECKPOINT_TYPE,
        "vision_palette": vision_palette,
        "vision_labels": vision_labels,
        "text_palette": text_palette,
        "text_labels": text_labels,
    }


def _base_query(data, pretrain_dataset, super_task):
    ct = data["CHECKPOINT_TYPE"]
    return (
        f"pretrain_dataset == '{pretrain_dataset}' and "
        f"super_task == '{super_task}' and "
        f"checkpoint_type == '{ct}' and "
        f"wandb_tags.str.contains('{pretrain_dataset}_architecture_joint')"
    )




def _wd_effective_from_name(run_name):
    """Derive (eff_text_wd, eff_vision_wd) by parsing an ablation run name.

    Fallback for when run.config is unavailable. Returns (None, None) if the
    name does not match a known WD-ablation pattern.
    """
    def _num(pattern):
        m = re.search(pattern, run_name)
        return float(m.group(1)) if m else None

    if "wd_joint_scale" in run_name:
        x = _num(r"wd_joint_scale([0-9.]+)")
        return (x, x) if x is not None else (None, None)

    text_scale = _num(r"wd_text_scale([0-9.]+)")
    vision_scale = _num(r"wd_vision_scale([0-9.]+)")
    old_text_scale = _num(r"(?<![a-z])wdscale([0-9.]+)")  # legacy text-scale runs
    if text_scale is None and vision_scale is None and old_text_scale is None:
        return (None, None)

    t = text_scale if text_scale is not None else (old_text_scale or 1.0)
    v = vision_scale if vision_scale is not None else 1.0
    return (_WD_BASE * t, _WD_BASE * v)


def _wd_effective(run_name, cfg):
    """Derive (eff_text_wd, eff_vision_wd), preferring run.config over the name."""
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except (ValueError, TypeError):
            cfg = None
    if isinstance(cfg, dict) and cfg.get("wd") is not None:
        wd = float(cfg["wd"])

        def _scale(key):
            val = cfg.get(key)
            return float(val) if val not in (None, "None", "") else 1.0

        return wd * _scale("wd_text_scale"), wd * _scale("wd_vision_scale")
    return _wd_effective_from_name(run_name)


def _wd_classify(eff_text, eff_vision, tol=1e-6):
    """Map effective WDs to (scenario, level). Level is the raised encoder's WD."""
    if eff_text is None or eff_vision is None:
        return (None, None)
    at_base_t = abs(eff_text - _WD_BASE) < tol
    at_base_v = abs(eff_vision - _WD_BASE) < tol
    if at_base_t and at_base_v:
        return ("baseline", _WD_BASE)
    if at_base_v and eff_text > _WD_BASE:
        return ("text", round(eff_text, 3))
    if at_base_t and eff_vision > _WD_BASE:
        return ("vision", round(eff_vision, 3))
    if abs(eff_text - eff_vision) < tol and eff_text > _WD_BASE:
        return ("joint", round(eff_text, 3))
    return (None, None)


def _load_run_configs():
    """Map run_name -> run.config (dict) from the wandb cache parquets."""
    cache_dir = DATA_DIR / "wandb_cache"
    configs = {}
    for f in cache_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(f)
        except Exception:
            continue
        if "run_name" not in df.columns or df["run_name"].dropna().empty:
            continue
        run_name = df["run_name"].dropna().iloc[0]
        cfg = None
        if "config" in df.columns and df["config"].notna().any():
            cfg = df["config"].dropna().iloc[0]
        configs[run_name] = cfg
    return configs


# ── Per-task zero-shot vectors ─────────────────────────
def model_task_vec(bench, ct, model, arch_tag=ARCH_TAG, pretrain_dataset=None):
    """Per-task zero-shot vector (index=``dataset|metric``) for an architecture-grid model.

    Averages the preferred-metric value per (dataset, metric) over the tag-filtered
    zero-shot rows of ``model`` at the ``ct`` checkpoint. ``pretrain_dataset`` adds an
    optional exact-match filter (the degradation / best-vs-largest tables pin it to
    ``cc12m``; the degraded/fixed table leaves it off).
    """
    mask = ((bench["model"] == model)
            & (bench["super_task"] == "zeroshot")
            & (bench["checkpoint_type"] == ct)
            & (bench["wandb_tags"].str.contains(arch_tag, na=False)))
    if pretrain_dataset is not None:
        mask = mask & (bench["pretrain_dataset"] == pretrain_dataset)
    g = bench[mask].copy()
    g["tc"] = g["dataset"] + "|" + g["preferred_metric"]
    return g.groupby("tc")["preferred_metric_value"].mean()


def run_task_vec(bench, ct, run_name):
    """Per-task zero-shot vector (index=``dataset|metric``) for one run."""
    g = bench[(bench["run_name"] == run_name)
              & (bench["super_task"] == "zeroshot")
              & (bench["checkpoint_type"] == ct)].copy()
    g["tc"] = g["dataset"] + "|" + g["preferred_metric"]
    return g.groupby("tc")["preferred_metric_value"].mean()


# ── Statistics ─────────────────────────────────────────
def sign_test(a, b):
    """Two-sided paired sign test over tasks; ties (d==0) excluded from n.

    Returns ``(mean_delta_pct, wins, n, p)``.
    """
    d = (a - b).dropna().values
    w = int((d > 0).sum())
    n = int((d != 0).sum())
    p = binomtest(w, n, 0.5).pvalue if n else float("nan")
    return d.mean() * 100.0, w, n, p


def bootstrap_ci(a, b, pct=(2.5, 97.5)):
    """Task-level paired bootstrap of the mean drop ``(a - b)``, in percent.

    Resamples the paired per-task differences with replacement; returns
    ``(ci_lo, ci_hi, se)`` for the mean, in benchmark points. Uses the module-global
    seeded RNG, so callers must invoke it in a fixed order for reproducibility.
    """
    d = (a - b).dropna().values
    n = len(d)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    means = d[_BOOT_RNG.integers(0, n, size=(N_BOOT, n))].mean(axis=1) * 100.0
    lo, hi = np.percentile(means, pct)
    return float(lo), float(hi), float(means.std(ddof=1))


def loocv_r2(model):
    """Leave-one-out CV R^2 for a fitted statsmodels OLS via the hat-matrix (PRESS)
    shortcut -- algebraically exact, no refitting loop."""
    h = model.get_influence().hat_matrix_diag
    resid = np.asarray(model.resid)
    y = np.asarray(model.model.endog)
    press = np.sum((resid / (1.0 - h)) ** 2)
    return 1 - press / np.sum((y - y.mean()) ** 2)


def json_num(x):
    """Cast to float for JSON output; None/NaN -> None. No rounding -- full precision
    (json.dumps cannot serialize numpy floats and emits an invalid ``NaN`` literal,
    so this minimal cast is still required for valid JSON)."""
    return None if x is None or x != x else float(x)


def dump_json(results, out):
    """Print ``results`` as JSON to stdout and persist them to ``out`` (pretty-printed).

    stdout stays pure JSON (so it can be redirected); the "wrote" note goes to stderr.
    """
    text = json.dumps(results, indent=2)
    print(text)
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(text + "\n")
    print(f"Wrote {outp}", file=sys.stderr)



# ── Figure data loaders ────────────────────────────────
# DataFrame producers read from disk (console logs, val-loss parquets, uni-align
# JSON, the geometry CSV, the wandb cache) for the figures in bin/generate_figures.py.
# They live here so that module stays purely presentational.
def _classify_ogr(run_name, model):
    """Classify an OGR run into best / degraded / improved by its config.

    degraded = {vision}--Giant (largest text encoder, overfit-prone); improved =
    a WD-ablation run (name contains '_wd'); best = the argmax-ZS text config.
    Colossal is excluded (its best text *is* Giant, so best == degraded).
    """
    parts = str(model).split("--")
    vision = VISION_NAME_MAP.get(parts[0], parts[0])
    text = parts[1] if len(parts) > 1 else ""
    if re.search(r"_wd", str(run_name)):
        return vision, text, "improved"
    if text == "Giant" and vision != "Colossal":
        return vision, text, "degraded"
    if vision != "Colossal":
        return vision, text, "best"
    return vision, text, None


# Matches an open_clip progress line and captures (console_epoch, loss_avg):
#   "... Train Epoch: 34 [...] Contrastive_loss: 0.0783 (0.0809) Loss: 0.0783 (0.0809)"
# The parenthetical after the standalone "Loss:" is the AverageMeter running mean
# (``.avg``) -- the batch-size-weighted epoch average -- and the last such line per
# epoch is that epoch's average. (Capital "Loss:" is the total loss logged to wandb
# as ``train/loss``; the lowercase "..._loss:" components are excluded.)
_TRAIN_LOG_LINE = re.compile(
    r"Train Epoch:\s*(\d+).*?(?<!\w)Loss:\s*[-+0-9.eE]+\s*\(([-+0-9.eE]+)\)")


def _load_train_loss_per_epoch():
    """Per-epoch *average* train loss, read from each run's console log (out.log).

    open_clip only logs the per-log-window sample (``.val``) to wandb, but prints
    ``Loss: <val> (<avg>)`` to the console, where ``<avg>`` is the AverageMeter
    running mean: the batch-size-weighted epoch average, reset each epoch. The last
    such line per epoch is that epoch's average, so we read it straight from the log.
    This is the authoritative number (it matches what the logs show, unlike an
    unweighted mean of the sparse wandb samples) and is unaffected by the
    wandb-history truncation that hits the Giant runs. Console epochs are 0-indexed;
    we shift to the 1-indexed ``epoch`` used everywhere else (checkpoint/val-loss
    numbering). One ``<run>/out.log`` per run under TRAIN_LOG_DIR.
    """
    frames = []
    for log in sorted(TRAIN_LOG_DIR.glob("*/out.log")):
        run = log.parent.name
        cp = f"/mnt/default/openclip_logs/{run}/checkpoints"
        per_epoch = {}  # console_epoch -> last-seen running-mean loss
        with open(log, errors="ignore") as fh:
            for line in fh:
                if "Train Epoch:" not in line:
                    continue
                m = _TRAIN_LOG_LINE.search(line)
                if m:
                    per_epoch[int(m.group(1))] = float(m.group(2))
        if not per_epoch:
            continue
        eps = sorted(per_epoch)
        frames.append(pd.DataFrame({
            "epoch": [e + 1 for e in eps],  # console 0-indexed -> 1-indexed
            "train/loss": [per_epoch[e] for e in eps],
            "checkpoint_path": cp,
        }))
    if not frames:
        return pd.DataFrame(columns=["checkpoint_path", "epoch", "train/loss"])
    return pd.concat(frames, ignore_index=True)


def _load_ogr_frame():
    """Build the OGR frame: holdout contrastive val loss + end-of-epoch train loss.

    Merges data/analysis/val_loss_ogr.parquet (from bin/compute_val_loss.py) with
    the per-epoch train loss and classifies each run (best/degraded/improved).
    """
    val_path = ANALYSIS_DIR / "val_loss_ogr.parquet"
    if not val_path.exists():
        raise FileNotFoundError(
            f"Missing {val_path}. Generate it first with:\n"
            "  python bin/compute_val_loss.py --models-file <models_list.txt> "
            "--models-root /mnt/blob --output data/analysis/val_loss_ogr.parquet"
        )
    ogr = pd.read_parquet(val_path)
    ogr["checkpoint_path"] = ogr["model_dir"].apply(lambda x: f"{x.replace('blob', 'default')}/checkpoints")
    val = ogr.pivot_table(index=["checkpoint_path", "run_name", "model", "epoch"],
                          columns="val_dataset", values="clip_val_loss")
    val.columns = [f"clip_val_loss_{d}" for d in val.columns]
    val = val.reset_index()
    train = _load_train_loss_per_epoch()
    df = train.merge(val, how="right", on=["checkpoint_path", "epoch"])
    cls = df.apply(lambda r: _classify_ogr(r["run_name"], r["model"]), axis=1, result_type="expand")
    df[["vision", "text", "group"]] = cls
    return df[df["group"].notna()].copy()


_DEGRADED_FIXED_VAL_PARQUET = "val_loss_degraded_fixed_v2.parquet"
_DEGRADED_FIXED_UNIALIGN_JSON = "uni_align_degraded_fixed_v2.json"


def _degraded_fixed_run_role():
    """run_name -> (vision_label, role) from _DEGRADED_FIXED_PAIRS."""
    m = {}
    for vis, pair in _DEGRADED_FIXED_PAIRS.items():
        m[pair["degraded"]] = (vis, "degraded")
        m[pair["fixed"]] = (vis, "fixed")
    return m


def _load_degraded_fixed_frame():
    """Final-epoch train loss + held-out CLIP loss for each degraded/fixed run.

    Merges data/analysis/val_loss_degraded_fixed_v2.parquet (from
    bin/compute_val_loss.py over a models-list file) with
    the per-epoch train loss, and tags each run with its (vision, role).
    """
    val_path = ANALYSIS_DIR / _DEGRADED_FIXED_VAL_PARQUET
    v = pd.read_parquet(val_path)
    v["checkpoint_path"] = v["model_dir"].apply(
        lambda x: f"{x.replace('blob', 'default')}/checkpoints")
    piv = v.pivot_table(index=["checkpoint_path", "run_name", "model", "epoch"],
                        columns="val_dataset", values="clip_val_loss")
    piv.columns = [f"clip_val_loss_{c}" for c in piv.columns]
    piv = piv.reset_index()
    train = _load_train_loss_per_epoch()
    df = piv.merge(train, how="left", on=["checkpoint_path", "epoch"])
    roles = _degraded_fixed_run_role()
    df[["vision", "role"]] = df["run_name"].apply(
        lambda r: pd.Series(roles.get(r, (None, None))))
    df = df[df["role"].notna()].copy()
    # keep the final epoch per run (the OGR point)
    df["_maxep"] = df.groupby("checkpoint_path")["epoch"].transform("max")
    return df[df["epoch"] == df["_maxep"]].copy()


def _degraded_fixed_avg_zs(data):
    """{vision: {"degraded": avg_zs, "fixed": avg_zs}} in %, from the benchmarks."""
    bench = data["benchmarks"]
    ct = data["CHECKPOINT_TYPE"]
    zs = bench.query(
        f"pretrain_dataset == 'cc12m' and super_task == 'zeroshot' and "
        f"checkpoint_type == '{ct}'")
    out = {}
    for vis, pair in _DEGRADED_FIXED_PAIRS.items():
        out[vis] = {}
        for role in ("degraded", "fixed"):
            g = zs[zs["run_name"] == pair[role]]
            out[vis][role] = g["preferred_metric_value"].mean() * 100 if len(g) else np.nan
    return out


_COLLAPSE_PARQUET = "collapse_train_loss.parquet"
# Global batch size for the CC12M architecture-grid runs. A collapsed CLIP model
# outputs a uniform distribution over the batch, so its contrastive loss saturates
# at ln(batch) -- the reference line in the collapse figure.
_COLLAPSE_BATCH_SIZE = 8192


def _load_collapse_train_loss():
    """Per-step CLIP train loss for the weight-decay ``collapse`` runs.

    Reads data/analysis/collapse_train_loss.parquet, holding the two clean
    collapse trajectories (Base--Giant on CC12M with weight decay raised to
    lambda=1.5, one level above the ablated range): text-only and joint. Each is
    tagged with its ``scenario`` and ``level``. The parquet is fetched from the
    wandb runs tagged ``collapse`` via :func:`refresh_collapse_cache`.
    """
    path = ANALYSIS_DIR / _COLLAPSE_PARQUET
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Regenerate it with "
            "`python -c 'from src.analysis import refresh_collapse_cache; refresh_collapse_cache()'`.")
    return pd.read_parquet(path)


def refresh_collapse_cache():
    """Fetch the two clean ``collapse``-tagged runs from wandb and cache the parquet.

    Uses WANDB_ENTITY / WANDB_PROJECT from the environment (loaded from .env). The
    vision-only lambda=1.5 run is excluded: it failed during warmup with the loss
    still declining, so it never shows the decline->spike->plateau collapse
    signature.
    """
    import os
    import wandb
    from dotenv import load_dotenv

    load_dotenv()
    entity = os.environ.get("WANDB_ENTITY")
    project = os.environ.get("WANDB_PROJECT", "clip_ablation")
    runs = {
        "cc12m_architecture_ViT-B-16--Giant_wd_vision_scale1_wd_text_scale3_1779556615": ("text", 1.5),
        "cc12m_architecture_ViT-B-16--Giant_wd_joint_scale1.5_1779556615": ("joint", 1.5),
    }
    api = wandb.Api()
    frames = []
    for run_name, (scenario, level) in runs.items():
        run = api.run(f"{entity}/{project}/{run_name}")
        hist = pd.DataFrame(list(run.scan_history(keys=["_step", "train/loss"])))
        hist = hist.dropna(subset=["train/loss"]).sort_values("_step").reset_index(drop=True)
        hist["run_name"] = run_name
        hist["scenario"] = scenario
        hist["level"] = level
        frames.append(hist[["run_name", "scenario", "level", "_step", "train/loss"]])
    df = pd.concat(frames, ignore_index=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ANALYSIS_DIR / _COLLAPSE_PARQUET, index=False)
    return df


def _load_degraded_fixed_geometry():
    """Alignment + image/text uniformity per degraded/fixed run, tagged (vision, role).

    Reads data/analysis/uni_align_degraded_fixed_v2.json (from bin/run_uni_align.py over
    a models-list file) and pivots per held-out dataset.
    """
    path = ANALYSIS_DIR / _DEGRADED_FIXED_UNIALIGN_JSON
    g = pd.read_json(path)
    g["run_name"] = (g["model_path"].astype(str)
                     .str.replace(r"/checkpoints.*", "", regex=True)
                     .str.split("openclip_logs/").str[-1])
    roles = _degraded_fixed_run_role()
    g[["vision", "role"]] = g["run_name"].apply(lambda r: pd.Series(roles.get(r, (None, None))))
    g = g[g["role"].notna()].copy()
    piv = g.pivot_table(index=["run_name", "vision", "role"], columns="dataset",
                        values=["alignment", "image_uniformity", "text_uniformity"])
    piv.columns = [f"{m}_{d}" for m, d in piv.columns]
    return piv.reset_index()


_GEOMETRY_FIG_DATA = ANALYSIS_DIR / "geometry_prediction_figure_data.csv"


def _load_geometry_fig_data():
    if not _GEOMETRY_FIG_DATA.exists():
        raise FileNotFoundError(
            f"{_GEOMETRY_FIG_DATA} not found; run "
            "`python bin/zs_perf_regression_geometry.py` first.")
    return pd.read_csv(_GEOMETRY_FIG_DATA)


def _load_learning_curves_frame(vision_encoder, text_encoders):
    """Per-epoch training metrics for one vision encoder, from the wandb cache.

    Concatenates data/wandb_cache/*.parquet, derives the dataset (cc3m/cc12m) and
    encoder names from the model string, and keeps the rows for ``vision_encoder``
    whose text encoder is in ``text_encoders`` and that have a recorded epoch.
    Consumed by the learning-curve figures.
    """
    combined = pd.concat(
        [pd.read_parquet(f) for f in WANDB_CACHE_DIR.glob("*.parquet")],
        ignore_index=True,
    )
    combined["dataset"] = combined["pretrain_dataset"].apply(
        lambda x: "cc3m" if "cc3m" in x else "cc12m"
    )
    combined["text_name"] = combined["model"].str.split("--").str[1]
    combined["vision_name_raw"] = combined["model"].str.split("--").str[0]
    return combined[
        (combined["text_name"].isin(text_encoders))
        & (combined["vision_name_raw"] == vision_encoder)
        & (combined["epoch"].notna())
    ]
