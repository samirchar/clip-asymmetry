#!/usr/bin/env python3
"""Compute the contrastive validation loss per epoch by calling bin/train.py.

This is a thin orchestrator: for every (run, epoch checkpoint, eval dataset) it
invokes ``bin/train.py`` in eval-only mode (``--val-data`` set, no ``--train-data``,
``--resume <epoch_N.pt>``), which loads the checkpoint, runs ``evaluate()`` and
appends the metrics (``clip_val_loss`` + retrieval Recall@K) to a ``results.jsonl``.
The orchestrator parses those and aggregates everything into a single tidy parquet.

Reusing train.py as a subprocess means the loss is computed by the *exact* training
code (no duplicated loss loop) and train.py itself is left untouched. The cost is one
process launch + model rebuild per (run, epoch, dataset); use ``--gpus`` to fan the
sweep out across GPUs.

Each output row is one (run, epoch, val_dataset) with columns:
    run_name, model, model_dir, epoch, checkpoint_file, val_dataset,
    clip_val_loss, image_to_text_R@1, text_to_image_R@1, num_samples,
    batch_size, precision.

The converted eval sets (see bin/create_eval_wds.py) are single-shard, so with
``--workers 1`` batching is deterministic. Keep ``--batch-size`` fixed across the
sweep -- the in-batch loss depends on it.

Sample usage (single GPU):
    python bin/compute_val_loss.py \
Sample usage (single GPU, first & last epoch, both datasets):
    python bin/compute_val_loss.py \
        --models-file <models_list.txt> \
        --eval-wds-root data/eval_wds \
        --val-datasets mscoco_captions flickr30k \
        --output data/analysis/val_loss_cc12m.parquet

By default only the first and last epoch are evaluated (enough for a scalar OGR).
Use --epochs all for the full curve, or --epochs 1 35 for specific epochs.
Default --batch-size is 1000 (full-set for Flickr; evenly divides COCO).

Fan out across 8 GPUs:
    python bin/compute_val_loss.py <same args> --gpus 0 1 2 3 4 5 6 7
"""

import argparse
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import pandas as pd

from src.utils import resolve_models, parse_webdataset_path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(REPO_ROOT, "bin", "train.py")


def parse_args():
    parser = argparse.ArgumentParser(description="Per-epoch contrastive val loss via train.py eval-only.")
    parser.add_argument("--models-file", type=str, default=None,
                        help="Text file listing run directories (one run dir per line).")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Explicit run directories (alternative to --models-file).")
    parser.add_argument("--models-root", type=str, default=None,
                        help="Prefix prepended to each (relative) run dir for LOCAL runs, e.g. /mnt/blob. "
                             "On AMLT, AMLT_DATA_DIR is used automatically (matches bin/benchmark.py). "
                             "Ignored for already-absolute paths that resolve as-is.")
    parser.add_argument("--eval-wds-root", type=str, default="data/eval_wds",
                        help="Root of converted eval webdatasets (<root>/<dataset>/<split>/*.tar).")
    parser.add_argument("--val-datasets", type=str, nargs="+", default=["mscoco_captions", "flickr30k"],
                        help="Eval dataset names under --eval-wds-root.")
    parser.add_argument("--split", type=str, default="test", help="Eval split (default: test).")
    parser.add_argument("--caption-key", type=str, default="caption",
                        help="JSON caption key in the eval shards (default: caption).")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Eval batch size (in-batch negatives). MUST be fixed across the sweep. "
                             "1000 is full-set for Flickr and evenly divides COCO (5000).")
    parser.add_argument("--epochs", type=str, nargs="+", default=["first-last"],
                        help="Which epochs to evaluate: 'first-last' (default = min & max checkpoints), "
                             "'all', or explicit epoch numbers e.g. --epochs 1 35.")
    parser.add_argument("--workers", type=int, default=1,
                        help="train.py dataloader workers (default 1 for deterministic batching).")
    parser.add_argument("--precision", type=str, default="amp", help="train.py precision (default: amp).")
    parser.add_argument("--gpus", type=str, nargs="+", default=None,
                        help="GPU ids to fan out across (e.g. 0 1 2 3). Default: single process, inherited env.")
    parser.add_argument("--output", type=str, required=True, help="Output parquet path.")
    parser.add_argument("--scratch-dir", type=str, default=None,
                        help="Scratch dir for per-eval train.py logs (default: a temp dir, cleaned up).")
    return parser.parse_args()


def apply_models_root(dirs, models_root):
    """Prefix run dirs following bin/benchmark.py's convention.

    On AMLT (AMLT_DATA_DIR set) prepend it; otherwise prepend --models-root if given.
    Paths are lstrip('/')-ed before joining so relative configs (e.g. openclip_logs/...)
    resolve the same way they do for benchmark.py. With no prefix, dirs are used as-is.
    """
    amlt_data_dir = os.environ.get("AMLT_DATA_DIR")
    root = amlt_data_dir if amlt_data_dir is not None else models_root
    if root is None:
        return [d.rstrip("/") for d in dirs]
    return [os.path.join(root, d.lstrip("/").rstrip("/")) for d in dirs]


def resolve_runs(args):
    """Return list of (architecture, run_dir), reusing src.utils.resolve_models.

    Applies the same AMLT/prefix path handling as bin/benchmark.py so relative run
    dirs in the models file resolve identically across both scripts.
    """
    if (args.models_file is None) == (args.models is None):
        raise ValueError("Provide exactly one of --models-file or --models.")
    if args.models_file is not None:
        with open(args.models_file) as f:
            dirs = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    else:
        dirs = list(args.models)
    dirs = apply_models_root(dirs, args.models_root)

    # Reuse resolve_models (which reads a file) by writing the prefixed dirs to a temp file.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        tf.write("\n".join(dirs))
        tmp_path = tf.name
    try:
        pairs = resolve_models(tmp_path)
    finally:
        os.remove(tmp_path)
    return [(arch, os.path.dirname(os.path.dirname(ckpt))) for arch, ckpt in pairs]


def discover_epoch_checkpoints(run_dir):
    """Return sorted [(epoch_int, checkpoint_path)] for epoch_<N>.pt files."""
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        return []
    epoch_re = re.compile(r"^epoch_(\d+)\.pt$")
    epochs = [(int(m.group(1)), os.path.join(ckpt_dir, f))
              for f in os.listdir(ckpt_dir) if (m := epoch_re.match(f))]
    epochs.sort(key=lambda x: x[0])
    return epochs


def select_epochs(epochs, spec):
    """Filter discovered epochs by --epochs spec: 'first-last', 'all', or explicit numbers."""
    if not epochs:
        return []
    if spec == ["all"]:
        return epochs
    if spec == ["first-last"]:
        chosen = {epochs[0][0], epochs[-1][0]}  # dedup if only one epoch exists
        return [e for e in epochs if e[0] in chosen]
    wanted = {int(s) for s in spec}
    selected = [e for e in epochs if e[0] in wanted]
    missing = wanted - {e[0] for e in selected}
    if missing:
        print(f"  Warning: requested epochs not found and skipped: {sorted(missing)}")
    return selected


def resolve_val_data(root, dataset, split):
    """Return (val_data_path, num_samples) for a converted eval dataset."""
    split_dir = os.path.join(root, dataset, split)
    shards = sorted(glob(os.path.join(split_dir, "*.tar")))
    if not shards:
        raise FileNotFoundError(f"No .tar shards found in {split_dir}")
    val_data = shards[0] if len(shards) == 1 else parse_webdataset_path(os.path.join(split_dir, "*.tar"))
    count_path = os.path.join(split_dir, "nsamples.txt")
    num_samples = int(open(count_path).read().strip()) if os.path.isfile(count_path) else 0
    return val_data, num_samples


def run_eval(task, args, scratch_root, gpu):
    """Invoke train.py eval-only for one (run, epoch, dataset); return a result row or None."""
    arch, run_dir, run_name, epoch, ckpt, dataset, val_data, num_samples = task
    name = f"{run_name}__{dataset}__ep{epoch}"
    logs_dir = tempfile.mkdtemp(prefix="valloss_", dir=scratch_root)
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        "--model", arch,
        "--dataset-type", "auto",
        "--val-data", val_data,
        "--val-num-samples", str(num_samples),
        "--caption-key", args.caption_key,
        "--batch-size", str(args.batch_size),
        "--precision", args.precision,
        "--workers", str(args.workers),
        "--val-frequency", "1",
        "--zeroshot-frequency", "0",
        "--resume", ckpt,
        "--logs", logs_dir,
        "--name", name,
    ]
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    try:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
        results_path = os.path.join(logs_dir, name, "checkpoints", "results.jsonl")
        if proc.returncode != 0 or not os.path.isfile(results_path):
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            print(f"  FAILED {name} (rc={proc.returncode}): {' | '.join(tail)}")
            return None
        val_rows = [json.loads(l) for l in open(results_path) if l.strip() and "clip_val_loss" in l]
        if not val_rows:
            print(f"  FAILED {name}: no clip_val_loss in results.jsonl")
            return None
        m = val_rows[-1]
        return {
            "run_name": run_name, "model": arch, "model_dir": run_dir,
            "epoch": epoch, "checkpoint_file": ckpt, "val_dataset": dataset,
            "clip_val_loss": m["clip_val_loss"],
            "image_to_text_R@1": m.get("image_to_text_R@1"),
            "text_to_image_R@1": m.get("text_to_image_R@1"),
            "num_samples": m.get("num_samples"),
            "batch_size": args.batch_size, "precision": args.precision,
        }
    finally:
        shutil.rmtree(logs_dir, ignore_errors=True)


def main():
    args = parse_args()
    runs = resolve_runs(args)
    if not runs:
        print("No runs resolved. Exiting.")
        return
    val_specs = {ds: resolve_val_data(args.eval_wds_root, ds, args.split) for ds in args.val_datasets}

    # Build the full task list.
    tasks = []
    for arch, run_dir in runs:
        run_name = os.path.basename(run_dir)
        epochs = select_epochs(discover_epoch_checkpoints(run_dir), args.epochs)
        if not epochs:
            print(f"{run_name}: no matching epoch_*.pt found, skipping.")
            continue
        for dataset, (val_data, num_samples) in val_specs.items():
            for epoch, ckpt in epochs:
                tasks.append((arch, run_dir, run_name, epoch, ckpt, dataset, val_data, num_samples))
    print(f"Total evals: {len(tasks)} ({len(runs)} runs x {len(val_specs)} datasets x epochs)")

    scratch_root = args.scratch_dir or tempfile.mkdtemp(prefix="valloss_sweep_")
    os.makedirs(scratch_root, exist_ok=True)
    gpus = args.gpus if args.gpus else [None]

    rows = []
    try:
        if len(gpus) == 1:
            for i, task in enumerate(tasks):
                row = run_eval(task, args, scratch_root, gpus[0])
                if row:
                    rows.append(row)
                print(f"[{i + 1}/{len(tasks)}] {task[2]} {task[5]} ep{task[3]} -> "
                      f"{row['clip_val_loss']:.4f}" if row else f"[{i + 1}/{len(tasks)}] {task[2]} failed")
        else:
            gpu_q = queue.Queue()
            for g in gpus:
                gpu_q.put(g)

            def worker(task):
                gpu = gpu_q.get()
                try:
                    return run_eval(task, args, scratch_root, gpu)
                finally:
                    gpu_q.put(gpu)

            done = 0
            with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                futures = {pool.submit(worker, t): t for t in tasks}
                for fut in as_completed(futures):
                    done += 1
                    row = fut.result()
                    if row:
                        rows.append(row)
                    t = futures[fut]
                    status = f"{row['clip_val_loss']:.4f}" if row else "failed"
                    print(f"[{done}/{len(tasks)}] {t[2]} {t[5]} ep{t[3]} -> {status}")
    finally:
        if not args.scratch_dir:
            shutil.rmtree(scratch_root, ignore_errors=True)

    if not rows:
        print("No successful evals. Nothing written.")
        return
    df = pd.DataFrame(rows).sort_values(["run_name", "val_dataset", "epoch"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"\nWrote {len(df)} rows to {args.output}")
    print(f"  runs={df['run_name'].nunique()} datasets={sorted(df['val_dataset'].unique())}")


if __name__ == "__main__":
    main()
