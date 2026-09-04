#!/usr/bin/env python
"""Export trained run checkpoints into an upload-ready folder tree for the Hugging Face Hub.

Copies each selected run's params.txt and latest checkpoint into
<out>/<folder>/{params.txt, checkpoints/epoch_latest.pt}. Full checkpoint by default;
--weights-only keeps just the model weights (~3x smaller). --tags filters by wandb tag;
--name-by model names folders by model id (requires one run per model).
"""
import argparse
import glob
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = "epoch_latest.pt"


def read_runs():
    """Return [{run, model, tags}] from the wandb cache."""
    import pandas as pd

    runs = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data/wandb_cache/*.parquet"))):
        df = pd.read_parquet(f, columns=["run_name", "model", "wandb_tags"])

        def first(col):
            s = df[col].dropna()
            return s.iloc[0] if len(s) else ""

        runs.append({
            "run": first("run_name") or os.path.basename(f)[:-8],
            "model": first("model"),
            "tags": {t for t in str(first("wandb_tags")).split("|") if t},
        })
    return runs


def write_checkpoint(src, dst, weights_only):
    if not weights_only:
        shutil.copyfile(src, dst)
        return
    import torch

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = {k: ckpt[k] for k in ("epoch", "name", "state_dict") if k in ckpt}
    torch.save(ckpt, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="Output directory (upload this to the HF Hub).")
    ap.add_argument("--ckpt-root", default=os.path.join(ROOT, "openclip_logs"),
                    help="Root holding <run>/checkpoints/epoch_latest.pt.")
    ap.add_argument("--tags", nargs="+", help="Only export runs with any of these wandb tags.")
    ap.add_argument("--weights-only", action="store_true", help="Keep only model weights (~3x smaller).")
    ap.add_argument("--name-by", choices=["run", "model"], default="run",
                    help="Folder name: run name (default) or model id (requires one run per model).")
    args = ap.parse_args()

    runs = read_runs()
    if args.tags:
        wanted = set(args.tags)
        runs = [r for r in runs if r["tags"] & wanted]
    if not runs:
        sys.exit("No runs selected.")

    if args.name_by == "model":
        models = [r["model"] for r in runs]
        dups = sorted({m for m in models if models.count(m) > 1})
        if dups:
            sys.exit(f"--name-by model requires one run per model; multiple runs for: {dups}")

    exported = 0
    for r in sorted(runs, key=lambda x: x["run"]):
        params = os.path.join(args.ckpt_root, r["run"], "params.txt")
        ckpt = os.path.join(args.ckpt_root, r["run"], "checkpoints", CKPT)
        if not (os.path.isfile(params) and os.path.isfile(ckpt)):
            print(f"skip (missing): {r['run']}")
            continue
        folder = os.path.join(args.out, r["model"] if args.name_by == "model" else r["run"])
        os.makedirs(os.path.join(folder, "checkpoints"), exist_ok=True)
        shutil.copyfile(params, os.path.join(folder, "params.txt"))
        write_checkpoint(ckpt, os.path.join(folder, "checkpoints", CKPT), args.weights_only)
        exported += 1
        print(os.path.basename(folder))

    print(f"Exported {exported} run(s) to {args.out}")


if __name__ == "__main__":
    main()
