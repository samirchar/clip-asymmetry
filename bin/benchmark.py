"""
Benchmark script for evaluating CLIP models using clip_benchmark.

This script provides two subcommands:

  eval  - Evaluates trained models on a set of datasets. For each model directory
          matching --models-pattern, it reads params.txt and results.jsonl to
          identify the best and latest checkpoints, generates a models.txt file
          compatible with clip_benchmark, and runs evaluation.
          Output: One JSON file per (dataset, model, checkpoint) combination in
          --output-dir, e.g. benchmarks/benchmark_cc3m_compression_{dataset}_{pretrained}_{model}_{language}_{task}.json

  build - Aggregates individual benchmark JSON outputs into a single CSV file.
          Output: A single CSV file in --output-dir, e.g. benchmarks/cc3m_compression_benchmark.json

Usage:
    torchrun --nproc_per_node=8 bin/benchmark.py eval --name cc3m_compression --models-pattern "openclip_logs/*/" --datasets-path configs/eval/webdatasets.txt --distributed
    python bin/benchmark.py build --name cc3m_compression --files benchmarks/benchmark_*.json

Note: 
  - Do not quote the glob pattern for --files so the shell can expand it.
  - No process group is initialized. Ranks coordinate via env vars (RANK, WORLD_SIZE)
    and filesystem signals. Each rank runs independently after models.txt is ready.

Prerequisites:
  - A webdatasets.txt file must exist in --datasets-path (default: configs/eval/)
    listing the datasets to evaluate on (one per line).

Custom model architectures are registered via register_models() so that
clip_benchmark / open_clip can load them by name.
"""

import ast
import glob
import json
import sys
from src.utils import register_models
import os
import shutil
import subprocess
import tarfile
import time
import csv
from clip_benchmark.cli import main as cb_main
import argparse
from src.utils import LATEST_CHECKPOINT_NAME, BEST_CHECKPOINT_NAME, read_openclip_params
import tqdm
is_amlt = os.environ.get("AMLT_DATA_DIR", None) is not None


def _wait_for_file(path: str, poll_interval: float = 5.0, timeout: float = 600.0):
    """Poll until a file exists on the filesystem, with a timeout."""
    elapsed = 0.0
    while not os.path.exists(path):
        if elapsed >= timeout:
            raise TimeoutError(f"Timed out after {timeout}s waiting for {path}")
        time.sleep(poll_interval)
        elapsed += poll_interval


def _curl_supports_retry_all_errors():
    """Return True if the system curl supports --retry-all-errors (curl >= 7.71)."""
    try:
        out = subprocess.run(
            ["curl", "--help", "all"],
            capture_output=True, text=True, timeout=10,
        ).stdout
        return "--retry-all-errors" in out
    except Exception:
        return False


def _is_complete_tar(path):
    """True iff `path` is a fully-downloaded, readable tar archive.

    Two independent checks: (1) walk every member -- a body truncated by a
    mid-stream drop raises ReadError; (2) the archive ends in the standard two
    512-byte zero blocks. This is a secondary guard in _robust_download_to_file;
    the primary guarantee is curl exiting 0 after receiving the full
    Content-Length.
    """
    try:
        if not os.path.exists(path) or os.path.getsize(path) < 1024:
            return False
        with open(path, "rb") as f:
            f.seek(-1024, os.SEEK_END)
            if f.read(1024) != b"\x00" * 1024:
                return False
        with tarfile.open(path, "r:") as tf:
            n = 0
            for _ in tf:
                n += 1
            return n > 0
    except Exception:
        return False


def patch_webdataset_safe_streaming():
    """Make webdataset stream shards via curl with *safe* retries.

    The default opener is ``curl -f -s -L`` with no retries; combined with the
    default reraise handler, a single transient connection error aborts the
    whole eval. We add retries for transient errors that occur *before* any
    body bytes are written (timeouts, connection failures, 5xx). We deliberately
    do NOT use ``-C -`` or ``--retry-all-errors`` here: retrying a *mid-stream*
    drop on a non-rewindable pipe concatenates duplicate bytes and corrupts the
    stream (manifesting as ``tarfile.ReadError`` or PIL ``broken data stream``).
    Mid-stream recovery is handled instead by the file cache (see
    patch_webdataset_robust_cache_download), which requires --wds-cache-dir.
    """
    try:
        from webdataset.gopen import gopen_schemes, Pipe
    except Exception as exc:  # defensive: never let the patch break eval
        print(f"[benchmark] Skipped webdataset streaming patch (import failed): {exc}")
        return

    retry_flags = "--retry 5 --retry-delay 2 --connect-timeout 30"

    def safe_gopen_curl(url, mode="rb", bufsize=8192):
        if mode[0] == "r":
            cmd = f"curl -f -s -L {retry_flags} '{url}'"
            return Pipe(cmd, mode=mode, shell=True, bufsize=bufsize,
                        ignore_status=[141, 23])
        if mode[0] == "w":
            cmd = f"curl -f -s -X PUT -L -T - '{url}'"
            return Pipe(cmd, mode=mode, shell=True, bufsize=bufsize,
                        ignore_status=[141, 26])
        raise ValueError(f"{mode}: unknown mode")

    for scheme in ("http", "https"):
        if scheme in gopen_schemes:
            gopen_schemes[scheme] = safe_gopen_curl
    print(f"[benchmark] Patched webdataset streaming: curl {retry_flags} (pipe-safe)")


def patch_webdataset_robust_cache_download():
    """Make the webdataset file cache download shards resiliently.

    When --wds-cache-dir is set, webdataset downloads each shard to a local file
    before reading it. We replace its ``download`` with a curl that writes
    directly to a *seekable* file and resumes from the partial file on every
    retry (``-C -`` + ``--retry-all-errors``). Because the output is a real file
    (not a pipe), resume is correct: HF's Xet CDN honors HTTP Range, so curl
    fetches only the missing bytes and continues until the full Content-Length
    is received. The shard is verified complete and only then promoted to its
    final cache path, so a truncated download is never cached or read.

    This is what actually recovers from the mid-stream drops seen on the cluster
    (``unexpected end of data`` / ``broken data stream``).
    """
    try:
        import webdataset.cache as wds_cache
        from webdataset.cache import pipe_cleaner
    except Exception as exc:  # defensive: never let the patch break eval
        print(f"[benchmark] Skipped webdataset cache patch (import failed): {exc}")
        return

    file_flags = "--retry 10 --retry-delay 3 --connect-timeout 30 --max-time 7200"
    if _curl_supports_retry_all_errors():
        file_flags += " --retry-all-errors"

    def robust_download(url, dest, chunk_size=1024 ** 2, verbose=False):
        real_url = pipe_cleaner(url)
        temp = dest + ".tmp"
        last = None
        for attempt in range(1, 5):
            # curl -> seekable file; -C - safely resumes the partial file.
            cmd = (f"curl -f -L --silent --show-error {file_flags} -C - "
                   f"-o '{temp}' '{real_url}'")
            rc = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
            if rc.returncode == 0 and _is_complete_tar(temp):
                os.replace(temp, dest)
                return dest
            last = f"rc={rc.returncode} stderr={rc.stderr.strip()[:200]}"
            if verbose:
                print(f"[benchmark] shard download attempt {attempt} failed: {last}")
            time.sleep(min(3 * attempt, 15))
        if os.path.exists(temp):
            os.remove(temp)
        raise IOError(f"Failed to download {real_url} after retries ({last})")

    wds_cache.download = robust_download
    print(f"[benchmark] Patched webdataset cache download: curl {file_flags} -C - (file-safe)")


register_models()
patch_webdataset_safe_streaming()
patch_webdataset_robust_cache_download()


def read_results(results_path: str) -> list:
    # results.jsonl is opened in append mode and flushed through blobfuse, so a
    # crashed/resumed run can leave NUL-byte gaps (partial writes) or truncated
    # lines. str.strip() does NOT remove NUL bytes, so a naive json.loads over
    # every non-blank line crashes on those rows. Drop NULs and skip any line
    # that isn't valid JSON, keeping the surrounding good rows.
    results = []
    with open(results_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.replace("\x00", "").strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"[benchmark] skipping corrupt line in {results_path}: {line[:60]!r}")
    # Newer runs (zero-shot-only, no val set) don't write an "epoch" key, but
    # append exactly one row per epoch in ascending order. Fall back to the
    # 1-based row index so downstream logic can still locate epoch_{N}.pt.
    for i, row in enumerate(results, start=1):
        row.setdefault("epoch", i)
    return results

def get_best_epoch(results: list = None, results_path: str = None, metric: str = "imagenet-zeroshot-val-top1", lower_is_better: bool = False) -> int:
    assert (results is not None) ^ (results_path is not None), "Provide either results or results_path, not both."
    if results is None:
        results = read_results(results_path)
    key = min if lower_is_better else max
    best = key(results, key=lambda r: r[metric])
    return best["epoch"]


def eval_benchmark(
    name: str,
    models_dir: str,
    datasets_path: str,
    output_dir: str = "benchmarks",
    dataset_root: str = "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main",
    task: str = "auto",
    distributed: bool = False,
    features_root: str = None,
    val_proportion: float = None,
    resume: bool = False,
    dynamic_load_balancing: bool = False,
    wds_cache_dir: str = None
):
    pretrained_model = os.path.join(models_dir, f"{name}_models.txt")
    output = os.path.join(output_dir, f"benchmark_{name}_"+"{dataset}_{pretrained_full_path}_{model}_{language}_{task}.json")

    original_argv = sys.argv
    try:
        sys.argv = [
            "clip_benchmark", "eval",
            "--pretrained_model", pretrained_model,
            "--dataset", datasets_path,
            "--dataset_root", dataset_root,
            "--output", output,
            "--task", task,
            "--recall_k", "1",
        ]
        if val_proportion is not None:
            sys.argv.extend(["--val-proportion", str(val_proportion)])
        if features_root is not None:
            sys.argv.extend(["--feature_root", features_root])
        if wds_cache_dir is not None:
            sys.argv.extend(["--wds_cache_dir", wds_cache_dir])
        if distributed:
            sys.argv.append("--distributed")
        if resume:
            sys.argv.append("--skip_existing")
        if dynamic_load_balancing:
            sys.argv.append("--dynamic_load_balancing")
        
        cb_main()
    finally:
        sys.argv = original_argv

def build_benchmark(name: str, output_dir: str, files: list):
    output = os.path.join(output_dir,f"{name}_benchmark.csv")
    original_argv = sys.argv
    try:
        sys.argv = [
            "clip_benchmark", "build",
            *files,
            "--output", output,
        ]
        cb_main()
    finally:
        sys.argv = original_argv

def main(args):

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    is_rank0 = (rank == 0)
    features_root = None # Only set for AMLT runs, points to a writable directory for storing extracted features during eval.
    if is_amlt:
        amlt_data_dir = os.environ.get("AMLT_DATA_DIR")
        amlt_output_dir = os.environ.get("AMLT_DATA_DIR")
        args.output_dir_output_path = os.path.join(amlt_output_dir, args.output_dir.lstrip('/'))
        args.output_dir = os.path.join(amlt_data_dir, args.output_dir.lstrip('/'))
        features_root = os.path.join(amlt_data_dir, "/scratch/features/") # Used by clip_benchmark to store extracted features. Must be on a writable path.
        os.makedirs(features_root, exist_ok=True)
        
        os.makedirs(args.output_dir_output_path, exist_ok=True)
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    
    if args.which == "eval":
        if is_amlt:
            if args.models_pattern is not None:
                args.models_pattern = os.path.join(amlt_data_dir, args.models_pattern.lstrip('/'))
            if args.models_list is not None:
                args.models_list = [os.path.join(amlt_data_dir, p.lstrip('/')) for p in args.models_list]
            # if args.models_list_file is not None:
            #     args.models_list_file = os.path.join(amlt_data_dir, args.models_list_file.lstrip('/'))
            args.models_dir = os.path.join(amlt_data_dir, args.models_dir.lstrip('/'))

        pretrained_model = os.path.join(args.models_dir, f"{args.name}_models.txt")

        if is_rank0:
            if args.models_list is not None:
                model_dirs = args.models_list
            elif args.models_list_file is not None:
                with open(args.models_list_file) as f:
                    model_dirs = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                if is_amlt:
                    model_dirs = [os.path.join(amlt_data_dir, p.lstrip('/')) for p in model_dirs]
            elif args.models_pattern is not None:
                model_dirs = glob.glob(args.models_pattern, recursive=False)
            else:
                raise ValueError("Either --models-pattern, --models-list, or --models-list-file must be provided.")
            
            print(f"Found {len(model_dirs)} model directories.")
            os.makedirs(args.models_dir, exist_ok=True)

            # Creates the models.txt if doesn't exist.
            # Write to a temp file first, then atomically rename so that
            # other ranks polling with _wait_for_file only see the complete file.
            if not os.path.isfile(pretrained_model):
                pretrained_model_tmp = pretrained_model + ".tmp"
                f = open(pretrained_model_tmp, "w", newline="")
                writer = csv.writer(f)
                print("Created models.txt, writing model checkpoints...")
                for model_dir in tqdm.tqdm(model_dirs, desc="Writing models.txt"):
                    best_epoch_checkpoint = os.path.join(model_dir,"checkpoints",BEST_CHECKPOINT_NAME)
                    last_epoch_checkpoint = os.path.join(model_dir,"checkpoints",LATEST_CHECKPOINT_NAME)
                    best_epoch_exists = os.path.isfile(best_epoch_checkpoint)
                    last_epoch_exists = os.path.isfile(last_epoch_checkpoint)

                    model_params = read_openclip_params(os.path.join(model_dir,"params.txt"))

                    if not best_epoch_exists or not last_epoch_exists:
                        results = read_results(f"{model_dir}/checkpoints/results.jsonl")
                        if not best_epoch_exists: 
                            best_epoch = get_best_epoch(results=results)
                            shutil.copy(f"{model_dir}/checkpoints/epoch_{best_epoch}.pt", best_epoch_checkpoint)
                            print(f"Copied epoch_{best_epoch}.pt to {BEST_CHECKPOINT_NAME} for {model_dir}")
                        if not last_epoch_exists:
                            last_epoch = results[-1]["epoch"]
                            shutil.copy(f"{model_dir}/checkpoints/epoch_{last_epoch}.pt", last_epoch_checkpoint)
                            print(f"Copied epoch_{last_epoch}.pt to {LATEST_CHECKPOINT_NAME} for {model_dir}")

                    rows = [
                        [model_params.get("model"),best_epoch_checkpoint],
                        [model_params.get("model"),last_epoch_checkpoint]
                        ]
                    writer.writerows(rows)
                f.close()
                os.replace(pretrained_model_tmp, pretrained_model)
        
        if is_amlt:
            # Wait for blobfuse to propagate copied checkpoints to blob storage
            print("Waiting for blobfuse to sync checkpoints...")
            time.sleep(60*5) #TODO: there must be a better way than sleep
            print("Done waiting for blobfuse sync.")


        # Wait for rank 0 to finish writing models.txt (file-based signal, no process group needed)
        if not is_rank0:
            print("Waiting for rank 0 to finish writing models.txt...")
            _wait_for_file(pretrained_model, poll_interval=5.0, timeout=600.0)
            print("models.txt found, proceeding with evaluation")

        #Run eval with clip benchmark
        print(f"[Rank {rank}] Running evaluation with CLIP Benchmark...")
        # Robust recovery from mid-stream HF shard drops requires downloading
        # shards to files (the cache path). On AMLT default to node-local
        # scratch so the eval is resilient without any extra flags; override
        # with --wds-cache-dir, or set it to "none" to disable.
        cache_root = args.wds_cache_dir
        if cache_root is None and is_amlt:
            cache_root = "/scratch/wds_cache"
        wds_cache_dir = None
        if cache_root is not None and str(cache_root).lower() != "none":
            # Per-rank subdir so concurrent ranks never write the same cache file.
            wds_cache_dir = os.path.join(cache_root, f"rank{rank}")
            os.makedirs(wds_cache_dir, exist_ok=True)
            print(f"[Rank {rank}] Caching webdataset shards under {wds_cache_dir}")
        eval_benchmark(
            name=args.name,
            models_dir=args.models_dir,
            datasets_path=args.datasets_path,
            output_dir=args.output_dir_output_path if is_amlt else args.output_dir,
            dataset_root=args.dataset_root,
            task=args.task,
            distributed=args.distributed,
            features_root=features_root,
            val_proportion=args.val_proportion,
            resume=args.resume,
            dynamic_load_balancing=args.dynamic_load_balancing,
            wds_cache_dir=wds_cache_dir
        )

    if args.which == "build" and is_rank0:
        #Build
        if is_amlt:
            args.files = [os.path.join(amlt_data_dir, f.lstrip('/')) for f in args.files] if args.files is not None else None
            
        if args.files is None:
            args.files = glob.glob(
                os.path.join(
                    args.output_dir_output_path if is_amlt else args.output_dir,f"benchmark_{args.name}_*")
                )

        print("Building benchmark with CLIP Benchmark...")
        build_benchmark(
            name=args.name,
            output_dir= args.output_dir_output_path if is_amlt else args.output_dir,
            files=args.files
        )
        print(f"Benchmark built successfully at {os.path.join(args.output_dir_output_path if is_amlt else args.output_dir,f'{args.name}_benchmark.csv')}")



if __name__ == "__main__":

    """
    Sample usage:
    torchrun --nproc_per_node=8 bin/benchmark.py eval --name cc3m_caption_131232 --models-pattern "openclip_logs/cc3m_caption*/" --datasets-path configs/eval/webdatasets.txt --distributed
    python bin/benchmark.py build --name cc3m_caption_131232 --files benchmarks/benchmark_cc3m_caption_131232*.json

    Note: Do not quote the glob pattern for --files so the shell can expand it.
    """
    parser = argparse.ArgumentParser(description="Evaluate and build CLIP benchmark.")
    subparsers = parser.add_subparsers()


    # Evaluation subcommand
    parser_eval = subparsers.add_parser("eval", help="Evaluate models using CLIP Benchmark.")
    parser_eval.add_argument("--name", type=str, help="Name for the benchmark.", required=True)
    parser_eval.add_argument("--datasets-path", type=str, help="Directory containing webdatasets.txt listing the datasets to evaluate on.", required=True)
    parser_eval.add_argument("--output-dir", type=str, default="benchmarks", help="Directory to save benchmark results.")
    parser_eval.add_argument("--models-pattern", type=str, help="Glob pattern to locate model directories.", default=None)
    parser_eval.add_argument("--models-list", type=str, nargs='+', default=None, help="Explicit list of model directory paths. Alternative to --models-pattern.")
    parser_eval.add_argument("--models-list-file", type=str, default=None, help="Path to a text file listing model directories (one per line). Alternative to --models-pattern.")
    parser_eval.add_argument("--models-dir", type=str, default="configs/eval/", help="Directory containing models.txt indicating the checkpoints to evaluate.")
    parser_eval.add_argument("--dataset-root", type=str, default="https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main", help="Root URL for dataset shards.")
    parser_eval.add_argument("--task", type=str, default="auto", help="Task type for evaluation (e.g., 'auto', 'linear_probe')")
    parser_eval.add_argument("--val-proportion", type=float, default=None, help="Proportion of training data to use for validation probing tasks")
    parser_eval.add_argument("--distributed", action="store_true", help="Whether to use distributed evaluation. Eval runs distributed across ranks")
    parser_eval.add_argument("--resume", action="store_true", help="Skip already completed benchmarks (uses clip_benchmark --skip_existing).")
    parser_eval.add_argument("--dynamic-load-balancing", action="store_true", help="Use file-based dynamic task claiming instead of static round-robin partitioning. Requires --distributed.")
    parser_eval.add_argument("--wds-cache-dir", type=str, default=None, help="Local directory to cache webdataset shards (a per-rank subdir is created). Required for robust recovery from mid-stream HF drops; also dedupes downloads across checkpoints. Defaults to /scratch/wds_cache on AMLT; pass 'none' to disable.")
    parser_eval.set_defaults(which="eval")
    
    # Build subcommand
    parser_build = subparsers.add_parser("build", help="Build benchmark using CLIP Benchmark.")
    parser_build.add_argument("--name", type=str, help="Name for the benchmark.", required=True)
    parser_build.add_argument("--output-dir", type=str, default="benchmarks", help="Directory to save benchmark results.")
    parser_build.add_argument("--files", type=str, nargs='+', help="List of input benchmark JSON files.", required=True)
    parser_build.set_defaults(which="build")

    args = parser.parse_args()
    main(args)
