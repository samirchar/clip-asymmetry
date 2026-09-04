#!/usr/bin/env python3
"""Convert clip_benchmark retrieval webdatasets into train.py-compatible val shards.

clip_benchmark serves retrieval datasets (e.g. mscoco_captions, flickr30k) as
webdatasets where each sample is an image plus a ``.txt`` field holding *multiple*
captions, one per line. ``bin/train.py``'s val loader (``src.loaders.get_wds_dataset``)
instead expects each sample to carry:

  - an image (``jpg``/``png``/``jpeg``/``webp``),
  - a ``txt`` field (required by ``filter_no_caption_no_image_no_json``), and
  - a ``json`` field whose ``caption_key`` entry is the caption that actually gets
    tokenized (``make_sample`` reads ``sample["json"][caption_key]``).

This script bridges the two: it streams the clip_benchmark shards, selects
``--captions-per-image`` captions per image, and writes new shards in the format
above so they can be plugged straight into ``train.py`` via ``--val-data`` in
eval-only mode.

Note on multiple captions: ``train.py``'s eval loss is standard in-batch InfoNCE
(``labels = arange(batch_size)``), which treats every other row in a batch as a
negative. Emitting more than one caption per image (``--captions-per-image > 1``)
produces multiple rows sharing the same image, so captions of the same image
become *false negatives* and inflate the loss. Use ``--captions-per-image 1``
(the default) for a clean, comparable contrastive validation loss.

Sample usage:
    python bin/create_eval_wds.py \
        --datasets mscoco_captions flickr30k \
        --split test \
        --captions-per-image 1 \
        --output-dir data/eval_wds
"""

import argparse
import json
import os
import shutil
import subprocess
import tempfile

import webdataset as wds

IMAGE_EXTENSIONS = ("webp", "png", "jpg", "jpeg")
CLIP_BENCHMARK_REPO = "https://huggingface.co/datasets/clip-benchmark/wds_{name}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert clip_benchmark retrieval wds into train.py val shards."
    )
    parser.add_argument(
        "--datasets", type=str, nargs="+", required=True,
        help="clip_benchmark retrieval dataset name(s), e.g. mscoco_captions flickr30k.",
    )
    parser.add_argument(
        "--split", type=str, default="test",
        help="Dataset split to convert (default: test, the Karpathy eval split).",
    )
    parser.add_argument(
        "--captions-per-image", type=int, default=1,
        help="Number of captions to keep per image. Use 1 for a clean contrastive "
             "val loss; >1 emits one (image, caption) sample per caption and "
             "reintroduces in-batch false negatives.",
    )
    parser.add_argument(
        "--output-dir", type=str, required=True,
        help="Directory to write converted shards into (one subdir per dataset).",
    )
    parser.add_argument(
        "--caption-key", type=str, default="caption",
        help="JSON key under which the caption is stored, matching train.py's "
             "--caption-key (default: caption).",
    )
    parser.add_argument(
        "--maxcount", type=int, default=100000,
        help="Max samples per output shard (default: 100000, so a ~5k-image eval "
             "set fits in a single shard for deterministic ordering).",
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Optional cap on number of source images per dataset (for quick tests).",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Directory to download source shards into (default: a temp dir that is "
             "deleted afterwards). Use to reuse downloads across runs.",
    )
    parser.add_argument(
        "--keep-cache", action="store_true",
        help="Keep downloaded source shards instead of deleting them.",
    )
    return parser.parse_args()


def read_repo_text(name, rel_path, token=None):
    """Read a small text file from a clip_benchmark wds repo via the raw endpoint."""
    url = f"{CLIP_BENCHMARK_REPO.format(name=name)}/raw/main/{rel_path}"
    cmd = ["curl", "-sL", "--fail", "--retry", "3", "--retry-delay", "5"]
    if token:
        cmd += ["-H", f"Authorization: Bearer {token}"]
    cmd.append(url)
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def download_shard(name, split, shard_idx, dest_dir, token=None):
    """Download a single source .tar shard via the resolve endpoint, if not present."""
    url = f"{CLIP_BENCHMARK_REPO.format(name=name)}/resolve/main/{split}/{shard_idx}.tar"
    dest = os.path.join(dest_dir, f"{name}_{split}_{shard_idx}.tar")
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        return dest
    cmd = ["curl", "-sL", "--fail", "-C", "-", "--retry", "3", "--retry-delay", "5"]
    if token:
        cmd += ["-H", f"Authorization: Bearer {token}"]
    cmd += [url, "-o", dest]
    subprocess.run(cmd, check=True)
    return dest


def iter_source_samples(tar_path):
    """Yield (key, image_ext, image_bytes, [captions]) from a raw clip_benchmark shard."""
    for sample in wds.WebDataset(tar_path, shardshuffle=False):
        image_ext = next((e for e in IMAGE_EXTENSIONS if e in sample), None)
        if image_ext is None or "txt" not in sample:
            continue
        raw_txt = sample["txt"]
        if isinstance(raw_txt, bytes):
            raw_txt = raw_txt.decode("utf-8")
        captions = [c.strip() for c in raw_txt.splitlines() if c.strip()]
        if not captions:
            continue
        yield sample["__key__"], image_ext, sample[image_ext], captions


def convert_dataset(name, args, token):
    """Convert one clip_benchmark dataset split into train.py-compatible shards."""
    print(f"\n{'=' * 60}\nConverting {name} [{args.split}]")

    dataset_type = read_repo_text(name, "dataset_type.txt", token=token).lower()
    if dataset_type != "retrieval":
        print(f"  Warning: dataset_type is '{dataset_type}', expected 'retrieval'. "
              f"Captions may not be present; skipping.")
        return
    nshards = int(read_repo_text(name, f"{args.split}/nshards.txt", token=token))
    print(f"  Source shards: {nshards}")

    out_dir = os.path.join(args.output_dir, name, args.split)
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, f"{name}-{args.split}-%04d.tar")

    if args.cache_dir:
        cache_dir = args.cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        cleanup_cache = not args.keep_cache
    else:
        cache_dir = tempfile.mkdtemp(prefix=f"{name}_{args.split}_")
        cleanup_cache = not args.keep_cache

    n_images = 0
    n_samples = 0
    stop = False
    try:
        with wds.ShardWriter(pattern, maxcount=args.maxcount) as sink:
            for shard_idx in range(nshards):
                if stop:
                    break
                tar_path = download_shard(name, args.split, shard_idx, cache_dir, token=token)
                for key, image_ext, image_bytes, captions in iter_source_samples(tar_path):
                    selected = captions[:args.captions_per_image]
                    for cap_idx, caption in enumerate(selected):
                        out_key = f"{n_samples:08d}"
                        record = {
                            "__key__": out_key,
                            image_ext: image_bytes,
                            "txt": caption,
                            "json": {
                                args.caption_key: caption,
                                "orig_key": key,
                                "caption_index": cap_idx,
                            },
                        }
                        sink.write(record)
                        n_samples += 1
                    n_images += 1
                    if args.max_images is not None and n_images >= args.max_images:
                        stop = True
                        break
                if not args.keep_cache and not args.cache_dir:
                    os.remove(tar_path)
    finally:
        if cleanup_cache and os.path.isdir(cache_dir) and not args.cache_dir:
            shutil.rmtree(cache_dir, ignore_errors=True)

    count_path = os.path.join(out_dir, "nsamples.txt")
    with open(count_path, "w") as f:
        f.write(str(n_samples))

    print(f"  Wrote {n_samples} samples from {n_images} images to {out_dir}")
    print(f"  --val-data '{pattern.replace('%04d', '{0000..%04d}' % (0))}'  (single-shard example)")
    print(f"  --val-num-samples {n_samples}")
    return n_samples


def main():
    args = parse_args()
    token = os.environ.get("HF_TOKEN")

    is_amlt = os.environ.get("AMLT_DATA_DIR", None) is not None
    if is_amlt:
        amlt_data_dir = os.environ.get("AMLT_DATA_DIR")
        args.output_dir = os.path.join(amlt_data_dir, args.output_dir.lstrip("/"))

    if args.captions_per_image > 1:
        print("Warning: --captions-per-image > 1 emits multiple samples per image, "
              "which reintroduces in-batch false negatives in train.py's eval loss. "
              "Use 1 for a clean contrastive validation loss.")

    os.makedirs(args.output_dir, exist_ok=True)
    for name in args.datasets:
        convert_dataset(name, args, token)


if __name__ == "__main__":
    main()
