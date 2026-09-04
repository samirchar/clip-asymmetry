import argparse
import json
import os
import random
import tempfile

import torch
import torch.distributed as dist
import torch.nn.functional as F
from braceexpand import braceexpand
from tqdm import tqdm
import open_clip

from src.geometry import uniformity, alignment
from src.utils import register_models, parse_webdataset_path, resolve_models
from src.loaders import get_dataset_fn
from src.distributed import environment_setup, is_master, cleanup
from datetime import timedelta
from clip_benchmark.datasets.builder import build_dataset


is_amlt = os.environ.get("AMLT_DATA_DIR", None) is not None




def parse_args():
    parser = argparse.ArgumentParser(description="Compute uniformity and alignment for CLIP models.")
    parser.add_argument("--models-file", type=str, help="Path to text file listing model directories to evaluate.")
    parser.add_argument("--models", type=str,nargs='+', help="Pairs of architecture and checkpoint_path, e.g. 'ViT-B-32=checkpoints/cc3m_architecture_joint/ViT-B-32_epoch_latest.pt, ViT-L-14=checkpoints/cc3m_architecture_joint/ViT-L-14_epoch_latest.pt'")
    parser.add_argument("--models-root", type=str, default=None,
                        help="Root prefix for relative run dirs in --models-file "
                             "(e.g. /mnt/blob). On AMLT, AMLT_DATA_DIR is used instead. "
                             "Mirrors bin/compute_val_loss.py / bin/benchmark.py.")
    parser.add_argument("--val-datasets", type=str, nargs='+', required=True, help="Names of CLIP benchmark wds datasets")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size per GPU.")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers per GPU.")
    parser.add_argument("--output", type=str, required=True, help="Path to save results JSON.")
    parser.add_argument("--max-samples", type=str, nargs='+', default=[5000], help="Max samples to use for uniformity/alignment computation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for distributed setup.")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in minutes for distributed setup.")
    return parser.parse_args()

def load_clipbenchmark_dataset(dataset_name, transform, batch_size, num_workers, seed=42):
  
    if not dataset_name.startswith("wds/"):
        dataset_name = "wds/" + dataset_name

    dataset_root = "https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main"
        
    dataset = build_dataset(
        root=dataset_root.format(dataset_cleaned=dataset_name.replace("wds/", "", 1).replace("/", "-")),
        dataset_name=dataset_name, 
        transform=transform, 
        download=True,
        task="auto"
    )
    dataloader = torch.utils.data.DataLoader(
        dataset.shuffle(50000, rng=random.Random(seed)).batched(batch_size), batch_size=None, 
        shuffle=False, num_workers=num_workers
    )
    return dataloader

def extract_features(model, dataloader, tokenizer, device, normalize=False, max_samples=None, max_captions_per_image = 1):

    image_embs, text_embs = [], []
    n = 0
    with torch.no_grad():
        for batch in tqdm(dataloader,desc="Extracting features"):
            images = batch[0].to(device)
            # Flatten all captions and repeat images to match
            captions_per_image = []
            flat_texts = []
            for texts in batch[1]:
                captions_per_image.append(min(len(texts), max_captions_per_image))
                flat_texts.extend(texts[:max_captions_per_image])

            flat_texts = tokenizer(flat_texts).to(device)

            repeated_images = images.repeat_interleave(torch.tensor(captions_per_image, device=device), dim=0)
            
            image_features = model.encode_image(repeated_images)
            text_features = model.encode_text(flat_texts)

            if normalize:
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
            image_embs.append(image_features.cpu())
            text_embs.append(text_features.cpu())

            n += repeated_images.shape[0]
            if max_samples is not None and n >= max_samples:
                break

    image_embs = torch.cat(image_embs)[:max_samples]
    text_embs = torch.cat(text_embs)[:max_samples]
    return image_embs, text_embs


def apply_models_root(dirs, models_root):
    """Prefix run dirs following bin/benchmark.py / bin/compute_val_loss.py convention.

    On AMLT (AMLT_DATA_DIR set) prepend it; otherwise prepend --models-root if given.
    Paths are lstrip('/')-ed before joining so relative configs (e.g. openclip_logs/...)
    resolve the same way they do for benchmark.py. With no prefix, dirs are used as-is.
    """
    amlt_data_dir = os.environ.get("AMLT_DATA_DIR")
    root = amlt_data_dir if amlt_data_dir is not None else models_root
    if root is None:
        return [d.rstrip("/") for d in dirs]
    return [os.path.join(root, d.lstrip("/").rstrip("/")) for d in dirs]


def resolve_models_file(models_file, models_root):
    """resolve_models() with the shared --models-root / AMLT prefix handling applied.

    Reads the run-dir list, prefixes each (skipping blanks and #-comments), then reuses
    src.utils.resolve_models via a temp file so path resolution matches benchmark.py.
    """
    with open(models_file) as f:
        dirs = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    dirs = apply_models_root(dirs, models_root)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
        tf.write("\n".join(dirs))
        tmp_path = tf.name
    try:
        return resolve_models(tmp_path)
    finally:
        os.remove(tmp_path)


def main():

    '''
    Sample usage:
    python bin/run_uni_align.py --models-file=<models_list.txt> --val-datasets=mscoco_captions flickr30k --batch-size=256 --output=data/analysis/uni_align_results.json --max-samples=5000 
    '''
    register_models()
    args = parse_args()

    device_type, local_rank, global_rank, world_size = environment_setup(args.seed, timeout=timedelta(minutes=args.timeout))

    if args.models_file is None and args.models is None:
        raise ValueError("Must specify either --models-file or --models")
    elif args.models_file is not None and args.models is not None:
        raise ValueError("Cannot specify both --models-file and --models")
    elif args.models is not None:
        resolved_models = []
        for pair in args.models:
            architecture, checkpoint_path = pair.split("=", 1)
            resolved_models.append((architecture.strip(), checkpoint_path.strip()))
        # Apply the same AMLT / --models-root prefixing to explicit ckpt paths.
        arches = [a for a, _ in resolved_models]
        ckpts = apply_models_root([c for _, c in resolved_models], args.models_root)
        resolved_models = list(zip(arches, ckpts))
    elif args.models_file is not None:
        resolved_models = resolve_models_file(args.models_file, args.models_root)

    # Get all runs (paths already prefixed above; no per-run rewriting needed)
    runs = []
    for val_data in args.val_datasets:
        for architecture, checkpoint_path in resolved_models:
            for max_samples in args.max_samples:
                runs.append((val_data, architecture, checkpoint_path, int(max_samples)))
    
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # Only keep runs for this global rank
    runs = [r for i, r in enumerate(runs) if i % world_size == global_rank]

    print(f"[Rank {global_rank}] Starting evaluation of {len(runs)} runs...")

    local_results = []
    for val_data, architecture, checkpoint_path, max_samples in runs:
        print(f"\n{'='*60}")
        print(f"[Rank {global_rank}] Evaluating model {architecture} on {val_data}...")
        
        model, _, preprocess_val = open_clip.create_model_and_transforms(architecture, pretrained=checkpoint_path, device=device)
        tokenizer = open_clip.get_tokenizer(architecture)
        model.eval()
        
        dataloader = load_clipbenchmark_dataset(val_data, preprocess_val, args.batch_size, args.workers, seed=args.seed)       
        image_embs, text_embs = extract_features(model, dataloader, tokenizer, device, normalize=True, max_samples=max_samples)

        print(f"  Image embeddings shape: {image_embs.shape}")
        print(f"  Text embeddings shape:  {text_embs.shape}")

        uniformity_image = uniformity(image_embs)
        uniformity_text = uniformity(text_embs)
        alignment_score = alignment(image_embs, text_embs)

        print(f"  Image uniformity: {uniformity_image:.2f}")
        print(f"  Text uniformity:  {uniformity_text:.2f}")
        print(f"  Alignment: {alignment_score:.2f}")

        local_results.append({
            "dataset": val_data,
            "architecture": architecture,
            "model_path": checkpoint_path,
            "image_uniformity": uniformity_image,
            "text_uniformity": uniformity_text,
            "alignment": alignment_score,
            "max_samples": max_samples
        })

        del model
        torch.cuda.empty_cache()
    
    # Gather results from all ranks to rank 0
    if dist.is_initialized():
        all_results_list = [None] * world_size
        dist.all_gather_object(all_results_list, local_results)
        if is_master():
            all_results = [r for rank_results in all_results_list for r in rank_results]
    else:
        all_results = local_results

    if is_master():
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {args.output} ({len(all_results)} entries)")

    cleanup()


if __name__ == "__main__":
    main()