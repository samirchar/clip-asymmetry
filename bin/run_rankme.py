import argparse
import json
import os
import random

import torch
import torch.distributed as dist
import torch.nn.functional as F
from braceexpand import braceexpand
from tqdm import tqdm
import open_clip

from src.geometry import rankme
from src.utils import register_models, parse_webdataset_path, resolve_models
from src.loaders import get_dataset_fn
from src.distributed import environment_setup, is_master, cleanup
from datetime import timedelta



is_amlt = os.environ.get("AMLT_DATA_DIR", None) is not None


def parse_args():
    parser = argparse.ArgumentParser(description="Compute RankMe for CLIP models.")
    parser.add_argument("--models-file", type=str, help="Path to text file listing model directories to evaluate.")
    parser.add_argument("--models", type=str,nargs='+', help="Pairs of architecture and checkpoint_path, e.g. 'ViT-B-32=checkpoints/cc3m_architecture_joint/ViT-B-32_epoch_best.pt, ViT-L-14=checkpoints/cc3m_architecture_joint/ViT-L-14_epoch_best.pt'")
    parser.add_argument("--val-datasets", type=str, nargs='+', required=True, help="Path(s) to validation data shards")
    parser.add_argument("--dataset-type", choices=["webdataset", "csv", "synthetic", "auto"], default="auto",
                        help="Which type of dataset to process.")
    parser.add_argument("--val-num-samples", type=int, default=None,
                        help="Number of samples in dataset. Useful for webdataset if not available in info file.")
    parser.add_argument("--caption-key", type=str, default="caption", help="The name of the key for the captions.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size per GPU.")
    parser.add_argument("--workers", type=int, default=4, help="Number of dataloader workers per GPU.")
    parser.add_argument("--normalize", action="store_true", help="Whether to normalize the embeddings.")
    parser.add_argument("--output", type=str, required=True, help="Path to save results JSON.")
    parser.add_argument("--max-samples", type=str, nargs='+', default=[25600], help="Max samples to use for RankMe computation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for distributed setup.")
    parser.add_argument("--timeout", type=int, default=60, help="Timeout in minutes for distributed setup.")
    return parser.parse_args()

def extract_features(model, dataloader, device, normalize=False, max_samples=None):

    image_embs, text_embs = [], []
    n = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            images, texts = batch[0].to(device), batch[1].to(device)
            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            if normalize:
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)
                
            image_embs.append(image_features.cpu())
            text_embs.append(text_features.cpu())

            n += images.shape[0]
            if max_samples is not None and n >= max_samples:
                break

    image_embs = torch.cat(image_embs)[:max_samples]
    text_embs = torch.cat(text_embs)[:max_samples]
    return image_embs, text_embs

def main():
    '''
    Sample usage:
    python bin/run_rankme.py --models-file=<models_list.txt> --val-datasets="data/cc12m-wds/cc12m-train-*.tar" --batch-size=4096 --output=data/analysis/rankme_results_random.json --max-samples=25600 
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
    elif args.models_file is not None:
        resolved_models = resolve_models(args.models_file)

    # Get all runs
    runs = []
    #TODO: Since models and datasets are tightly coupled for this eval, the .txt should also contain the dataset info
    for val_data in args.val_datasets:
        val_data = parse_webdataset_path(val_data)
        if is_amlt:
            val_data = os.path.join(os.environ.get("AMLT_DATA_DIR"), val_data.lstrip('/'))
        # Expand brace pattern into individual shard URLs and shuffle deterministically
        shard_urls = list(braceexpand(val_data))
        random.seed(args.seed)
        random.shuffle(shard_urls)
        for architecture, checkpoint_path in resolved_models:
            for max_samples in args.max_samples:
                if is_amlt:
                    checkpoint_path = os.path.join(os.environ.get("AMLT_DATA_DIR"), checkpoint_path.lstrip('/'))
                runs.append((shard_urls, val_data, architecture, checkpoint_path, int(max_samples)))
    
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    # Only keep runs for this global rank
    runs = [r for i, r in enumerate(runs) if i % world_size == global_rank]

    print(f"[Rank {global_rank}] Starting evaluation of {len(runs)} runs...")

    local_results = []
    for shard_urls, val_data, architecture, checkpoint_path, max_samples in runs:
        print(f"\n{'='*60}")
        print(f"[Rank {global_rank}] Evaluating model {architecture} on {val_data}...")
        
        model, _, preprocess_val = open_clip.create_model_and_transforms(architecture, pretrained=checkpoint_path, device=device)
        tokenizer = open_clip.get_tokenizer(architecture)
        model.eval()

        # Pass shuffled shard list so SimpleShardList reads them in random order
        args.val_data = shard_urls
        
        dataloader = get_dataset_fn(val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer).dataloader
        
        image_embs, text_embs = extract_features(model, dataloader, device, normalize=args.normalize, max_samples=max_samples)

        print(f"  Image embeddings shape: {image_embs.shape}")
        print(f"  Text embeddings shape:  {text_embs.shape}")

        image_rankme = rankme(image_embs)
        text_rankme = rankme(text_embs)
        print(f"  RankMe (image): {image_rankme:.2f}")
        print(f"  RankMe (text):  {text_rankme:.2f}")

        local_results.append({
            "dataset": val_data,
            "architecture": architecture,
            "model_path": checkpoint_path,
            "image_rankme": image_rankme,
            "text_rankme": text_rankme,
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