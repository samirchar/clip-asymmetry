'''
Adapted from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/main.py
'''

# Dump the Python stack on a native crash (e.g. SIGSEGV) instead of failing
# silently. Free insurance for the earlier one-off single-rank segfault.
import faulthandler
faulthandler.enable(all_threads=True)

import copy
import glob
import logging
import math
import os
import re
import subprocess
import sys
import random
from datetime import datetime
is_amlt = os.environ.get("AMLT_DATA_DIR", None) is not None

import torch
from torch import optim


try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip import create_model_and_transforms, trace_model, get_tokenizer, create_loss
from open_clip_train.distributed import is_master, broadcast_object, is_device_available, is_using_distributed, set_device, world_info_from_env
from open_clip_train.logger import setup_logging
from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown, _warmup_lr
from open_clip_train.train import train_one_epoch
from open_clip_train.file_utils import pt_load, check_exists, start_sync_process, remote_sync
import argparse
import ast
from dotenv import load_dotenv
from src.utils import config_parser, parse_webdataset_path, register_models
from src.loaders import get_data

import json
import logging
import torch.nn.functional as F

from open_clip import get_input_dtype
from open_clip_train.zero_shot import build_zero_shot_classifier, IMAGENET_CLASSNAMES, OPENAI_IMAGENET_TEMPLATES, accuracy
from open_clip_train.precision import get_autocast
import torch.distributed as dist
from tqdm import tqdm
from src.utils import LATEST_CHECKPOINT_NAME, BEST_CHECKPOINT_NAME
import warnings
from datetime import timedelta
from typing import Optional
import numpy as np

# init_distributed_device / init_distributed_device_so are adapted verbatim from
# open_clip_train.distributed, with one change: init_process_group receives an
# explicit timeout. openclip uses the NCCL default (10 min); when rank 0's startup
# lags (e.g. wandb.init on a flaky server) the other ranks otherwise abort at the
# first collective. _PG_TIMEOUT bounds the collective watchdog, and we also raise
# the separate NCCL heartbeat monitor (env-only, ~8 min default) to match.
_PG_TIMEOUT = timedelta(minutes=30)


def init_distributed_device_so(
        device: str = 'cuda',
        dist_backend: Optional[str] = None,
        dist_url: Optional[str] = None,
        horovod: bool = False,
        no_set_device_rank: bool = False,
):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    distributed = False
    world_size = 1
    global_rank = 0
    local_rank = 0
    device_type, *device_idx = device.split(':', maxsplit=1)
    is_avail, is_known = is_device_available(device_type)
    if not is_known:
        warnings.warn(f"Device {device} was not known and checked for availability, trying anyways.")
    elif not is_avail:
        warnings.warn(f"Device {device} was not available, falling back to CPU.")
        device_type = device = 'cpu'

    if horovod:
        import horovod.torch as hvd
        assert hvd is not None, "Horovod is not installed"
        hvd.init()
        local_rank = int(hvd.local_rank())
        global_rank = hvd.rank()
        world_size = hvd.size()
        distributed = True
    elif is_using_distributed():
        if dist_backend is None:
            dist_backends = {
                "cuda": "nccl",
                "hpu": "hccl",
                "npu": "hccl",
                "xpu": "ccl",
            }
            dist_backend = dist_backends.get(device_type, 'gloo')

        dist_url = dist_url or 'env://'
        os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", str(int(_PG_TIMEOUT.total_seconds())))

        if 'SLURM_PROCID' in os.environ:
            # DDP via SLURM
            local_rank, global_rank, world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ['LOCAL_RANK'] = str(local_rank)
            os.environ['RANK'] = str(global_rank)
            os.environ['WORLD_SIZE'] = str(world_size)
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
                world_size=world_size,
                rank=global_rank,
                timeout=_PG_TIMEOUT,
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            local_rank, _, _ = world_info_from_env()
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
                timeout=_PG_TIMEOUT,
            )
            world_size = torch.distributed.get_world_size()
            global_rank = torch.distributed.get_rank()
        distributed = True

    if distributed and not no_set_device_rank and device_type not in ('cpu', 'mps'):
        # Ignore manually specified device index in distributed mode and
        # override with resolved local rank, fewer headaches in most setups.
        if device_idx:
            warnings.warn(f'device index {device_idx[0]} removed from specified ({device}).')
        device = f'{device_type}:{local_rank}'
        set_device(device)

    return dict(
        device=device,
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        distributed=distributed,
    )


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0
    result = init_distributed_device_so(
        device=getattr(args, 'device', 'cuda'),
        dist_backend=getattr(args, 'dist_backend', None),
        dist_url=getattr(args, 'dist_url', None),
        horovod=getattr(args, 'horovod', False),
        no_set_device_rank=getattr(args, 'no_set_device_rank', False),
    )
    args.device = result['device']
    args.world_size = result['world_size']
    args.rank = result['global_rank']
    args.local_rank = result['local_rank']
    args.distributed = result['distributed']
    device = torch.device(args.device)
    return device


def decoupled_cosine_lr(optimizer, warmup_length, steps):
    """Cosine LR scheduler that respects per-group initial_lr for decoupled learning rates."""
    def _lr_adjuster(step):
        if step < warmup_length:
            scale = _warmup_lr(1, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            scale = 0.5 * (1 + math.cos(math.pi * e / es))

        for param_group in optimizer.param_groups:
            param_group["lr"] = param_group["initial_lr"] * scale
        return scale
    return _lr_adjuster


def _is_gain_or_bias_param(name, param):
    """Match OpenCLIP's exclude rule for weight decay (bias/norm/logit_scale get WD=0)."""
    return param.ndim < 2 or "bn" in name or "ln" in name or "bias" in name or 'logit_scale' in name


def _is_text_param(name):
    """Text-tower parameter classification. Vision = visual.*, logit_scale excluded, rest = text."""
    name = name.removeprefix("module.")
    return not name.startswith("visual.") and name != "logit_scale"


def build_param_groups(named_parameters, lr, wd, lr_text_scale=None,
                       wd_text_scale=None, wd_vision_scale=None,
                       compensate_wd="false"):
    """Build AdamW param groups for vanilla or decoupled LR/WD training.

    Returns a list of dicts compatible with optim.AdamW. Pure function for testability.

    Modes:
        - Vanilla (no scales set): 2 groups (bias/norm wd=0, weights wd=wd)
        - Decoupled (any scale set): 4 groups split by text/non-text x bias-norm/weights

    WD priority for text weights:
        --wd-text-scale > --compensate-wd > default (same as vision)
    """
    decoupled = (lr_text_scale is not None or
                 wd_text_scale is not None or
                 wd_vision_scale is not None)

    gain_or_bias_params = [(n, p) for n, p in named_parameters
                           if _is_gain_or_bias_param(n, p) and p.requires_grad]
    rest_params = [(n, p) for n, p in named_parameters
                   if not _is_gain_or_bias_param(n, p) and p.requires_grad]

    if not decoupled:
        return [
            {"params": [p for _, p in gain_or_bias_params], "weight_decay": 0.,
             "lr": lr, "initial_lr": lr},
            {"params": [p for _, p in rest_params], "weight_decay": wd,
             "lr": lr, "initial_lr": lr},
        ]

    # Decoupled mode
    effective_lr_text_scale = lr_text_scale if lr_text_scale is not None else 1.0
    text_lr = lr * effective_lr_text_scale

    if wd_text_scale is not None:
        text_wd = wd * wd_text_scale
    elif compensate_wd == "true" and lr_text_scale is not None:
        text_wd = wd / lr_text_scale
    else:
        text_wd = wd

    vision_wd = wd * wd_vision_scale if wd_vision_scale is not None else wd

    text_ids = {id(p) for n, p in named_parameters if _is_text_param(n)}
    gob_text = [p for _, p in gain_or_bias_params if id(p) in text_ids]
    gob_rest = [p for _, p in gain_or_bias_params if id(p) not in text_ids]
    weight_text = [p for _, p in rest_params if id(p) in text_ids]
    weight_rest = [p for _, p in rest_params if id(p) not in text_ids]

    return [
        {"params": gob_text, "weight_decay": 0.,
         "lr": text_lr, "initial_lr": text_lr},
        {"params": gob_rest, "weight_decay": 0.,
         "lr": lr, "initial_lr": lr},
        {"params": weight_text, "weight_decay": text_wd,
         "lr": text_lr, "initial_lr": text_lr},
        {"params": weight_rest, "weight_decay": vision_wd,
         "lr": lr, "initial_lr": lr},
    ]


def run(model, classifier, dataloader, args):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    logging.info(f"eval device: {device}")
    logging.info(f"model device: {next(model.parameters()).device}")
    logging.info(f"classifier device: {classifier.device}")

    with torch.inference_mode():
        top1, top5, n = 0., 0., 0.
        for images, target in tqdm(dataloader, unit_scale=args.batch_size):
            images = images.to(device=device, dtype=input_dtype)
            target = target.to(device)

            with autocast():
                # predict
                output = model(image=images)
                image_features = output['image_features'] if isinstance(output, dict) else output[0]
                logits = 100. * image_features @ classifier

            # measure accuracy
            acc1, acc5 = accuracy(logits, target, topk=(1, 5))
            top1 += acc1
            top5 += acc5
            n += images.size(0)

    top1 = (top1 / n)
    top5 = (top5 / n)
    return top1, top5

def zero_shot_eval(model, data, epoch, args, tokenizer=None):
    if 'imagenet-val' not in data and 'imagenet-v2' not in data:
        return {}
    if args.zeroshot_frequency == 0:
        return {}
    if (epoch % args.zeroshot_frequency) != 0 and epoch != args.epochs:
        return {}

    logging.info('Starting zero-shot imagenet.')
    if tokenizer is None:
        tokenizer = get_tokenizer(args.model)

    logging.info('Building zero-shot classifier')
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    with autocast():
        classifier = build_zero_shot_classifier(
            model,
            tokenizer=tokenizer,
            classnames=IMAGENET_CLASSNAMES,
            templates=OPENAI_IMAGENET_TEMPLATES,
            num_classes_per_batch=10,
            device=device,
            use_tqdm=True,
        )

    logging.info('Using classifier')
    results = {}
    if 'imagenet-val' in data:
        top1, top5 = run(model, classifier, data['imagenet-val'].dataloader, args)
        results['imagenet-zeroshot-val-top1'] = top1
        results['imagenet-zeroshot-val-top5'] = top5
    if 'imagenet-v2' in data:
        top1, top5 = run(model, classifier, data['imagenet-v2'].dataloader, args)
        results['imagenetv2-zeroshot-val-top1'] = top1
        results['imagenetv2-zeroshot-val-top5'] = top5

    logging.info('Finished zero-shot imagenet.')

    return results

def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics

def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)
    

def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None):
    metrics = {}
    if not is_master(args):
        return 
    
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        eval_model = model.module
    else:
        eval_model = model

    device = torch.device(args.device)
    eval_model.eval()

    zero_shot_metrics = zero_shot_eval(eval_model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = eval_model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"]
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            # Persist the epoch explicitly so best-checkpoint selection stays
            # correct even when a resumed run re-appends rows (row position no
            # longer equals the epoch number). Written only to the file, not to
            # the returned/logged metrics dict, to avoid side effects.
            f.write(json.dumps({**metrics, "epoch": epoch}))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    
    return metrics


def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def add_args_fn(parser):
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
    )

    parser.add_argument(
        "--train-data-upsampling-factors",
        type=str,
        default=None,
        help=(
            "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
            "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
            "By default, datapoints are sampled uniformly regardless of the dataset sizes."
        )
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to file(s) with validation data",
    )

    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "synthetic", "auto"],
        default="auto",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection."
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--caption-key",
        type=str,
        default="caption",
        help="The name of the key for the captions."
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override system default cache path for model & tokenizer file downloads.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs/",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--epochs-cooldown", type=int, default=None,
        help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--lr-text-scale", type=float, default=None,
        help="If set, text encoder LR = lr * lr_text_scale. Vision encoder uses lr. "
             "Requires fresh training (cannot resume from checkpoints without this flag).",
    )
    parser.add_argument(
        "--compensate-wd", type=str, default="false", choices=["true", "false"],
        help="If true and --lr-text-scale is set, scale text weight decay by 1/lr_text_scale "
             "to keep effective regularization (lr * wd) constant. "
             "Ignored if --wd-text-scale is also set (explicit scale takes precedence).",
    )
    parser.add_argument(
        "--wd-text-scale", type=float, default=None,
        help="If set, text encoder weight decay = wd * wd_text_scale. "
             "Can be used with or without --lr-text-scale.",
    )
    parser.add_argument(
        "--wd-vision-scale", type=float, default=None,
        help="If set, vision encoder weight decay = wd * wd_vision_scale. "
             "Can be used with or without other decoupled flags.",
    )
    parser.add_argument("--momentum", type=float, default=None, help="Momentum (for timm optimizers).")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--opt", type=str, default='adamw',
        help="Which optimizer to use. Choices are ['adamw', or any timm optimizer 'timm/{opt_name}']."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default='cosine',
        help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
    )
    parser.add_argument(
        "--lr-cooldown-end", type=float, default=0.0,
        help="End learning rate for cooldown schedule. Default: 0"
    )
    parser.add_argument(
        "--lr-cooldown-power", type=float, default=1.0,
        help="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)"
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action='store_true',
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action='store_true',
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset')
    parser.add_argument(
        '--image-interpolation',
        default=None, type=str, choices=['bicubic', 'bilinear', 'random'],
        help="Override default image resize interpolation"
    )
    parser.add_argument(
        '--image-resize-mode',
        default=None, type=str, choices=['shortest', 'longest', 'squash'],
        help="Override default image resize (& crop) mode during inference"
    )
    parser.add_argument('--aug-cfg', nargs='*', default={}, action=ParseKwargs)
    parser.add_argument(
        '--train-pre-resize',
        type=int,
        default=None,
        help='If set, prepend a Resize(size=N) before the training transforms (e.g. 256). Useful for small-resolution datasets like CC3M/CC12M.'
    )
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    parser.add_argument(
        '--force-context-length', type=int, default=None,
        help='Override default context length'
    )
    parser.add_argument(
        '--force-image-size', type=int, nargs='+', default=None,
        help='Override default image size'
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--force-patch-dropout",
        default=None,
        type=float,
        help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
    )
    parser.add_argument(
        "--force-custom-text",
        default=False,
        action='store_true',
        help="Force use of CustomTextCLIP model (separate text-tower).",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--torchcompile",
        default=False,
        action='store_true',
        help="torch.compile() the model, requires pytorch 2.0 or later.",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only",
    )
    parser.add_argument(
        "--accum-freq", type=int, default=1, help="Update the model every --acum-freq steps."
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="Accelerator to use."
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend",
        default=None,
        type=str,
        help="distributed backend. \"nccl\" for GPU, \"hccl\" for Ascend NPU"
    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default='open-clip',
        help="Name of the project if logging with wandb.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log directory, and execute from there."
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip."
    )
    parser.add_argument(
        "--lock-text",
        default=False,
        action='store_true',
        help="Lock full text tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-text-unlocked-layers",
        type=int,
        default=0,
        help="Leave last n text tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-text-freeze-layer-norm",
        default=False,
        action='store_true',
        help="Freeze LayerNorm running stats in text tower for any locked layers.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Log every n steps to tensorboard/console/wandb.",
    )
    parser.add_argument(
        "--coca-caption-loss-weight",
        type=float,
        default=2.0,
        help="Weight assigned to caption loss in CoCa."
    )
    parser.add_argument(
        "--coca-contrastive-loss-weight",
        type=float,
        default=1.0,
        help="Weight assigned to contrastive loss when training CoCa."
    )
    parser.add_argument(
        "--remote-sync",
        type=str,
        default=None,
        help="Optinoally sync with a remote path specified by this arg",
    )
    parser.add_argument(
        "--remote-sync-frequency",
        type=int,
        default=300,
        help="How frequently to sync to a remote directly if --remote-sync is not None.",
    )
    parser.add_argument(
        "--remote-sync-protocol",
        choices=["s3", "fsspec"],
        default="s3",
        help="How to do the remote sync backup if --remote-sync is not None.",
    )
    parser.add_argument(
        "--delete-previous-checkpoint",
        default=False,
        action="store_true",
        help="If true, delete previous checkpoint after storing a new one."
    )
    parser.add_argument(
        "--distill-model",
        default=None,
        help='Which model arch to distill from, if any.'
    )
    parser.add_argument(
        "--distill-pretrained",
        default=None,
        help='Which pre-trained weights to distill from, if any.'
    )
    parser.add_argument(
        "--use-bnb-linear",
        default=None,
        help='Replace the network linear layers from the bitsandbytes library. '
        'Allows int8 training/inference, etc.'
    )
    parser.add_argument(
        "--siglip",
        default=False,
        action="store_true",
        help='Use SigLip (sigmoid) loss.'
    )
    parser.add_argument(
        "--loss-dist-impl",
        default=None,
        type=str,
        help='A string to specify a specific distributed loss implementation.'
    )
    parser.add_argument(
        "--save-best",
        default=False,
        action="store_true",
        help="Save the best checkpoint based on --best-metric to epoch_best.pt."
    )
    parser.add_argument(
        "--best-metric",
        type=str,
        default="imagenet-zeroshot-val-top1",
        help="Metric key to track for --save-best (e.g. 'imagenet-zeroshot-val-top1', 'clip_val_loss'). Default: imagenet-zeroshot-val-top1",
    )
    parser.add_argument(
        "--best-metric-lower-is-better",
        default=False,
        action="store_true",
        help="If set, lower metric values are better (e.g. for loss). Default: False (higher is better).",
    )
    parser.add_argument(
        "--track-dynamics",
        default=False,
        action="store_true",
        help="Track per-epoch NWC/SI training dynamics. Off by default: it snapshots "
             "the full state_dict to CPU each epoch on the master rank, adding "
             "end-of-epoch overhead while other ranks wait at the barrier.",
    )


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def _sync_offline_wandb(local_wandb_dir, blob_wandb_dir):
    """Best-effort mirror of locally-written offline wandb runs to blob storage.

    Offline wandb writes to fast local disk (reliable flushing); blobfuse buffers
    the long-open .wandb file and would lose it on a crash. This rsyncs the local
    runs to the blob-mounted run dir so they survive VM teardown and can be
    `wandb sync`-ed later. Never raises -- logging must not interrupt training.
    """
    try:
        if not os.path.isdir(local_wandb_dir):
            return
        if os.path.abspath(local_wandb_dir) == os.path.abspath(blob_wandb_dir):
            return
        os.makedirs(blob_wandb_dir, exist_ok=True)
        subprocess.run(["rsync", "-a", local_wandb_dir + "/", blob_wandb_dir + "/"], check=False)
    except Exception as e:
        logging.warning(f"offline wandb -> blob sync failed: {e}")


def main(args):
    register_models()
    load_dotenv()
    
    args, resolved = config_parser(add_args_fn=add_args_fn)

    if is_amlt:
        amlt_data_dir = os.environ.get("AMLT_DATA_DIR")
        if args.train_data is not None:
            args.train_data = os.path.join(amlt_data_dir, args.train_data.lstrip('/'))
        if args.val_data is not None:
            args.val_data = os.path.join(amlt_data_dir, args.val_data.lstrip('/'))
        if args.imagenet_val is not None:
            # args.imagenet_val = os.path.join(amlt_data_dir, args.imagenet_val.lstrip('/'))
            args.imagenet_val = "/scratch/imagenet/val/"
        if args.imagenet_v2 is not None:
            args.imagenet_v2 = os.path.join(amlt_data_dir, args.imagenet_v2.lstrip('/'))
        if args.logs is not None:
            args.logs = os.path.join(amlt_data_dir, args.logs.lstrip('/'))
        
    #TODO: may have to include other args here. Wildcard may not work with non-webdataset datasets
    if args.dataset_type == 'webdataset':
        args.train_data = parse_webdataset_path(args.train_data)
        if args.val_data is not None:
            args.val_data = parse_webdataset_path(args.val_data)

    if 'timm' not in args.opt:
        # set default opt params based on model name (only if timm optimizer not used)
        default_params = get_default_params(args.model)
        for name, val in default_params.items():
            if getattr(args, name) is None:
                setattr(args, name, val)
    
    # Add csv_caption_key for compatiblity
    args.csv_caption_key = args.caption_key


    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)


    # if is_master(args) and is_amlt:
    #     if args.imagenet_val is not None:
    #         shutil.copytree(args.imagenet_val, "/scratch/imagenet_val/")
    #         args.imagenet_val = "/scratch/imagenet_val/"

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = '-'.join([
            date_str,
            f"model_{model_name_safe}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    dist_model = None
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        #FIXME: support distillation with grad accum.
        assert args.accum_freq == 1
        #FIXME: support distillation with coca.
        assert 'coca' not in args.model.lower()

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)
    model_kwargs = {}
    if args.siglip:
        model_kwargs['init_logit_scale'] = np.log(10)  # different from CLIP
        model_kwargs['init_logit_bias'] = -10
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        force_context_length=args.force_context_length,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        cache_dir=args.cache_dir,
        **model_kwargs,
    )

    if args.train_pre_resize is not None:
        from torchvision import transforms
        preprocess_train = transforms.Compose(
            [transforms.Resize(size=args.train_pre_resize),
            *preprocess_train.transforms]
        )

    #Print pretrain transform config
    logging.info(f"Pretrain transform config:")
    logging.info(preprocess_train)

    if args.distill:
        # FIXME: currently assumes the model you're distilling from has the same tokenizer & transforms.
        dist_model, _, _ = create_model_and_transforms(
            args.distill_model, 
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
            cache_dir=args.cache_dir,
        )
    if args.use_bnb_linear is not None:
        print('=> using a layer from bitsandbytes.\n'
              '   this is an experimental feature which requires two extra pip installs\n'
              '   pip install bitsandbytes triton'
              '   please make sure to use triton 2.0.0')
        import bitsandbytes as bnb
        from open_clip.utils import replace_linear
        print(f'=> replacing linear layers with {args.use_bnb_linear}')
        linear_replacement_cls = getattr(bnb.nn.triton_based_modules, args.use_bnb_linear)
        replace_linear(model, linear_replacement_cls)
        model = model.to(device)

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
    
        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'

        opt = getattr(args, 'opt', 'adamw').lower()
        if opt.startswith('timm/'):
            from timm.optim import create_optimizer_v2
            timm_opt = opt.split('timm/')[-1]
            opt_kwargs = {}
            assert (args.beta1 is None) == (args.beta2 is None), \
                'When using timm optimizer, BOTH beta1 and beta2 must be specified (or not specified).'
            if args.beta1 is not None:
                opt_kwargs['betas'] = (args.beta1, args.beta2)
            if args.momentum is not None:
                opt_kwargs['momentum'] = args.momentum
            optimizer = create_optimizer_v2(
                model,
                timm_opt,
                lr=args.lr,
                weight_decay=args.wd,
                eps=args.eps,
                **opt_kwargs,
            )
        else:
            named_parameters = list(model.named_parameters())

            if opt == 'adamw':
                param_groups = build_param_groups(
                    named_parameters,
                    lr=args.lr,
                    wd=args.wd,
                    lr_text_scale=args.lr_text_scale,
                    wd_text_scale=args.wd_text_scale,
                    wd_vision_scale=args.wd_vision_scale,
                    compensate_wd=args.compensate_wd,
                )
                optimizer = optim.AdamW(
                    param_groups,
                    lr=args.lr,
                    betas=(args.beta1, args.beta2),
                    eps=args.eps,
                )
                if is_master(args):
                    decoupled = (args.lr_text_scale is not None or
                                 args.wd_text_scale is not None or
                                 args.wd_vision_scale is not None)
                    if decoupled:
                        # Groups: 0=text bias/norm, 1=vision bias/norm, 2=text weights, 3=vision weights
                        text_lr = param_groups[0]["lr"]
                        text_wd = param_groups[2]["weight_decay"]
                        vision_wd = param_groups[3]["weight_decay"]
                        n_text = sum(p.numel() for p in param_groups[0]["params"]) \
                                 + sum(p.numel() for p in param_groups[2]["params"])
                        n_rest = sum(p.numel() for p in param_groups[1]["params"]) \
                                 + sum(p.numel() for p in param_groups[3]["params"])
                        logging.info(
                            f'Decoupled optimizer: text_lr={text_lr}, vision_lr={args.lr}, '
                            f'text_wd={text_wd}, vision_wd={vision_wd}, '
                            f'text_params={n_text:,}, vision_params={n_rest:,}'
                        )
            else:
                assert False, f'Unknown optimizer {opt}'

        if is_master(args):
            defaults = copy.deepcopy(optimizer.defaults)
            defaults['weight_decay'] = args.wd
            defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
            logging.info(
                f'Created {type(optimizer).__name__} ({args.opt}) optimizer: {defaults}'
            )

        if args.horovod:
            optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(optimizer, root_rank=0)

        scaler = None
        if args.precision == "amp":
            try:
                scaler = torch.amp.GradScaler(device=device)
            except (AttributeError, TypeError) as e:
                scaler = torch.cuda.amp.GradScaler()

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if optimizer is not None:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # initialize datasets
    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir, context_length=args.force_context_length)
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )
    
    # print one train batch for verification
    # tokenizer
    # sample_text = next(iter(data['train'].dataloader))[-1]
    # print(sample_text)
    # print(tokenizer.decode(sample_text[0].detach().cpu().tolist()))
    
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        if args.lr_scheduler == "cosine":
            if args.lr_text_scale is not None or args.wd_text_scale is not None or args.wd_vision_scale is not None:
                scheduler = decoupled_cosine_lr(optimizer, args.warmup, total_steps)
            else:
                scheduler = cosine_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const":
            scheduler = const_lr(optimizer, args.lr, args.warmup, total_steps)
        elif args.lr_scheduler == "const-cooldown":
            assert args.epochs_cooldown is not None,\
                "Please specify the number of cooldown epochs for this lr schedule."
            cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
            scheduler = const_lr_cooldown(
                optimizer, args.lr, args.warmup, total_steps,
                cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
        else:
            logging.error(
                f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
            exit(1)

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        logging.info(f"wandb project name: {args.wandb_project_name}, run name: {args.name}, id: {args.name}")
        # WANDB_PROJECT,WANDB_RUN_GROUP,WANDB_NOTES,WANDB_NAME,WANDB_RUN_ID,WANDB_PROJECT,WANDB_ENTITY
        # NOTE: never log WANDB_API_KEY -- it is a secret and would leak into the run's stdout logs.
        logging.info(f"Wandb env vars. Project: {os.getenv('WANDB_PROJECT')}, Run Group: {os.getenv('WANDB_RUN_GROUP')}, Notes: {os.getenv('WANDB_NOTES')}, Name: {os.getenv('WANDB_NAME')}, Run ID: {os.getenv('WANDB_RUN_ID')}, Entity: {os.getenv('WANDB_ENTITY')}")

        # wandb.init() runs on rank 0 only, before the first training collective.
        # The Singularity compute nodes have flaky egress to api.wandb.ai, so online
        # init can hang. We bound it with init_timeout and, on failure, fall back to
        # OFFLINE logging so training proceeds regardless. This fallback is only safe
        # because init_distributed_device (above) raises the NCCL timeouts to 30 min.
        #
        # wandb writes to fast LOCAL disk (/scratch): blobfuse buffers the long-open
        # .wandb file and would lose it on a crash. For offline runs we rsync that
        # local dir to the blob run dir every epoch (in the checkpoint block) and
        # once at the end, so offline metrics survive VM teardown -- sync afterwards
        # with `wandb sync <log_base_path>/wandb/offline-run-*`.
        local_wandb_base = "/scratch" if os.path.isdir("/scratch") else log_base_path
        os.makedirs(local_wandb_base, exist_ok=True)
        os.environ["WANDB_DIR"] = local_wandb_base
        # Verbose wandb client + wandb-core (Go) debug logs -> <run>/logs/debug.log
        # and debug-internal.log (rsynced to blob each epoch). Diagnoses the online
        # init timeout on Singularity's flaky egress to api.wandb.ai.
        os.environ["WANDB_DEBUG"] = "true"
        os.environ["WANDB_CORE_DEBUG"] = "true"
        args.local_wandb_dir = os.path.join(local_wandb_base, "wandb")
        args.blob_wandb_dir = os.path.join(log_base_path, "wandb")
        args.wandb_offline = False
        # init_timeout bounds the run<->backend handshake (the network step that
        # times out on flaky egress); x_service_wait bounds startup of the local
        # wandb-core service process. Both = 2/3 of the NCCL timeout so the offline
        # fallback fires well inside the 30-min collective watchdog. NOTE: the field
        # is x_service_wait -- wandb >=0.19 renamed the old _service_wait.
        wandb_timeout_s = (_PG_TIMEOUT * 2 / 3).seconds
        wandb_init_kwargs = dict(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
            settings=wandb.Settings(
                init_timeout=wandb_timeout_s,
                x_service_wait=wandb_timeout_s,
            ),
        )
        try:
            wandb.init(**wandb_init_kwargs)
        except Exception as exc:
            logging.warning(f"wandb online init failed ({exc}); retrying in offline mode.")
            # teardown() is required: once the online session has started, switching
            # to offline via an env var is ignored -- only teardown + mode="offline"
            # forces it. teardown can take a few minutes to flush the dead online run,
            # which the 30 min NCCL timeout absorbs. If offline init also fails, let
            # the exception propagate so the job fails cleanly (max_attempts retries).
            try:
                wandb.teardown()
            except Exception:
                pass
            wandb.init(mode="offline", **wandb_init_kwargs)
            args.wandb_offline = True

        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:
        logging.info('Compiling model...')

        if args.grad_checkpointing and args.distributed:
            logging.info('Disabling DDP dynamo optimizer when grad checkpointing enabled.')
            # As of now (~PyTorch 2.4/2.5), compile + grad checkpointing work, but DDP optimizer must be disabled
            torch._dynamo.config.optimize_ddp = False

        model = torch.compile(original_model)

    if 'train' not in data:
        # If using int8, convert to inference mode.
        if args.use_bnb_linear is not None:
            from open_clip.utils import convert_int8_model_to_inference_mode
            convert_int8_model_to_inference_mode(model)
        # Evaluate.
        evaluate(model, data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer)
        return

    loss = create_loss(args)

    # Best metric tracking
    best_metric_value = None

    # NWC/SI tracking: snapshot state_dict to compute per-epoch weight changes
    prev_state_dict = None
    si_ratios = []  # collects per-epoch NWC_text / NWC_vision for running SI
    nwc_vision_keys, nwc_text_keys = [], []
    if args.track_dynamics and is_master(args):
        prev_state_dict = {k: v.clone().cpu() for k, v in original_model.state_dict().items()}
        for k, v in prev_state_dict.items():
            if not torch.is_tensor(v) or not torch.is_floating_point(v):
                continue
            name = k.removeprefix("module.")
            if name.startswith("visual."):
                nwc_vision_keys.append(k)
            elif name != "logit_scale":
                nwc_text_keys.append(k)

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)
        completed_epoch = epoch + 1

        # Compute NWC and SI (master only)
        if args.track_dynamics and is_master(args) and prev_state_dict is not None:
            curr_state_dict = {k: v.clone().cpu() for k, v in original_model.state_dict().items()}

            def _compute_nwc(keys):
                sum_sq, n = 0.0, 0
                for k in keys:
                    delta = curr_state_dict[k].float() - prev_state_dict[k].float()
                    sum_sq += delta.double().pow(2).sum().item()
                    n += delta.numel()
                return math.sqrt(sum_sq / n) if n > 0 else 0.0

            nwc_vision = _compute_nwc(nwc_vision_keys)
            nwc_text = _compute_nwc(nwc_text_keys)
            si_epoch = nwc_text / nwc_vision if nwc_vision > 0 else float('inf')
            si_ratios.append(si_epoch)
            si_running = sum(si_ratios) / len(si_ratios)

            logging.info(
                f'Epoch {completed_epoch} dynamics: NWC_vision={nwc_vision:.6f}, '
                f'NWC_text={nwc_text:.6f}, SI={si_epoch:.4f}, SI_running={si_running:.4f}'
            )

            if args.wandb:
                num_batches_per_epoch = data['train'].dataloader.num_batches // args.accum_freq
                step = num_batches_per_epoch * completed_epoch
                wandb.log({
                    'dynamics/nwc_vision': nwc_vision,
                    'dynamics/nwc_text': nwc_text,
                    'dynamics/si_epoch': si_epoch,
                    'dynamics/si_running': si_running,
                    'epoch': completed_epoch,
                }, step=step)

            prev_state_dict = curr_state_dict

        eval_metrics = {}
        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            eval_metrics = evaluate(model, data, completed_epoch, args, tb_writer=writer, tokenizer=tokenizer)
            if eval_metrics is None:
                eval_metrics = {}

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)

            # Save best checkpoint based on validation metric
            if args.save_best and args.best_metric in eval_metrics:
                current_value = eval_metrics[args.best_metric]
                is_new_best = (
                    best_metric_value is None
                    or (args.best_metric_lower_is_better and current_value < best_metric_value)
                    or (not args.best_metric_lower_is_better and current_value > best_metric_value)
                )

                if is_new_best:
                    best_metric_value = current_value
                    logging.info(
                        f"New best {args.best_metric}: {best_metric_value:.4f} at epoch {completed_epoch}. "
                        f"Saving {BEST_CHECKPOINT_NAME}"
                    )
                    best_checkpoint_dict = {
                        **checkpoint_dict,
                        "best_metric_name": args.best_metric,
                        "best_metric_value": best_metric_value,
                        "best_metric_epoch": completed_epoch,
                    }
                    tmp_best_path = os.path.join(args.checkpoint_path, "tmp_best.pt")
                    best_save_path = os.path.join(args.checkpoint_path, BEST_CHECKPOINT_NAME)
                    torch.save(best_checkpoint_dict, tmp_best_path)
                    os.replace(tmp_best_path, best_save_path)

            # Mirror offline wandb (written to local disk) to blob each epoch, so a
            # crash loses at most this epoch's metrics -- training itself is already
            # protected by the checkpoint just written above.
            if getattr(args, "wandb_offline", False):
                _sync_offline_wandb(args.local_wandb_dir, args.blob_wandb_dir)

        # Barrier to prevent non-master ranks from racing ahead to the next
        # epoch while rank 0 is still writing checkpoints to blob storage,
        # which would cause NCCL collective timeouts.
        if args.distributed:
            dist.barrier()

    if args.wandb and is_master(args):
        wandb.finish()
        # Final mirror of the offline run to blob after wandb closes/flushes it.
        if getattr(args, "wandb_offline", False):
            _sync_offline_wandb(args.local_wandb_dir, args.blob_wandb_dir)

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')
    

def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    import signal

    def _sigterm_handler(signum, frame):
        logging.warning(f"Caught signal {signum} (preemption). Exiting gracefully.")
        raise SystemExit(1)

    signal.signal(signal.SIGTERM, _sigterm_handler)
    signal.signal(signal.SIGINT, _sigterm_handler)

    try:
        main(sys.argv[1:])
    except SystemExit as e:
        logging.warning(f"Exiting with code {e.code}")
        sys.exit(e.code)
    except Exception as e:
        import traceback
        logging.error(f"FATAL ERROR: {traceback.format_exc()}")
        raise