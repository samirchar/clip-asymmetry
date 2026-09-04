'''
Adapted from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/data.py
'''


from open_clip_train.data import (ResampledShards2,
                                  SharedEpoch,
                                  detshuffle2,
                                  DataInfo,
                                  _SHARD_SHUFFLE_SIZE,
                                  _SHARD_SHUFFLE_INITIAL,
                                  _SAMPLE_SHUFFLE_SIZE,
                                  _SAMPLE_SHUFFLE_INITIAL,
                                  tarfile_to_samples_nothrow,
                                  log_and_continue,
                                  get_imagenet,
                                  expand_urls,
                                  get_dataset_size,
                                  get_csv_dataset,
                                  get_synthetic_dataset)
import math
import os
import logging
from functools import partial
from glob import glob
import webdataset as wds
from ast import literal_eval
import numpy as np

def update_wds_zip(src,extra_data):
    for s , extra in zip(src,extra_data):
        assert s["key"] == extra["key"], f"Keys do not match: {s['key']} vs {extra['key']}"
        s.update(extra)
        yield s

def find_extra_data_url(url:str, prefix:str, mode:str, extra_data_dir: str = None)-> str:
    """Given a URL, find the corresponding URL for the extra column data."""
    assert url.endswith(".tar"), f"URL does not end with .tar: {url}"
    base = os.path.basename(url)
    new_base = f"{prefix}-{mode}-{base}"
    if extra_data_dir is None:
        extra_data_dir = os.path.dirname(url)
    new_url = os.path.join(extra_data_dir, new_base)
    return new_url

def update_wds(src, find_extra_data_url_func):
    """Given an iterator over a dataset, add an extra column from a separate dataset."""
    last_url = None
    column_src = None
    for sample in src:
        # We use the __url__ field to keep track of which shard we are working on.
        # We then open the corresponding URL for the extra column data if necessary.
        if last_url != sample["__url__"]:
            column_url = find_extra_data_url_func(sample["__url__"])
            column_src = iter(wds.WebDataset(column_url, shardshuffle=False))
            last_url = sample["__url__"]
        # Read the next sample from the extra column data.
        extra = next(column_src)
        # Check that the keys match.
        assert extra["__key__"] == sample["__key__"], f"Keys do not match: {extra['__key__']} vs {sample['__key__']}"
        # Update the sample with the extra data.
        sample.update(extra)
        # Yield the updated sample.
        yield sample

def filter_no_caption_no_image_no_json(sample):
    has_caption = ('txt' in sample)
    has_image = ('png' in sample or 'jpg' in sample or 'jpeg' in sample or 'webp' in sample)
    has_json = ('json' in sample)
    return has_caption and has_image and has_json

def get_columnar_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):
    input_shards = args.train_data if is_train else args.val_data
    input_shards_extra = args.train_data_extra_column if is_train else args.val_data_extra_column


    update_wds_partial = partial(
        update_wds,
        find_extra_data_url_func=partial(
            find_extra_data_url,
            prefix="text_only",
            mode="",
            extra_data_dir="data/mscoco/"
        )
    )

    input_shards = wds.WebDataset(
        sorted(glob("data/mscoco/0*.tar"))
    ).compose(update_wds_partial)


    assert input_shards is not None

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 
    
    pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    # implement with high level api instead of pipeline
    if is_train:
        input_shards = input_shards.shuffle(
            seed=args.seed
        ).split_by_node().split_by_worker()


    if is_train:
        pipeline.extend([
            detshuffle2(
                bufsize=_SHARD_SHUFFLE_SIZE,
                initial=_SHARD_SHUFFLE_INITIAL,
                seed=args.seed,
                epoch=shared_epoch,
            ),
            wds.split_by_node,
            wds.split_by_worker,
        ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.map_dict(image=preprocess_img, text=lambda text: tokenizer(text)[0]),
        wds.to_tuple("image", "text"),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_wds_dataset(args, preprocess_img, is_train, epoch=0, floor=False, tokenizer=None):

    def make_sample(sample):
        image=preprocess_img(sample["image"])
        text=tokenizer(sample["json"][caption_column])[0]
        return image, text

    def filter_none_captions(sample):
        """Skip samples where the caption key is None."""
        if sample["json"].get(caption_column) is None:
            logging.warning(
                f"Skipping sample with None caption: key='{sample.get('__key__', '?')}', "
                f"shard='{sample.get('__url__', '?')}', caption_key='{caption_column}'"
            )
            return False
        return True
    
    caption_column = args.caption_key
    
    input_shards = args.train_data if is_train else args.val_data
    assert input_shards is not None
    resampled = getattr(args, 'dataset_resampled', False) and is_train

    num_shards = None
    if is_train:
        if args.train_num_samples is not None:
            num_samples = args.train_num_samples
        else:
            num_samples, num_shards = get_dataset_size(input_shards)
            if not num_samples:
                raise RuntimeError(
                    'Currently, the number of dataset samples must be specified for the training dataset. '
                    'Please specify it via `--train-num-samples` if no dataset length info is present.')
    else:
        # Eval will just exhaust the iterator if the size is not specified.
        num_samples = args.val_num_samples or 0 

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc

    if is_train and args.train_data_upsampling_factors is not None:
        assert resampled, "--train_data_upsampling_factors is only supported when sampling with replacement (with --dataset-resampled)."
    
    if resampled:
        pipeline = [ResampledShards2(
            input_shards,
            weights=args.train_data_upsampling_factors,
            deterministic=True,
            epoch=shared_epoch,
        )]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if is_train:
        if not resampled:
            pipeline.extend([
                detshuffle2(
                    bufsize=_SHARD_SHUFFLE_SIZE,
                    initial=_SHARD_SHUFFLE_INITIAL,
                    seed=args.seed,
                    epoch=shared_epoch,
                ),
                wds.split_by_node,
                wds.split_by_worker,
            ])
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
        
        
    else:
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            wds.tarfile_to_samples(handler=log_and_continue),
        ])
    pipeline.extend([
        wds.select(filter_no_caption_no_image_no_json),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt"),
        wds.select(filter_none_captions),
        wds.map(make_sample),
        wds.batched(args.batch_size, partial=not is_train)
    ])

    dataset = wds.DataPipeline(*pipeline)

    if is_train:
        if not resampled:
            num_shards = num_shards or len(expand_urls(input_shards)[0])
            assert num_shards >= args.workers * args.world_size, 'number of shards must be >= total workers'
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = args.batch_size * args.world_size
        num_batches = round_fn(num_samples / global_batch_size)
        num_workers = max(1, args.workers)
        num_worker_batches = round_fn(num_batches / num_workers)  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this
    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / args.batch_size)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=args.workers,
        persistent_workers=args.workers > 0,
    )

    # FIXME not clear which approach is better, with_epoch before vs after dataloader?
    # hoping to resolve via https://github.com/webdataset/webdataset/issues/169
    # if is_train:
    #     # roll over and repeat a few samples to get same number of full batches on each node
    #     global_batch_size = args.batch_size * args.world_size
    #     num_batches = math.ceil(num_samples / global_batch_size)
    #     num_workers = max(1, args.workers)
    #     num_batches = math.ceil(num_batches / num_workers) * num_workers
    #     num_samples = num_batches * global_batch_size
    #     dataloader = dataloader.with_epoch(num_batches)
    # else:
    #     # last batches are partial, eval is done on single (master) node
    #     num_batches = math.ceil(num_samples / args.batch_size)

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)

def get_dataset_fn(data_path, dataset_type):
    if dataset_type == "webdataset":
        return get_wds_dataset
    elif dataset_type == "csv":
        return get_csv_dataset
    elif dataset_type == "synthetic":
        return get_synthetic_dataset
    elif dataset_type == "auto":
        ext = data_path.split('.')[-1]
        if ext in ['csv', 'tsv']:
            return get_csv_dataset
        elif ext in ['tar']:
            return get_wds_dataset
        else:
            raise ValueError(
                f"Tried to figure out dataset type, but failed for extension {ext}.")
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
def get_data(args, preprocess_fns, epoch=0, tokenizer=None):
    preprocess_train, preprocess_val = preprocess_fns
    data = {}

    if args.train_data or args.dataset_type == "synthetic":
        data["train"] = get_dataset_fn(args.train_data, args.dataset_type)(
            args, preprocess_train, is_train=True, epoch=epoch, tokenizer=tokenizer)

    if args.val_data:
        data["val"] = get_dataset_fn(args.val_data, args.dataset_type)(
            args, preprocess_val, is_train=False, tokenizer=tokenizer)

    if args.imagenet_val is not None:
        data["imagenet-val"] = get_imagenet(args, preprocess_fns, "val")

    if args.imagenet_v2 is not None:
        data["imagenet-v2"] = get_imagenet(args, preprocess_fns, "v2")

    return data