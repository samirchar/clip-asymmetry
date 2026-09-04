
import os
from typing import Any, Dict, Iterator
import pandas as pd
import csv 
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import logging
import argparse
import json
from hashlib import blake2b
import pyarrow.parquet as pq
import pandas as pd
from glob import glob, has_magic
import re
import random
import numpy as np
import torch
from src import ARCHITECTURES_PATH
import transformers
import ast
from open_clip import add_model_config
import torch.nn.functional as F

LATEST_CHECKPOINT_NAME = "epoch_latest.pt"
BEST_CHECKPOINT_NAME = "epoch_best.pt"

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    transformers.set_seed(seed)


def create_csv_dataset_from_files(
    metadata_files: list,
    output_file:str,
    image_column:str="filepath",
    caption_column:str="title"
    )->None:
    """
    Create a CSV dataset with image paths and captions from parquet metadata files created by img2dataset in "files" mode.
    Assumes that the image files are stored in a directory structure where each shard has its own subdirectory named after the shard ID.
    
    Args:
        metadata_files (list): List of paths to parquet metadata files.
        output_file (str): Path to the output CSV file.
        image_column (str): Name of the image path column in the output CSV.
        caption_column (str): Name of the caption column in the output CSV.
    Returns:
        None
    """
    
    with open(output_file, "w") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([image_column, caption_column])

        for meta_file in tqdm(metadata_files):
            shard_id = os.path.basename(meta_file).split('.')[0]
            file_dir = os.path.dirname(meta_file)
            df = pd.read_parquet(meta_file)[['caption','key']]
            df['image_path'] = df['key'].apply(lambda x: os.path.join(file_dir, shard_id, f"{x}.jpg"))
            csv_writer.writerows(df[['image_path','caption']].values.tolist())


#TODO: This will have to change because the augment function processes one caption at a time. 
# This should all be done with DDP.

class CaptionTransformer:
    """
    Intended for simple transformations

    sample usage:

    Example 1:
    ct = CaptionTransformer(mode="test_123",augmented_text_key="text aug 1")
    ct.augment_write_all_shards(
        input_shards="data/mscoco/{00000..00003}.tar",
        output_dir="data/mscocotests/",
        prefix="test_123"
    )

    creates files like "data/mscocotests/test_123-00000.tar" with the augmented text in the "text aug 1" key.

    Example 2:
    ct = CaptionTransformer(mode="")
    ct.augment_write_all_shards(
        input_shards=sorted(glob("data/mscoco/*.tar")),
        output_dir="data/mscoco/",
        prefix="text_only"
    )

    creates files like "data/mscoco/text_only-00000.tar" with the augmented text in the "txt" key.

    """
    def __init__(self, mode: str,augmented_text_key: str = 'txt'):
        self.mode = mode
        self.augmented_text_key = augmented_text_key
    
    # This is a a placeholder for the actual augmentation function.
    def augment(self, text):
        return self.mode.encode() + b'-' + text

    def augment_write_shard(
        self,
        input_shard: str,
        output_shard: str):
        '''
        Uses the augmenter to read an input shard, augment the text, and write to an output shard.

        input_shard: e.g. "data/mscoco/00000.tar"
        output_shard: e.g. "data/mscoco/text-0.009-00000.tar"
        ''' 
        with wds.TarWriter(output_shard) as sink:
            for sample in wds.WebDataset(input_shard,shardshuffle=False):
                    sink.write(
                        {'__key__':sample['__key__'],
                        '__url__':sample['__url__'],
                        self.augmented_text_key:self.augment(text = sample['txt'])
                        }
                    )

    def augment_write_all_shards(
        self,
        input_shards: list[str] | str,
        output_dir: str,
        prefix: str,
        ) -> None:
        '''
        input_path_pattern: e.g. "data/mscoco/{00000..00003}.tar" or list of paths for example using sorted(glob("data/mscoco/*.tar"))
        '''
        
        os.makedirs(output_dir,exist_ok=True)

        input_shards = wds.SimpleShardList(input_shards)
        for input_shard in tqdm(input_shards):
            url = input_shard['url']
            base = os.path.basename(url)
            output_shard = os.path.join(output_dir,f"{prefix}-{self.mode}-{base}")
            self.augment_write_shard(input_shard=url,output_shard=output_shard)
            


    """
    It is also possible to do this by writing a consolidated webdataset with all the extra columns. It can be done efficiently without duplicating the data in the following way:
    1) Download the parquet,csv,tsv, etc. file with the urls and original captions
    2) Use the captions to created augmented version and add them as columns to the dataframe from step 1.
    3) Use download from img2dataset and pass the modified parquet from step 2 as input, and pass the additional columns to save as a list of strings to the save_additional_columns argument.
    4) This will create a webdataset with all the columns in the parquet file, including the augmented captions. The original caption and additional columns can be accessed via the "json" key
    5) I can use this format to create a custom dataloader that handles this


    from img2dataset import downloads

    d = pd.read_parquet("data/mscoco/mscoco.parquet")
    d = d.iloc[:10]
    d['TEXTAUG1'] = d['TEXT'].apply(lambda x: "AUG 1: " + x)
    d.to_parquet("data/mscoco/mscoco_small.parquet",index=False)

    download(
        processes_count=256,
        thread_count=4,
        url_list="data/mscoco/mscoco_small.parquet",
        image_size=224,
        output_folder='data/mscoco_small/',
        output_format="webdataset",
        input_format="parquet",
        url_col="URL",
        caption_col="TEXT",
        enable_wandb=False,
        retries=4,
        timeout=10,
        max_shard_retry=4,
        incremental_mode="incremental",
        save_additional_columns=['TEXTAUG1']

    )

    #!python bin/download_data.py --dataset mscoco --output_dir data/mscoco_small/ --save_additional_columns "TEXTAUG1"


    for i in wds.WebDataset("data/mscoco_small/00000.tar"):
        print(i['json'].decode())
        break

    """


#TODO: Remove these generators since they will not be used and save in personal repo.
def csv_generator(filepath:str, as_dict:bool = False):
    """
    Reads a CSV file line by line using a generator for memory efficiency.
    Args:
        filepath (str): The path to the CSV file.
        as_dict (bool): If True, yields each row as a dictionary.
    Yields:
        list: A list representing a row from the CSV file.
    """
    with open(filepath, 'r', newline='', encoding='utf-8') as csvfile:
        csv_reader = csv.reader(csvfile)
        headers = next(csv_reader)
        for row in csv_reader:
            if as_dict:
                yield {header: value for header, value in zip(headers, row)}
            else:
                yield row

def tsv_generator(filepath:str, as_dict:bool = False):
    """
    Reads a TSV file line by line using a generator for memory efficiency.
    Args:
        filepath (str): The path to the TSV file.
        as_dict (bool): If True, yields each row as a dictionary.
    Yields:
        list: A list representing a row from the TSV file.
    """
    with open(filepath, 'r', newline='', encoding='utf-8') as tsvfile:
        tsv_reader = csv.reader(tsvfile, delimiter='\t')
        headers = next(tsv_reader)
        for row in tsv_reader:
            if as_dict:
                yield {header: value for header, value in zip(headers, row)}
            else:
                yield row
        for row in tsv_reader:
            yield row

class ShardWritter:
    def __init__(self, output_dir: str, base_name: str, max_chunk_mb: int = 512, fmt: str = "parquet"):
        """
        WARNING: This class is not thread safe and using max_workers > 1 can lead to data corruption. 
        Additionally, to correctly work on multiple processes, each process should have their own base_name so they write to different files.

        fmt can be "parquet", "csv", or "tsv"
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.base_name = base_name
        self.max_bytes = max_chunk_mb * 1024 * 1024
        self.fmt = fmt.lower()

        self.current_chunk_idx = 0
        self.buffer = []
        self.buffer_size = 0  # estimated bytes in buffer
        

    def _chunk_path(self) -> str:
        return os.path.join(self.output_dir, f"{self.base_name}_{self.current_chunk_idx:05d}.{self.fmt}")

    def _write_shard(self):
        if not self.buffer:
            return
        df = pd.DataFrame(self.buffer)
        path = self._chunk_path()

        if self.fmt == "parquet":
            df.to_parquet(path, engine="pyarrow", index=False)
        elif self.fmt == "csv":
            df.to_csv(path, index=False)
        elif self.fmt == "tsv":
            df.to_csv(path, index=False, sep="\t")
        else:
            raise ValueError(f"Unsupported format {self.fmt}")

        # print(f"Wrote {path} with {len(df)} rows")
        self.buffer.clear()
        self.buffer_size = 0
        self.current_chunk_idx += 1

    def _estimate_record_size(self, record: Dict[str, Any]) -> int:
        return sum(len(str(v).encode("utf-8")) for v in record.values())
    
    def write_record(self, record: Dict[str, Any]):
        est_size = self._estimate_record_size(record)
        if self.buffer_size + est_size > self.max_bytes and self.buffer:
            self._write_shard()
        self.buffer.append(record)
        self.buffer_size += est_size

    def close(self):
        self._write_shard()
        
    def shard(self, records: Iterator[Dict[str, Any]], write_async: bool = False, max_workers: int = 1):
        if write_async:                    
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for record in records:
                    executor.submit(self.write_record, record)
            self.close()
        else:
            for rec in records:
                self.write_record(rec)
            self.close()


def get_logger():
    # Create a custom logger
    logger = logging.getLogger(__name__)

    # Set the logging level to INFO
    logger.setLevel(logging.INFO)

    # Create a console handler and set its level to INFO
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create a formatter that includes the current date and time
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set the formatter for the console handler
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

    # Example usage
    logger.info("This is an info message.")
    return logger


def config_parser(
    add_args_fn=None,
    config_file_arg_name:str="--config-file",
    raise_error_on_unknown:bool = False
    )->tuple[argparse.ArgumentParser, Dict[str, Any], str]:
    '''
    Config parser that can take arguments from a config file and command line.
    Command line arguments override config file arguments. Command line arguments from parser can have their own defaults, but will be overridden by config file if provided.
    
    WARNING: Only the arguments from the parser will be considered. If the config file has extra arguments that are not in the parser, they will be ignored or raise an error based on raise_error_on_unknown.
    
    Priority: command line args > config file args > parser defaults
    
    Arguments:
        add_args_fn: function that takes in a parser and adds arguments to it.
        config_file_arg_name: name of the argument to specify the config file path.
        raise_error_on_unknown: if True, raises an error if there are unknown arguments in the config file.
    Returns:
        args: argparse.Namespace with all arguments
    '''
    parser_first = argparse.ArgumentParser(add_help=False)
    parser_first.add_argument(config_file_arg_name, type=str, help="Path to config file", required=False)
    initial_args, _ = parser_first.parse_known_args()
    config = {}

    if initial_args.config_file is not None:
        if initial_args.config_file.endswith(".json"):
            with open(initial_args.config_file, "r") as f:
                config = json.load(f)
        else:
            raise ValueError("Only JSON config files are supported")

    parser = argparse.ArgumentParser(parents=[parser_first])
    
    if add_args_fn is not None:
        add_args_fn(parser)
        
    if config:
        # Validate config keys match parser arguments
        valid_keys = {a.dest for a in parser._actions}
        unknown = set(config.keys()) - valid_keys
        
        if unknown:
            if raise_error_on_unknown:
                raise ValueError(f"Unknown config keys {unknown}")
            else:
                print(f"Warning: Unknown config keys {unknown} will be ignored")
                for key in unknown:
                    config.pop(key)

        parser.set_defaults(**config)

    args = parser.parse_args()
    resolved = vars(args).copy()
    # resolved["_config_file"] = initial_args.config_file
    args.config_hash = hash_dict(resolved)
    return args, resolved


def hash_dict(dictionary: dict, digest_size: int = 12) -> str:
    # Drop non-essential or ephemeral keys (like paths you don’t want in the hash)
    clean = {k: dictionary[k] for k in sorted(dictionary)}
    # Canonical JSON: sorted keys, no whitespace
    blob = json.dumps(clean, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return blake2b(blob, digest_size=digest_size).hexdigest() 



def get_failed_img2ds_downloads(output_dir:str)->tuple[bool, pd.DataFrame]:
    rows = []
    for p in glob(f"{output_dir}/[0-9]*.parquet", recursive=True):
        df = pq.read_table(p).to_pandas()
        failed_temp = df[df["status"]!="success"][["url","caption"]].rename(columns={"url":"URL","caption":"TEXT"})
        if not failed_temp.empty:
            rows.append(failed_temp)

    has_failures = rows != []
    
    if not has_failures:
        return False

    failed = pd.concat(rows, ignore_index=True).drop_duplicates("URL")
    failed.to_parquet(os.path.join(output_dir,"failed.parquet"), index=False)
    return has_failures, failed

def parse_webdataset_path(path):
    import os, re
    from glob import glob, has_magic

    def shard_id(fname):
        matches = re.findall(r"\d+", fname)
        if not matches:
            return None
        # pick the longest run of digits
        return int(max(matches, key=len))

    if has_magic(path):
        paths = glob(path)
        if not paths:
            raise FileNotFoundError(f"No files matched glob: {path}")
        
        base_path = os.path.dirname(paths[0])
        sorted_shards = sorted(os.path.basename(p).split(".")[0] for p in paths)

        first_num = shard_id(sorted_shards[0])
        last_num  = shard_id(sorted_shards[-1])

        # ensure contiguity
        assert last_num - first_num + 1 == len(paths), \
            "Shards are not contiguous or parsing of path failed"

        # NEW: build {0001..1000} using numeric ids + preserved width
        # width from the longest digit run in the first shard stem
        first_stem = sorted_shards[0]
        first_digits = max(re.findall(r"\d+", first_stem), key=len)
        prefix,postfix = first_stem.split(first_digits,1)
        print(prefix,postfix)

        width = len(first_digits)
        
        return os.path.join(
            base_path,
            f"{prefix}"                      # prefix (e.g., 'cc3m-train-')
            f"{{{first_num:0{width}d}..{last_num:0{width}d}}}"  # {0001..1000}
            f"{postfix}.tar"                 # optional suffix + extension
        )
    else:
        return path


def register_models(architectures_path: str = ARCHITECTURES_PATH):
    print(f'Registering models from {architectures_path}')
    assert os.path.exists(f'{architectures_path}'), f'Path {architectures_path} does not exist'
    models = glob(f'{architectures_path}/*')
    for model in models:
        add_model_config(path=model)


def count_parameters(model):
    """
    Counts the total and trainable parameters in a PyTorch model.
    Args:
        model (torch.nn.Module): The PyTorch model to analyze.
    Returns:
        total_params (int): The total number of parameters in the model.
        trainable_params (int): The number of trainable parameters in the model.

    """
    total_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        num_params = param.numel()

        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
    return total_params, trainable_params

def read_openclip_params(path: str) -> dict:    
    params = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            key, _, value = line.partition(": ")
            try:
                params[key] = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                params[key] = value
    return params

def resolve_models(models_file: str, checkpoint_name: str = LATEST_CHECKPOINT_NAME) -> list[tuple[str, str]]:
    """Read a text file of openclip model directories and return (architecture, checkpoint_path) tuples.

    Each line in the file should be a path to a model directory containing
    params.txt and a checkpoints/ subfolder.
    """
    with open(models_file) as f:
        dirs = f.read().splitlines()

    models = []
    not_found = 0
    for d in dirs:
        params_path = os.path.join(d, "params.txt")
        ckpt_path = os.path.join(d, "checkpoints", checkpoint_name)

        if not os.path.isfile(params_path):
            print(f"Warning: params.txt not found in {d}, skipping")
            not_found += 1
            continue
        if not os.path.isfile(ckpt_path):
            print(f"Warning: {checkpoint_name} not found in {d}/checkpoints, skipping")
            not_found += 1
            continue

        params = read_openclip_params(params_path)
        architecture = params.get("model")

        if architecture is None:
            print(f"Warning: 'model' key not found in {params_path}, skipping")
            not_found += 1
            continue

        models.append((architecture, ckpt_path))

    if not_found:
        print(f"\n{not_found} model folder(s) skipped (not found or incomplete).")
    print(f"Resolved {len(models)} model(s).")
    return models


