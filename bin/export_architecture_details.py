import argparse
import json
import os
from pathlib import Path

import pandas as pd
from open_clip.factory import create_model

from src import ARCHITECTURES_PATH
from src.utils import count_parameters, register_models

def _load_arch_config(arch_name, architectures_path):
    with open(Path(architectures_path) / f"{arch_name}.json", "r") as handle:
        return json.load(handle)


def _extract_vision_heads(module):
    if module is None:
        return None

    if hasattr(module, "transformer") and hasattr(module.transformer, "resblocks"):
        blocks = module.transformer.resblocks
    else:
        return None

    if len(blocks) > 0 and hasattr(blocks[0], "attn"):
        return getattr(blocks[0].attn, "num_heads", None)
    return None


def main(output_path: str, base_architecture: str, architectures_path: str) -> None:
    register_models(architectures_path)

    architectures = [f for f in os.listdir(architectures_path) if f.endswith(".json")]

    rows = []
    for arch in architectures:
        print(arch)
        arch = arch.partition(".")[0]
        model = create_model(model_name=arch)
        arch_config = _load_arch_config(arch, architectures_path)
        vision_cfg = arch_config.get("vision_cfg", {})
        text_cfg = arch_config.get("text_cfg", {})

        trainable_params_visual = count_parameters(model.visual)[-1]
        trainable_params_all = count_parameters(model)[-1]
        trainable_params_text = trainable_params_all - trainable_params_visual - 1

        vision_layers = vision_cfg.get("layers")
        vision_width = vision_cfg.get("width")
        vision_heads = _extract_vision_heads(model.visual)
        text_layers = text_cfg.get("layers")
        text_width = text_cfg.get("width")
        text_heads = text_cfg.get("heads")

        rows.append([
            arch,
            trainable_params_visual,
            trainable_params_text,
            trainable_params_all,
            trainable_params_text / trainable_params_visual,
            vision_layers,
            vision_width,
            vision_heads,
            text_layers,
            text_width,
            text_heads,
        ])

    df = pd.DataFrame(
        rows,
        columns=[
            "architecture",
            "vision_params",
            "text_params",
            "total_params",
            "text_vision_ratio",
            "vision_layers",
            "vision_width",
            "vision_heads",
            "text_layers",
            "text_width",
            "text_heads",
        ],
    ).sort_values(by="text_params")

    base_text_params = df.query(f"architecture=='{base_architecture}'")["text_params"].iloc[0]
    df["text_params_relative_to_base"] = df["text_params"] / base_text_params

    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export architecture parameter counts to CSV.")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="architecture_details.csv",
        help="Path for the output CSV file (default: architecture_details.csv)",
    )
    parser.add_argument(
        "-b", "--base-architecture",
        type=str,
        default="ViT-B-16--Base",
        help="Base architecture for relative text param comparison (default: ViT-B-16--Base)",
    )
    parser.add_argument(
        "-a", "--architectures-path",
        type=str,
        default=ARCHITECTURES_PATH,
        help="Path to the architectures directory (default: src/architectures/)",
    )
    args = parser.parse_args()

    
    main(args.output, args.base_architecture, args.architectures_path)