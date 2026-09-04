#!/usr/bin/env python3
"""
Combine vision and text encoder JSONs into CLIP architecture configs.

Usage:
    # Generate a single combination:
    python combine_architectures.py --vision ViT-B-16 --text Base

    # Generate all combinations in the grid:
    python combine_architectures.py --all

    # Generate all combinations for specific vision/text lists:
    python combine_architectures.py --vision ViT-Atto-16 ViT-Tiny-16 ViT-B-16 ViT-Giant-16 --text Femto Atto Tiny Base Giant
"""

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ARCH_DIR = SCRIPT_DIR.parent / "src" / "architectures"
VISION_DIR = ARCH_DIR / "vision"
TEXT_DIR = ARCH_DIR / "text"
OUTPUT_DIR = ARCH_DIR

EMBED_DIM = 512

ALL_VISION = sorted([p.stem for p in VISION_DIR.glob("*.json")])
ALL_TEXT = sorted([p.stem for p in TEXT_DIR.glob("*.json")])


def combine(vision_name: str, text_name: str, output_dir: Path = OUTPUT_DIR) -> Path:
    """Combine a vision and text encoder config into a full CLIP architecture JSON."""
    vision_path = VISION_DIR / f"{vision_name}.json"
    text_path = TEXT_DIR / f"{text_name}.json"

    if not vision_path.exists():
        raise FileNotFoundError(f"Vision config not found: {vision_path}")
    if not text_path.exists():
        raise FileNotFoundError(f"Text config not found: {text_path}")

    with open(vision_path) as f:
        vision_cfg = json.load(f)
    with open(text_path) as f:
        text_cfg = json.load(f)

    architecture = {
        "embed_dim": EMBED_DIM,
        "vision_cfg": vision_cfg,
        "text_cfg": text_cfg,
    }

    out_name = f"{vision_name}--{text_name}.json"
    out_path = output_dir / out_name
    with open(out_path, "w") as f:
        json.dump(architecture, f, indent=4)
        f.write("\n")

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Combine vision + text encoder configs into CLIP architectures")
    parser.add_argument("--vision", nargs="+", help="Vision encoder name(s), e.g. ViT-B-16")
    parser.add_argument("--text", nargs="+", help="Text encoder name(s), e.g. Base")
    parser.add_argument("--all", action="store_true", help="Generate all combinations in the grid")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory (default: architectures/)")
    args = parser.parse_args()

    if args.all:
        vision_list = ALL_VISION
        text_list = ALL_TEXT
    elif args.vision and args.text:
        vision_list = args.vision
        text_list = args.text
    else:
        parser.error("Provide --vision and --text, or use --all")

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    created = []
    for v in vision_list:
        for t in text_list:
            path = combine(v, t, output_dir)
            created.append(path)
            print(f"Created: {path.name}")

    print(f"\n{len(created)} architecture(s) generated in {output_dir}")


if __name__ == "__main__":
    main()
