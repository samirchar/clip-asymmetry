# Bigger Text Encoders Can Hurt CLIP Zero-Shot Performance

Contrastive Language-Image Pretraining (CLIP) is a building block of many machine
learning applications. Scaling laws have guided resource allocation for large-scale
training, yet prior work treats total CLIP model size as a single variable, without
exploring how the capacity split between encoders impacts downstream performance.
Here, we train multiple CLIP models with different vision and text encoder sizes,
revealing that for most vision encoders, there is an optimal text encoder size beyond
which zero-shot performance degrades, even as total parameter count increases.
Exploiting this behavior yields efficient configurations that match the zero-shot
performance of the standard ViT-B/16 architecture with up to 55% fewer parameters. We
further show that this degradation stems from overfitting induced by the oversized text
encoder, and that using modality-specific weight decay coefficients not only recovers but
improves performance across all degraded configurations. A geometric analysis reveals a
trade-off in which scaling the text encoder improves embedding uniformity but worsens
cross-modal alignment; we further show that these metrics are predictive of zero-shot
performance. We hope these findings motivate CLIP architectures and training methods that
counteract this degradation, a prerequisite for scaling CLIP reliably and efficiently.

## Table of Contents

- [Installation](#installation)
- [Data](#data)
- [Reproducing Paper Figures](#reproducing-paper-figures-from-provided-results)
- [Secrets (`.env`)](#secrets-env)
- [Reproducing the Analysis](#reproducing-the-analysis)
- [Creating Architectures](#creating-architectures)
- [Training a Model](#training-a-model)
- [Evaluating a Model](#evaluating-a-model)

## Installation

**Requirements**
- Python 3.10
- PyTorch 2.12 with CUDA 13 or above

```bash
git clone https://github.com/samirchar/clip-asymmetry
cd clip-asymmetry

python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

(On Windows, activate with `.venv\Scripts\activate`.)

In that environment, install PyTorch 2.12.1 using the index URL for your installed CUDA
version (see [pytorch.org](https://pytorch.org/get-started/locally/)):

```bash
pip install torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu130
```

Then install the package. The first line installs the core (training and analysis); the
second adds the plotting extras used by `bin/generate_figures.py`.

```bash
pip install -e .
pip install -e ".[figures]"
```

## Data

The external data you may download yourself: the training data (CC3M / CC12M), optionally
ImageNet, and the released model checkpoints (only needed to re-run the analysis). Zero-shot
and linear-probe evaluation datasets are fetched automatically by `clip-benchmark`, so they
are not listed here.

### CC3M / CC12M (training)

The models train on the original captions of the public `pixparse` webdatasets:

```bash
huggingface-cli download pixparse/cc3m-wds --repo-type dataset --local-dir data/cc3m-wds
huggingface-cli download pixparse/cc12m-wds --repo-type dataset --local-dir data/cc12m-wds
```

This downloads CC3M as `cc3m-train-{0000..0575}.tar` (plus `cc3m-validation-{0000..0015}.tar`)
and CC12M as `cc12m-train-{0000..2175}.tar` (train split only). The provided train configs
point `train_data`/`val_data` at these paths.

### ImageNet (optional)

ImageNet powers the periodic zero-shot readout during training (`imagenet_val` in the
config). Register at [image-net.org](https://image-net.org), download the validation set, and place it at
`data/imagenet/val/` in class-folder format (`data/imagenet/val/<wnid>/*.JPEG`). To skip
it, set `"imagenet_val": null` in the train config.

### Model checkpoints

The trained model checkpoints are available for download on the Hugging Face Hub
([`samirchar/clip-asymmetry`](https://huggingface.co/samirchar/clip-asymmetry)). This release
contains the main **CC12M architecture grid**: 30 CLIP models spanning 5 vision encoders x 6
text encoders. Download them into `openclip_logs/`, so each run directory contains a
`params.txt` and a
`checkpoints/` folder:

```bash
huggingface-cli download samirchar/clip-asymmetry --repo-type model --local-dir openclip_logs
```

Each model lives in its own folder named by its model id (e.g. `ViT-B-16--Base/`). To
download a single model instead of all 30, pass an `--include` pattern:

```bash
huggingface-cli download samirchar/clip-asymmetry --repo-type model --local-dir openclip_logs --include "ViT-B-16--Base/*"
```

The 30 released models (each named `{VisionEncoder}--{TextEncoder}`):

| Model | Vision encoder | Text encoder |
|---|---|---|
| `ViT-Tiny-16--Femto` | ViT-Tiny-16 | Femto |
| `ViT-Tiny-16--Nano` | ViT-Tiny-16 | Nano |
| `ViT-Tiny-16--Atto` | ViT-Tiny-16 | Atto |
| `ViT-Tiny-16--Tiny` | ViT-Tiny-16 | Tiny |
| `ViT-Tiny-16--Base` | ViT-Tiny-16 | Base |
| `ViT-Tiny-16--Giant` | ViT-Tiny-16 | Giant |
| `ViT-Atto-16--Femto` | ViT-Atto-16 | Femto |
| `ViT-Atto-16--Nano` | ViT-Atto-16 | Nano |
| `ViT-Atto-16--Atto` | ViT-Atto-16 | Atto |
| `ViT-Atto-16--Tiny` | ViT-Atto-16 | Tiny |
| `ViT-Atto-16--Base` | ViT-Atto-16 | Base |
| `ViT-Atto-16--Giant` | ViT-Atto-16 | Giant |
| `ViT-B-16--Femto` | ViT-B-16 | Femto |
| `ViT-B-16--Nano` | ViT-B-16 | Nano |
| `ViT-B-16--Atto` | ViT-B-16 | Atto |
| `ViT-B-16--Tiny` | ViT-B-16 | Tiny |
| `ViT-B-16--Base` | ViT-B-16 | Base |
| `ViT-B-16--Giant` | ViT-B-16 | Giant |
| `ViT-Giant-16--Femto` | ViT-Giant-16 | Femto |
| `ViT-Giant-16--Nano` | ViT-Giant-16 | Nano |
| `ViT-Giant-16--Atto` | ViT-Giant-16 | Atto |
| `ViT-Giant-16--Tiny` | ViT-Giant-16 | Tiny |
| `ViT-Giant-16--Base` | ViT-Giant-16 | Base |
| `ViT-Giant-16--Giant` | ViT-Giant-16 | Giant |
| `ViT-Colossal-16--Femto` | ViT-Colossal-16 | Femto |
| `ViT-Colossal-16--Nano` | ViT-Colossal-16 | Nano |
| `ViT-Colossal-16--Atto` | ViT-Colossal-16 | Atto |
| `ViT-Colossal-16--Tiny` | ViT-Colossal-16 | Tiny |
| `ViT-Colossal-16--Base` | ViT-Colossal-16 | Base |
| `ViT-Colossal-16--Giant` | ViT-Colossal-16 | Giant |

## Reproducing Paper Figures (from provided results)

All figure and table inputs are committed under `data/` and `benchmarks/`, so every figure
regenerates **offline**, with no GPU, no data download, and no Weights & Biases account needed:

```bash
python bin/generate_figures.py
python bin/generate_figures.py --list
python bin/generate_figures.py --fig rankme_heatmap modality_gap
```

Run with no arguments to regenerate every figure into `figures/`; `--list` prints the
available figure names; `--fig <names>` regenerates only the named subset.


## Secrets (`.env`)

To run the training script (`bin/train.py`) with W&B logging (with the `report_to="wandb"` argument), create a git-ignored `.env` file in the repository root with your W&B credentials:

```bash
WANDB_API_KEY=your_wandb_api_key
WANDB_ENTITY=your_wandb_entity
WANDB_PROJECT=your_wandb_project
```

The file is loaded automatically via `python-dotenv`.

## Reproducing the Analysis

The following scripts reproduce the different analyses done throughout the paper. Some
scripts require a list of trained models, passed as a models file (`--models-file`): a text
file with one run directory per line, where each line is the directory that holds a run's
checkpoints (it contains the run's `params.txt` and a `checkpoints/` folder). See
[`configs/eval/example_models_list.txt`](configs/eval/example_models_list.txt) for the
format. You can obtain these checkpoints by downloading the released checkpoints from the
Hugging Face Hub (see [Data](#data)) or by training your own models with the training script
(see [Training a Model](#training-a-model)). Each script writes its artifact into
`data/analysis/`.

### Embedding Geometry

The following scripts compute the RankMe, the uniformity and alignment, and the modality gap
of the models specified in the models file:

```bash
python bin/run_rankme.py --models-file=<models_list.txt> \
  --val-datasets="data/cc12m-wds/cc12m-train-*.tar" \
  --output=data/analysis/rankme_results_cc12m.json --max-samples=25600

python bin/run_uni_align.py --models-file=<models_list.txt> \
  --val-datasets mscoco_captions flickr30k \
  --output=data/analysis/uni_align_cc12m.json --max-samples=5000

python bin/run_modality_gap.py --models-file <models_list.txt> \
  --val-datasets mscoco_captions flickr30k \
  --output=data/analysis/modality_gap_results.json --max-samples=5000
```

### Contrastive loss on held-out datasets

First convert the held-out retrieval datasets into webdataset format with
`bin/create_eval_wds.py`:

```bash
python bin/create_eval_wds.py --datasets mscoco_captions flickr30k --output-dir data/eval_wds
```

Then compute the contrastive validation loss per checkpoint (used for the overfitting
analysis):

```bash
python bin/compute_val_loss.py --models-file <models_list.txt> \
  --eval-wds-root data/eval_wds --val-datasets mscoco_captions flickr30k \
  --output data/analysis/val_loss_ogr.parquet
```

### Regressions

`zs_perf_regression_params.py` regresses average zero-shot accuracy on the vision and text
encoder parameter counts; `zs_perf_regression_geometry.py` regresses it on the embedding
geometry (alignment and uniformity):

```bash
python bin/zs_perf_regression_params.py
python bin/zs_perf_regression_geometry.py
```

### Significance tests

Statistical tests behind the paper's claims. `significance_degradation_seeds.py` is a
training-seed variance test that checks the zero-shot degradation is not just seed noise.
The other three are per-dataset two-sided sign tests: `significance_degradation_tasks.py`
compares the peak text encoder against the oversized Giant text encoder;
`significance_wd_ablation_tasks.py` tests the modality-specific weight-decay ablation
asymmetry; and `significance_wd_intervention_tasks.py` tests the improvement from
modality-specific weight decay:

```bash
python bin/significance_degradation_seeds.py
python bin/significance_degradation_tasks.py
python bin/significance_wd_ablation_tasks.py
python bin/significance_wd_intervention_tasks.py
```

## Creating Architectures

An architecture is an OpenCLIP model config named `{Vision}--{Text}.json` in
`src/architectures/`, combining a **vision base** (`src/architectures/vision/`) with a
**text base** (`src/architectures/text/`). Models are registered at runtime via
`register_models()` (`open_clip.add_model_config`).

### Method 1: programmatic

A single combination (writes `src/architectures/ViT-B-16--Base.json`):

```bash
python bin/create_architecture_config.py --vision ViT-B-16 --text Base --output-dir src/architectures
```

Or a full grid:

```bash
python bin/create_architecture_config.py \
  --vision ViT-Atto-16 ViT-Tiny-16 ViT-B-16 ViT-Giant-16 \
  --text Femto Atto Tiny Base Giant --output-dir src/architectures
```

### Method 2: by hand

Create `src/architectures/ViT-B-16--Base.json`, where `vision_cfg` comes from a vision base
(e.g. `src/architectures/vision/ViT-B-16.json`), `text_cfg` from a text base
(e.g. `src/architectures/text/Base.json`), and `embed_dim` is the shared projection size:

```json
{
    "embed_dim": 512,
    "vision_cfg": { "image_size": 224, "layers": 12, "width": 768, "patch_size": 16 },
    "text_cfg":   { "context_length": 77, "vocab_size": 49408, "width": 512, "heads": 8, "layers": 12 }
}
```

## Training a Model

```bash
python bin/train.py --config-file configs/train/cc12m-base-original.json \
  --name my_run --resume latest --logs openclip_logs
```

Settings come from the JSON `--config-file` merged with CLI flags (CLI wins). Key fields in
`configs/train/*.json`:

| Field | Meaning |
|---|---|
| `model` | Architecture name `{Vision}--{Text}` (must exist in `src/architectures/`) |
| `train_data` / `val_data` | Local webdataset `.tar` glob (e.g. `data/cc12m-wds/cc12m-train-*.tar`) |
| `caption_key` | JSON metadata key holding the caption (`"caption"`) |
| `train_num_samples` | Number of samples per epoch |
| `batch_size`, `epochs`, `lr`, `warmup`, `eps` | Core optimization settings |
| `wd` | Weight decay (modality-specific weight decay is a central ablation of this paper) |
| `precision` | `amp` (mixed precision) or `fp32` |
| `imagenet_val` | ImageNet val dir for the periodic zero-shot readout, or `null` to skip |
| `zeroshot_frequency` | Epochs between zero-shot readouts |
| `report_to` / `wandb_project_name` | Weights & Biases logging (`report_to: ""` to disable) |
| `logs` | Output directory for checkpoints and logs |
| `train_pre_resize`, `aug_cfg` | Image pre-resize and augmentation |

Checkpoints are written to `<logs>/<name>/checkpoints/`.

## Evaluating a Model

Evaluation uses `clip-benchmark` (patched fork, see [Installation](#installation)). It runs
in two stages, `eval` (compute per-dataset results) then `build` (aggregate to a CSV), or
`all` for both. Combined:

```bash
python bin/benchmark.py all --name my_bench \
  --models-pattern "openclip_logs/my_run*/" --distributed
```

Or in two steps:

```bash
python bin/benchmark.py eval  --name my_bench \
  --models-pattern "openclip_logs/my_run*/" --datasets-path configs/eval/webdatasets_zero_shot.txt --distributed
python bin/benchmark.py build --name my_bench \
  --files benchmarks/benchmark_my_bench*.json
```

The eval dataset lists live in `configs/eval/`:

| File | Role |
|---|---|
| `webdatasets_zero_shot.txt` | Zero-shot datasets (`mscoco_captions`, `flickr30k`, `imagenet1k`, ...) |
| `webdatasets_probe.txt` | Linear-probe datasets (`cars`, `food101`, `country211`, ...) |
| `tasklist.yml` | Dataset taxonomy (name, size, main metric, tags) used to group results in the analysis |

Select a list with `--datasets-path configs/eval/webdatasets_zero_shot.txt`. Per-run results
are written to `benchmarks/*.json` and aggregated into `benchmarks/merged_benchmark.csv`,
which the analysis and figure scripts consume.
