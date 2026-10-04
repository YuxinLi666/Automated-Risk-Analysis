# Engine Fault NLP

Maintenance work orders → fault-chain extraction → fault knowledge graph → risk ranking.

This repository contains the core research software and input datasets for automated maintenance-oriented risk analysis. The current model predicts **subsystem, component, and failure mode**. Accepted predictions form a four-layer path under a fixed equipment root: **E0 → S → C → F**. Maintenance costs are excluded from the text encoder and linked to paths only for downstream risk analysis.

**Private research repository.** The `data/` directory contains restricted, non-anonymized maintenance records. Keep this repository private. The code license does not authorize redistribution of the data. See [Data access](DATA_AVAILABILITY.md).

## Included and excluded

Included: preprocessing, input schemas, annotation utilities, baseline models, MacBERT–MGP training and module ablations, checkpoint export and inference, evaluation, four-layer graph construction, Balanced Risk ranking, configuration, tests, and research inputs.

Excluded: plotting and presentation scripts, manuscript files, figures, previously computed predictions and metrics, ranking tables, training histories, pretrained or fine-tuned weights, virtual environments, and caches. The machine-generated labels retained in `data/` are reference-supervision inputs, not outputs presented as new experimental results.

## Installation

Python 3.11 is recommended. From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e .
python run.py doctor
python run.py test
python run.py audit
```

Install neural-model dependencies separately:

```bash
python -m pip install -e ".[ml]"
```

The retained model implementation targets PyTorch 2.5.1 and Transformers 4.46.3. Install a CUDA-compatible PyTorch build for your hardware using the [official installation instructions](https://pytorch.org/get-started/locally/). The controlled training experiment requires a CUDA GPU. Checkpoint inference also supports CPU execution. No API-based language model is needed for the MacBERT–MGP workflow.

## Quick offline check

```bash
python run.py demo
```

This uses synthetic Chinese examples and a character TF–IDF centroid model. It checks the basic data/model/export pipeline without downloading a pretrained model. It is not a MacBERT experiment or a claim about paper performance. Generated files go to ignored `outputs/` directories.

## Main research workflow

### 1. Inspect the input data

See [data/README.md](data/README.md) and [the input manifest](docs/data_manifest.json). The collection has 2,638 document-level work orders. Model development uses the existing 800/100/100 split of a 1,000-record preannotated seed set. A separate set of 727 work orders supplies the unseen evaluation cohort. These subsets overlap the overall collection and must not be added together as independent dataset sizes.

The original Chinese texts, labels, entity offsets, and domain dictionaries are preserved. Translating them would change the trained model's inputs. Repository documentation and user-facing workflow instructions are in English.

### 2. Train the current four-layer extractor

```bash
python run.py train --epochs 8
```

This runs the full MacBERT–MGP variant with seed 42 and explicitly saves the minimum-validation-loss checkpoint. The encoder is initialized from `hfl/chinese-macbert-base`. The existing experiment checks the 800/100/100/727 split sizes. It does not use the test or unseen results to choose a checkpoint. See [Reproducibility](docs/REPRODUCIBILITY.md) for implementation boundaries.

To run all six variants over the three configured seeds:

```bash
python run.py ablate --variants full,no_globalpointer,no_relation,no_auxiliary,no_class_reweighting,no_mean_pooling --seeds 42,52,62
```

Checkpoints are saved only when `--save-checkpoints` is supplied. The original experimental training code discarded its best state after evaluation. This repository adds checkpoint serialization and a matching inference entry point without changing the training objective or validation-selection rule. No trained checkpoints are bundled.

### 3. Predict and rank

```bash
python run.py predict --help
python run.py rank --help

python run.py predict --checkpoint outputs/module_ablation/runs/full/seed_42/checkpoint --records data/work_orders.jsonl --output outputs/macbert_mgp_predictions.jsonl --device cpu
python run.py rank --predictions outputs/macbert_mgp_predictions.jsonl --cohort data/unseen/labels.jsonl --output-dir outputs/unseen_ranking
python run.py rank --predictions outputs/macbert_mgp_predictions.jsonl --canonicalize --screen-materials --output-dir outputs/full_ranking
```

Use `predict` to apply a saved four-layer checkpoint to work-order JSONL data. Use `rank` to combine those predictions with the work-order costs. Ranking exports machine-readable path, node, and edge data without generating plots. Input paths and output directories are command-line arguments rather than dependencies on previous paper figures or result folders.

Balanced Risk combines empirical ranks of Wilson occurrence evidence, shrinkage-adjusted winsorized cost severity, and cumulative economic burden. Defaults are a 95% Wilson interval, 5th/95th-percentile winsorization, shrinkage strength 5, and contribution coefficients 0.45/0.35/0.20. These are study settings, not learned failure probabilities. The extracted code retains the historical positive-cost-only severity policy. See the ranking CLI help and output summary for its treatment of incomplete paths and costs.

## Repository layout

```text
engine-fault-nlp-core/
├── src/engine_fault_nlp/    # Core package and four-layer ranking
├── scripts/                # Training, inference, evaluation, ablation, ranking
├── data/                   # Restricted research inputs, tracked in this private repo
├── config/                 # Configuration
├── examples/               # Small illustrative inputs
├── tests/                  # Core and integration tests
├── tools/                  # Repository audit and release packaging
├── docs/                   # Reproducibility, provenance, and verification
├── .github/workflows/      # Offline core checks
├── run.py                  # Repository-local command dispatcher
└── pyproject.toml          # Package metadata and dependency groups
```

## Current versus legacy code

The package also retains earlier rule/TF–IDF, weak-supervision, teacher-annotation, five-layer cause extraction, and TOPSIS utilities to preserve their dependencies and tests. They are **not** added to the current neural model's predictions. In particular, `engine-fault-nlp build-graph-rank` is the legacy graph branch. Use `python run.py rank` for the current four-layer Balanced Risk branch. The historical monitored Transformer script includes a cause head and is not interchangeable with the cause-free module-ablation model.

Teacher-assisted annotation is opt-in and can send text to an externally configured endpoint. Do not use it with restricted records unless the data owner authorizes that transfer.

## License and provenance

Source code retains the original [MIT license](LICENSE). Data, pretrained checkpoints, and third-party resources are not relicensed by this repository. [Verification](docs/VERIFICATION.md) records the checks performed during packaging. The original working project and its results remain unchanged outside this clean copy.
