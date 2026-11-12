# Automated Risk Analysis

Code for **Automated risk analysis via language model–based hierarchical fault knowledge extraction from maintenance work orders**.

**Authors:** Yuxin Li, Rongxi Wang, Yonggang Song, Yufan Li, Jianmin Gao, and Zhao Wang.

[Application video](videos/)

## Overview

Maintenance work orders contain fault descriptions, diagnostic findings, repair actions, part information, and maintenance costs. Nonstandard terminology, dispersed descriptions, and implicit fault information make it difficult to organize these records consistently for maintenance risk analysis.

This study develops an integrated framework that extracts standardized fault knowledge from maintenance work orders, constructs a hierarchical fault knowledge graph, and ranks fault paths using occurrence evidence and economic consequences. The framework connects each identified fault path to its supporting work orders, providing traceable evidence for maintenance prioritization and resource allocation.

The framework is applied to a heavy-duty natural gas engine fleet, with **2,439 valid maintenance work orders** retained for risk analysis.

## Main contributions

1. **An integrated framework for automated maintenance risk analysis.** Maintenance work orders are transformed into a hierarchical fault knowledge graph that links fault information to recorded occurrences and maintenance costs.
2. **A multi-granularity fault knowledge extraction model.** MacBERT–MGP combines document-level label prediction with auxiliary entity span extraction and relation classification to identify standardized subsystem, component, and failure mode labels.
3. **A balanced risk score for fault path prioritization.** Occurrence evidence, per-occurrence cost severity, and cumulative economic burden are combined to identify both infrequent costly faults and recurrent faults with substantial maintenance expenditure.

## Framework

```text
Maintenance work orders
        │
        ▼
Aggregation, cleaning, and field-tagged text serialization
        │
        ▼
MacBERT–MGP: standardized fault label prediction
        │
        ▼
Hierarchical fault knowledge graph
Equipment → Subsystem → Component → Failure mode
        │
        ▼
Fault path retrieval and work-order cost aggregation
        │
        ▼
Balanced risk ranking and maintenance prioritization
```

### 1. Maintenance work order processing

Records sharing the same work-order identifier are aggregated and cleaned. Available text fields are serialized into a field-tagged sequence that preserves the origin of each text fragment. Model training and inference use the original Chinese text. Settlement costs are linked through work-order identifiers for downstream risk analysis.

### 2. MacBERT–MGP knowledge extraction

**MacBERT–MGP** is a MacBERT-based multi-granularity GlobalPointer model with a shared Chinese MacBERT-base encoder.

| Branch | Function |
| --- | --- |
| Document-level prediction | Predicts standardized subsystem, component, and failure mode labels |
| Entity span extraction | Uses GlobalPointer to provide local entity supervision |
| Relation classification | Provides supervision from relations between annotated entities |

The document representation combines the `[CLS]` token representation with masked mean pooling. Three classification heads produce scores for the subsystem, component, and failure mode label sets. All branches are trained jointly, allowing document-level context and local entity–relation evidence to shape the shared representation.

At inference, each document-level head selects its highest-scoring label. The label is accepted when its score is at least **0.5**. Work orders with accepted predictions at all three levels form complete fault paths.

### 3. Hierarchical fault knowledge graph

Each complete prediction is organized as:

**Equipment root E0 → Subsystem S → Component C → Failure mode F**

Nodes with the same standardized identifier are merged within each level. The graph retains work-order identifiers and complete node combinations, linking each fault path to its supporting records. These supported paths serve as the common units for counting fault occurrences, aggregating maintenance costs, and reviewing the evidence underlying maintenance priorities.

### 4. Balanced risk ranking

Fault paths are evaluated along two complementary dimensions: **occurrence evidence** and **economic consequences**. Three indicators operationalize these dimensions:

| Indicator | Definition |
| --- | --- |
| Occurrence evidence | Lower bound of the two-sided 95% Wilson score interval for the path's work-order frequency |
| Cost severity | Shrinkage-adjusted winsorized mean maintenance cost per occurrence |
| Economic burden | Cumulative original maintenance cost associated with the path |

Cost severity uses path-specific 5th/95th-percentile winsorization and a shrinkage strength of **5**. The three indicators are normalized using ascending ranks with average ranks for ties. Their combination is:

```text
Balanced risk = 0.45 × sqrt(max(occurrence_rank, ε) × max(severity_rank, ε))
                + 0.35 × severity_rank
                + 0.20 × burden_rank

ε = 1e-12
```

Paths are ranked by decreasing unrounded score to establish relative maintenance priorities within the analysis set. The occurrence–severity interaction emphasizes paths with both substantial occurrence evidence and high costs, while the separate severity and burden terms retain the importance of infrequent costly faults and recurrent maintenance demands.

## Case study and evaluation

The study uses industrial maintenance work orders from a heavy-duty natural gas engine fleet. The datasets serve three purposes:

| Purpose | Dataset |
| --- | --- |
| Model development | 1,000 work orders, divided into 800 training, 100 validation, and 100 test records |
| Generalization evaluation | 727 labeled work orders excluded from training and model selection |
| Fleet-level risk analysis | 2,439 valid maintenance work orders from 1,454 vehicles recorded between 2022 and 2024 |

The manuscript evaluates node-level extraction, complete-path prediction, performance across work-order lengths, module ablations, and comparisons with lexical classifiers and a compact pretrained Transformer. Risk analysis highlights infrequent faults with high per-occurrence costs and recurrent faults with substantial cumulative economic burden.

## Engineering application

The web application implements the proposed framework for maintenance risk analysis. It connects work-order import, MacBERT–MGP inference, hierarchical fault knowledge graph construction, and maintenance prioritization in a single workflow.

The application provides four functional areas:

| Functional area | Capabilities |
| --- | --- |
| Import work orders | Import TXT, Word (DOCX), CSV, Excel, or JSON records, map data fields, and preview work orders |
| Knowledge graph | Explore a radial graph with the equipment at its center and subsystem, component, and failure mode nodes on concentric rings; highlight Top K fault paths and inspect node information |
| Risk and cost | Rank fault paths by occurrence count, cost severity, economic burden, or balanced risk |
| Work-order evidence | Trace fault paths to supporting records, inspect prediction scores and maintenance costs, review incomplete predictions, and export analysis results |

The application supports engineering interpretation by linking maintenance priorities to the recorded fault descriptions and costs that underlie them.

### Application video

**Supplementary Video S1 (in preparation)** covers the engineering application workflow: work-order import, fault knowledge extraction, graph exploration, risk ranking, evidence tracing, and result export.

The [application video page](videos/) provides the video description and upload location.

## Implementation

The repository implements the following workflow:

- Work-order preprocessing and field-tagged input construction.
- MacBERT–MGP training, checkpoint export, inference, and evaluation.
- Auxiliary-task and model-component ablation experiments.
- Four-level fault graph construction with work-order provenance.
- Fault path occurrence and cost aggregation.
- Balanced risk scoring and export of path rankings, graph nodes, and graph edges.

### Installation

Python 3.11 is recommended. The neural model uses **PyTorch 2.5.1** and **Transformers 4.46.3**.

```bash
python -m venv .venv
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -e ".[ml]"
python run.py doctor
```

The training entry point uses a CUDA GPU. Inference supports CPU and CUDA.

### Train

```bash
python run.py train --epochs 8
```

This command trains the complete model with seed 42 and saves the checkpoint with the lowest validation loss. Default settings include a learning rate of `2e-5`, a batch size of `4`, and a maximum sequence length of `512` tokens.

### Predict

```bash
python run.py predict --checkpoint outputs/module_ablation/runs/full/seed_42/checkpoint --records data/work_orders.jsonl --output outputs/macbert_mgp_predictions.jsonl --device cpu
```

### Construct the graph and rank fault paths

```bash
python run.py rank --predictions outputs/macbert_mgp_predictions.jsonl --cohort data/unseen/labels.jsonl --output-dir outputs/generalization_ranking
python run.py rank --predictions outputs/macbert_mgp_predictions.jsonl --canonicalize --screen-materials --output-dir outputs/full_ranking
```

### Run ablations

```bash
python run.py ablate --variants full,no_globalpointer,no_relation,no_auxiliary,no_class_reweighting,no_mean_pooling --seeds 42,52,62 --save-checkpoints
```

The `--seeds` argument sets the random seeds for ablation experiments. Tables 4 and 7 report mean and sample standard deviation across 30 random seeds. Figure 9 presents a single evaluation run for each method.

## Repository structure

```text
Automated-Risk-Analysis/
├── src/engine_fault_nlp/    # Data processing, extraction, graph, and ranking modules
├── scripts/                # Training, inference, evaluation, and ablation entry points
├── data/                   # Research inputs and label resources
├── config/                 # Configuration
├── examples/               # Illustrative input
├── tests/                  # Core and integration checks
├── tools/                  # Repository verification utilities
├── docs/                   # Technical documentation and data provenance
├── videos/                 # Application video and accompanying description
├── run.py                  # Workflow command dispatcher
└── pyproject.toml          # Package metadata and dependencies
```

## License

Source code is provided under the [MIT License](LICENSE). Industrial data and third-party model resources are subject to their respective access conditions and licenses.
