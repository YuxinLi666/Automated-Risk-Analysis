# Reproducibility and implementation boundaries

## Five-stage workflow

1. Aggregate and clean maintenance rows at the work-order level. The package's `prepare` command accepts the original XLSX schema. A document-level collection is included in `data/work_orders.jsonl`.
2. Serialize available text fields with source tags. Costs and administrative identifiers are not encoder features. Chinese text is preserved.
3. Train and apply the cause-free MacBERT–MGP model. Three document heads predict subsystem, component and failure mode. Span and relation branches supply auxiliary training supervision. No rule or TF–IDF predictions are fused into this neural decoding path.
4. Combine accepted nodes with the common root and merge identical canonical nodes within each layer. Retain the observed work-order/path association rather than enumerating arbitrary graph routes.
5. Allocate maintenance costs across distinct accepted paths and compute occurrence, cost severity, economic burden and Balanced Risk. Export data rather than plots.

## Controlled model experiment

The main experiment retains the fixed 800 training / 100 validation / 100 internal-test / 727 unseen document protocol. The script intentionally rejects different counts. For a new study, revise this check explicitly and document the new split protocol rather than disguising another dataset as this experiment.

Defaults: MacBERT base, 512 tokens, batch size 4, AdamW learning rate 2e-5, weight decay 0.01, gradient norm limit 1.0, 8 epochs, acceptance threshold 0.5, and seeds 42/52/62. The seed-42 full-model convenience command is `python run.py train`. A different number of epochs can be supplied explicitly.

Checkpoint selection uses minimum validation loss. Test and unseen labels are used for reporting after selection. The dataset contains preannotations and silver reference labels, not an independently adjudicated expert test set. Document-level non-overlap alone does not establish vehicle-level or temporal independence.

The class vocabulary and class weights are constructed from training annotations. Cost metadata are not supplied to the encoder. The model produces at most one accepted node per layer. Missing accepted nodes prevent a complete path, rather than introducing a guessed label.

## Release changes

- Input defaults point to `data/`, not old result folders.
- Plotting and rendering code has been excluded, including automatic Graphviz rendering.
- Four-layer graph aggregation and Balanced Risk computation have separate non-plotting entry points.
- The module-ablation script can now serialize its selected model using an explicit checkpoint option. A matching predictor reconstructs the same model. This is a packaging/inference capability, not a new model variant or training objective.
- Model weights, tokenizers downloaded during experiments and previously computed research results are not included in the release.

The retained source includes legacy cause extraction, hybrid baselines, weak labeling, uncertainty-aware graphs and TOPSIS. They remain available for historical code compatibility but are not the default current-model workflow. The monitored Transformer script is a legacy four-head model and should not be mixed with the cause-free model's checkpoints.

## Cost and cohort policy

The non-plotting ranking branch retains the historical positive-cost-only severity estimate to avoid silently changing computations during packaging. This differs from a formulation using all nonnegative cost observations when zero-cost records are present. Report the implemented policy explicitly. Burden and allocation diagnostics are exported separately.

The complete input collection contains 2,638 orders, whereas the manuscript's 2,439-record statistical population was obtained by a particular extraction and screening run. Since historical predictions are excluded, the release does not claim to reconstruct that exact population without rerunning the relevant model and screening policy. The ranking entry point does not hard-code the published path counts or top-ranked results.

## Safe verification

```bash
python run.py test
python run.py audit
python run.py doctor
python run.py train --help
python run.py predict --help
python run.py rank --help
```

Core tests and the synthetic demo do not require model downloads. Neural checkpoint round-trip tests, when the optional ML stack is installed, use a tiny local model rather than the full research training run. A test pass does not reproduce the reported paper metrics.
