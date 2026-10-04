# Research inputs (restricted)

These files contain **non-anonymized, private research data**. Repository access must remain private. The source-code license does not grant rights to share these records. See [data access conditions](../DATA_AVAILABILITY.md).

| File | Records | Role |
| --- | ---: | --- |
| `work_orders.jsonl` | 2,638 | Aggregated document-level orders, including maintenance text and settlement costs |
| `seed/all.jsonl` | 1,000 | Full preannotated model-development seed |
| `seed/train.jsonl` | 800 | Parameter optimization |
| `seed/validation.jsonl` | 100 | Checkpoint selection |
| `seed/test.jsonl` | 100 | Internal held-out evaluation |
| `seed/expert_review_pending.csv` | 1,024 rows | Path-level review worksheet for the seed, not 1,024 independent orders |
| `unseen/labels.jsonl` | 727 | Reference labels for the unseen evaluation cohort |
| `labels/all_work_orders_silver.jsonl` | 2,638 | Historical machine-generated reference annotations used by baseline workflows |
| `knowledge_base.json` | — | Domain vocabulary, label identifiers, and matching resources |

The seed and unseen cohorts are subsets of `work_orders.jsonl`, not additional independent datasets. Training, validation, and test document IDs are disjoint. The unseen IDs are disjoint from the entire seed. This does not establish vehicle-level, temporal, or text-template independence.

The seed files retain their `PREANNOTATED_NEEDS_EXPERT_REVIEW` status. Expert-review columns in the original worksheet are unfilled. Neither filenames nor repository packaging turn these annotations into expert-adjudicated gold labels. The unseen references are silver labels.

Original Chinese text, identifiers, annotations, entity offsets, and numeric values are preserved byte-for-byte. The [manifest](../docs/data_manifest.json) records source-relative paths, byte sizes, record counts, and SHA-256 hashes. Run `python run.py audit` to verify integrity without printing private record contents.

The 2,439-record analysis population in the manuscript is a downstream model-extraction and screening result, not the size of the raw document input. Published predictions and screening outputs are deliberately not included in this source/data-only release. Re-running a different checkpoint must not be described as an exact reproduction of those historical counts.

Work-order identifiers, vehicle/engine identifiers, locations, dates, expenses, and potentially contact details remain in the inputs. Do not send them to external teacher APIs without separate data-owner authorization. Do not publish this directory or the release archive.
