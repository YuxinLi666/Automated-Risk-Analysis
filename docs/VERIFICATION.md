# Packaging verification

Verified locally on 5 October 2026. This is a source/data packaging check, not a rerun of the manuscript experiments.

| Check | Outcome |
| --- | --- |
| Core unittest suite | `OK (skipped=2)`, with 61 tests reported by unittest |
| Four-layer ranking tests | Passed support deduplication, equal cost allocation, adjacent-layer structure, threshold handling, tied empirical ranks, empty inputs, missing/zero costs, and plot-free export checks |
| Synthetic baseline demo | Passed as part of the test suite |
| Prediction and ranking command-line help | Passed without model download |
| Data integrity | Nine preserved inputs matched their original SHA-256 hashes and byte sizes |
| Split checks | 800/100/100 disjoint document splits, 727 unseen IDs disjoint from the seed, all IDs join the 2,638-order collection |
| Release scan | No plotting-library imports, binary results, or common credential/private-key patterns detected among the allowlisted release files |

## Unverified neural runtime

The existing local Python environment raises Windows error 1114 while loading PyTorch's `c10.dll`. The optional checkpoint round-trip test class was therefore skipped with an explicit runtime-load reason. Its tiny offline test covers all six architecture variants when the optional neural dependencies load correctly. Successful core checks must not be interpreted as successful neural retraining or checkpoint validation on this machine.

The other skip is the optional contract test for a historical private Excel workbook that is not included in this release. The aggregated JSONL research inputs are included instead. The original Python environment and historical model/result files were not changed.

## Re-run

```bash
python run.py test
python run.py audit
python run.py predict --help
python run.py rank --help
```

Use a working PyTorch/Transformers installation to execute the optional neural checks. No paper performance numbers are asserted by these packaging tests. A limited token-pattern scan does not constitute a security certification, and the research data remain non-anonymized and restricted.
