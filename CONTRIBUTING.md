# Contributing

Keep code, comments, documentation, commit messages and CLI help in English. Preserve Chinese text wherever it is model input, a domain dictionary, an annotation or a test fixture.

Run `python run.py test` and `python run.py audit` before committing. Do not commit generated metrics, predictions, figures, checkpoints, caches, credentials or environments. Add tests for changes to decoding, cohort selection, path deduplication, cost allocation or ranking.

Do not use held-out evaluation results to select checkpoints or seeds. Document any changes to label provenance, data splits, thresholds, ranking weights or cost policies rather than silently overwriting their meaning.

This repository includes restricted data. Keep it private. Avoid printing identifying values in CI logs or issue discussions.
