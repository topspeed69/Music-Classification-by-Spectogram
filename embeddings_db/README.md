# Embeddings database

This directory contains precomputed embedding files used by the project for similarity search and recommendations.

These files are deliberately not tracked by git to avoid committing large binary or generated files (e.g. .npy, .csv, .json). If you need to regenerate them, see the project's README or the `embeddings/` source code.

Files that should be ignored by git (examples):

- embeddings.npy
- embeddings.csv
- embeddings_database.json
- metadata.csv
- metadata.json

If you want to re-create these files:

1. Run the embedding extraction scripts located in `embeddings/extract_embeddings.py`.
2. Confirm any required datasets are available under `AudioToSpectogram/fma_small_dataset` or other data directories.

Keep this file committed so git preserves the `embeddings_db/` directory even when the large data files are ignored.
