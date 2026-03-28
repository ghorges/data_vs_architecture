# data-vs-architecture

Code repository for the analysis pipeline used in the data-versus-architecture materials discovery study.

## Requirements

- Python `3.11`
- `uv`

## Setup

```bash
uv sync
```

## Main Entry Point

```bash
uv run python scripts/run_evidence_pipeline.py --list
```

Run the default evidence pipeline with:

```bash
uv run python scripts/run_evidence_pipeline.py
```

## Repository Contents

- `src/`: shared project code
- `scripts/`: data preparation, analysis, figure, and packaging scripts
- `data/`, `results/`, `external/`: empty placeholders only

## Data And Outputs

- Processed data are distributed separately in the dataset archive.
- Generated results are not stored in this repository upload.
- Manuscript transfer files are not included here.
