# Quickstart

This repository now follows a standard Python package layout so it can be used as the public interface for the private `glycomaps` pipeline code.

## 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

## 2) Install glycographer in editable mode

```bash
pip install -e .
```

## 3) Initialize a project workspace

```bash
glycographer init /path/to/my-run
```

This creates:

- `configs/`
- `data/input/`
- `data/output/`
- `scripts/`
- `configs/pipeline.example.yaml`

## 4) Validate workspace structure

```bash
glycographer validate /path/to/my-run
```

## 5) Integrate private `glycomaps` modules

Move or mirror modeling modules from `glycomaps` into `src/glycographer/` by feature area (for example `preprocess.py`, `docking.py`, `analysis.py`) and wire them into the CLI workflow.
