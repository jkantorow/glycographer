# glycographer

Simulation tools and pipeline for generating receptor-glycoligand binding affinity maps powered by PyRosetta.

## Goal

`glycographer` is the public, polished interface for the private `glycomaps` molecular modeling pipeline. This repository now uses a standard Python layout and includes onboarding docs so new users can install, initialize, and validate a working project structure quickly.

## Repository layout

```text
glycographer/
├── docs/                    # User/developer documentation
├── examples/                # Example datasets and tutorials
├── scripts/                 # Utility scripts
├── src/glycographer/        # Python package source
├── tests/                   # Automated tests
├── pyproject.toml           # Packaging and CLI entry points
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

## Quick usage

```bash
glycographer init ./my-run
glycographer validate ./my-run
```

For a full walkthrough, see [`docs/quickstart.md`](docs/quickstart.md).

## Next migration step from `glycomaps`

As private modules are brought over, place them under `src/glycographer/` by pipeline stage and expose stable commands through the CLI. This preserves a clean user-facing interface while allowing internal refactors.
