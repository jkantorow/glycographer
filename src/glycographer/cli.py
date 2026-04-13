from __future__ import annotations

import argparse
from pathlib import Path

REQUIRED_DIRS = (
    Path("configs"),
    Path("data") / "input",
    Path("data") / "output",
    Path("scripts"),
)


def init_project(root: Path) -> int:
    for rel_path in REQUIRED_DIRS:
        (root / rel_path).mkdir(parents=True, exist_ok=True)

    example_config = root / "configs" / "pipeline.example.yaml"
    if not example_config.exists():
        example_config.write_text(
            "# Example pipeline configuration\n"
            "run_name: example-run\n"
            "input_dir: data/input\n"
            "output_dir: data/output\n"
            "# Add PyRosetta and modeling options here\n",
            encoding="utf-8",
        )

    print(f"Initialized glycographer project at {root}")
    return 0


def validate_project(root: Path) -> int:
    missing = [str(rel_path) for rel_path in REQUIRED_DIRS if not (root / rel_path).exists()]
    if missing:
        print("Missing required paths:")
        for path in missing:
            print(f" - {path}")
        return 1

    print("Project layout looks good.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="glycographer project utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Create the standard glycographer project layout")
    init_parser.add_argument("path", nargs="?", default=".", help="Project directory")

    validate_parser = subparsers.add_parser("validate", help="Validate the standard glycographer project layout")
    validate_parser.add_argument("path", nargs="?", default=".", help="Project directory")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    root = Path(args.path).resolve()

    if args.command == "init":
        return init_project(root)
    if args.command == "validate":
        return validate_project(root)

    parser.error(f"Unsupported command: {args.command}")
    return 2
