import tempfile
import unittest
from pathlib import Path

from glycographer import cli


class CLITests(unittest.TestCase):
    def test_init_creates_expected_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            result = cli.init_project(root)

            self.assertEqual(result, 0)
            self.assertTrue((root / "configs").is_dir())
            self.assertTrue((root / "data" / "input").is_dir())
            self.assertTrue((root / "data" / "output").is_dir())
            self.assertTrue((root / "scripts").is_dir())
            self.assertTrue((root / "configs" / "pipeline.example.yaml").is_file())

    def test_validate_fails_when_layout_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = cli.validate_project(Path(tmp_dir))
            self.assertEqual(result, 1)

    def test_validate_passes_after_init(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            self.assertEqual(cli.init_project(root), 0)
            result = cli.validate_project(root)
            self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
