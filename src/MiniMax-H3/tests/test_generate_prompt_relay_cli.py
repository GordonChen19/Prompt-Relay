"""CLI tests that do not require PyTorch or model weights."""

import importlib.util
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "generate_prompt_relay.py"
SPEC = importlib.util.spec_from_file_location("generate_prompt_relay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class GeneratePromptRelayCliTests(unittest.TestCase):
    def test_native_first_and_last_images_are_supported(self):
        args = MODULE.parse_args(
            [
                "--prompt",
                "test",
                "--image",
                "first.png",
                "--last_image",
                "last.png",
            ]
        )
        self.assertEqual(args.image, "first.png")
        self.assertEqual(args.last_image, "last.png")
        self.assertIsNone(args.height)
        self.assertIsNone(args.width)

    def test_arbitrary_keyframe_option_is_removed(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            MODULE.parse_args(
                ["--prompt", "test", "--keyframe", "middle.png", "18"]
            )

    def test_explicit_canvas_dimensions_must_be_paired(self):
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            MODULE.parse_args(["--prompt", "test", "--height", "768"])

        args = MODULE.parse_args(
            [
                "--prompt",
                "test",
                "--height",
                "768",
                "--width",
                "1344",
            ]
        )
        self.assertEqual((args.height, args.width), (768, 1344))


if __name__ == "__main__":
    unittest.main()
