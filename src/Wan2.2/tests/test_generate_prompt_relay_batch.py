import ast
import random
import sys
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


GENERATE_PATH = Path(__file__).parents[1] / "generate.py"
SOURCE = GENERATE_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)


def load_output_helpers():
    names = {
        "_conditioning_config_for_prompt_mode",
        "_default_output_filename",
        "_output_path_for_job",
        "_prompt_relay_output_filename",
    }
    functions = [
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    module = ast.Module(body=functions, type_ignores=[])
    namespace = {"datetime": datetime, "Path": Path, "sys": sys}
    exec(compile(module, str(GENERATE_PATH), "exec"), namespace)
    return namespace


def load_validate_args():
    functions = [
        node
        for node in TREE.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "_prompt_relay_output_filename",
            "_validate_args",
        }
    ]

    class RelayConfig:
        @classmethod
        def from_json(cls, path):
            file_name = "duplicate.mp4" if path.startswith("duplicate") else None
            return SimpleNamespace(
                full_prompt=f"prompt from {path}",
                file_name=file_name,
            )

    config = SimpleNamespace(
        sample_steps=40,
        sample_shift=12.0,
        sample_guide_scale=(3.0, 4.0),
        frame_num=81,
    )
    namespace = {
        "EXAMPLE_PROMPT": {"t2v-A14B": {"prompt": "example"}},
        "PromptRelayConfig": RelayConfig,
        "Path": Path,
        "SUPPORTED_SIZES": {"t2v-A14B": ("832*480",)},
        "WAN_CONFIGS": {"t2v-A14B": config},
        "random": random,
        "sys": sys,
    }
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]),
            str(GENERATE_PATH),
            "exec",
        ),
        namespace,
    )
    return namespace["_validate_args"]


def batch_args(**overrides):
    values = dict(
        ckpt_dir="checkpoint",
        task="t2v-A14B",
        prompt_filepath=["shot_01.json", "shot_02.json"],
        prompt_mode="relay",
        use_prompt_extend=False,
        prompt=None,
        output_dir="outputs",
        save_file=None,
        image=None,
        audio=None,
        enable_tts=False,
        tts_prompt_audio=None,
        tts_prompt_text=None,
        tts_text=None,
        sample_steps=None,
        sample_shift=None,
        sample_guide_scale=None,
        frame_num=None,
        base_seed=123,
        size="832*480",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class PromptRelayBatchCliTests(unittest.TestCase):
    def test_validation_loads_every_prompt_file(self):
        args = batch_args()
        load_validate_args()(args)

        self.assertEqual(len(args.prompt_relay_jobs), 2)
        self.assertEqual(args.prompt_relay_jobs[0][0], "shot_01.json")
        self.assertEqual(args.prompt_relay_jobs[1][0], "shot_02.json")
        self.assertEqual(args.prompt, "prompt from shot_01.json")

    def test_multiple_prompt_files_require_output_directory(self):
        args = batch_args(output_dir=None)
        with self.assertRaisesRegex(ValueError, "output_dir is required"):
            load_validate_args()(args)

    def test_duplicate_configured_filenames_are_rejected(self):
        args = batch_args(
            prompt_filepath=["duplicate_1.json", "duplicate_2.json"]
        )
        with self.assertRaisesRegex(ValueError, "must be unique"):
            load_validate_args()(args)

    def test_prompt_filepath_is_repeatable_and_output_dir_exists(self):
        parser_calls = [
            node
            for node in ast.walk(TREE)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ]
        arguments = {
            node.args[0].value: node
            for node in parser_calls
            if node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        }

        prompt_filepath = arguments["--prompt_filepath"]
        keywords = {keyword.arg: keyword.value for keyword in prompt_filepath.keywords}
        self.assertEqual(keywords["action"].value, "append")
        self.assertIn("--output_dir", arguments)
        self.assertIn("--prompt_mode", arguments)

    def test_concat_mode_disables_temporal_routing(self):
        conditioning_config = load_output_helpers()[
            "_conditioning_config_for_prompt_mode"
        ]
        relay_config = object()

        self.assertIs(
            conditioning_config("relay", relay_config),
            relay_config,
        )
        self.assertIsNone(conditioning_config("concat", relay_config))

    def test_output_paths_share_directory_and_have_stable_indices(self):
        output_path_for_job = load_output_helpers()["_output_path_for_job"]
        args = SimpleNamespace(
            output_dir="outputs/prompt_relay_batch",
            save_file=None,
            task="t2v-A14B",
            size="832*480",
            ulysses_size=4,
        )

        first = output_path_for_job(
            args,
            "first",
            prompt_filepath="prompts/shot_01.json",
            prompt_relay_config=SimpleNamespace(file_name=None),
            job_index=0,
            job_count=3,
        )
        second = output_path_for_job(
            args,
            "second",
            prompt_filepath="another/shot_02.json",
            prompt_relay_config=SimpleNamespace(file_name=None),
            job_index=1,
            job_count=3,
        )

        self.assertEqual(
            first, "outputs/prompt_relay_batch/001_shot_01.mp4"
        )
        self.assertEqual(
            second, "outputs/prompt_relay_batch/002_shot_02.mp4"
        )

    def test_json_filename_is_used_exactly(self):
        output_path_for_job = load_output_helpers()["_output_path_for_job"]
        args = SimpleNamespace(
            output_dir="outputs/prompt_relay_batch",
            save_file=None,
            task="t2v-A14B",
            size="832*480",
            ulysses_size=4,
        )

        output = output_path_for_job(
            args,
            "prompt",
            prompt_filepath="prompts/arbitrary_source_name.json",
            prompt_relay_config=SimpleNamespace(file_name="p001.mp4"),
            job_index=0,
            job_count=3,
        )

        self.assertEqual(output, "outputs/prompt_relay_batch/p001.mp4")

    def test_pipeline_is_created_before_batch_loop(self):
        generate = next(
            node
            for node in TREE.body
            if isinstance(node, ast.FunctionDef) and node.name == "generate"
        )
        pipeline_calls = [
            node
            for node in ast.walk(generate)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "WanT2V"
        ]
        job_loops = [
            node
            for node in ast.walk(generate)
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "enumerate"
        ]

        self.assertEqual(len(pipeline_calls), 1)
        self.assertEqual(len(job_loops), 1)
        self.assertLess(pipeline_calls[0].lineno, job_loops[0].lineno)


if __name__ == "__main__":
    unittest.main()
