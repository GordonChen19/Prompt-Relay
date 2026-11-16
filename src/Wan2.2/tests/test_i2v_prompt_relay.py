import ast
import unittest
from pathlib import Path


I2V_PATH = Path(__file__).parents[1] / "wan" / "image2video.py"
SOURCE = I2V_PATH.read_text(encoding="utf-8")
TREE = ast.parse(SOURCE)
WAN_ROOT = I2V_PATH.parent
MODEL_SOURCE = (WAN_ROOT / "modules" / "model.py").read_text(encoding="utf-8")
ATTENTION_SOURCE = (WAN_ROOT / "modules" / "attention.py").read_text(
    encoding="utf-8"
)
SP_SOURCE = (WAN_ROOT / "distributed" / "sequence_parallel.py").read_text(
    encoding="utf-8"
)
ULYSSES_SOURCE = (WAN_ROOT / "distributed" / "ulysses.py").read_text(
    encoding="utf-8"
)


class I2VPromptRelayStructureTests(unittest.TestCase):
    def test_conditional_branch_keeps_image_input_and_adds_relay(self):
        wan_i2v = next(
            node
            for node in TREE.body
            if isinstance(node, ast.ClassDef) and node.name == "WanI2V"
        )
        generate = next(
            node
            for node in wan_i2v.body
            if isinstance(node, ast.FunctionDef) and node.name == "generate"
        )
        dictionaries = {}
        for node in ast.walk(generate):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Dict):
                dictionaries[target.id] = {
                    key.value
                    for key in node.value.keys
                    if isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                }

        self.assertIn("y", dictionaries["common"])
        source = ast.unparse(generate)
        self.assertIn("prompt_relay_segments=current_relay", source)
        self.assertIn("context=[contexts[0]]", source)
        self.assertIn("context=[null_contexts[0]]", source)

    def test_generate_accepts_relay_config_and_query_chunk_size(self):
        wan_i2v = next(
            node
            for node in TREE.body
            if isinstance(node, ast.ClassDef) and node.name == "WanI2V"
        )
        generate = next(
            node
            for node in wan_i2v.body
            if isinstance(node, ast.FunctionDef) and node.name == "generate"
        )
        argument_names = [argument.arg for argument in generate.args.args]

        self.assertIn("prompt_relay_config", argument_names)
        self.assertIn("prompt_filepath", argument_names)
        self.assertIn("query_chunk_size", argument_names)

    def test_visual_relay_handoff_flushes_solver_history(self):
        self.assertIn(
            "_flush_multistep_solver_history(sample_scheduler)", SOURCE
        )
        self.assertIn(
            "b0_schedulers = [copy.deepcopy(sample_scheduler)", SOURCE
        )

    def test_discarded_anchor_uses_neighbor_predictions(self):
        handoff = SOURCE.split(
            "if visual_relay and not guidance_active:", 1
        )[1].split("next_latent = sample_scheduler.step", 1)[0]
        self.assertIn("expanded[:, left:left + 1]", handoff)
        self.assertIn("expanded[:, right:right + 1]", handoff)
        self.assertNotIn("latent[:, left:left + 1]", handoff)

    def test_i2v_rejects_a_non_aligned_requested_frame_count(self):
        self.assertIn(
            "(frame_num - 1) % temporal_stride != 0", SOURCE
        )

    def test_visual_relay_uses_shifted_anchor_rope(self):
        self.assertIn("'visual_relay_anchor_rope_offset': 0.5", SOURCE)
        self.assertIn("visual_relay_shifted_attention", MODEL_SOURCE)
        self.assertIn("anchor_rope_offset=visual_relay_anchor_rope_offset", MODEL_SOURCE)
        self.assertIn("shifted_rope_positions", ATTENTION_SOURCE)

    def test_shifted_anchor_attention_is_available_with_ulysses(self):
        self.assertIn("visual_relay_anchor_indices", SP_SOURCE)
        self.assertIn("visual_relay_anchor_indices", ULYSSES_SOURCE)
        self.assertIn("visual_relay_shifted_attention", ULYSSES_SOURCE)


if __name__ == "__main__":
    unittest.main()
