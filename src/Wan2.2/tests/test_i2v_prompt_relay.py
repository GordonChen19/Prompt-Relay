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
        self.assertIn("prompt_relay_segments=main_relay", source)
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

    def test_intermediate_images_use_in_place_pixel_slots(self):
        self.assertIn(
            "model_latent_frames = output_latent_frames", SOURCE
        )
        self.assertIn(
            "pixel_index = temporal_stride * output_index", SOURCE
        )
        self.assertIn("joint_video[:, pixel_index] = resized[position]", SOURCE)
        self.assertIn("joint_encoded = self.vae.encode([joint_video])[0]", SOURCE)

    def test_intermediate_images_enable_only_final_mask_channel(self):
        self.assertIn("joint_mask[3, output_index].fill_(1)", SOURCE)
        self.assertNotIn("joint_mask[:, output_index].fill_(1)", SOURCE)

    def test_in_place_conditioning_has_no_temporary_anchor_denoising(self):
        self.assertNotIn("VisualRelayLayout", SOURCE)
        self.assertNotIn("visual_relay_anchor_indices", SOURCE)
        self.assertNotIn("b0_latents", SOURCE)
        self.assertNotIn("VISUAL_RELAY_HANDOFF_STEP", SOURCE)
        self.assertNotIn("single_frame_indices=", SOURCE)

    def test_i2v_rejects_a_non_aligned_requested_frame_count(self):
        self.assertIn(
            "(frame_num - 1) % temporal_stride != 0", SOURCE
        )

    def test_in_place_conditioning_uses_standard_attention(self):
        self.assertNotIn("visual_relay_anchor_rope_offset", SOURCE)
        self.assertNotIn("visual_relay_shifted_attention", SOURCE)

    def test_old_temporary_anchor_attention_is_removed(self):
        for source in (MODEL_SOURCE, ATTENTION_SOURCE, SP_SOURCE, ULYSSES_SOURCE):
            self.assertNotIn("visual_relay_anchor_indices", source)
            self.assertNotIn("visual_relay_shifted_attention", source)

    def test_image_conditioning_does_not_add_self_attention_masking(self):
        for source in (MODEL_SOURCE, ATTENTION_SOURCE, SP_SOURCE, ULYSSES_SOURCE):
            self.assertNotIn("image_conditioned_frame_attention", source)
            self.assertNotIn("image_condition_indices", source)

    def test_distributed_prompt_relay_routing_is_preserved(self):
        self.assertIn("prompt_relay_segments=None", SP_SOURCE)
        self.assertIn("prompt_relay_query_chunk_size=128", SP_SOURCE)
        self.assertIn("prompt_relay_query_token_offset", SP_SOURCE)
        self.assertIn("build_temporal_penalty", SP_SOURCE)
        self.assertIn("prompt_relay_segments=prompt_relay_segments", SP_SOURCE)
        self.assertIn("prompt_relay_attention", MODEL_SOURCE)


if __name__ == "__main__":
    unittest.main()
