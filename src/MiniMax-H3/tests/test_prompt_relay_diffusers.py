"""Integration tests for the pinned Diffusers H3 API; no weights required."""

import unittest

try:
    import torch
    from diffusers import MiniMaxH3Transformer3DModel
    from diffusers.modular_pipelines.minimax_h3.before_denoise import (
        MiniMaxH3PrepareLayoutStep,
    )
    from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (
        MiniMaxH3ModularPipeline,
    )
except (ImportError, ModuleNotFoundError) as error:
    raise unittest.SkipTest(
        f"Install the pinned MiniMax H3 Diffusers dependencies: {error}"
    )

from h3_prompt_relay import PromptRelayConfig, enable_prompt_relay
from h3_prompt_relay.attention import (
    prepare_attention_state,
    use_prompt_relay_attention,
)


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def tiny_model():
    return MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=16,
        hidden_size=24,
        num_layers=2,
        num_refiner_layers=1,
        ffn_dim=32,
        in_channels=24,
        audio_in_channels=32,
        patch_size=(1, 2, 2),
        text_dim=8,
        freq_dim=8,
        time_embed_hidden_dim=24,
        time_embed_dim=16,
        rope_freq_dim=2,
    ).eval()


class DiffusersIntegrationTests(unittest.TestCase):
    def test_native_first_and_last_keyframes_use_official_rope_times(self):
        layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags=torch.tensor([1, 1, 1]),
            num_latent_frames=7,
            latent_height=2,
            latent_width=4,
            num_audio_latents=3,
            patch_size=(1, 2, 2),
            audio_channels=2,
            audio_tag=2,
            video_tag=0,
            keyframe_anchors=("first", "last"),
        )
        positions, _, video, _, _, condition_rows, _ = layout
        rows_per_frame = 2
        target_rows = video[condition_rows:]
        first_condition_row = 3
        first_target_row = target_rows[0]
        self.assertEqual(
            positions[first_condition_row, 0].item(),
            positions[first_target_row, 0].item(),
        )

        # The pinned official implementation places the last anchor at the
        # reference endpoint: text origin + sum(1,4,4,4,4,1,4)*5/3 - 5/3.
        last_condition_row = 3 + rows_per_frame
        expected_last_time = 3 + (sum((1, 4, 4, 4, 4, 1, 4)) - 1) * 5 / 3
        self.assertAlmostEqual(
            positions[last_condition_row, 0].item(), expected_last_time
        )

    def test_installs_idempotently_for_all_h3_workflows(self):
        for workflow in (None, "t2va", "fl2va", "ref2va"):
            with self.subTest(workflow=workflow):
                pipe = MiniMaxH3ModularPipeline(workflow=workflow)
                self.assertIs(enable_prompt_relay(pipe), pipe)
                self.assertIs(enable_prompt_relay(pipe), pipe)
                names = {parameter.name for parameter in pipe.blocks.inputs}
                self.assertIn("prompt_relay_config", names)
                self.assertIn("query_chunk_size", names)
                self.assertNotIn("indexed_keyframes", names)

    def test_real_transformer_routes_and_restores_attention(self):
        torch.manual_seed(11)
        model = tiny_model()
        layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            text_token_tags=torch.tensor([0, 0, 1, 1, 1, 1, 1]),
            num_latent_frames=7,
            latent_height=2,
            latent_width=4,
            num_audio_latents=9,
            patch_size=(1, 2, 2),
            audio_channels=2,
            audio_tag=2,
            video_tag=0,
            keyframe_anchors=(),
        )
        positions, tags, video, audio, text, num_condition_rows, _ = layout
        config = PromptRelayConfig.from_mapping(
            {
                "global_prompt": "g",
                "local_prompts": ["a", "b"],
                "frame_intervals": [[0, 5], [3, 7]],
            }
        )
        state = prepare_attention_state(
            position_ids=positions,
            video_indices=video,
            text_indices=text,
            num_condition_video_rows=num_condition_rows,
            num_latent_frames=7,
            tokenizer=CharacterTokenizer(),
            config=config,
        )
        inputs = {
            "hidden_states": torch.randn(1, len(video), 96),
            "audio_hidden_states": torch.randn(1, len(audio), 32),
            "encoder_hidden_states": torch.randn(1, len(text), 8),
            "timestep": torch.tensor([0.6]),
            "timestep_indices": torch.zeros(len(tags), dtype=torch.long),
            "token_tags": tags,
            "position_ids": positions,
            "video_indices": video,
            "audio_indices": audio,
            "text_indices": text,
        }
        original = [block.attn.processor for block in model.transformer_blocks]
        with torch.no_grad():
            baseline = model(**inputs)
            with use_prompt_relay_attention(model, state, query_chunk_size=5):
                routed = model(**inputs)
            restored = model(**inputs)

        self.assertFalse(torch.allclose(routed.sample, baseline.sample))
        self.assertTrue(torch.isfinite(routed.sample).all())
        torch.testing.assert_close(restored.sample, baseline.sample, atol=0, rtol=0)
        self.assertEqual(
            original,
            [block.attn.processor for block in model.transformer_blocks],
        )


if __name__ == "__main__":
    unittest.main()
