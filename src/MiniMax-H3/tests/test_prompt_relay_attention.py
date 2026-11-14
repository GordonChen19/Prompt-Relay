import unittest

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError as error:
    raise unittest.SkipTest(f"PyTorch is required for attention tests: {error}")

from h3_prompt_relay.attention import (
    AttentionState,
    _segment_costs,
    relay_attention,
    token_segment_ids,
    use_prompt_relay_attention,
)
from h3_prompt_relay.config import PromptRelayConfig, resolve_schedule


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return {
            "input_ids": list(range(len(text))),
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def make_state(mode="hard"):
    config = PromptRelayConfig.from_mapping(
        {
            "global_prompt": "global",
            "local_prompts": ["first", "second"],
            "frame_intervals": [[0, 3], [2, 4]],
            "mode": mode,
            "sigma": 1.0,
        }
    )
    return AttentionState(
        video_rows=torch.tensor([0, 1, 2, 3]),
        frame_ids=torch.tensor([0, 1, 2, 3]),
        other_rows=torch.tensor([4, 5, 6]),
        key_segment_ids=torch.tensor([-1, -1, -1, -1, -1, 0, 1]),
        schedule=resolve_schedule(config, 4),
        sequence_length=7,
    )


class PromptRelayAttentionTests(unittest.TestCase):
    def test_hard_overlap_enables_both_local_prompts(self):
        state = make_state("hard")
        query = torch.zeros(1, 7, 1, 2)
        key = torch.zeros_like(query)
        value = torch.zeros_like(query)
        value[:, 5, :, 0] = 1
        value[:, 6, :, 1] = 1
        output = relay_attention(query, key, value, state, query_chunk_size=2)

        self.assertGreater(output[0, 2, 0, 0].item(), 0.1)
        self.assertGreater(output[0, 2, 0, 1].item(), 0.1)
        self.assertGreater(
            output[0, 0, 0, 0].item(),
            output[0, 0, 0, 1].item() * 100,
        )

    def test_gaussian_cost_matches_wan_formula(self):
        config = PromptRelayConfig.from_mapping(
            {
                "global_prompt": "global",
                "local_prompts": ["event"],
                "frame_intervals": [[0, 8]],
                "mode": "gaussian",
                "sigma": 1.0,
            }
        )
        schedule = resolve_schedule(config, 8)
        frames = torch.arange(8)
        actual = _segment_costs(frames, schedule, device=torch.device("cpu"))
        expected = torch.relu((frames.float() - 4).abs() - 2).square() / 2
        torch.testing.assert_close(actual[:, 0], expected)

    def test_non_video_queries_keep_native_attention(self):
        torch.manual_seed(7)
        state = make_state("gaussian")
        query, key, value = [torch.randn(1, 7, 2, 4) for _ in range(3)]
        actual = relay_attention(query, key, value, state, query_chunk_size=2)
        native = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            key.transpose(1, 2),
            value.transpose(1, 2),
        ).transpose(1, 2)
        torch.testing.assert_close(
            actual[:, state.other_rows], native[:, state.other_rows]
        )

    def test_repeated_local_text_maps_to_distinct_token_rows(self):
        config = PromptRelayConfig.from_mapping(
            {
                "global_prompt": "same",
                "local_prompts": ["same", "same"],
            }
        )
        text = config.full_prompt
        ids = token_segment_ids(CharacterTokenizer(), config, len(text) + 3)
        self.assertTrue(torch.any(ids == 0))
        self.assertTrue(torch.any(ids == 1))
        self.assertTrue(torch.all(ids[:3] == -1))

    def test_attention_processors_are_restored_after_exception(self):
        class FakeAttention:
            def __init__(self, processor):
                self.processor = processor

            def set_processor(self, processor):
                self.processor = processor

        original = object()
        attention = FakeAttention(original)
        transformer = type(
            "Transformer",
            (),
            {"transformer_blocks": [type("Block", (), {"attn": attention})()]},
        )()
        state = make_state("gaussian")
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with use_prompt_relay_attention(transformer, state):
                self.assertIsNot(attention.processor, original)
                raise RuntimeError("intentional")
        self.assertIs(attention.processor, original)


if __name__ == "__main__":
    unittest.main()
