import importlib.util
import json
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path


try:
    import torch
except ModuleNotFoundError:
    torch = types.ModuleType("torch")
    sys.modules["torch"] = torch
    TORCH_AVAILABLE = False
else:
    TORCH_AVAILABLE = True


MODULE_PATH = Path(__file__).parents[1] / "wan" / "prompt_relay.py"
SPEC = importlib.util.spec_from_file_location("prompt_relay_for_tests", MODULE_PATH)
PROMPT_RELAY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PROMPT_RELAY
SPEC.loader.exec_module(PROMPT_RELAY)

PromptRelayConfig = PROMPT_RELAY.PromptRelayConfig
PromptRelaySegment = PROMPT_RELAY.PromptRelaySegment
build_temporal_penalty = PROMPT_RELAY.build_temporal_penalty
build_prompt_relay_segments = PROMPT_RELAY.build_prompt_relay_segments


def load_model_module_for_tests():
    """Load wan.modules.model without requiring an installed diffusers package."""

    package_name = "wan_model_for_tests"
    package = types.ModuleType(package_name)
    package.__path__ = []
    modules_package = types.ModuleType(f"{package_name}.modules")
    modules_package.__path__ = []

    configuration_utils = types.ModuleType("diffusers.configuration_utils")
    modeling_utils = types.ModuleType("diffusers.models.modeling_utils")
    diffusers = types.ModuleType("diffusers")
    diffusers_models = types.ModuleType("diffusers.models")

    class ConfigMixin:
        pass

    class ModelMixin(torch.nn.Module):
        pass

    def register_to_config(function):
        return function

    configuration_utils.ConfigMixin = ConfigMixin
    configuration_utils.register_to_config = register_to_config
    modeling_utils.ModelMixin = ModelMixin

    attention = types.ModuleType(f"{package_name}.modules.attention")
    attention.flash_attention = lambda *args, **kwargs: None

    module_path = Path(__file__).parents[1] / "wan" / "modules" / "model.py"
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.modules.model", module_path
    )
    module = importlib.util.module_from_spec(spec)
    replacements = {
        package_name: package,
        f"{package_name}.modules": modules_package,
        f"{package_name}.prompt_relay": PROMPT_RELAY,
        f"{package_name}.modules.attention": attention,
        "diffusers": diffusers,
        "diffusers.configuration_utils": configuration_utils,
        "diffusers.models": diffusers_models,
        "diffusers.models.modeling_utils": modeling_utils,
        spec.name: module,
    }
    with mock.patch.dict(sys.modules, replacements):
        spec.loader.exec_module(module)
    return module


class PromptRelayConfigTests(unittest.TestCase):

    def test_visual_relay_images_and_indices(self):
        config = PromptRelayConfig.from_mapping({
            "global_prompt": "global",
            "local_prompts": ["first", "second"],
            "images": ["start.png", "middle.png", "end.png"],
            "image_index": [0, 10, 16],
        })
        self.assertEqual(config.images, (
            "start.png", "middle.png", "end.png"))
        self.assertEqual(config.image_indices, (0, 10, 16))
        self.assertTrue(config.visual_relay_enabled)

    def test_visual_relay_rejects_mismatched_arrays(self):
        with self.assertRaisesRegex(ValueError, "equal lengths"):
            PromptRelayConfig.from_mapping({
                "global_prompt": "global",
                "local_prompts": ["local"],
                "images": ["start.png", "middle.png"],
                "image_index": [0],
            })

    def test_visual_relay_rejects_unsorted_indices(self):
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            PromptRelayConfig.from_mapping({
                "global_prompt": "global",
                "local_prompts": ["local"],
                "images": ["a.png", "b.png", "c.png"],
                "image_index": [0, 10, 5],
            })
    def test_shared_segment_builder_maps_local_tokens_and_frames(self):
        class TokenIds(list):
            def tolist(self):
                return list(self)

        class CharacterTokenizer:
            def __call__(self, text, **kwargs):
                return [TokenIds(ord(character) for character in text)]

        config = PromptRelayConfig.from_mapping(
            {
                "global_prompt": "g",
                "local_prompts": ["a", "b"],
                "frame_intervals": [[0, 3], [2, 5]],
            }
        )
        segments, full_prompt = build_prompt_relay_segments(
            CharacterTokenizer(),
            config,
            latent_frames=5,
            tokens_per_frame=12,
            text_len=32,
        )

        self.assertEqual(full_prompt, "g a b")
        self.assertEqual(
            [(segment.frame_start, segment.frame_end) for segment in segments],
            [(0, 3), (2, 5)],
        )
        self.assertEqual(
            [segment.tokens_per_frame for segment in segments],
            [12, 12],
        )
        self.assertEqual(segments[0].token_indices, (1, 2))
        self.assertEqual(segments[1].token_indices, (3, 4))

    def test_loads_json_and_normalizes_prompts(self):
        payload = {
            "id": " p001 ",
            "file_name": "p001.mp4",
            "global_prompt": " Global prompt ",
            "local_prompts": [" First action ", "Second action"],
            "segment_lengths": [2, 3],
            "mode": "GAUSSIAN",
            "sigma": 0.5,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            config = PromptRelayConfig.from_json(path)

        self.assertEqual(config.global_prompt, "Global prompt")
        self.assertEqual(config.id, "p001")
        self.assertEqual(config.file_name, "p001.mp4")
        self.assertEqual(config.local_prompts, ("First action", "Second action"))
        self.assertEqual(
            config.full_prompt, "Global prompt First action Second action"
        )
        self.assertEqual(config.mode, "gaussian")
        self.assertEqual(config.frame_intervals(5), ((0, 2), (2, 5)))

    def test_rejects_file_name_containing_a_directory(self):
        with self.assertRaisesRegex(ValueError, "not a directory path"):
            PromptRelayConfig.from_mapping(
                {
                    "global_prompt": "Global",
                    "local_prompts": ["one"],
                    "file_name": "nested/p001.mp4",
                }
            )

    def test_rejects_non_mp4_file_name(self):
        with self.assertRaisesRegex(ValueError, "must end with .mp4"):
            PromptRelayConfig.from_mapping(
                {
                    "global_prompt": "Global",
                    "local_prompts": ["one"],
                    "file_name": "p001.mov",
                }
            )

    def test_equal_intervals_match_existing_ceil_behavior(self):
        config = PromptRelayConfig.from_mapping(
            {
                "global_prompt": "Global",
                "local_prompts": ["one", "two", "three", "four"],
            }
        )
        self.assertEqual(
            config.frame_intervals(21),
            ((0, 6), (6, 12), (12, 18), (18, 21)),
        )
        self.assertEqual(config.mode, "gaussian")

    def test_rejects_mismatched_segment_lengths(self):
        with self.assertRaisesRegex(ValueError, "one entry per local prompt"):
            PromptRelayConfig.from_mapping(
                {
                    "global_prompt": "Global",
                    "local_prompts": ["one", "two"],
                    "segment_lengths": [5],
                }
            )

    def test_rejects_segment_lengths_that_do_not_cover_timeline(self):
        config = PromptRelayConfig.from_mapping(
            {
                "global_prompt": "Global",
                "local_prompts": ["one", "two"],
                "segment_lengths": [4, 4],
            }
        )
        with self.assertRaisesRegex(ValueError, "must sum"):
            config.frame_intervals(9)

    def test_rejects_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "mode must be one of"):
            PromptRelayConfig.from_mapping(
                {
                    "global_prompt": "Global",
                    "local_prompts": ["one"],
                    "mode": "sharp",
                }
            )

    def test_explicit_frame_intervals_can_overlap(self):
        config = PromptRelayConfig.from_mapping(
            {
                "global_prompt": "Global",
                "local_prompts": ["one", "two", "three"],
                "frame_intervals": [[0, 7], [3, 8], [8, 21]],
            }
        )
        self.assertEqual(
            config.frame_intervals(21),
            ((0, 7), (3, 8), (8, 21)),
        )

    def test_rejects_intervals_beyond_latent_timeline(self):
        config = PromptRelayConfig.from_mapping(
            {
                "global_prompt": "Global",
                "local_prompts": ["one"],
                "frame_intervals": [[0, 22]],
            }
        )
        with self.assertRaisesRegex(ValueError, "cannot extend beyond"):
            config.frame_intervals(21)

    def test_rejects_lengths_and_explicit_intervals_together(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            PromptRelayConfig.from_mapping(
                {
                    "global_prompt": "Global",
                    "local_prompts": ["one", "two"],
                    "segment_lengths": [4, 4],
                    "frame_intervals": [[0, 4], [3, 8]],
                }
            )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class TemporalPenaltyTests(unittest.TestCase):
    def test_attention_rejects_nonpositive_chunk_size(self):
        model = load_model_module_for_tests()
        q = torch.randn(1, 2, 1, 4)
        k = torch.randn(1, 2, 1, 4)
        v = torch.randn(1, 2, 1, 4)
        segment = PromptRelaySegment(
            frame_start=0,
            frame_end=1,
            token_indices=(1,),
            tokens_per_frame=1,
            mode="hard",
            sigma=1.0,
        )

        with self.assertRaisesRegex(ValueError, "positive integer"):
            model.prompt_relay_attention(
                q,
                k,
                v,
                (segment,),
                chunk_size=0,
            )

    def test_precomputed_penalty_preserves_attention_result(self):
        model = load_model_module_for_tests()
        torch.manual_seed(7)
        q = torch.randn(1, 6, 2, 4)
        k = torch.randn(1, 5, 2, 4)
        v = torch.randn(1, 5, 2, 4)
        segment = PromptRelaySegment(
            frame_start=1,
            frame_end=3,
            token_indices=(2, 3),
            tokens_per_frame=2,
            mode="gaussian",
            sigma=0.932,
        )
        segments = (segment,)
        query_token_offset = 4
        penalty = build_temporal_penalty(
            segments,
            q.size(1),
            k.size(1),
            device=q.device,
            query_token_offset=query_token_offset,
        )

        original = model.prompt_relay_attention(
            q,
            k,
            v,
            segments,
            query_token_offset=query_token_offset,
            chunk_size=2,
        )
        precomputed = model.prompt_relay_attention(
            q,
            k,
            v,
            segments,
            query_token_offset=query_token_offset,
            chunk_size=3,
            temporal_penalty=penalty,
        )

        self.assertTrue(torch.allclose(original, precomputed, atol=1e-6))

    def test_overlapping_intervals_enable_both_local_prompts(self):
        segments = (
            PromptRelaySegment(
                frame_start=0,
                frame_end=4,
                token_indices=(1,),
                tokens_per_frame=1,
                mode="hard",
                sigma=1.0,
            ),
            PromptRelaySegment(
                frame_start=2,
                frame_end=6,
                token_indices=(2,),
                tokens_per_frame=1,
                mode="hard",
                sigma=1.0,
            ),
        )
        penalty = build_temporal_penalty(
            segments, 6, 4, device=torch.device("cpu")
        )
        self.assertEqual(penalty[3, 1].item(), 0.0)
        self.assertEqual(penalty[3, 2].item(), 0.0)
        self.assertEqual(penalty[1, 2].item(), 1e9)
        self.assertEqual(penalty[5, 1].item(), 1e9)

    def test_hard_mode_masks_only_local_tokens_outside_interval(self):
        segment = PromptRelaySegment(
            frame_start=0,
            frame_end=4,
            token_indices=(1,),
            tokens_per_frame=1,
            mode="hard",
            sigma=1.0,
        )
        penalty = build_temporal_penalty(
            (segment,), 6, 3, device=torch.device("cpu")
        )
        self.assertTrue(torch.equal(penalty[:, 0], torch.zeros(6)))
        self.assertEqual(penalty[3, 1].item(), 0.0)
        self.assertEqual(penalty[4, 1].item(), 1e9)
        self.assertEqual(penalty[5, 1].item(), 1e9)

    def test_query_offset_preserves_global_frame_coordinates(self):
        segment = PromptRelaySegment(
            frame_start=0,
            frame_end=4,
            token_indices=(1,),
            tokens_per_frame=2,
            mode="hard",
            sigma=1.0,
        )
        penalty = build_temporal_penalty(
            (segment,),
            2,
            3,
            device=torch.device("cpu"),
            query_token_offset=10,
        )
        self.assertTrue(torch.equal(penalty[:, 1], torch.full((2,), 1e9)))

    def test_gaussian_mode_is_soft_outside_inner_window(self):
        segment = PromptRelaySegment(
            frame_start=0,
            frame_end=8,
            token_indices=(1,),
            tokens_per_frame=1,
            mode="gaussian",
            sigma=1.0,
        )
        penalty = build_temporal_penalty(
            (segment,), 8, 3, device=torch.device("cpu")
        )
        self.assertEqual(penalty[4, 1].item(), 0.0)
        self.assertGreater(penalty[0, 1].item(), 0.0)
        self.assertLess(penalty[0, 1].item(), 1e9)


if __name__ == "__main__":
    unittest.main()
