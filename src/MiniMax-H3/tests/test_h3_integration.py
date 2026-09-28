"""Tests use the real upstream H3 model/layout/schedulers, without weights."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from diffusers import MiniMaxH3Transformer3DModel, MiniMaxH3Scheduler
from diffusers.modular_pipelines.modular_pipeline import ModularPipelineBlocks, PipelineState
from diffusers.modular_pipelines.modular_pipeline_utils import InputParam, OutputParam
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep, MiniMaxH3Ref2VAPrepareLayoutStep,
)
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import MiniMaxH3Blocks
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import MiniMaxH3ModularPipeline

from h3_prompt_relay import enable_prompt_relay
from h3_prompt_relay.attention import SlidingWindowConfig, prepare_attention_state, use_relay_attention
from h3_prompt_relay.pipeline import RelayDenoiseStep, RelayRef2VADenoiseStep
from h3_prompt_relay.schedule import build_prompt_text, token_segment_ids
from test_schedule_attention import CharacterTokenizer


def tiny_model():
    return MiniMaxH3Transformer3DModel(
        num_attention_heads=2, attention_head_dim=16, hidden_size=24,
        num_layers=2, num_refiner_layers=1, ffn_dim=32, in_channels=24,
        audio_in_channels=32, patch_size=(1, 2, 2), text_dim=8,
        freq_dim=8, time_embed_hidden_dim=24, time_embed_dim=16, rope_freq_dim=2,
    ).eval()


class UpstreamModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = tiny_model()
        self.config = dict(global_prompt="g", local_prompts=["a", "b"], segment_intervals=[[0, 5], [3, 7]])
        self.layout_args = dict(text_token_tags=torch.tensor([0, 0, 1, 1, 1, 1, 1]), num_latent_frames=7,
                                latent_height=2, latent_width=4, num_audio_latents=9,
                                patch_size=(1, 2, 2), audio_channels=2, audio_tag=2, video_tag=0)

    def run_layout(self, layout):
        positions, tags, video, audio, text, ncond, _ = layout
        state = prepare_attention_state(position_ids=positions, video_indices=video, text_indices=text,
                                        num_condition_video_rows=ncond, num_latent_frames=7, duration=22/24,
                                        tokenizer=CharacterTokenizer(), config=self.config)
        inputs = dict(hidden_states=torch.randn(1, len(video), 96), audio_hidden_states=torch.randn(1, len(audio), 32),
                      encoder_hidden_states=torch.randn(1, len(text), 8), timestep=torch.tensor([0.6]),
                      timestep_indices=torch.zeros(len(tags), dtype=torch.long), token_tags=tags, position_ids=positions,
                      video_indices=video, audio_indices=audio, text_indices=text)
        originals = [b.attn.processor for b in self.model.transformer_blocks]
        refiners = [b.attn.processor for b in self.model.token_refiner.refiner_blocks]
        with torch.no_grad():
            baseline = self.model(**inputs)
            # No-op routing over the entire time range must match native H3.
            active_schedule = state.schedule
            state.schedule = None
            with use_relay_attention(self.model, state, SlidingWindowConfig(31, 16), 5):
                whole = self.model(**inputs)
            torch.testing.assert_close(whole.sample, baseline.sample, atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(whole.audio_sample, baseline.audio_sample, atol=3e-6, rtol=3e-5)
            state.schedule = active_schedule
            with use_relay_attention(self.model, state, SlidingWindowConfig(4, 2), 5):
                relayed = self.model(**inputs)
            self.assertFalse(torch.allclose(relayed.sample, baseline.sample))
            self.assertTrue(torch.isfinite(relayed.sample).all())
            self.assertTrue(torch.isfinite(relayed.audio_sample).all())
            restored = self.model(**inputs)
            torch.testing.assert_close(restored.sample, baseline.sample, atol=0, rtol=0)
        self.assertEqual(originals, [b.attn.processor for b in self.model.transformer_blocks])
        self.assertEqual(refiners, [b.attn.processor for b in self.model.token_refiner.refiner_blocks])
        with self.assertRaisesRegex(RuntimeError, "intentional"):
            with use_relay_attention(self.model, state):
                raise RuntimeError("intentional")
        self.assertEqual(originals, [b.attn.processor for b in self.model.transformer_blocks])
        return state

    def test_text_and_first_last_keyframes(self):
        for anchors in [(), ("first",), ("last",), ("first", "last")]:
            with self.subTest(anchors=anchors):
                state = self.run_layout(MiniMaxH3PrepareLayoutStep.build_packed_sequence(
                    **self.layout_args, keyframe_anchors=anchors))
                self.assertEqual(len(state.video_rows), 14)

    def test_reference_clock_offset_and_mixed_media(self):
        refs = [SimpleNamespace(kind="image", has_audio=False), SimpleNamespace(kind="audio", has_audio=True),
                SimpleNamespace(kind="video", has_audio=True)]
        layout = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
            **self.layout_args, references=refs,
            condition_latents=[torch.zeros(1, 24, 1, 2, 2), torch.zeros(1, 24, 7, 2, 4)],
            audio_condition_latents=[torch.zeros(10, 32), torch.zeros(20, 32)])
        state = self.run_layout(layout)
        self.assertEqual(state.schedule.intervals, [(0, 5), (3, 7)])
        self.assertEqual(int(state.key_segment_ids[:2].max()), -1)

    def test_fused_projections(self):
        self.model.fuse_qkv_projections()
        self.run_layout(MiniMaxH3PrepareLayoutStep.build_packed_sequence(**self.layout_args))

    def test_context_parallel_rejected(self):
        layout = MiniMaxH3PrepareLayoutStep.build_packed_sequence(**self.layout_args)
        state = prepare_attention_state(position_ids=layout[0], video_indices=layout[2], text_indices=layout[4],
                                        num_condition_video_rows=0, num_latent_frames=7, duration=22/24)
        with patch.object(self.model.transformer_blocks[0].attn.processor, "_parallel_config", object()):
            with self.assertRaisesRegex(ValueError, "unsharded"):
                with use_relay_attention(self.model, state, SlidingWindowConfig(4, 2)):
                    pass


class TinyTextStep(ModularPipelineBlocks):
    """Deterministic embeddings substitute ONLY the heavyweight Qwen encoder."""
    model_name = "minimax-h3"

    @property
    def inputs(self):
        return [InputParam("prompt", type_hint=str, required=True)]

    @property
    def intermediate_outputs(self):
        return [OutputParam("prompt_embeds", type_hint=torch.Tensor), OutputParam("text_token_tags", type_hint=torch.Tensor)]

    def __call__(self, components, state):
        tokens = torch.tensor([ord(c) for c in state.get("prompt")]).float()
        state.set("prompt_embeds", torch.sin(tokens[:, None] + torch.arange(8)[None])[None])
        state.set("text_token_tags", torch.ones(len(tokens), dtype=torch.long))
        return components, state


class PipelineTests(unittest.TestCase):
    def test_all_workflows_install_and_idempotence(self):
        for workflow in [None, "t2va", "fl2va", "ref2va"]:
            with self.subTest(workflow=workflow):
                pipe = MiniMaxH3ModularPipeline(workflow=workflow)
                enable_prompt_relay(pipe)
                self.assertIs(enable_prompt_relay(pipe), pipe)
                names = {p.name for p in pipe.blocks.inputs}
                self.assertTrue({"prompt_relay_config", "sliding_window", "window_length", "window_stride"} <= names)

    def test_real_denoising_loop_no_weights(self):
        blocks = MiniMaxH3Blocks().get_workflow("t2va")
        blocks.sub_blocks["text_encoder"] = TinyTextStep()
        blocks.sub_blocks.pop("decode.video")
        blocks.sub_blocks.pop("decode.audio")
        pipe = MiniMaxH3ModularPipeline(blocks=blocks)
        torch.manual_seed(42)
        pipe.update_components(transformer=tiny_model(), scheduler=MiniMaxH3Scheduler(shift=12),
                               audio_scheduler=MiniMaxH3Scheduler(shift=3))
        pipe.tokenizer = CharacterTokenizer()
        pipe.set_progress_bar_config(disable=True)
        config = dict(global_prompt="g", local_prompts=["a", "b"], segment_intervals=[[0, 24], [18, 37]])
        text = build_prompt_text(config)[0]

        def run(**options):
            return pipe(num_frames=124, height=32, width=32, num_inference_steps=3,
                        generator=torch.Generator().manual_seed(7), output=["latents", "audio_latents"], **options)

        baseline = run(prompt=text)
        enable_prompt_relay(pipe)
        disabled = run(prompt=text)
        for name in baseline:
            torch.testing.assert_close(disabled[name], baseline[name], rtol=0, atol=0)
        overlap = run(prompt_relay_config=config, sliding_window=True, window_length=21, window_stride=12,
                      query_chunk_size=32)
        self.assertEqual(overlap["latents"].shape, (1, 24, 37, 2, 2))
        self.assertEqual(overlap["audio_latents"].shape[0], 2)
        self.assertTrue(torch.isfinite(overlap["latents"]).all())
        self.assertFalse(torch.allclose(overlap["latents"], baseline["latents"]))
        restored = run(prompt=text)
        for name in baseline:
            torch.testing.assert_close(restored[name], baseline[name], rtol=0, atol=0)

    def test_real_byte_bpe_offsets(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        from transformers import PreTrainedTokenizerFast

        config = dict(global_prompt="A robot waves. 机器人挥手。", local_prompts=["A robot waves.", "机器人挥手。", "A robot waves."])
        text, _ = build_prompt_text(config)
        backend = Tokenizer(models.BPE(unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.train_from_iterator([text], trainers.BpeTrainer(vocab_size=270, special_tokens=["[UNK]"],
                                                               initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend)
        n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        ids = token_segment_ids(tokenizer, config, n + 12)
        self.assertTrue(torch.all(ids[:12] == -1))
        for i in range(3):
            self.assertTrue(torch.any(ids == i))

    def test_conditioned_workflow_loops_preserve_anchors(self):
        for workflow in ["fl2va", "ref2va"]:
            with self.subTest(workflow=workflow):
                blocks = MiniMaxH3Blocks().get_workflow(workflow)
                blocks.sub_blocks["text_encoder"] = TinyTextStep()
                # Substitute media ENCODING with synthetic encoded tensors, but
                # retain the real layout, noise augmentation and both schedulers.
                for name in ["before_encode", "vae_encoder", "decode.video", "decode.audio", "denoise.after_denoise"]:
                    blocks.sub_blocks.pop(name)
                pipe = MiniMaxH3ModularPipeline(blocks=blocks)
                model_name = "transformer_ref" if workflow == "ref2va" else "transformer"
                pipe.update_components(**{model_name: tiny_model()}, scheduler=MiniMaxH3Scheduler(shift=12),
                                       audio_scheduler=MiniMaxH3Scheduler(shift=3))
                pipe.tokenizer = CharacterTokenizer()
                enable_prompt_relay(pipe)
                pipe.set_progress_bar_config(disable=True)
                state = PipelineState()
                if workflow == "fl2va":
                    state.set("keyframe_anchors", ("first", "last"))
                    state.set("condition_latents", [torch.zeros(1, 24, 1, 2, 2), torch.ones(1, 24, 1, 2, 2)])
                else:
                    state.set("normalized_references", [SimpleNamespace(kind="image", has_audio=False),
                                                        SimpleNamespace(kind="audio", has_audio=True)])
                    state.set("condition_latents", [torch.zeros(1, 24, 1, 2, 2)])
                    state.set("audio_condition_latents", [torch.ones(10, 32)])
                config = dict(global_prompt="g", local_prompts=["a", "b"], segment_intervals=[[0, 24], [18, 37]])
                result = pipe(state=state, prompt_relay_config=config, sliding_window=True, window_length=21,
                              window_stride=12, num_frames=124, height=32, width=32, num_inference_steps=3,
                              generator=torch.Generator().manual_seed(7))
                n = result.num_condition_video_rows
                torch.testing.assert_close(result.latents[:n], result.condition_rows, rtol=0, atol=0)
                self.assertTrue(torch.isfinite(result.latents).all())
                if workflow == "ref2va":
                    torch.testing.assert_close(result.audio_latents[:10], torch.ones(10, 32), rtol=0, atol=0)


class CLITests(unittest.TestCase):
    def test_switches_and_invalid_window(self):
        from generate import parse_args

        self.assertFalse(parse_args(["--prompt", "test"]).sliding_window)
        self.assertTrue(parse_args(["--prompt", "test", "--sliding_window"]).sliding_window)
        self.assertFalse(parse_args(["--prompt", "test", "--sliding_window", "false"]).sliding_window)
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            parse_args(["--prompt", "test", "--sliding_window", "--window_length", "4", "--window_stride", "5"])


if __name__ == "__main__":
    unittest.main()
