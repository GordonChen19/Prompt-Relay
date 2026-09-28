import math
import unittest

import torch
import torch.nn.functional as F

from h3_prompt_relay.attention import SlidingWindowConfig, prepare_attention_state, relay_attention
from h3_prompt_relay.schedule import build_prompt_text, prepare_schedule, token_segment_ids


class CharacterTokenizer:
    def __call__(self, text, **kwargs):
        return dict(input_ids=[ord(c) for c in text], offset_mapping=[(i, i + 1) for i in range(len(text))])


def fixture(config=None):
    config = config or dict(global_prompt="g", local_prompts=["a", "b"], segment_intervals=[[0, 5], [3, 7]])
    text, _ = build_prompt_text(config)
    # Two Qwen vision/presentation tokens, two VAE keyframe rows, stereo audio,
    # then seven generated frames with two spatial patches each.
    ntext = len(text) + 2
    video = torch.cat((torch.arange(ntext, ntext + 2), torch.arange(ntext + 8, ntext + 22)))
    positions = torch.zeros(ntext + 22, 3, dtype=torch.float64)
    positions[video[:2], 0] = 999  # must NOT be mistaken for generated frames
    times = torch.tensor([0, 1, 5, 9, 13, 17, 18], dtype=torch.float64) / 24
    positions[video[2:], 0] = (times * 40 + 101.25).repeat_interleave(2)
    state = prepare_attention_state(position_ids=positions, video_indices=video,
                                    text_indices=torch.arange(ntext), num_condition_video_rows=2,
                                    num_latent_frames=7, duration=22/24,
                                    tokenizer=CharacterTokenizer(), config=config)
    return state


def dense_reference(q, k, v, state, window=None):
    """Small explicit softmax oracle, evaluating one query and window at a time."""
    result = torch.empty_like(q)
    frame_by_row = dict(zip(state.video_rows.tolist(), state.frame_ids.tolist()))
    windows = list(window.windows(state.num_frames)) if window else [(0, state.num_frames)]
    for row in range(q.shape[1]):
        frame = frame_by_row.get(row)
        applicable = [(a, b) for a, b in windows if frame is not None and a <= frame < b]
        if frame is None:
            applicable = [(0, state.num_frames)]
        values = []
        for start, end in applicable:
            keys = [i for i in range(q.shape[1]) if i not in frame_by_row or start <= frame_by_row[i] < end]
            logits = torch.einsum("bhd,bkhd->bhk", q[:, row], k[:, keys]) / math.sqrt(q.shape[-1])
            if frame is not None and state.schedule is not None:
                s = state.schedule
                coordinate = float(s.coordinates[frame])
                for j, key_row in enumerate(keys):
                    segment = int(state.key_segment_ids[key_row])
                    if segment >= 0:
                        a, b = s.intervals[segment]
                        distance = max(float(s.coordinates[a]) - coordinate, coordinate - float(s.coordinates[b-1]), 0)
                        logits[:, :, j] -= distance**2 / (2 * s.sigma**2)
            values.append(torch.einsum("bhk,bkhd->bhd", logits.softmax(-1), v[:, keys]))
        result[:, row] = torch.stack(values).mean(0)
    return result


class ScheduleTests(unittest.TestCase):
    def test_nonuniform_seconds_and_overlap(self):
        config = dict(global_prompt="g", local_prompts=["a", "b"], time_unit="seconds",
                      segment_intervals=[[0, 17/24], [13/24, 22/24]])
        state = fixture(config)
        self.assertEqual(state.schedule.intervals, [(0, 5), (4, 7)])
        torch.testing.assert_close(state.schedule.coordinates, torch.tensor([0, 1, 5, 9, 13, 17, 18]) / 24)
        self.assertEqual(len(state.video_rows), 14)
        self.assertEqual(int(state.key_segment_ids[:2].max()), -1)

    def test_repeated_and_unicode_prompt_spans(self):
        config = dict(global_prompt="挥手", local_prompts=["挥手", "挥手"])
        text, spans = build_prompt_text(config)
        ids = token_segment_ids(CharacterTokenizer(), config, len(text) + 3)
        self.assertTrue(torch.all(ids[:5] == -1))
        for index, (a, b) in enumerate(spans):
            self.assertTrue(torch.all(ids[3+a:3+b] == index))

    def test_default_consecutive_and_gap(self):
        s = prepare_schedule(dict(local_prompts=["a", "b", "c"]), list(range(7)), 7)
        self.assertEqual(s.intervals, [(0, 3), (3, 5), (5, 7)])
        s = prepare_schedule(dict(local_prompts=["a", "b"], segment_intervals=[[4, 7], [0, 2]]), list(range(7)), 7)
        self.assertEqual(s.intervals, [(4, 7), (0, 2)])

    def test_rejects_token_crossing_global_local_content(self):
        config = dict(global_prompt="g", local_prompts=["a"])
        tokenizer = lambda text, **kwargs: dict(input_ids=[1], offset_mapping=[(0, len(text))])
        with self.assertRaisesRegex(ValueError, "global/local"):
            token_segment_ids(tokenizer, config, 1)

    def test_invalid_schedules(self):
        base = dict(local_prompts=["a", "b"], segment_intervals=[[0, 5], [3, 7]])
        invalid = [dict(epsilon=0), dict(epsilon=1), dict(tail_width=0), dict(tail_width=float("nan")),
                   dict(segment_intervals=[[0, 8], [1, 2]]), dict(segment_intervals=[[True, 2], [1, 2]]),
                   dict(segment_intervals=[[1, 1], [1, 2]]), dict(segment_intervals=[[0.1, 2], [1, 2]]),
                   dict(segment_lengths=[3, 4]), dict(fps=30), dict(auto_overlap=True),
                   dict(time_unit="seconds", segment_intervals=[[0.1, 0.2], [1, 2]])]
        for change in invalid:
            with self.subTest(change=change), self.assertRaises(ValueError):
                prepare_schedule(dict(base, **change), list(range(7)), 7)

    def test_window_coverage_and_validation(self):
        self.assertEqual(list(SlidingWindowConfig(4, 3).windows(8)), [(0, 4), (3, 7), (6, 8)])
        self.assertEqual(list(SlidingWindowConfig(31, 16).windows(7)), [(0, 7)])
        for args in [(0, 1), (4, 5), (4, 0), (True, 1)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                SlidingWindowConfig(*args)


class AttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.state = fixture()
        self.q, self.k, self.v = [torch.randn(1, self.state.sequence_length, 2, 8) for _ in range(3)]

    def test_matches_dense_oracle(self):
        for window in [None, SlidingWindowConfig(4, 2), SlidingWindowConfig(3, 3)]:
            for chunk in [1, 5, 128]:
                with self.subTest(window=window, chunk=chunk):
                    actual = relay_attention(self.q, self.k, self.v, self.state, window, chunk)
                    expected = dense_reference(self.q, self.k, self.v, self.state, window)
                    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_full_window_and_other_queries(self):
        relay = relay_attention(self.q, self.k, self.v, self.state)
        full = relay_attention(self.q, self.k, self.v, self.state, SlidingWindowConfig(31, 16))
        torch.testing.assert_close(relay, full)
        baseline = F.scaled_dot_product_attention(self.q.transpose(1, 2), self.k.transpose(1, 2),
                                                   self.v.transpose(1, 2)).transpose(1, 2)
        window = relay_attention(self.q, self.k, self.v, self.state, SlidingWindowConfig(4, 2))
        torch.testing.assert_close(window[:, self.state.other_rows], baseline[:, self.state.other_rows])
        self.assertFalse(torch.allclose(window[:, self.state.video_rows], baseline[:, self.state.video_rows]))

    def test_sliding_only(self):
        self.state.schedule = None
        actual = relay_attention(self.q, self.k, self.v, self.state, SlidingWindowConfig(4, 2))
        torch.testing.assert_close(actual, dense_reference(self.q, self.k, self.v, self.state, SlidingWindowConfig(4, 2)))

    def test_overlap_has_two_unpenalized_local_prompts(self):
        q, k = torch.zeros_like(self.q), torch.zeros_like(self.k)
        v = torch.zeros_like(self.v)
        v[:, self.state.key_segment_ids == 0, :, 0] = 1
        v[:, self.state.key_segment_ids == 1, :, 1] = 1
        output = relay_attention(q, k, v, self.state)
        both = output[:, self.state.video_rows[self.state.frame_ids == 4]]
        torch.testing.assert_close(both[..., 0], both[..., 1])
        first = output[:, self.state.video_rows[self.state.frame_ids == 0]]
        self.assertTrue(torch.all(first[..., 0] > first[..., 1] * 100))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_bfloat16_is_finite_and_close(self):
        args = [x.cuda().bfloat16() for x in (self.q, self.k, self.v)]
        actual = relay_attention(*args, self.state, SlidingWindowConfig(4, 2), 5)
        expected = dense_reference(*(x.float().cpu() for x in args), self.state, SlidingWindowConfig(4, 2))
        self.assertTrue(torch.isfinite(actual).all())
        torch.testing.assert_close(actual.float().cpu(), expected, atol=0.008, rtol=0.03)


if __name__ == "__main__":
    unittest.main()
