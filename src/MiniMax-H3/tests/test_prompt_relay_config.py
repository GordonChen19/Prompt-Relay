import json
import tempfile
import unittest
from pathlib import Path

from h3_prompt_relay import PromptRelayConfig
from h3_prompt_relay.config import resolve_schedule


class PromptRelayConfigTests(unittest.TestCase):
    def test_json_defaults_match_wan(self):
        payload = {
            "global_prompt": " Global ",
            "local_prompts": [" First ", "Second"],
            "frame_intervals": [[0, 7], [3, 8]],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            config = PromptRelayConfig.from_json(path)

        self.assertEqual(config.global_prompt, "Global")
        self.assertEqual(config.local_prompts, ("First", "Second"))
        self.assertEqual(config.full_prompt, "Global First Second")
        self.assertEqual(config.mode, "gaussian")
        self.assertEqual(config.sigma, 0.932)
        self.assertEqual(config.interval_expansion, 2)
        self.assertEqual(config.frame_intervals(8), ((0, 8), (1, 8)))

    def test_automatic_intervals_keep_wan_ceil_allocation(self):
        config = PromptRelayConfig.from_mapping(
            {
                "global_prompt": "Global",
                "local_prompts": ["one", "two", "three", "four"],
            }
        )
        self.assertEqual(
            config.frame_intervals(21),
            ((0, 8), (4, 14), (10, 20), (16, 21)),
        )

    def test_segment_lengths_must_cover_internal_timeline(self):
        config = PromptRelayConfig.from_mapping(
            {
                "global_prompt": "Global",
                "local_prompts": ["one", "two"],
                "segment_lengths": [4, 4],
            }
        )
        with self.assertRaisesRegex(ValueError, "must sum"):
            config.frame_intervals(9)

    def test_intervals_support_overlap_gaps_and_unsorted_order(self):
        intervals = [[8, 15], [0, 10], [17, 21]]
        config = PromptRelayConfig.from_mapping(
            {
                "global_prompt": "Global",
                "local_prompts": ["one", "two", "three"],
                "frame_intervals": intervals,
                "interval_expansion": 0,
            }
        )
        self.assertEqual(config.frame_intervals(21), tuple(map(tuple, intervals)))

    def test_interval_expansion_is_configurable_and_clamped(self):
        config = PromptRelayConfig.from_mapping(
            {
                "global_prompt": "Global",
                "local_prompts": ["one", "two", "three"],
                "frame_intervals": [[0, 8], [8, 14], [14, 21]],
                "interval_expansion": 3,
            }
        )
        self.assertEqual(
            config.frame_intervals(21),
            ((0, 11), (5, 17), (11, 21)),
        )
        self.assertEqual(config.to_mapping()["interval_expansion"], 3)

    def test_lengths_and_intervals_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            PromptRelayConfig.from_mapping(
                {
                    "global_prompt": "Global",
                    "local_prompts": ["one", "two"],
                    "segment_lengths": [4, 4],
                    "frame_intervals": [[0, 4], [3, 8]],
                }
            )

    def test_hard_and_gaussian_schedule_parameters(self):
        gaussian = PromptRelayConfig.from_mapping(
            {
                "global_prompt": "Global",
                "local_prompts": ["one"],
                "frame_intervals": [[0, 8]],
            }
        )
        schedule = resolve_schedule(gaussian, 8)
        self.assertEqual(schedule.midpoints, (4,))
        self.assertEqual(schedule.inner_windows, (2,))
        self.assertEqual(schedule.mode, "gaussian")

        hard = PromptRelayConfig.from_mapping(
            {
                "global_prompt": "Global",
                "local_prompts": ["one"],
                "frame_intervals": [[0, 8]],
                "mode": "hard",
            }
        )
        self.assertEqual(resolve_schedule(hard, 8).mode, "hard")

    def test_invalid_intervals_and_modes_are_rejected(self):
        invalid = (
            {"frame_intervals": [[0, 0]]},
            {"frame_intervals": [[-1, 2]]},
            {"frame_intervals": [[0, 2.5]]},
            {"frame_intervals": [[False, 2]]},
            {"mode": "sharp"},
            {"sigma": 0},
            {"interval_expansion": -1},
            {"interval_expansion": 1.5},
            {"interval_expansion": True},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                PromptRelayConfig.from_mapping(
                    {
                        "global_prompt": "Global",
                        "local_prompts": ["one"],
                        **changes,
                    }
                )


if __name__ == "__main__":
    unittest.main()
