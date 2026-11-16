import importlib.util
import sys
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "wan" / "visual_relay.py"
SPEC = importlib.util.spec_from_file_location("visual_relay", MODULE_PATH)
VISUAL_RELAY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = VISUAL_RELAY
SPEC.loader.exec_module(VISUAL_RELAY)
VisualRelayLayout = VISUAL_RELAY.VisualRelayLayout
shifted_rope_positions = VISUAL_RELAY.shifted_rope_positions
ordinary_anchor_attention_regions = (
    VISUAL_RELAY.ordinary_anchor_attention_regions
)
anchor_suffix_attention_frames = VISUAL_RELAY.anchor_suffix_attention_frames


class VisualRelayLayoutTests(unittest.TestCase):

    def test_expands_and_restores_output_length(self):
        layout = VisualRelayLayout.build(21, (0, 10, 16))
        self.assertEqual(layout.internal_latent_frames, 23)
        self.assertEqual(layout.temporary_internal_indices, (10, 17))
        self.assertEqual(len(layout.retained_internal_indices), 21)

    def test_each_auxiliary_stream_keeps_only_its_anchor(self):
        layout = VisualRelayLayout.build(21, (0, 10, 16, 19))
        temporary = set(layout.temporary_internal_indices)
        for anchor_number, own_anchor in enumerate(
            layout.temporary_internal_indices
        ):
            selected = set(layout.auxiliary_internal_indices(anchor_number))
            self.assertEqual(selected & temporary, {own_anchor})
            self.assertEqual(len(selected), 22)

    def test_compacted_anchor_uses_final_output_coordinate(self):
        layout = VisualRelayLayout.build(21, (0, 10, 16))
        self.assertEqual(layout.compact_anchor_index(0), 10)
        self.assertEqual(layout.compact_anchor_index(1), 16)
        self.assertEqual(layout.history_output_indices(1), (12, 13, 14, 15))

    def test_shifted_rope_places_every_anchor_halfway_between_outputs(self):
        layout = VisualRelayLayout.build(21, (0, 10, 16))
        positions = shifted_rope_positions(
            layout.internal_latent_frames,
            layout.temporary_internal_indices,
            anchor_rope_offset=0.5,
        )
        self.assertEqual(positions[9:12], (9.0, 9.5, 10.0))
        self.assertEqual(positions[16:19], (15.0, 15.5, 16.0))
        retained_positions = tuple(
            positions[index] for index in layout.retained_internal_indices
        )
        self.assertEqual(
            retained_positions,
            tuple(float(index) for index in range(21)),
        )

    def test_ordinary_queries_see_only_neighboring_anchors(self):
        self.assertEqual(
            ordinary_anchor_attention_regions(27, (10, 16, 22)),
            (
                (0, 10, (10,)),
                (11, 16, (10, 16)),
                (17, 22, (16, 22)),
                (23, 27, (22,)),
            ),
        )

    def test_anchor_suffix_excludes_every_other_temporary_anchor(self):
        visible = anchor_suffix_attention_frames(27, (10, 16, 22), 10)
        self.assertEqual(visible[0], 10)
        self.assertNotIn(16, visible)
        self.assertNotIn(22, visible)
        self.assertEqual(
            visible,
            (10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 23, 24, 25, 26),
        )


if __name__ == "__main__":
    unittest.main()
