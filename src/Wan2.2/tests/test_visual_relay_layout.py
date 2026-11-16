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


if __name__ == "__main__":
    unittest.main()
