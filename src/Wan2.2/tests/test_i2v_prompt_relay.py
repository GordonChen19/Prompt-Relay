import ast
import unittest
from pathlib import Path


I2V_PATH = Path(__file__).parents[1] / "wan" / "image2video.py"
TREE = ast.parse(I2V_PATH.read_text(encoding="utf-8"))


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

        self.assertIn("y", dictionaries["arg_c"])
        self.assertIn("prompt_relay_segments", dictionaries["arg_c"])
        self.assertIn("prompt_relay_query_chunk_size", dictionaries["arg_c"])
        self.assertIn("y", dictionaries["arg_null"])
        self.assertNotIn("prompt_relay_segments", dictionaries["arg_null"])

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


if __name__ == "__main__":
    unittest.main()
