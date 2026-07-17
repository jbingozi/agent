import unittest
from pathlib import Path

from utilss.path_tool import get_abs_path
from utilss.prompt_loader import load_system_prompts, load_rag_prompts, load_report_prompts


class SmokeTest(unittest.TestCase):
    def test_core_paths_exist(self):
        self.assertTrue(Path(get_abs_path("config/agent.yml")).exists())
        self.assertTrue(Path(get_abs_path("prompts/main_prompt.txt")).exists())

    def test_prompts_load(self):
        self.assertGreater(len(load_system_prompts()), 0)
        self.assertGreater(len(load_rag_prompts()), 0)
        self.assertGreater(len(load_report_prompts()), 0)


if __name__ == "__main__":
    unittest.main()
