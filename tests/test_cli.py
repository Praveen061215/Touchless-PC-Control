"""
test_cli.py
------------
Unit tests for command line parsing.
"""

import unittest
from cli import parse_arguments


class TestCLI(unittest.TestCase):
    def test_default_arguments(self):
        args = parse_arguments([])
        self.assertEqual(args.config, "config.json")
        self.assertIsNone(args.camera)
        self.assertIsNone(args.width)
        self.assertFalse(args.no_hud)
        self.assertFalse(args.diagnostics)

    def test_custom_arguments(self):
        args = parse_arguments([
            "--camera", "1",
            "--width", "1280",
            "--height", "720",
            "--filter", "one-euro",
            "--no-hud",
            "--debug"
        ])
        self.assertEqual(args.camera, 1)
        self.assertEqual(args.width, 1280)
        self.assertEqual(args.height, 720)
        self.assertEqual(args.filter, "one-euro")
        self.assertTrue(args.no_hud)
        self.assertTrue(args.debug)

    def test_diagnostics_flag(self):
        args = parse_arguments(["--diagnostics"])
        self.assertTrue(args.diagnostics)


if __name__ == "__main__":
    unittest.main()
