"""The small GUI delegates validated inputs to the supported CLI."""
import argparse
import unittest

from almanac.cli import build_parser
from almanac.gui import generation_command


class GuiTests(unittest.TestCase):
    def test_command_round_trip_and_star_switch(self):
        for stars in (True, False):
            command = generation_command("2028", " -05:30 ", "-30.5", stars)
            args = build_parser().parse_args(command[3:])
            self.assertEqual((args.year, args.utc_offset, args.latitude, args.elevation),
                             (2028, -330, -30.5, 0))
            self.assertEqual(args.no_stars, not stars)

    def test_invalid_inputs_are_rejected_before_launch(self):
        for values in (("x", "+8", "40"), ("2028", "8.5", "40"),
                       ("2028", "+8", "nan"), ("2028", "+8", "91")):
            with self.assertRaises((ValueError, argparse.ArgumentTypeError)):
                generation_command(*values, False)

    def test_gui_and_render_commands(self):
        self.assertEqual(build_parser().parse_args(["gui"]).command, "gui")
        args = build_parser().parse_args(["render", "--run-dir", "example", "--no-stars"])
        self.assertTrue(args.no_stars)


if __name__ == "__main__":
    unittest.main()
