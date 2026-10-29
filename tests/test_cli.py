"""The CLI has one source of observation inputs and no render-time overrides."""

import argparse
import contextlib
import io
import unittest

from almanac.cli import build_parser, parse_utc_offset


class CliTests(unittest.TestCase):
    def test_fixed_offsets(self):
        for text, minutes in [("+8", 480), ("+08:00", 480), ("-05:30", -330),
                              ("+05:45", 345), ("+14:00", 840), ("-12", -720)]:
            self.assertEqual(parse_utc_offset(text), minutes)
        for text in ["8.5", "+14:01", "-12:01", "NaN", "Asia/Shanghai", "+08:60"]:
            with self.assertRaises(argparse.ArgumentTypeError):
                parse_utc_offset(text)

    def test_required_parameters_and_default_height(self):
        parser = build_parser()
        args = parser.parse_args(["all", "--year", "2027", "--utc-offset=+08:00", "--latitude", "40"])
        self.assertEqual((args.year, args.utc_offset, args.latitude, args.elevation),
                         (2027, 480, 40, 0))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["generate", "--year", "2027"])

    def test_render_cannot_override_observer_or_year(self):
        for name, value in [("--year", "2028"), ("--latitude", "52"),
                            ("--utc-offset", "+08:00"), ("--elevation", "40")]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                build_parser().parse_args(["render", "--run-dir", "example", name, value])


if __name__ == "__main__":
    unittest.main()
