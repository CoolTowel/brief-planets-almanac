"""Rendering boundary tests; no PDF files are authored by this suite."""
import inspect
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from astropy.table import Table

from almanac import render
from almanac._vendor.astronomy import Time
from almanac.moon_symbol import draw_moon_symbol
from almanac.topo import TopoContext


def dataset_fixture(year=2027):
    topo = TopoContext(year, 480, 40)
    count = topo.days_in_year
    days = np.arange(1, count + 1)
    tables = {}
    bodies = ("Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn", *render.STAR_NAMES)
    for event in ("rise", "set", "transit"):
        table = Table({"Day": days})
        hour = {"rise": 6.0, "set": -6.0, "transit": 0.0}[event]
        for body in bodies:
            table[body] = np.full(count, hour)
        tables[event] = table
    tables["twilight"] = Table({"Day": days, "Dusk": np.full(count, -4.0),
                                "Dawn": np.full(count, 4.0)})
    return SimpleNamespace(topo=topo, run_dir=Path("test-run"), manifest={}, tables=tables,
                           arrays={"EOT": np.zeros(count), "sun_alts": np.full((count, 1440), -18.0)})


class RenderBoundaryTests(unittest.TestCase):
    def test_polar_night_rejected_before_directory_creation(self):
        data = dataset_fixture()
        data.tables["rise"]["Sun"][59] = -999
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "output"
            with self.assertRaisesRegex(ValueError, "2027-03-01"):
                render.render_almanac(data, output)
            self.assertFalse(output.exists())

    def test_reversed_solar_events_rejected(self):
        data = dataset_fixture()
        data.tables["rise"]["Sun"][0] = -7
        with self.assertRaisesRegex(ValueError, "2027-01-01"):
            render._validate_render_dataset(data)

    def test_existing_output_rejected_before_plotting(self):
        data = dataset_fixture()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / data.run_dir.name
            directory.mkdir()
            existing = directory / "almanac.png"
            existing.write_bytes(b"preserve-me")
            with patch.object(render._AlmanacPlot, "draw") as draw:
                with self.assertRaisesRegex(FileExistsError, "--overwrite"):
                    render.render_almanac(data, temp)
                draw.assert_not_called()
            self.assertEqual(existing.read_bytes(), b"preserve-me")

    def test_wrong_year_dimensions_rejected(self):
        data = dataset_fixture(2028)
        data.arrays["EOT"] = np.zeros(365)
        with self.assertRaisesRegex(ValueError, "EOT dimensions"):
            render._validate_render_dataset(data)

    def test_missing_twilight_values_break_lines(self):
        data = dataset_fixture()
        data.tables["twilight"]["Dusk"][5] = -999
        plot = render._AlmanacPlot(data, render.RenderOptions(moon=False))
        ax = Mock()
        plot.grid(ax)
        dusk_x, dusk_y = ax.plot.call_args_list[0].args
        self.assertTrue(np.isnan(dusk_x[5]))
        self.assertTrue(np.isnan(dusk_y[5]))
        self.assertEqual(dusk_x[4], -4)
        self.assertEqual(dusk_x[6], -4)

    def test_all_moon_calls_use_dataset_observer(self):
        data = dataset_fixture()
        plot = render._AlmanacPlot(data, render.RenderOptions(moon=False))
        quarter = SimpleNamespace(time=data.topo.midnight(3), quarter=0)
        after_end = SimpleNamespace(time=Time(data.topo.end_utc.ut + 1), quarter=1)
        with patch.object(render.astro, "SearchMoonQuarter", return_value=quarter), \
                patch.object(render.astro, "NextMoonQuarter", return_value=after_end), \
                patch.object(render.astro, "MoonPhase", return_value=90), \
                patch.object(render, "draw_moon_symbol") as symbol:
            plot.moons(Mock())
        # Two ordinary rise symbols after the new-moon rise and two set
        # symbols before its set are now intentionally suppressed.
        self.assertEqual(symbol.call_count, 2 * data.topo.days_in_year - 4)
        quarter_calls = 0
        for call in symbol.call_args_list:
            obs = call.kwargs["obs"]
            self.assertEqual((obs.latitude, obs.longitude, obs.height), (40, 120, 0))
            quarter_calls += int(call.kwargs["mares"])
        self.assertEqual(quarter_calls, 2)

    def test_moon_observer_is_required_keyword_only(self):
        parameter = inspect.signature(draw_moon_symbol).parameters["obs"]
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(parameter.default, inspect.Parameter.empty)

    def test_date_labels_cross_leap_day_and_year_boundary(self):
        data = dataset_fixture(2028)
        plot = render._AlmanacPlot(data, render.RenderOptions(moon=False))
        plot.days_7 = np.array([59, 60, 61, 366])
        ax = Mock()
        plot.text(ax)
        labels = [call.args[2] for call in ax.text.call_args_list[:8]]
        self.assertEqual(labels, ["28", "29", "29", "1", "1", "2", "31", "1"])

    def test_overwrite_and_both_output_paths(self):
        data = dataset_fixture()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / data.run_dir.name
            directory.mkdir()
            existing = directory / "almanac.png"
            existing.write_bytes(b"existing")
            # Mock both writers: this test checks orchestration, not PDF output.
            with patch("matplotlib.figure.Figure.savefig") as save, \
                    patch.object(render._AlmanacPlot, "draw"):
                paths = render.render_almanac(data, temp, render.RenderOptions(overwrite=True))
            self.assertEqual(set(paths), {"png", "pdf"})
            self.assertEqual([call.args[0].suffix for call in save.call_args_list], [".png", ".pdf"])
            self.assertEqual(paths["png"], existing)


if __name__ == "__main__":
    unittest.main()
