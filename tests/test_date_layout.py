"""Date layout uses rendered moon geometry and leaves calendar values intact."""
import unittest
from unittest.mock import patch

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Ellipse

from almanac.date_layout import DateLayout, DATE_RAISE_POINTS
from almanac.render import ROOT, RenderOptions, _AlmanacPlot
from tests.test_render import dataset_fixture


class DateLayoutTests(unittest.TestCase):
    def test_ink_bounds_match_rasterized_source_han_digits(self):
        for dpi in (72, 300):
            figure = Figure(figsize=(2, 2), dpi=dpi)
            FigureCanvasAgg(figure)
            try:
                ax = figure.add_axes((0, 0, 1, 1))
                ax.axis("off")
                text = ax.text(0.5, 0.5, "31", ha="center", va="center",
                               fontproperties=FontProperties(
                                   fname=str(ROOT / "fonts/SourceHanSansSC-Regular.otf"), size=9))
                figure.canvas.draw()
                layout = DateLayout([], [], [], np.empty((0, 3)), 0.8)
                ink = layout.ink_box(text, figure.canvas.get_renderer())
                pixels = np.asarray(figure.canvas.buffer_rgba())
                rows, cols = np.nonzero(np.min(pixels[:, :, :3], axis=2) < 128)
                raster = [cols.min(), pixels.shape[0] - rows.max() - 1,
                          cols.max() + 1, pixels.shape[0] - rows.min()]
                # Vertical accuracy controls whether a rule breaks. Allow one
                # pixel for raster hinting; outlines themselves are unhinted.
                np.testing.assert_allclose(ink.extents[[1, 3]], np.array(raster)[[1, 3]],
                                           atol=1, rtol=0)
            finally:
                figure.clear()

    def test_rules_ignore_font_whitespace_but_clear_digit_ink(self):
        for dpi in (72, 300):
            with self.subTest(dpi=dpi):
                figure = Figure(figsize=(15, 15), dpi=dpi)
                FigureCanvasAgg(figure)
                try:
                    ax = figure.subplots()
                    ax.set_xlim(-8, 8)
                    ax.set_ylim(366, 1)
                    date = ax.text(-6.4, 120, "31", ha="center", va="center",
                                   fontproperties=FontProperties(
                                       fname=str(ROOT / "fonts/SourceHanSansSC-Regular.otf"),
                                       size=9), zorder=30)
                    bounds = np.array([[1, -6, 6], [366, -6, 6]])
                    layout = DateLayout([(date, 120, -1)], [], [], bounds, 0.8)
                    ax.add_artist(layout)
                    figure.canvas.draw()
                    ink = layout.ink_box(date, figure.canvas.get_renderer())
                    scale = dpi / 72
                    # Clear ink by 1 pt above/below; intersect its middle once.
                    for display_y, split in ((ink.y0 - scale, False),
                                             (ink.y1 + scale, False),
                                             ((ink.y0 + ink.y1) / 2, True)):
                        y = ax.transData.inverted().transform((0, display_y))[1]
                        line, = ax.plot([-6.79, -6.01], [y, y], lw=0.5, zorder=19)
                        layout.rules = [(line, np.array(line.get_xdata()),
                                         np.array(line.get_ydata()))]
                        figure.canvas.draw()
                        self.assertEqual(bool(np.isnan(line.get_xdata()).any()), split)
                        if split:
                            xy = ax.transData.transform(np.column_stack(line.get_data()))
                            self.assertAlmostEqual(xy[1, 0], ink.x0 - scale)
                            self.assertAlmostEqual(xy[3, 0], ink.x1 + scale)
                        else:
                            np.testing.assert_allclose(line.get_xdata(), [-6.79, -6.01])
                        line.remove()
                finally:
                    figure.clear()

    def test_raise_horizontal_avoidance_rule_gap_and_repeat_draw(self):
        figure = Figure(figsize=(15, 15), dpi=72)
        FigureCanvasAgg(figure)
        try:
            ax = figure.subplots()
            ax.set_xlim(-8, 8)
            ax.set_ylim(366, 1)
            ax.set_aspect(1 / 14)
            date = ax.text(-6.2, 120.5, "31", ha="center", va="center", fontsize=10, zorder=30)
            moon = Ellipse((-6.12, 120), 0.26, 3.64, zorder=999)
            ax.add_patch(moon)
            line, = ax.plot([-6.79, -6.01], [120, 120], zorder=19)
            bounds = np.column_stack((np.arange(1, 367), np.full(366, -6), np.full(366, 6)))
            layout = DateLayout([(date, 120, -1)], [line], [(moon,)], bounds, 0.8)
            ax.add_artist(layout)
            figure.canvas.draw()
            renderer = figure.canvas.get_renderer()
            box = date.get_window_extent(renderer)
            original = ax.transData.transform((-6.2, 120.5))
            actual = date.get_transform().transform(date.get_position())
            self.assertEqual(date.get_position(), (-6.2, 120.5))
            self.assertAlmostEqual(actual[1] - original[1], DATE_RAISE_POINTS)
            self.assertLess(actual[0], original[0])
            self.assertFalse(box.overlaps(moon.get_window_extent(renderer)))
            xy = ax.transData.transform(np.column_stack((line.get_xdata(), line.get_ydata())))
            for start, end in zip(xy[:-1], xy[1:]):
                if np.isfinite([start, end]).all():
                    self.assertTrue(max(start[0], end[0]) <= box.x0
                                    or min(start[0], end[0]) >= box.x1)
            before = box.extents.copy()
            before_line = np.array(line.get_xdata())
            figure.canvas.draw()
            np.testing.assert_array_equal(date.get_window_extent(renderer).extents, before)
            np.testing.assert_array_equal(line.get_xdata(), before_line)
        finally:
            figure.clear()

    def test_only_existing_full_moon_patches_are_registered(self):
        data = dataset_fixture()
        plot = _AlmanacPlot(data, RenderOptions(moon=False))
        figure = Figure()
        FigureCanvasAgg(figure)
        try:
            ax = figure.subplots()
            def symbol(**kwargs):
                kwargs["ax"].add_patch(Ellipse((kwargs["hour"], kwargs["day"]), 0.26, 3.64))
            with patch("almanac.render.draw_moon_symbol", side_effect=symbol):
                for phase in (0, 90, 180, 270):
                    plot.moon_symbol(ax, data.topo.midnight(1), phase, 1, -6.2, quarter=True)
            self.assertEqual(len(ax.patches), 4)
            self.assertEqual(plot.full_moon_patches, [(ax.patches[2],)])
        finally:
            figure.clear()


if __name__ == "__main__":
    unittest.main()
