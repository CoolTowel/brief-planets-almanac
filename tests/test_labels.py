"""Renderer-scale geometry, masked gaps, and display-only star selection."""
import unittest
from unittest.mock import Mock

import numpy as np
from matplotlib import font_manager, rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D, Bbox

from almanac.calculations import compute_eot
from almanac.labels import CurveLabel, CurveLabels, split_visible_curve
from almanac.render import ROOT, RenderOptions, _AlmanacPlot
from tests.test_render import dataset_fixture


class CurveLabelTests(unittest.TestCase):
    def test_visible_ink_clearance_is_two_points_for_all_reading_directions(self):
        for dpi in (72, 300):
            figure, artist = self.make_figure(dpi)
            try:
                scale = dpi / 72
                for angle in (25, 65, 115, 155, 205, 245, 295, 335):
                    vector = np.array([np.cos(np.deg2rad(angle)),
                                       np.sin(np.deg2rad(angle))])
                    origin = np.array([200, 200]) * scale
                    points = np.array([origin - 100 * scale * vector,
                                       origin + 100 * scale * vector])
                    for side in (-1, 0, 1):
                        spec = CurveLabel("金星没" if side else "土星出", points, "gold",
                                          horizontal_side=side, vertical=side == 1)
                        glyphs = artist._glyphs(figure.canvas.get_renderer(), spec, points,
                                               100 * scale, -6.7 if side < 0 else 6.7)
                        self.assertEqual(len(glyphs), 3)
                        normal = np.array([-vector[1], vector[0]])
                        if side:
                            normal *= side * np.sign(normal[0])
                        else:
                            # Automatic reading may reverse its normal.
                            if np.dot(glyphs[0]._curve_ink.mean(axis=0) - origin, normal) < 0:
                                normal *= -1
                        for glyph in glyphs:
                            ink = TextPath((0, 0), glyph.get_text(), prop=glyph.get_fontproperties())
                            ink = ink.transformed(Affine2D().scale(scale).rotate_deg(
                                glyph.get_rotation()).translate(*glyph.get_position()))
                            projected = ink.transformed(Affine2D.from_values(
                                vector[0], normal[0], vector[1], normal[1],
                                -origin @ vector, -origin @ normal)).get_extents()
                            self.assertAlmostEqual(projected.y0 / scale - 0.4, 2, places=7)
            finally:
                figure.clear()

    def test_mercury_sixty_degree_rule_in_all_tangent_quadrants(self):
        figure, artist = self.make_figure(72)
        try:
            for angle in (1, 59, 60, 61, 119, 120, 121, 179,
                          181, 239, 240, 241, 299, 300, 301, 359):
                for reverse in (False, True):
                    for side in (-1, 1):
                        with self.subTest(angle=angle, reverse=reverse, side=side):
                            radians = np.deg2rad(angle)
                            vector = np.array([np.cos(radians), np.sin(radians)]) * 100
                            points = np.array([np.array([200, 200]) - vector,
                                               np.array([200, 200]) + vector])
                            if reverse:
                                points = points[::-1]
                            spec = CurveLabel("水星出", points, "aqua", horizontal_side=side)
                            glyphs = artist._glyphs(figure.canvas.get_renderer(), spec, points,
                                                   100, side * 6.7)
                            inclination = min(angle % 180, 180 - angle % 180)
                            vertical = inclination >= 60
                            self.assertEqual(len(glyphs), 3)
                            expected_rotation = angle + (90 if vertical else 0)
                            for glyph in glyphs:
                                difference = (glyph.get_rotation() - expected_rotation) % 180
                                self.assertLess(min(difference, 180 - difference), 1e-8)
                            positions = np.array([g._curve_ink.mean(axis=0) for g in glyphs])
                            self.assertTrue(np.all(np.diff(positions[:, 1]) < 0) if vertical
                                            else np.all(np.diff(positions[:, 0]) > 0))
                            order = np.argsort(points[:, 1])
                            curve_x = np.interp(positions[:, 1], points[order, 1], points[order, 0])
                            self.assertTrue(np.all(side * (positions[:, 0] - curve_x) > 0))
        finally:
            figure.clear()

    def test_inner_planet_event_labels_use_vertical_outer_side(self):
        data = dataset_fixture()
        for event in ("rise", "set"):
            for body in ("Mercury", "Venus"):
                data.tables[event][body] = np.full(data.topo.days_in_year, -3.0)
        plot = _AlmanacPlot(data, RenderOptions(moon=False, stars=False))
        plot.lines(Mock())
        labels = [spec for spec in plot.curve_labels if spec.text[:2] in ("水星", "金星")]
        self.assertEqual(len(labels), 4)
        for spec in labels:
            self.assertEqual(spec.vertical, spec.text.startswith("金星"))
            side = -1 if spec.text.endswith("出") else 1
            self.assertEqual(spec.horizontal_side, side)
            self.assertIsNone(spec.normal_offset)

    def test_forced_vertical_planet_labels_keep_side_and_reading_order(self):
        for dpi in (72, 300):
            figure, artist = self.make_figure(dpi)
            try:
                scale = dpi / 72
                for dx in (-200, 200):
                    for reverse in (False, True):
                        points = np.array([[100, 300], [100 + dx, 150]], dtype=float) * scale
                        if reverse:
                            points = points[::-1]
                        length = np.linalg.norm(points[1] - points[0])
                        for side in (-1, 1):
                            spec = CurveLabel("水星出" if side < 0 else "金星没", points, "aqua",
                                              vertical=True, horizontal_side=side)
                            glyphs = artist._glyphs(figure.canvas.get_renderer(), spec, points,
                                                   length / 2, side * 6.7)
                            self.assertEqual(len(glyphs), 3)
                            self.assertTrue(np.all(np.diff([g.get_position()[1] for g in glyphs]) < 0))
                            for glyph in glyphs:
                                x, y = glyph._curve_ink.mean(axis=0)
                                order = np.argsort(points[:, 1])
                                curve_x = np.interp(y, points[order, 1], points[order, 0])
                                self.assertGreater(side * (x - curve_x), 0)
            finally:
                figure.clear()

    def test_masked_gaps_do_not_get_joined_or_mutated(self):
        points = np.array([[0, 1], [1, 2], [np.nan, np.nan], [3, 4], [4, 5]])
        before = points.copy()
        pieces = split_visible_curve(points)
        self.assertEqual([len(piece) for piece in pieces], [2, 2])
        pieces[0][0] = 99
        np.testing.assert_equal(points, before)

    def make_figure(self, dpi):
        figure = Figure(figsize=(6, 6), dpi=dpi)
        FigureCanvasAgg(figure)
        ax = figure.subplots()
        ax.set_xlim(-8, 8)
        ax.set_ylim(366, 1)
        days = np.arange(1, 367)
        points = np.column_stack((-4 + np.sin(days / 90), days))
        spec = CurveLabel("水星没", points, "aqua", repeat=False)
        artist = CurveLabels([spec], ROOT / "fonts/SourceHanSansSC-Regular.otf",
                             np.column_stack((days, days * 0 - 7, days * 0 + 7)))
        ax.add_artist(artist)
        figure.canvas.draw()
        return figure, artist

    def test_physical_label_positions_survive_dpi_change(self):
        figures = []
        try:
            locations = []
            for dpi in (72, 300):
                figure, artist = self.make_figure(dpi)
                figures.append(figure)
                self.assertEqual(len(artist.placed_labels), 1)
                boxes = artist.placed_labels[0][1]
                locations.append(np.array([box.get_points() for box in boxes]) * 72 / dpi)
            np.testing.assert_allclose(*locations, atol=1.2, rtol=0)
        finally:
            for figure in figures:
                figure.clear()

    def test_glyphs_follow_tangent_and_use_regular_nine_point_font(self):
        figure, artist = self.make_figure(100)
        try:
            spec = artist.labels[0]
            points = artist.axes.transData.transform(spec.points)
            length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
            glyphs = artist._glyphs(figure.canvas.get_renderer(), spec, points, length / 2, 2.2)
            self.assertEqual("".join(glyph.get_text() for glyph in glyphs), "水星没")
            self.assertTrue(all(glyph.get_fontsize() == 9 for glyph in glyphs))
            self.assertTrue(all(glyph.get_fontproperties().get_file().endswith(
                "SourceHanSansSC-Regular.otf") for glyph in glyphs))
            self.assertGreater(np.ptp([glyph.get_rotation() for glyph in glyphs]), 0.01)
        finally:
            figure.clear()

    def test_month_font_and_expanded_vertical_tracking(self):
        figure, artist = self.make_figure(72)
        try:
            points = np.array([[200, 500], [200, 100]], dtype=float)
            spec = CurveLabel("十一月", points, "black", vertical=True, month=True)
            glyphs = artist._glyphs(figure.canvas.get_renderer(), spec, points, 200, 0)
            self.assertEqual(len(glyphs), 3)
            self.assertTrue(all(g.get_fontsize() == 10 for g in glyphs))
            self.assertTrue(all(g.get_fontproperties().get_file().endswith(
                "SourceHanSansSC-Medium.otf") for g in glyphs))
            np.testing.assert_allclose(np.diff([g.get_position()[1] for g in glyphs]), -11.5)
            self.assertTrue(all(g.get_rotation() == 0 for g in glyphs))
        finally:
            figure.clear()

    def test_repeats_require_long_curve_and_widely_separated_positions(self):
        for dpi in (72, 144):
            for length, expected in ((550, 1), (800, 2)):
                with self.subTest(dpi=dpi, length=length):
                    figure = Figure(figsize=(15, 15), dpi=dpi)
                    FigureCanvasAgg(figure)
                    try:
                        ax = figure.add_axes((0.1, 0.1, 0.8, 0.8))
                        ax.set_xlim(-8, 8)
                        ax.set_ylim(366, 1)
                        centre = ax.transData.transform((-4, 183))
                        half = length * dpi / 72 / 2
                        points = ax.transData.inverted().transform(
                            [centre + (0, half), centre - (0, half)])
                        spec = CurveLabel("木星上中天", points, "purple")
                        artist = CurveLabels([spec], ROOT / "fonts/SourceHanSansSC-Regular.otf",
                                             np.array([[1, -7, 7], [366, -7, 7]]))
                        ax.add_artist(artist)
                        figure.canvas.draw()
                        self.assertEqual(len(artist.placed_labels), expected)
                        if expected == 2:
                            centres = [Bbox.union(boxes).get_points().mean(axis=0)
                                       for _, boxes in artist.placed_labels]
                            self.assertGreater(np.linalg.norm(centres[1] - centres[0]) * 72 / dpi,
                                               280)
                    finally:
                        figure.clear()

    def test_label_moves_away_from_another_curve_crossing(self):
        figure, artist = self.make_figure(72)
        try:
            first = CurveLabel("木星没", np.array([[-4, 30], [-4, 335]]), "purple",
                               repeat=False)
            crossing = CurveLabel("火星出", np.array([[-6, 183], [-2, 183]]), "red",
                                  repeat=False)
            artist.labels = [first, crossing]
            figure.canvas.draw()
            boxes = artist.placed_labels[0][1]
            crossing_y = artist.axes.transData.transform((-4, 183))[1]
            self.assertTrue(all(not box.y0 <= crossing_y <= box.y1 for box in boxes))
        finally:
            figure.clear()

    def test_star_switch_only_removes_star_plot_calls(self):
        data = dataset_fixture()
        before = {key: table.copy() for key, table in data.tables.items()}
        for enabled, expected in ((False, 13), (True, 23)):
            plot = _AlmanacPlot(data, RenderOptions(moon=False, stars=enabled))
            ax = Mock()
            plot.lines(ax)
            self.assertEqual(ax.plot.call_count, expected)
            names = [label.text for label in plot.curve_labels]
            self.assertEqual(any("天狼" in name for name in names), enabled)
        for key in before:
            np.testing.assert_array_equal(data.tables[key].as_array(), before[key].as_array())

    def test_steep_chinese_labels_always_read_top_down(self):
        figure, artist = self.make_figure(100)
        try:
            spec = CurveLabel("水星没", np.empty((0, 2)), "aqua", repeat=False)
            for dx in (-100, 0, 100):
                for reverse in (False, True):
                    with self.subTest(dx=dx, reverse=reverse):
                        points = np.array([[200, 600], [200 + dx, 200]], dtype=float)
                        if reverse:
                            points = points[::-1]
                        length = np.linalg.norm(points[1] - points[0])
                        glyphs = artist._glyphs(figure.canvas.get_renderer(), spec,
                                                points, length / 2, 2.2)
                        self.assertEqual("".join(g.get_text() for g in glyphs), "水星没")
                        self.assertTrue(np.all(np.diff([g.get_position()[1] for g in glyphs]) < 0))
                        angles = [(g.get_rotation() + 180) % 360 - 180 for g in glyphs]
                        self.assertTrue(all(abs(angle) < 30 for angle in angles))
        finally:
            figure.clear()

    def test_shallow_chinese_labels_always_read_left_to_right(self):
        figure, artist = self.make_figure(100)
        try:
            spec = CurveLabel("火星出", np.empty((0, 2)), "firebrick", repeat=False)
            for dy in (-150, 150):
                for reverse in (False, True):
                    with self.subTest(dy=dy, reverse=reverse):
                        points = np.array([[100, 300], [500, 300 + dy]], dtype=float)
                        if reverse:
                            points = points[::-1]
                        length = np.linalg.norm(points[1] - points[0])
                        glyphs = artist._glyphs(figure.canvas.get_renderer(), spec,
                                                points, length / 2, 2.2)
                        self.assertEqual("".join(g.get_text() for g in glyphs), "火星出")
                        self.assertTrue(np.all(np.diff([g.get_position()[0] for g in glyphs]) > 0))
                        angles = [(g.get_rotation() + 180) % 360 - 180 for g in glyphs]
                        self.assertTrue(all(abs(angle) < 30 for angle in angles))
        finally:
            figure.clear()

    def test_months_include_leap_february_on_both_sides(self):
        data = dataset_fixture(2028)
        plot = _AlmanacPlot(data, RenderOptions(moon=False))
        ax = Mock()
        plot.annotations(ax)
        labels = ax.add_artist.call_args.args[0].labels
        self.assertEqual(sum(label.end is None for label in labels), 24)
        february = [label for label in labels if label.text == "二月"]
        np.testing.assert_array_equal(february[0].points[:, 1], np.arange(32, 61))
        np.testing.assert_array_equal(february[1].points[:, 1], np.arange(31, 60))

    def test_month_centres_are_fixed_with_or_without_moon_obstacles(self):
        data = dataset_fixture(2028)
        plot = _AlmanacPlot(data, RenderOptions(moon=False, grid=False, eot=False))
        collector = Mock()
        plot.annotations(collector)
        specifications = collector.add_artist.call_args.args[0].labels
        layouts = []
        for crowded in (False, True):
            figure = Figure(figsize=(15, 15), dpi=72)
            FigureCanvasAgg(figure)
            try:
                ax = figure.subplots()
                ax.set_xlim(-8, 8)
                ax.set_ylim(366, 1)
                obstacles = (np.vstack([spec.points for spec in specifications])
                             if crowded else ())
                artist = CurveLabels(specifications, ROOT / "fonts/SourceHanSansSC-Regular.otf",
                                     np.column_stack((np.arange(1, 367), np.full(366, -6),
                                                      np.full(366, 6))), obstacles)
                ax.add_artist(artist)
                figure.canvas.draw()
                self.assertEqual(len(artist.placed_labels), 24)
                layouts.append(np.concatenate([box.extents for _, boxes in artist.placed_labels
                                               for box in boxes]))
                for spec, (_, boxes) in zip(specifications, artist.placed_labels):
                    centre_y = np.mean([(box.y0 + box.y1) / 2 for box in boxes])
                    day = ax.transData.inverted().transform((0, centre_y))[1]
                    self.assertAlmostEqual(day, np.mean(spec.points[[0, -1], 1]), places=6)
            finally:
                figure.clear()
        np.testing.assert_array_equal(*layouts)

    def test_eot_and_twilight_labels_at_both_ends_and_layer_switches(self):
        data = dataset_fixture()
        figure = Figure(figsize=(15, 15), dpi=72)
        FigureCanvasAgg(figure)
        try:
            plot = _AlmanacPlot(data, RenderOptions(moon=False, lines=False, background=False))
            plot.draw(figure)
            figure.canvas.draw()
            artist = next(a for a in figure.axes[0].artists if isinstance(a, CurveLabels))
            inverse = figure.axes[0].transData.inverted()
            for name in ("均时差", "天文晨光始", "天文昏影终"):
                placements = [boxes for text, boxes in artist.placed_labels if text == name]
                self.assertEqual(len(placements), 2, name)
                days = [inverse.transform(boxes[0].get_points().mean(axis=0))[1]
                        for boxes in placements]
                self.assertLess(min(days), 35)
                self.assertGreater(max(days), 330)
                if name == "均时差":
                    for boxes in placements:
                        self.assertTrue(all(inverse.transform(box.get_points().mean(axis=0))[0] > 0
                                            for box in boxes))
            ax = Mock()
            _AlmanacPlot(data, RenderOptions(moon=False, grid=False, eot=False)).annotations(ax)
            self.assertFalse(any(label.end for label in ax.add_artist.call_args.args[0].labels))
        finally:
            figure.clear()

    def test_eot_clears_quarter_symbol_even_when_ra_boxes_penalize_other_positions(self):
        data = dataset_fixture(2026)
        data.arrays["EOT"] = compute_eot(data.topo)
        for font in font_manager.findSystemFonts(
                fontpaths=[str(ROOT / "fonts/Libertinus-7.040/static/OTF")]):
            font_manager.fontManager.addfont(font)
        figure = Figure(figsize=(15, 15), dpi=72)
        FigureCanvasAgg(figure)
        try:
            with rc_context({"font.family": "Libertinus Sans"}):
                plot = _AlmanacPlot(data, RenderOptions(moon=False, lines=False))
                plot.draw(figure)
                ax = figure.axes[0]
                artist = next(a for a in ax.artists if isinstance(a, CurveLabels))
                # Selected 2026-01-26 quarter set; a 10 pt square conservatively
                # encloses the enlarged symbol. Previously two EOT glyphs hit it.
                artist.obstacles = np.array([[0.491642241177, 10],
                                             [0.201835780248, 25],
                                             [0.347467351996, 39]])
                figure.canvas.draw()
                moons = [Bbox.from_bounds(x - 5, y - 5, 10, 10)
                         for x, y in ax.transData.transform(artist.obstacles)]
                placements = [boxes for text, boxes in artist.placed_labels if text == "均时差"]
                self.assertEqual(len(placements), 2)
                self.assertFalse(any(box.overlaps(moon) for boxes in placements
                                     for box in boxes for moon in moons))
        finally:
            figure.clear()


if __name__ == "__main__":
    unittest.main()
