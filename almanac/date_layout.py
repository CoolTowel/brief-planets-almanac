"""Renderer-space date placement and gaps in calendar rules."""
from copy import copy

import numpy as np
from matplotlib.artist import Artist
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D, Bbox, IdentityTransform


DATE_RAISE_POINTS = 2
DATE_CLEARANCE_POINTS = 1
RULE_CLEARANCE_POINTS = 0.35


class DateLayout(Artist):
    """Lay out existing date artists before calendar rules and text are drawn.

    ``dates`` is a sequence of (Text, day index, side), where side is -1/1.
    ``rules`` contains horizontal Line2D artists; ``moons`` contains groups of
    patches for already drawn full moons only. ``bounds`` is an N×3 array of
    (day, sunset hour, sunrise hour); ``month_width`` is the date-band width in
    hours. No astronomy is evaluated here. Each renderer resets base transforms
    so PNG/PDF and repeated draws cannot accumulate shifts.
    """

    def __init__(self, dates, rules, moons, bounds, month_width):
        super().__init__()
        self.dates = [(text, side, text.get_transform()) for text, _, side in dates]
        self.rules = [(line, np.array(line.get_xdata()), np.array(line.get_ydata()))
                      for line in rules]
        self.moons = moons
        self.bounds = bounds
        self.month_width = month_width
        self._ink_paths = {}
        self.set_zorder(18)

    def ink_box(self, text, renderer):
        """Return the unrotated digit ink bounds, excluding font whitespace.

        Date labels are single-line, centred, unrotated text. Measure a copy
        aligned to the baseline so renderer/font line metrics, including the
        font-table metrics used by Matplotlib 3.11, match the actual text.
        Glyph outlines are cached by font and date string in points.
        """
        font = text.get_fontproperties()
        key = (font.get_fontconfig_pattern(), text.get_text())
        if key not in self._ink_paths:
            self._ink_paths[key] = TextPath((0, 0), text.get_text(), prop=font).get_extents()
        ink = self._ink_paths[key]
        scale = renderer.points_to_pixels(1)
        box = text.get_window_extent(renderer)
        probe = copy(text)
        probe.set_transform(IdentityTransform())
        probe.set_position((0, 0))
        probe.set_verticalalignment("baseline")
        baseline = box.y0 - probe.get_window_extent(renderer).y0
        return Bbox.from_extents(box.x0 + ink.x0 * scale, baseline + ink.y0 * scale,
                                 box.x0 + ink.x1 * scale, baseline + ink.y1 * scale)

    @staticmethod
    def _shift(box, obstacles, left, right, gap):
        """Choose the nearest collision-free horizontal position analytically."""
        if not len(obstacles):
            return 0.0
        intervals = obstacles[(obstacles[:, 1] < box.y1 + gap)
                              & (obstacles[:, 3] > box.y0 - gap)][:, [0, 2]]
        if not len(intervals):
            return 0.0
        if not np.any((intervals[:, 0] < box.x1 + gap)
                      & (intervals[:, 1] > box.x0 - gap)):
            return 0.0
        low, high = left, right - box.width
        if high < low:
            return 0.0
        candidates = np.clip(np.r_[box.x0, intervals[:, 1] + gap,
                                    intervals[:, 0] - gap - box.width], low, high)
        best = None
        for start in candidates:
            overlap = np.maximum(0, np.minimum(start + box.width + gap, intervals[:, 1])
                                 - np.maximum(start - gap, intervals[:, 0])).sum()
            score = (float(overlap), abs(start - box.x0))
            if best is None or score < best[0]:
                best = score, start - box.x0
        return float(best[1])

    def draw(self, renderer):
        if not self.get_visible():
            return
        scale = renderer.points_to_pixels(1)
        gap = DATE_CLEARANCE_POINTS * scale
        moon_boxes = np.array([
            Bbox.union([patch.get_window_extent(renderer) for patch in group]).extents
            for group in self.moons if group
        ]).reshape(-1, 4)
        inverse = self.axes.transData.inverted()
        date_boxes = []
        for text, side, base_transform in self.dates:
            raised = base_transform + Affine2D().translate(0, DATE_RAISE_POINTS * scale)
            text.set_transform(raised)
            box = self.ink_box(text, renderer)
            # Constrain the whole glyph box to its curved date band.
            days = inverse.transform([[box.x0, box.y0], [box.x1, box.y1]])[:, 1]
            inner = np.interp(days, self.bounds[:, 0], self.bounds[:, 1 if side < 0 else 2])
            outer = inner + side * self.month_width
            inner_x = self.axes.transData.transform(np.column_stack((inner, days)))[:, 0]
            outer_x = self.axes.transData.transform(np.column_stack((outer, days)))[:, 0]
            left = np.maximum.reduce(np.minimum(inner_x, outer_x)) + gap
            right = np.minimum.reduce(np.maximum(inner_x, outer_x)) - gap
            shift = self._shift(box, moon_boxes, left, right, gap)
            if shift:
                text.set_transform(raised + Affine2D().translate(shift, 0))
            # Font line boxes include substantial empty space above/below the
            # digits. Only actual ink close to a stroke should interrupt it.
            date_boxes.append(self.ink_box(text, renderer).extents)

        date_boxes = np.array(date_boxes).reshape(-1, 4)
        for line, original_x, original_y in self.rules:
            points = self.axes.transData.transform(np.column_stack((original_x, original_y)))
            left, right = sorted(points[:, 0])
            y = points[0, 1]
            clearance = (RULE_CLEARANCE_POINTS + line.get_linewidth() / 2) * scale
            overlaps = date_boxes[(date_boxes[:, 1] - clearance <= y)
                                  & (date_boxes[:, 3] + clearance >= y)
                                  & (date_boxes[:, 0] - gap < right)
                                  & (date_boxes[:, 2] + gap > left)]
            cursor = left
            segments = []
            for box in overlaps[np.argsort(overlaps[:, 0])]:
                start, end = max(left, box[0] - gap), min(right, box[2] + gap)
                if start > cursor:
                    segments.extend([[cursor, y], [start, y], [np.nan, np.nan]])
                cursor = max(cursor, end)
            if cursor < right:
                segments.extend([[cursor, y], [right, y]])
            if segments:
                data = inverse.transform(np.asarray(segments))
                line.set_data(data[:, 0], data[:, 1])
            else:
                line.set_data([], [])
        self.stale = False
