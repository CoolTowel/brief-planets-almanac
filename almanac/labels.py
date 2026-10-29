"""Display-space lettering along existing curves; no astronomical calculations."""
from dataclasses import dataclass

import numpy as np
from matplotlib.artist import Artist
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path
from matplotlib.text import Text
from matplotlib.textpath import TextPath
from matplotlib.transforms import Affine2D, Bbox, IdentityTransform


VERTICAL_LABEL_ANGLE_DEGREES = 60
REPEAT_MIN_LENGTH_POINTS = 650
REPEAT_MIN_SEPARATION_POINTS = 240
MONTH_TRACKING_POINTS = 1.5
CURVE_TEXT_GAP_POINTS = 2


def split_visible_curve(points):
    """Return finite contiguous (hour, day) segments without joining masked gaps.

    ``points`` is an N×2 display-data array, with NaN for hidden events.
    The input is never modified. Segments with fewer than two points are omitted.
    """
    points = np.asarray(points, dtype=float)
    valid = np.isfinite(points).all(axis=1)
    edges = np.flatnonzero(np.diff(np.r_[False, valid, False]))
    return [points[start:end].copy() for start, end in zip(edges[::2], edges[1::2])
            if end - start >= 2]


@dataclass
class CurveLabel:
    """Label specification in the same data coordinates as its plotted curve."""

    text: str
    points: np.ndarray
    color: str
    vertical: bool = False
    repeat: bool = True
    end: str | None = None
    avoid_meridian: bool = True
    normal_offset: float | None = None
    fixed_position: bool = False
    month: bool = False
    horizontal_side: int = 0  # -1: left of curve, +1: right; 0: automatic.
    curve_linewidth: float = 0.8  # Points; clearance is measured from the stroke edge.


def _overlaps(boxes, targets):
    """Compare display-space rectangles in bulk with Bbox.overlaps semantics."""
    a = np.asarray([box.extents for box in boxes])[:, None, :]
    b = targets[None, :, :]
    return ((a[..., 0] <= b[..., 2]) & (a[..., 2] >= b[..., 0])
            & (a[..., 1] <= b[..., 3]) & (a[..., 3] >= b[..., 1]))


class CurveLabels(Artist):
    """Draw Regular 9 pt text and Medium 10 pt months with renderer metrics.

    ``labels`` contains CurveLabel objects; ``font_path`` is the Regular font.
    ``month_font_path`` is the Medium font (defaults to its sibling font file).
    ``bounds`` contains (day, sunset hour, sunrise hour) rows for label clearance.
    ``obstacles`` contains (hour, day) moon centres. All offsets and glyph spacing
    are in typographic points, so saving at another DPI keeps the same layout.
    Short segments are left unlabelled rather than spilling into the frame.
    """

    def __init__(self, labels, font_path, bounds, obstacles=(), *, month_font_path=None):
        super().__init__()
        self.labels = labels
        self.font = FontProperties(fname=str(font_path), size=9, weight="normal")
        if month_font_path is None:
            from pathlib import Path as FilePath

            month_font_path = FilePath(font_path).with_name("SourceHanSansSC-Medium.otf")
        self.month_font = FontProperties(fname=str(month_font_path), size=10, weight="medium")
        self.bounds = np.asarray(bounds)
        self.obstacles = np.asarray(obstacles).reshape(-1, 2)
        self.placed_labels = []
        self._glyph_paths = {}
        self.set_zorder(35)

    def _glyphs(self, renderer, spec, points, centre, offset):
        """Choose reading direction from the local display-space tangent.

        Steep labels read top down with upright Chinese glyphs; shallow labels
        read left to right. Orientation applies to the whole word, even when
        the input curve has been reversed or bends elsewhere in the year.
        """
        scale = renderer.points_to_pixels(1)
        font = self.month_font if spec.month else self.font
        distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        keep = np.r_[True, np.diff(distances) > 1e-8]
        points, distances = points[keep], distances[keep]
        if len(points) < 2:
            return []

        def interpolate(position):
            return np.array([np.interp(position, distances, points[:, dim]) for dim in (0, 1)])

        tangent = interpolate(centre + 4 * scale) - interpolate(centre - 4 * scale)
        inclination = np.degrees(np.arctan2(abs(tangent[1]), abs(tangent[0])))
        # Fold all quadrants onto 0..90 degrees; keep exactly 60 degrees
        # vertical despite floating-point interpolation roundoff.
        vertical = spec.vertical or inclination >= VERTICAL_LABEL_ANGLE_DEGREES - 1e-10
        reverse = tangent[1] > 0 if vertical else tangent[0] < 0
        if reverse:
            points = points[::-1]
            distances = distances[-1] - distances[::-1]
            centre = distances[-1] - centre
        widths = np.array([renderer.get_text_width_height_descent(c, font, False)[0]
                           for c in spec.text])
        if vertical:
            widths[:] = font.get_size_in_points() * scale
        tracking = MONTH_TRACKING_POINTS if spec.month else 0.25
        advances = widths + tracking * scale
        total = advances.sum()
        start = centre - total / 2
        if start < 3 * scale or start + total > distances[-1] - 3 * scale:
            return []
        positions = start + np.cumsum(advances) - advances / 2
        glyphs = []
        for char, position in zip(spec.text, positions):
            xy = np.array([np.interp(position, distances, points[:, dim]) for dim in (0, 1)])
            # Average the tangent across one glyph to suppress daily sampling noise.
            before = np.array([np.interp(position - 4 * scale, distances, points[:, dim])
                               for dim in (0, 1)])
            after = np.array([np.interp(position + 4 * scale, distances, points[:, dim])
                              for dim in (0, 1)])
            tangent = after - before
            angle = np.degrees(np.arctan2(tangent[1], tangent[0]))
            normal = np.array([-tangent[1], tangent[0]]) / np.linalg.norm(tangent)
            rotation = angle + 90 if vertical else angle
            event_label = not spec.month and spec.end is None
            if spec.horizontal_side:
                # Reading direction can reverse the normal for horizontal
                # lettering. Its side must not depend on that direction.
                normal *= spec.horizontal_side * np.sign(normal[0])
            elif event_label and offset < 0:
                normal *= -1
            if event_label:
                # Place real glyph ink, not its baseline/line box, at a fixed
                # distance from the stroke. Horizontal and vertical text use
                # the same rule, including lettering on the opposite side.
                key = (font.get_fontconfig_pattern(), char)
                if key not in self._glyph_paths:
                    self._glyph_paths[key] = TextPath((0, 0), char, prop=font)
                ink = self._glyph_paths[key].transformed(
                    Affine2D().rotate_deg(rotation))
                # Project the actual outline onto the curve normal/tangent;
                # rotating an axis-aligned text box overestimates clearance.
                vertices = ink.vertices[ink.codes != Path.CLOSEPOLY] * scale
                unit_tangent = tangent / np.linalg.norm(tangent)
                projected = ink.transformed(Affine2D.from_values(
                    unit_tangent[0], normal[0], unit_tangent[1], normal[1], 0, 0)).get_extents()
                gap = (CURVE_TEXT_GAP_POINTS + spec.curve_linewidth / 2) * scale
                xy += (normal * (gap - projected.y0 * scale)
                       - unit_tangent * (projected.x0 + projected.x1) * scale / 2)
            else:
                xy += offset * scale * normal
            glyph = Text(*xy, text=char, fontproperties=font, color=spec.color,
                         rotation=rotation,
                         rotation_mode="anchor", ha="left" if event_label else "center",
                         va="baseline" if event_label else ("center" if vertical else "baseline"),
                         transform=IdentityTransform())
            glyph.set_figure(self.figure)
            if event_label:
                glyph._curve_ink = vertices + xy
            glyphs.append(glyph)
        return glyphs

    def draw(self, renderer):
        if not self.get_visible():
            return
        scale = renderer.points_to_pixels(1)
        occupied = np.empty((0, 4))
        self.placed_labels = []
        moon_centres = self.axes.transData.transform(self.obstacles)
        moon_boxes = [Bbox.from_bounds(x - 3 * scale, y - 3 * scale, 6 * scale, 6 * scale)
                      for x, y in moon_centres]
        moon_boxes = np.array([box.extents for box in moon_boxes]).reshape(-1, 4)
        # Keep EOT lettering off central RA numbers and horizontal tick strokes.
        meridian_boxes = np.array([
            text.get_window_extent(renderer).padded(scale).extents
            for text in self.axes.texts if text.get_zorder() == 0
        ] + [
            line.get_window_extent(renderer).padded(0.5 * scale).extents
            for line in self.axes.lines if line.get_zorder() == 0
        ]).reshape(-1, 4)
        inverse = self.axes.transData.inverted()
        curve_paths = [(spec, Path(self.axes.transData.transform(spec.points)))
                       for spec in self.labels if not spec.fixed_position]
        for spec in self.labels:
            points = self.axes.transData.transform(spec.points)
            # Horizontal curve lettering reads left to right. Months read top down.
            if not spec.vertical and points[-1, 0] < points[0, 0]:
                points = points[::-1]
            length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
            word_length = sum(renderer.get_text_width_height_descent(c, self.font, False)[0]
                              + 0.25 * scale for c in spec.text)
            repeat_length = max(REPEAT_MIN_LENGTH_POINTS * scale, 12 * word_length)
            separation = max(REPEAT_MIN_SEPARATION_POINTS * scale, 6 * word_length)
            count = 2 if spec.repeat and length >= repeat_length else 1
            targets = [0.25, 0.75] if count == 2 else [0.5]
            placed_centres = []
            if spec.end is not None:
                targets = [0.1 if spec.end == "top" else 0.9]
            for target in targets:
                if spec.fixed_position:
                    # Month labels belong at the calendar interval's midpoint,
                    # not the arc-length midpoint of its seasonally bent frame.
                    distances = np.r_[0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
                    middle_y = (points[0, 1] + points[-1, 1]) / 2
                    order = np.argsort(points[:, 1])
                    centre = np.interp(middle_y, points[order, 1], distances[order])
                    candidates = [centre / length]
                elif spec.end is not None:
                    # EOT text must fit in the small gap between RA numbers;
                    # a coarse candidate grid can miss that gap entirely.
                    candidate_count = 209 if not spec.avoid_meridian else 27
                    fractions = (np.linspace(0.08, 0.6, candidate_count) if spec.end == "top"
                                 else np.linspace(0.4, 0.92, candidate_count))
                    candidates = sorted(fractions, key=lambda value: abs(value - target))
                else:
                    # Keep repeats in separate parts of the curve, even when
                    # one part is crowded. Never cluster duplicate words.
                    low, high = 0.08, 0.92
                    if count == 2:
                        low, high = (0.08, 0.45) if target < 0.5 else (0.55, 0.92)
                    candidates = sorted(np.linspace(low, high, 49),
                                        key=lambda value: abs(value - target))
                placements = [(fraction, spec.normal_offset, 0.0) for fraction in candidates]
                if not spec.avoid_meridian and spec.normal_offset is not None:
                    # A crowded EOT endpoint may have no clear 3-glyph gap.
                    # Stay on the same side of its curve, trying 5.5 pt
                    # more clearance before accepting a moon intersection.
                    farther = spec.normal_offset + np.copysign(5.5, spec.normal_offset)
                    placements.extend((fraction, farther, 0.055) for fraction in candidates)
                    placements.sort(key=lambda value: abs(value[0] - target) + value[2])
                best = None
                for fraction, offset, offset_penalty in placements:
                    if any(abs(fraction * length - centre) < separation
                           for centre in placed_centres):
                        continue
                    if offset is None:
                        offset = 0 if spec.vertical else 2.2
                    glyphs = self._glyphs(renderer, spec, points, fraction * length,
                                          offset)
                    if not glyphs:
                        continue
                    boxes = [glyph.get_window_extent(renderer).padded(0.8 * scale)
                             for glyph in glyphs]
                    if not spec.month:
                        corners = inverse.transform(np.concatenate([box.corners() for box in boxes]))
                        x, day = corners.T
                        left = np.interp(day, self.bounds[:, 0], self.bounds[:, 1])
                        right = np.interp(day, self.bounds[:, 0], self.bounds[:, 2])
                        if (np.any(day < self.bounds[0, 0] + 1)
                                or np.any(day > self.bounds[-1, 0] - 1)
                                or np.any(x < left + 0.08) or np.any(x > right - 0.08)
                                or (spec.avoid_meridian
                                    and np.any((x > -0.2) & (x < 0.5)))):
                            continue
                    if spec.horizontal_side:
                        # Compare every glyph box against the curve at the same
                        # display height. This also rejects words straddling a
                        # bend, where a centre-only side check is insufficient.
                        order = np.argsort(points[:, 1])
                        corners = np.concatenate([glyph._curve_ink for glyph in glyphs])
                        curve_x = np.interp(corners[:, 1], points[order, 1], points[order, 0])
                        if np.any(spec.horizontal_side * (corners[:, 0] - curve_x) < 0.5 * scale):
                            continue
                    if not spec.fixed_position and _overlaps(boxes, occupied).any():
                        continue
                    tick_penalty = 0.0
                    if not spec.avoid_meridian:
                        # Text extents include font ascender/descender whitespace.
                        # Minimize overlap area instead of dropping a required
                        # endpoint label when those conservative boxes cannot fit.
                        a = np.asarray([box.extents for box in boxes])[:, None, :]
                        b = meridian_boxes[None, :, :]
                        widths = np.maximum(0, np.minimum(a[..., 2], b[..., 2])
                                            - np.maximum(a[..., 0], b[..., 0]))
                        heights = np.maximum(0, np.minimum(a[..., 3], b[..., 3])
                                             - np.maximum(a[..., 1], b[..., 1]))
                        tick_penalty = float((widths * heights).sum() / scale**2) * 100
                    # EOT's tight RA spacing uses conservative font boxes for
                    # tick penalties. Keep an actual moon clear before trading
                    # off that whitespace; enlarged quarter symbols need 5 pt.
                    moon_targets = (moon_boxes + np.array([-2, -2, 2, 2]) * scale
                                    if not spec.avoid_meridian else moon_boxes)
                    collisions = (0 if spec.fixed_position
                                  else int(_overlaps(boxes, moon_targets).sum()))
                    crossings = 0
                    if spec.end is None and not spec.fixed_position:
                        crossings = sum(any(path.intersects_bbox(box, filled=False)
                                            for box in boxes)
                                        for other, path in curve_paths if other is not spec)
                    score = (collisions if not spec.avoid_meridian else 0,
                             tick_penalty + collisions * 5 + crossings * 2
                             + abs(fraction - target) + offset_penalty)
                    if best is None or score < best[0]:
                        best = score, glyphs, boxes, fraction * length
                    # Candidates are ordered by target distance. A zero-collision
                    # result is optimal; further search cannot improve the score.
                    if collisions == 0 and tick_penalty == 0 and crossings == 0:
                        break
                if best is not None:
                    _, glyphs, boxes, centre = best
                    placed_centres.append(centre)
                    for glyph in glyphs:
                        glyph.draw(renderer)
                    occupied = np.vstack((occupied, [box.extents for box in boxes]))
                    self.placed_labels.append((spec.text, boxes))
        self.stale = False
