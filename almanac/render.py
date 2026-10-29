"""Render a complete almanac using only the configuration in its dataset."""
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
import math

import numpy as np

from ._vendor import astronomy as astro
from .background import getbackground
from .moon_symbol import draw_moon_symbol
from .stars import STAR_NAMES


ROOT = Path(__file__).resolve().parents[1]
# Physical dash lengths (points), measured from the supplied 2026 PDF.
EVENT_DASHES_POINTS = {
    "rise": (6, 1.75, 1, 1.75),
    "set": (8, 4),
    "transit": (),
}


@dataclass(frozen=True)
class RenderOptions:
    """Display options; observation and time configuration belong to the dataset."""

    frame: bool = True
    lines: bool = True
    moon: bool = True
    background: bool = True
    grid: bool = True
    eot: bool = True
    meridian_ra: bool = True
    text: bool = True
    stars: bool = True
    dpi: int = 300
    figsize: tuple[float, float] = (15, 15)
    overwrite: bool = False

    def __post_init__(self):
        if isinstance(self.dpi, bool) or not isinstance(self.dpi, int) or self.dpi <= 0:
            raise ValueError("dpi must be a positive integer")
        if len(self.figsize) != 2 or any(not math.isfinite(v) or v <= 0 for v in self.figsize):
            raise ValueError("figsize must contain two positive finite values")


def _event_valid(hours):
    return np.isfinite(hours) & (hours >= -12) & (hours < 12)


def _validate_render_dataset(dataset):
    """Reject a mismatched dataset or a night the current frame cannot represent."""
    count = dataset.topo.days_in_year
    expected_days = np.arange(1, count + 1)
    for name in ("rise", "set", "transit", "twilight"):
        table = dataset.tables[name]
        if len(table) != count or not np.array_equal(table["Day"], expected_days):
            raise ValueError(f"{name} table days do not match dataset TopoContext")
    if np.shape(dataset.arrays["sun_alts"]) != (count, 1440):
        raise ValueError("Solar altitude dimensions do not match dataset TopoContext")
    if np.shape(dataset.arrays["EOT"]) != (count,):
        raise ValueError("EOT dimensions do not match dataset TopoContext")
    rise = np.asarray(dataset.tables["rise"]["Sun"])
    setting = np.asarray(dataset.tables["set"]["Sun"])
    invalid = ~(_event_valid(rise) & _event_valid(setting) & (setting <= rise))
    if invalid.any():
        dates = [dataset.topo.dates[i].isoformat() for i in np.flatnonzero(invalid)]
        shown = ", ".join(dates[:8])
        if len(dates) > 8:
            shown += f", ... ({len(dates)} nights)"
        raise ValueError(
            "Current almanac layout requires a sunset followed by sunrise in each "
            f"night; polar-day/polar-night layout is not supported. Dates: {shown}"
        )


class _AlmanacPlot:
    """Per-render state replacing the legacy module globals."""

    def __init__(self, dataset, options):
        self.dataset = dataset
        self.topo = dataset.topo
        self.obs = self.topo.observer
        self.options = options
        self.days = self.topo.days_in_year
        self.y_scale = 1 / 14
        self.month_width = 0.8
        from matplotlib.font_manager import FontProperties

        self.fontsize = 9
        self.font = FontProperties(fname=str(ROOT / "fonts/SourceHanSansSC-Regular.otf"),
                                   size=self.fontsize)
        self.sun_rise = self.event("rise", "Sun")
        self.sun_set = self.event("set", "Sun")
        self.min_set_hour = -int(-np.min(self.sun_set[:, 1])) - 1
        self.max_rise_hour = int(np.max(self.sun_rise[:, 1])) + 1
        self.days_7 = np.arange(1, self.days, 7)
        self.lms = []
        self.curve_labels = []
        self.date_labels = []
        self.calendar_rules = []
        self.full_moon_patches = []
        self.moon_positions = []
        if options.moon:
            for filename in ("lm1.txt", "lm2.txt", "lm3.txt"):
                outline = np.loadtxt(ROOT / "data" / filename)
                outline[:, 1] = -outline[:, 1]
                self.lms.append(outline)

    def event(self, table_name, body):
        table = self.dataset.tables[table_name]
        return np.column_stack((table["Day"], table[body])).astype(float)

    def night_mask(self, events, bleed=0.1):
        hours = events[:, 1]
        return (_event_valid(hours)
                & (hours >= self.sun_set[:, 1] - bleed)
                & (hours <= self.sun_rise[:, 1] + bleed))

    @staticmethod
    def patch(ax, vertices, color=(1, 1, 1), zorder=1):
        from matplotlib.path import Path as PlotPath
        from matplotlib.patches import PathPatch

        codes = np.full(len(vertices), PlotPath.LINETO, dtype=PlotPath.code_type)
        codes[0] = PlotPath.MOVETO
        ax.add_patch(PathPatch(PlotPath(vertices, codes), facecolor=color,
                               linewidth=0, zorder=zorder))

    def month_background(self, ax, events, left):
        shift = -self.month_width if left else self.month_width
        first = np.copy(events[:, ::-1])
        second = np.copy(events[::-1, ::-1])
        second[:, 0] += shift
        self.patch(ax, np.vstack((first, second)),
                   color=(232 / 255, 246 / 255, 252 / 255), zorder=10)

    def frame(self, ax):
        rise, setting = self.sun_rise, self.sun_set
        self.patch(ax, np.vstack((setting[:, ::-1], [[-12, self.days], [-12, 1]])), zorder=10)
        self.patch(ax, np.vstack((rise[:, ::-1], [[12, self.days], [12, 1]])), zorder=10)
        self.month_background(ax, setting, True)
        self.month_background(ax, rise, False)
        for events, direction in ((setting, -1), (rise, 1)):
            ax.plot(events[:, 1], events[:, 0], c="k", lw=1, zorder=20)
            ax.plot(events[:, 1] + direction * self.month_width, events[:, 0],
                    c="k", lw=0.5, zorder=20)
        # Preserve the original frame endpoints and graphic offsets.
        self.calendar_rules.extend(ax.plot(
            [setting[1, 1] - self.month_width, rise[1, 1] + self.month_width - 0.015],
            [1, 1], c="k", lw=1, zorder=20))
        self.calendar_rules.extend(ax.plot([setting[-1, 1] - self.month_width + 0.015,
                 rise[-1, 1] + self.month_width - 0.015],
                [self.days, self.days], c="k", lw=1, zorder=20))
        for index, date in enumerate(self.topo.dates):
            if date.day != 1 or index == 0:
                continue
            set_day = index + 1
            rise_day = set_day - 1
            set_hour = setting[index, 1]
            rise_hour = rise[rise_day - 1, 1]
            self.calendar_rules.extend(ax.plot(
                [set_hour - self.month_width + 0.01, set_hour - 0.01],
                [set_day, set_day], c="black", lw=0.5, zorder=19))
            self.calendar_rules.extend(ax.plot(
                [rise_hour + 0.01, rise_hour + self.month_width - 0.01],
                [rise_day, rise_day], c="black", lw=0.5, zorder=19))

    def meridian_ra(self, ax):
        astro.DefineStar(astro.Body.Star8, ra=0, dec=0, distanceLightYears=1000)
        offset = astro.HourAngle(astro.Body.Star8,
                                 astro.Time(self.topo.origin_utc.ut - 0.5), self.obs)
        sidereal_year = 365.25636042
        for ra in np.arange(0, 24, 0.25):
            imaginary = ra - offset
            if imaginary < 0:
                imaginary += 24
            y = imaginary / 24 * sidereal_year
            small = ra % 1 > 0.1
            width = 0.075 if small else 0.175
            ax.plot([-width, width], [y, y], zorder=0, c="white", lw=0.5)
            if not small:
                ax.text(0.225, y, str(int(ra)), fontproperties=self.font,
                        c="white", va="center", ha="left", zorder=0)

    def text(self, ax):
        for day in self.days_7:
            date = self.topo.dates[day - 1]
            next_date = date + timedelta(days=1)
            setting = ax.text(self.sun_set[day - 1, 1] - 0.2, day + 0.5, str(date.day),
                    va="center", ha="center", fontproperties=self.font, zorder=30)
            rising = ax.text(self.sun_rise[day - 1, 1] + 0.2, day + 0.5, str(next_date.day),
                    va="center", ha="center", fontproperties=self.font, zorder=30)
            self.date_labels.extend(((setting, day, -1), (rising, day, 1)))
        for hour in np.arange(self.min_set_hour + 1, self.max_rise_hour):
            label = hour + 12 if hour <= 0 else hour
            for y, alignment in ((1 - 0.75, "bottom"), (self.days + 0.75, "top")):
                ax.text(hour, y, str(label), va=alignment, ha="center",
                        fontproperties=self.font, zorder=30)

    def grid(self, ax):
        xs = np.arange(self.min_set_hour, self.max_rise_hour, self.y_scale)
        xs = xs[(xs < -0.01) | (xs > 0.01)]
        for day in self.days_7:
            ax.scatter(xs, np.full(xs.shape, day), c=[[0.75] * 3], zorder=-1,
                       s=0.5, edgecolors="none", lw=0)
        ys = np.arange(1, self.days + 1)
        vertical = np.arange(self.min_set_hour, self.max_rise_hour, 0.5)
        for x in vertical[vertical != 0]:
            ax.scatter(np.full(ys.shape, x), ys, c=[[0.75] * 3], zorder=-1,
                       s=0.5, edgecolors="none", lw=0)
        for name in ("Dusk", "Dawn"):
            events = self.event("twilight", name)
            events[~_event_valid(events[:, 1])] = np.nan
            ax.plot(events[:, 1], events[:, 0], zorder=1, lw=0.5,
                    c=[0.75] * 3, ls=(0, (8, 5)))

    def plot_events(self, ax, event, bodies, colors, zorder, linewidth):
        from matplotlib import rcParams
        from .labels import CurveLabel, split_visible_curve
        from .stars import STARS

        names = {"Mercury": "水星", "Venus": "金星", "Mars": "火星",
                 "Jupiter": "木星", "Saturn": "土星"}
        names.update({star.body.name: star.name for star in STARS})
        suffix = {"rise": "出", "set": "没", "transit": "上中天"}[event]
        # Matplotlib normally multiplies dash lengths by linewidth. Normalize
        # here to retain the reference's physical spacing, including thin stars.
        dash_scale = linewidth if rcParams["lines.scale_dashes"] else 1
        linestyle = (0, tuple(length / dash_scale for length in EVENT_DASHES_POINTS[event]))
        for body, color in zip(bodies, colors):
            events = self.event(event, body)
            events[~self.night_mask(events, bleed=0.2)] = np.nan
            ax.plot(events[:, 1], events[:, 0], zorder=zorder, lw=linewidth, c=color,
                    linestyle=linestyle)
            if self.options.text:
                for segment in split_visible_curve(events[:, ::-1]):
                    inner_planet = body in ("Mercury", "Venus") and event in ("rise", "set")
                    side = (-1 if event == "rise" else 1) if inner_planet else 0
                    self.curve_labels.append(CurveLabel(
                        names[body] + suffix, segment, color,
                        vertical=body == "Venus", horizontal_side=side,
                        curve_linewidth=linewidth,
                    ))

    def lines(self, ax):
        planets = ("Mercury", "Venus", "Mars", "Jupiter", "Saturn")
        colors = ("aqua", "darkorange", "firebrick", "mediumorchid", "gold")
        self.plot_events(ax, "rise", planets, colors, 5, 0.8)
        self.plot_events(ax, "transit", planets[2:], colors[2:], 6, 0.8)
        self.plot_events(ax, "set", planets, colors, 7, 0.8)
        if self.options.stars:
            self.plot_events(ax, "rise", STAR_NAMES, ["white"] * len(STAR_NAMES), 4, 0.5)
            self.plot_events(ax, "transit", STAR_NAMES, ["white"] * len(STAR_NAMES), 4, 0.5)

    def annotations(self, ax):
        """Add month and event lettering without changing any plotted samples."""
        from .labels import CurveLabel, CurveLabels, split_visible_curve

        font_path = ROOT / "fonts" / "SourceHanSansSC-Regular.otf"
        month_font_path = ROOT / "fonts" / "SourceHanSansSC-Medium.otf"
        for path in (font_path, month_font_path):
            if not path.is_file():
                raise FileNotFoundError(f"标注字体不存在：{path}")
        months = ("一月", "二月", "三月", "四月", "五月", "六月",
                  "七月", "八月", "九月", "十月", "十一月", "十二月")
        labels = []
        if self.options.frame:
            for events, direction in ((self.sun_set, -1), (self.sun_rise, 1)):
                # Use the same dates as the existing left/right day numbers.
                dates = [date + timedelta(days=int(direction == 1)) for date in self.topo.dates]
                for month in range(1, 13):
                    mask = np.array([date.year == self.topo.year and date.month == month
                                     for date in dates])
                    points = events[mask][:, ::-1].copy()
                    points[:, 0] += direction * 0.57
                    labels.append(CurveLabel(months[month - 1], points, "black",
                                             vertical=True, repeat=False, fixed_position=True,
                                             month=True))
        def add_end_labels(text, points, color, offset, avoid_meridian=True):
            # Reserve the first/last 45 rows, so a missing endpoint never moves
            # its label to the middle of the year or bridges missing events.
            for end, window in (("top", points[:45]), ("bottom", points[-45:])):
                segments = split_visible_curve(window)
                if segments:
                    segment = segments[0] if end == "top" else segments[-1]
                    normal_offset = offset
                    if normal_offset is None:
                        normal_offset = 6.5 if np.median(segment[:, 0]) >= 0 else -6.5
                    labels.append(CurveLabel(
                        text, segment, color, vertical=True, repeat=False,
                        end=end, normal_offset=normal_offset, avoid_meridian=avoid_meridian,
                    ))

        if self.options.eot:
            points = np.column_stack((self.dataset.arrays["EOT"], self.sun_set[:, 0]))
            # Hug the curve on the side away from the central RA spine: right
            # at the top of the 2026 reference, left at its bottom.
            add_end_labels("均时差", points, "white", None, avoid_meridian=False)
        if self.options.grid:
            for name, text, offset in (("Dusk", "天文昏影终", 5), ("Dawn", "天文晨光始", -5)):
                events = self.event("twilight", name)
                events[~_event_valid(events[:, 1])] = np.nan
                add_end_labels(text, events[:, ::-1], "0.75", offset)
        labels.extend(self.curve_labels)
        # Only visible symbols are obstacles, including nearest-event markers
        # that have moved to another night. Do not reserve discarded symbols.
        obstacles = self.moon_positions if self.options.moon else []
        bounds = np.column_stack((self.sun_set[:, 0], self.sun_set[:, 1], self.sun_rise[:, 1]))
        ax.add_artist(CurveLabels(labels, font_path, bounds, obstacles,
                                  month_font_path=month_font_path))

    def moon_symbol(self, ax, time, phase, day, hour, *, quarter=False):
        # Reuse the actual symbol's patches, including its enlargement. Do not
        # recompute lunar phases/positions for date layout or move the symbol.
        start = len(ax.patches) if quarter and phase == 180 else None
        self.moon_positions.append((hour, day))
        draw_moon_symbol(ax=ax, time=time, lms=self.lms, moon_phase=phase,
                         day=day, hour=hour, obs=self.obs,
                         y_zoom_factor=1 / self.y_scale,
                         zoom_factor=0.1 if quarter else 0.075, mares=quarter)
        if start is not None:
            self.full_moon_patches.append(tuple(ax.patches[start:]))

    def _phase_matched_moon_event(self, events, time, direction, target_phase):
        """Compare the lunar phases of the adjacent before/after events.

        ``events`` contains (one-based night row, midnight-centred hour).
        ``time`` is the exact quarter's UTC instant; ``target_phase`` is its
        0/90/180/270 degree geocentric MoonPhase angle. Only the immediately
        preceding/following valid event of ``direction`` is compared, before
        display filtering. The smaller circular angular difference wins;
        equal differences prefer the earlier event, regardless of time gap.

        At year boundaries, look up a missing neighbour outside the cached
        year with the same engine and observer, searching up to one year's
        duration. An outside winner has no plotted row; no in-year substitute
        is chosen. Return (UTC Time, row) or (None, None) for no valid candidates.
        """
        valid = events[_event_valid(events[:, 1])]
        if not len(valid):
            return None, None
        times = (self.topo.origin_utc.ut + valid[:, 0] - 1
                 + (valid[:, 1] + 12) / 24)
        candidates = {}
        for before, boundary, sign in ((True, self.topo.origin_utc, -1),
                                       (False, self.topo.end_utc, 1)):
            indices = np.flatnonzero(times <= time.ut if before else times >= time.ut)
            if len(indices):
                index = indices[np.argmax(times[indices]) if before else np.argmin(times[indices])]
                event_time, row = astro.Time(float(times[index])), valid[index]
            else:
                event_time = astro.SearchRiseSet(astro.Body.Moon, self.obs, direction,
                                                 boundary, sign * self.days)
                row = None
            if event_time is not None:
                candidates[event_time.ut] = (event_time, row)

        def phase_difference(candidate):
            event_time, _ = candidate
            angle = astro.MoonPhase(event_time)
            difference = abs((angle - target_phase + 180) % 360 - 180)
            return difference, event_time.ut

        return min(candidates.values(), key=phase_difference) if candidates else (None, None)

    def moons(self, ax):
        """Place all quarter symbols by phase at adjacent rise/set events.

        Ordinary symbols are suppressed on the preceding two local dates for
        full-moon rise/new-moon set, and the following two local dates for
        full-moon set/new-moon rise. Other event types are unaffected. Quarter
        times, event times, symbol geometry and visibility margins are retained.
        """
        rise, setting = self.event("rise", "Moon"), self.event("set", "Moon")
        all_events = (rise, setting)
        visible_quarters = [self.night_mask(events, bleed=2) for events in all_events]
        # Midnight is x=0 within a noon-to-noon row: nonnegative hours belong
        # to the next local calendar date. Compare dates, not elapsed 48 hours.
        event_dates = [self.topo.dates[0].toordinal() + events[:, 0] - 1
                       + (events[:, 1] >= 0) for events in all_events]
        special_events = set()
        suppressed_events = set()
        quarter = astro.SearchMoonQuarter(self.topo.origin_utc)
        while quarter.time.ut < self.topo.end_utc.ut:
            for event_index, (events, direction) in enumerate(zip(
                    all_events, (astro.Direction.Rise, astro.Direction.Set))):
                event_time, row = self._phase_matched_moon_event(
                    events, quarter.time, direction, 90 * quarter.quarter)
                if event_time is not None and quarter.quarter % 2 == 0:
                    local_date = event_time.Utc().astimezone(self.topo.timezone).date()
                    date_deltas = event_dates[event_index] - local_date.toordinal()
                    before = (quarter.quarter == 2) == (event_index == 0)
                    dates_to_hide = (-2, -1) if before else (1, 2)
                    hide = _event_valid(events[:, 1]) & np.isin(date_deltas, dates_to_hide)
                    suppressed_events.update((event_index, int(day)) for day in events[hide, 0])
                if row is None:
                    continue
                event_day, hour = int(row[0]), float(row[1])
                special_events.add((event_index, event_day))
                if visible_quarters[event_index][event_day - 1]:
                    self.moon_symbol(ax, quarter.time, 90 * quarter.quarter,
                                     event_day, hour, quarter=True)
            quarter = astro.NextMoonQuarter(quarter)
        for event_index, events in enumerate(all_events):
            for day, hour in events[self.night_mask(events)]:
                key = (event_index, int(day))
                if key in special_events or key in suppressed_events:
                    continue
                time = self.topo.time_from_plot(int(day), float(hour))
                self.moon_symbol(ax, time, astro.MoonPhase(time), day, hour)

    def draw(self, fig):
        ax = fig.subplots()
        options = self.options
        if options.frame:
            self.frame(ax)
        if options.meridian_ra:
            self.meridian_ra(ax)
        if options.eot:
            ax.plot(self.dataset.arrays["EOT"], np.arange(1, self.days + 1),
                    color="white", lw=0.5, zorder=1)
            ax.plot([0, 0], [1, self.days], color="white", lw=0.5, zorder=1)
        if options.text:
            self.text(ax)
        if options.background:
            ax.imshow(getbackground(self.dataset.arrays["sun_alts"]), zorder=-2,
                      extent=[-12, 12, self.days + 0.5, 0.5])
        if options.grid:
            self.grid(ax)
        if options.lines:
            self.lines(ax)
        if options.moon:
            self.moons(ax)
        ax.set_xlim(-12, 12)
        ax.set_ylim(self.days, 1)
        ax.axis("off")
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)
        ax.set_aspect(self.y_scale)
        if options.text:
            from .date_layout import DateLayout

            bounds = np.column_stack((self.sun_set[:, 0], self.sun_set[:, 1], self.sun_rise[:, 1]))
            ax.add_artist(DateLayout(self.date_labels, self.calendar_rules,
                                     self.full_moon_patches, bounds, self.month_width))
            self.annotations(ax)
        if not options.background:
            fig.set_facecolor("grey")


def render_almanac(dataset, output_root, render_options=None):
    """Write PNG and PDF using the dataset's sole observation/time configuration.

    Returns ``{'png': Path, 'pdf': Path}``. Existing outputs raise
    ``FileExistsError`` unless ``RenderOptions.overwrite`` is explicitly true.
    Missing solar events raise ``ValueError`` before any output is created.
    """
    options = render_options if render_options is not None else RenderOptions()
    _validate_render_dataset(dataset)
    output_dir = Path(output_root) / dataset.run_dir.name
    paths = {extension: output_dir / f"almanac.{extension}" for extension in ("png", "pdf")}
    if not options.overwrite:
        for path in paths.values():
            if path.exists():
                raise FileExistsError(f"Output exists; use --overwrite to replace it: {path}")

    # Import and configure plotting only when actually rendering, never on import.
    from matplotlib import rc_context
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    output_dir.mkdir(parents=True, exist_ok=True)
    with rc_context({"font.size": 9}):
        figure = Figure(figsize=options.figsize, dpi=options.dpi)
        FigureCanvasAgg(figure)
        try:
            _AlmanacPlot(dataset, options).draw(figure)
            for path in paths.values():
                figure.savefig(path, dpi=options.dpi)
        finally:
            # No pyplot manager is created; clearing releases artists and arrays.
            figure.clear()
    return paths
