"""Astronomical calculations with explicit context and no file side effects."""
from collections.abc import Callable

import numpy as np
from astropy.table import Table

from ._vendor import astronomy
from .stars import STAR_NAMES, register_stars
from .topo import TopoContext

MISSING_EVENT = -999.0
SOLAR_SAMPLE_MINUTES = 3
SOLAR_EXACT_EXTREME_DEGREES = 85.0
RISE_SET_BODIES = (
    "Sun", "Moon", "Mercury", "Venus", "Mars", "Jupiter", "Saturn",
    *STAR_NAMES,
)
TRANSIT_BODIES = ("Mars", "Jupiter", "Saturn", *STAR_NAMES)
TABLE_COLUMNS = {
    "rise": ("Day", *RISE_SET_BODIES),
    "set": ("Day", *RISE_SET_BODIES),
    "transit": ("Day", *TRANSIT_BODIES),
    "twilight": ("Day", "Dusk", "Dawn"),
}


def _plot_hour(event, noon):
    if event is None:
        return MISSING_EVENT
    return (event.ut - noon.ut) * 24 - 12


def compute_ephemeris(topo: TopoContext) -> dict[str, Table]:
    """Return one row per local noon-to-noon night, with midnight at hour zero.

    The returned keys are rise, set, transit and twilight. Day is one-based;
    event columns are hours relative to that night's midnight, or -999 when
    no event occurs. Search intervals, refraction and twilight altitude retain
    the existing implementation's conventions.
    """
    register_stars()
    observer = topo.observer
    noons = [topo.noon(day) for day in range(1, topo.days_in_year + 1)]
    tables = {}
    for key in TABLE_COLUMNS:
        table = Table(meta={"info": f"{key} times for year {topo.year}"})
        table["Day"] = np.arange(topo.days_in_year, dtype=float) + 1
        tables[key] = table

    for name in RISE_SET_BODIES:
        body = astronomy.BodyCode(name)
        for key, direction in (
            ("set", astronomy.Direction.Set),
            ("rise", astronomy.Direction.Rise),
        ):
            tables[key][name] = [
                _plot_hour(
                    astronomy.SearchRiseSet(body, observer, direction, noon, 1),
                    noon,
                )
                for noon in noons
            ]

    for name in TRANSIT_BODIES:
        values = []
        for noon in noons:
            transit = astronomy.SearchHourAngle(
                astronomy.BodyCode(name), observer, 0.0, noon, direction=1
            )
            values.append(
                _plot_hour(transit.time, noon)
                if transit.time.ut - noon.ut < 1 else MISSING_EVENT
            )
        tables["transit"][name] = values

    for name, direction in (
        ("Dusk", astronomy.Direction.Set), ("Dawn", astronomy.Direction.Rise)
    ):
        tables["twilight"][name] = [
            _plot_hour(
                astronomy.SearchAltitude(
                    astronomy.Body.Sun, observer, direction, noon, 1, -18.0
                ),
                noon,
            )
            for noon in noons
        ]
    return tables


def compute_eot(topo: TopoContext) -> np.ndarray:
    """Return the existing 12-minus-solar-hour-angle curve, in hours.

    Each value is sampled at the midnight within its noon-to-noon night.
    The existing sign convention is intentionally retained.
    """
    observer = topo.observer
    return np.asarray([
        12 - astronomy.HourAngle(astronomy.Body.Sun, topo.midnight(day), observer)
        for day in range(1, topo.days_in_year + 1)
    ])


def compute_sun_alts(
    topo: TopoContext, progress: Callable[[str], None] | None = None
) -> np.ndarray:
    """Return airless solar altitude degrees with shape (days, 1440).

    Exact engine samples start at local January 1 noon, every three minutes,
    with the final minute included. A cubic spline of sine altitude restores
    the existing one-minute grid across nights; absolute altitudes above 85
    degrees are evaluated exactly to avoid amplification near zenith/nadir.
    Exact sample values, equator of date, aberration and airless flags are
    preserved. The returned array is an approximation between exact samples.
    """
    from scipy.interpolate import CubicSpline

    observer = topo.observer
    samples = topo.days_in_year * 1440
    indices = np.unique(np.r_[np.arange(0, samples, SOLAR_SAMPLE_MINUTES), samples - 1])

    def exact_altitude(index):
        instant = topo.time_at_minute(int(index))
        equatorial = astronomy.Equator(
            astronomy.Body.Sun, instant, observer, True, False
        )
        return astronomy.Horizon(
            instant, observer, equatorial.ra, equatorial.dec,
            astronomy.Refraction.Airless,
        ).altitude

    exact = np.empty(len(indices), dtype=float)
    for sample, index in enumerate(indices):
        if progress is not None and sample % 50000 == 0:
            progress(f"太阳高度：{sample}/{len(indices)} 个采样（每3分钟）")
        exact[sample] = exact_altitude(index)
    spline = CubicSpline(indices, np.sin(np.deg2rad(exact)))
    altitudes = np.rad2deg(np.arcsin(np.clip(spline(np.arange(samples)), -1, 1)))
    altitudes[indices] = exact
    extremes = np.setdiff1d(
        np.flatnonzero(np.abs(altitudes) > SOLAR_EXACT_EXTREME_DEGREES),
        indices, assume_unique=True,
    )
    if progress is not None and len(extremes):
        progress(f"太阳高度：精确回算 {len(extremes)} 个天顶/天底附近采样")
    for index in extremes:
        altitudes[index] = exact_altitude(index)
    if progress is not None:
        progress(f"太阳高度：完成 {len(indices)} 个采样，恢复 {samples} 个分钟网格点")
    return altitudes.reshape((topo.days_in_year, 1440))
