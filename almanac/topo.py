"""Immutable observing configuration and the almanac's shared time mapping.

Rows represent local noon through the following local noon. A row's x=0 is
the intervening midnight, not the midnight at the start of its calendar date.
The fixed UTC offset determines the observer's nominal meridian; there is no
daylight-saving-time or geographic time-zone lookup.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
import math
from numbers import Integral, Real

from ._vendor.astronomy import Observer, Time


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not {type(value).__name__}")
    return int(value)


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


@dataclass(frozen=True)
class TopoContext:
    """One year's fixed-offset observing context.

    Parameters are the local Gregorian year (2..9998), UTC offset in whole
    minutes (-720..840), latitude in degrees north (-90..90), and elevation
    above sea level in metres. Longitude is derived from the offset and must
    not be supplied separately. The original offset is retained even when its
    meridian is normalized across the date line.

    Public row arguments are one-based; minute sample indices are zero-based.
    Astronomy Engine ``Time`` and ``Observer`` values are returned as fresh
    objects because those vendor types are mutable.
    """

    year: int
    utc_offset_minutes: int
    latitude_deg: float
    elevation_m: float = 0
    _origin_ut: float = field(init=False, repr=False, compare=False)
    _dates: tuple[date, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        year = _integer(self.year, "year")
        offset = _integer(self.utc_offset_minutes, "utc_offset_minutes")
        latitude = _finite_number(self.latitude_deg, "latitude_deg")
        elevation = _finite_number(self.elevation_m, "elevation_m")
        if not 2 <= year <= 9998:
            raise ValueError("year must be between 2 and 9998")
        if not -720 <= offset <= 840:
            raise ValueError("utc_offset_minutes must be between -720 and 840")
        if not -90 <= latitude <= 90:
            raise ValueError("latitude_deg must be between -90 and 90")
        object.__setattr__(self, "year", year)
        object.__setattr__(self, "utc_offset_minutes", offset)
        object.__setattr__(self, "latitude_deg", latitude)
        object.__setattr__(self, "elevation_m", elevation)

        first = date(year, 1, 1)
        count = (date(year + 1, 1, 1) - first).days
        object.__setattr__(self, "_dates", tuple(
            first + timedelta(days=index) for index in range(count)
        ))
        local_noon = datetime(year, 1, 1, 12, tzinfo=self.timezone)
        utc = local_noon.astimezone(timezone.utc)
        # Calendar normalization belongs to datetime, not undocumented
        # overflow of Time.Make's day/hour/minute parameters.
        origin = Time.Make(utc.year, utc.month, utc.day, utc.hour, utc.minute,
                           utc.second + utc.microsecond / 1_000_000)
        object.__setattr__(self, "_origin_ut", origin.ut)

    @property
    def utc_offset_hours(self) -> float:
        return self.utc_offset_minutes / 60

    @property
    def timezone(self) -> timezone:
        return timezone(timedelta(minutes=self.utc_offset_minutes))

    @property
    def longitude_deg(self) -> float:
        return (self.utc_offset_minutes / 4 + 180) % 360 - 180

    @property
    def observer(self) -> Observer:
        return Observer(self.latitude_deg, self.longitude_deg, self.elevation_m)

    @property
    def dates(self) -> tuple[date, ...]:
        return self._dates

    @property
    def days_in_year(self) -> int:
        return len(self._dates)

    @property
    def origin_utc(self) -> Time:
        """UTC instant corresponding to local January 1 noon."""
        return Time(self._origin_ut)

    @property
    def end_utc(self) -> Time:
        """Exclusive end: local January 1 noon of the next year."""
        return Time(self._origin_ut + self.days_in_year)

    def _row_index(self, day: int) -> int:
        number = _integer(day, "day")
        if not 1 <= number <= self.days_in_year:
            raise ValueError(f"day must be between 1 and {self.days_in_year}")
        return number - 1

    def noon(self, day: int) -> Time:
        """Return the local noon starting one-based row ``day`` as UTC."""
        return Time(self._origin_ut + self._row_index(day))

    def midnight(self, day: int) -> Time:
        """Return the following local midnight in one-based row ``day``."""
        return Time(self._origin_ut + self._row_index(day) + 0.5)

    def time_at_minute(self, index: int) -> Time:
        """UTC time of zero-based annual sample, at one-minute intervals."""
        number = _integer(index, "index")
        if not 0 <= number < self.days_in_year * 1440:
            raise ValueError(f"index must be in [0, {self.days_in_year * 1440})")
        return Time(self._origin_ut + number / 1440)

    def time_from_plot(self, day: int, x: float) -> Time:
        """Convert a one-based row and hour coordinate in [-12, 12) to UTC.

        Like all vendor Time operations, the result is limited by floating
        point resolution; sub-resolution distances from a boundary cannot
        be represented and are not artificially rounded or clamped here.
        """
        index = self._row_index(day)
        hour = _finite_number(x, "x")
        if not -12 <= hour < 12:
            raise ValueError("x must be in [-12, 12)")
        return Time(self._origin_ut + index + (hour + 12) / 24)

    def plot_coordinates(self, time: Time) -> tuple[int, float]:
        """Convert UTC into (one-based row, midnight-centred hour).

        Times outside this year's half-open noon-to-noon window raise
        ValueError. Exact represented noon boundaries belong to the next row.
        """
        if not isinstance(time, Time):
            raise TypeError("time must be an Astronomy Engine Time")
        ut = _finite_number(time.ut, "time.ut")
        if not self._origin_ut <= ut < self._origin_ut + self.days_in_year:
            raise ValueError("time is outside this almanac's noon-to-noon year")
        index = math.floor(ut - self._origin_ut)
        # When the magnitude of UT crosses a binary exponent boundary,
        # subtracting the origin can lie just below an integer even for a
        # canonical noon. Compare actual adjacent noons instead of applying
        # an arbitrary epsilon, which would misclassify real nearby events.
        if ut < self._origin_ut + index:
            index -= 1
        elif ut >= self._origin_ut + (index + 1):
            index += 1
        noon_ut = self._origin_ut + index
        return index + 1, (ut - noon_ut) * 24 - 12

    def to_dict(self) -> dict[str, int | float]:
        """Serialize only the four authoritative configuration inputs."""
        return {
            "year": self.year,
            "utc_offset_minutes": self.utc_offset_minutes,
            "latitude_deg": self.latitude_deg,
            "elevation_m": self.elevation_m,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> "TopoContext":
        """Restore the exact configuration schema, rejecting missing/extra keys."""
        if not isinstance(values, Mapping):
            raise TypeError("TopoContext data must be a mapping")
        fields = {"year", "utc_offset_minutes", "latitude_deg", "elevation_m"}
        if set(values) != fields:
            raise ValueError("TopoContext data must contain exactly " + ", ".join(sorted(fields)))
        return cls(**dict(values))


if __name__ == "__main__":
    example = TopoContext(year=2027, utc_offset_minutes=480, latitude_deg=40)
    print(example.to_dict())
