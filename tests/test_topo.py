"""Shared calendar/observer contract tests; no ephemeris files are generated."""

from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta, timezone
import math
import os
import time as system_time
import unittest
from unittest.mock import patch

from almanac._vendor.astronomy import Time
from almanac.topo import TopoContext


class TopoContextTests(unittest.TestCase):
    def test_gregorian_year_lengths_and_date_labels(self):
        for year, days in [(1900, 365), (2000, 366), (2027, 365),
                           (2028, 366), (2100, 365)]:
            with self.subTest(year=year):
                topo = TopoContext(year, 480, 40)
                self.assertEqual(topo.days_in_year, days)
                self.assertEqual(len(topo.dates), days)
                self.assertEqual(topo.dates[0], date(year, 1, 1))
                self.assertEqual(topo.dates[-1], date(year, 12, 31))
                self.assertEqual(topo.dates[-1] + timedelta(days=1), date(year+1, 1, 1))
                self.assertEqual(topo.dates[59], date(year, 2, 29) if days == 366 else date(year, 3, 1))
                self.assertEqual(topo.end_utc.ut - topo.origin_utc.ut, days)

    def test_offset_meridian_and_utc_origin(self):
        cases = [
            (-720, -180, datetime(2027, 1, 2, 0, tzinfo=timezone.utc)),
            (-210, -52.5, datetime(2027, 1, 1, 15, 30, tzinfo=timezone.utc)),
            (0, 0, datetime(2027, 1, 1, 12, tzinfo=timezone.utc)),
            (330, 82.5, datetime(2027, 1, 1, 6, 30, tzinfo=timezone.utc)),
            (345, 86.25, datetime(2027, 1, 1, 6, 15, tzinfo=timezone.utc)),
            (480, 120, datetime(2027, 1, 1, 4, tzinfo=timezone.utc)),
            (840, -150, datetime(2026, 12, 31, 22, tzinfo=timezone.utc)),
        ]
        for offset, longitude, expected in cases:
            with self.subTest(offset=offset):
                topo = TopoContext(2027, offset, 40)
                self.assertEqual(topo.longitude_deg, longitude)
                self.assertEqual(topo.utc_offset_minutes, offset)
                self.assertEqual(topo.origin_utc.Utc(), expected)
                self.assertEqual(topo.origin_utc.Utc().astimezone(topo.timezone),
                                 datetime(2027, 1, 1, 12, tzinfo=topo.timezone))

    def test_noon_midnight_and_minutes_share_the_same_origin(self):
        topo = TopoContext(2028, 480, 40)
        self.assertEqual(topo.noon(1).ut, topo.origin_utc.ut)
        self.assertEqual(topo.midnight(1).Utc(), datetime(2028, 1, 1, 16, tzinfo=timezone.utc))
        for row in [1, 59, 60, 61, 366]:
            self.assertEqual(topo.noon(row).ut, topo.time_at_minute((row-1)*1440).ut)
            self.assertEqual(topo.midnight(row).ut, topo.time_at_minute((row-1)*1440+720).ut)
        last = topo.time_at_minute(366*1440-1).Utc().astimezone(topo.timezone)
        self.assertEqual(last, datetime(2029, 1, 1, 11, 59, tzinfo=topo.timezone))

    def test_plot_roundtrip(self):
        for year in [1900, 2000, 2027, 2028, 2044, 2045, 2100]:
            for offset in [-720, -210, 0, 330, 345, 480, 840]:
                topo = TopoContext(year, offset, -24, 15)
                for day in [1, 59, 60, topo.days_in_year]:
                    for x in [-12, -6, 0, 6, 12-1/3600]:
                        with self.subTest(year=year, offset=offset, day=day, x=x):
                            actual_day, actual_x = topo.plot_coordinates(topo.time_from_plot(day, x))
                            self.assertEqual(actual_day, day)
                            # 36 microseconds: above double-UT rounding over
                            # the tested 1900..2100 range, below event precision.
                            self.assertAlmostEqual(actual_x, x, delta=1e-8)

    def test_half_open_boundaries_without_epsilon_snapping(self):
        # These years span different binary magnitudes of the UT day count.
        for year in [1955, 1999, 2000, 2001, 2044, 2045, 9998]:
            topo = TopoContext(year, 345, 40)
            for day in range(2, topo.days_in_year+1):
                boundary = topo.noon(day).ut
                before = Time(math.nextafter(boundary, -math.inf))
                after = Time(math.nextafter(boundary, math.inf))
                with self.subTest(year=year, day=day):
                    self.assertEqual(topo.plot_coordinates(before)[0], day-1)
                    self.assertEqual(topo.plot_coordinates(Time(boundary)), (day, -12))
                    self.assertEqual(topo.plot_coordinates(after)[0], day)
            self.assertEqual(topo.plot_coordinates(topo.origin_utc), (1, -12))
            self.assertEqual(topo.plot_coordinates(Time(math.nextafter(topo.end_utc.ut, -math.inf)))[0],
                             topo.days_in_year)
            with self.assertRaises(ValueError):
                topo.plot_coordinates(Time(math.nextafter(topo.origin_utc.ut, -math.inf)))
            with self.assertRaises(ValueError):
                topo.plot_coordinates(topo.end_utc)

    def test_frozen_inputs_and_independent_vendor_objects(self):
        topo = TopoContext(2027, 480, 40, 0)
        with self.assertRaises(FrozenInstanceError):
            topo.year = 2028
        observer = topo.observer
        observer.longitude = 0
        observer.height = 999
        self.assertEqual(topo.observer.longitude, 120)
        self.assertEqual(topo.observer.height, 0)
        origin = topo.origin_utc
        original_ut = origin.ut
        origin.ut += 10
        self.assertEqual(topo.origin_utc.ut, original_ut)

    def test_exact_serialization_contract(self):
        topo = TopoContext(2027, 345, -24.5, -10)
        self.assertEqual(set(topo.to_dict()),
                         {'year', 'utc_offset_minutes', 'latitude_deg', 'elevation_m'})
        self.assertEqual(TopoContext.from_dict(topo.to_dict()), topo)
        for key in topo.to_dict():
            partial = topo.to_dict()
            del partial[key]
            with self.assertRaises(ValueError):
                TopoContext.from_dict(partial)
        extra = dict(topo.to_dict(), longitude_deg=86.25)
        with self.assertRaises(ValueError):
            TopoContext.from_dict(extra)
        with self.assertRaises(TypeError):
            TopoContext.from_dict([])

    def test_invalid_configuration(self):
        valid = dict(year=2027, utc_offset_minutes=480, latitude_deg=40, elevation_m=0)
        invalid = {
            'year': [True, 2027.0, '2027', 1, 9999],
            'utc_offset_minutes': [True, 480.0, '480', -721, 841],
            'latitude_deg': [True, '40', -91, 91, math.nan, math.inf],
            'elevation_m': [True, '0', math.nan, -math.inf],
        }
        for key, values in invalid.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    with self.assertRaises((TypeError, ValueError)):
                        TopoContext(**dict(valid, **{key: value}))
        # Both supported extreme years remain safe through UTC conversion.
        for year in [2, 9998]:
            for offset in [-720, 840]:
                TopoContext(year, offset, 90).origin_utc.Utc()
                TopoContext(year, offset, -90).end_utc.Utc()

    def test_invalid_indices_and_coordinates(self):
        topo = TopoContext(2027, 480, 40)
        for day in [True, 0, 366, 1.0, '1']:
            for function in [topo.noon, topo.midnight]:
                with self.assertRaises((TypeError, ValueError)):
                    function(day)
        for index in [True, -1, 365*1440, 0.0]:
            with self.assertRaises((TypeError, ValueError)):
                topo.time_at_minute(index)
        for x in [True, -12.1, 12, math.nan, math.inf]:
            with self.assertRaises((TypeError, ValueError)):
                topo.time_from_plot(1, x)
        with self.assertRaises(TypeError):
            topo.plot_coordinates(datetime.now(timezone.utc))

    def test_host_timezone_does_not_change_result(self):
        reference = TopoContext(2027, 840, 40).origin_utc.ut
        try:
            for host_zone in ['UTC0', 'EST5EDT', 'JST-9']:
                with patch.dict(os.environ, {'TZ': host_zone}):
                    if hasattr(system_time, 'tzset'):
                        system_time.tzset()
                    self.assertEqual(TopoContext(2027, 840, 40).origin_utc.ut, reference)
        finally:
            if hasattr(system_time, 'tzset'):
                system_time.tzset()


if __name__ == '__main__':
    unittest.main()
