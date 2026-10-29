"""Small physical checks and controlled two-row computation isolation tests."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from almanac._vendor import astronomy
from almanac.calculations import (
    TABLE_COLUMNS, compute_ephemeris, compute_eot, compute_sun_alts,
)
from almanac.stars import STARS, STAR_NAMES, register_stars
from almanac.topo import TopoContext


class SmallWindow:
    """Expose two real nights to avoid running annual event searches in tests."""
    def __init__(self, topo):
        self.topo = topo
        self.days_in_year = 2
        self.year = topo.year

    @property
    def observer(self):
        return self.topo.observer

    def noon(self, day):
        return self.topo.noon(day)


class CalculationTests(unittest.TestCase):
    def test_eot_respects_requested_leap_year_and_existing_sign(self):
        topo = TopoContext(2024,480,40,40)
        values = compute_eot(topo)
        self.assertEqual(values.shape, (366,))
        for day in [1,59,60,61,366]:
            old_time = astronomy.Time.Make(2024,1,day,16,0,0)
            expected = 12-astronomy.HourAngle(astronomy.Body.Sun,old_time,topo.observer)
            self.assertLess(abs(values[day-1]-expected)*3600, 0.001)
        later = compute_eot(TopoContext(2027,480,40,40))
        self.assertEqual(later.shape, (365,))
        np.testing.assert_array_equal(values,compute_eot(topo))

    def test_catalogue_ids_and_negative_declination_correction(self):
        self.assertEqual(STAR_NAMES,tuple(f"Star{i}" for i in range(1,6)))
        self.assertEqual(STARS[3].dec,-(29+37/60))
        self.assertEqual(STARS[4].dec,-(16+43/60))
        register_stars()
        observer = astronomy.Observer(40,116,40)
        noon = astronomy.Time.Make(2026,1,1,12-116/15,0,0)
        corrected = astronomy.SearchRiseSet(
            astronomy.Body.Star5,observer,astronomy.Direction.Rise,noon,1)
        astronomy.DefineStar(astronomy.Body.Star5,6+45/60,-16+43/60,1000)
        old = astronomy.SearchRiseSet(
            astronomy.Body.Star5,observer,astronomy.Direction.Rise,noon,1)
        register_stars()
        self.assertGreater((corrected.ut-old.ut)*1440,5)
        self.assertLess((corrected.ut-old.ut)*1440,6)

    def test_two_night_calculation_matches_direct_search(self):
        topo = TopoContext(2027,480,40,0)
        window = SmallWindow(topo)
        tables = compute_ephemeris(window)
        for key, columns in TABLE_COLUMNS.items():
            self.assertEqual(tables[key].colnames,list(columns))
            np.testing.assert_array_equal(tables[key]["Day"],[1.,2.])
        for day in [1,2]:
            noon = topo.noon(day)
            rise = astronomy.SearchRiseSet(
                astronomy.Body.Sun,topo.observer,astronomy.Direction.Rise,noon,1)
            self.assertEqual(tables["rise"]["Sun"][day-1],(rise.ut-noon.ut)*24-12)

    def test_consecutive_sites_and_poisoned_star_slots_are_isolated(self):
        first = SmallWindow(TopoContext(2027,480,40,0))
        second = SmallWindow(TopoContext(2024,-300,-30,100))
        initial = compute_ephemeris(first)
        different = compute_ephemeris(second)
        astronomy.DefineStar(astronomy.Body.Star5,0,0,1)
        repeated = compute_ephemeris(first)
        for key in initial:
            for column in initial[key].colnames:
                np.testing.assert_array_equal(initial[key][column],repeated[key][column])
        self.assertFalse(np.array_equal(initial["rise"]["Sun"],different["rise"]["Sun"]))

    def test_missing_events_and_transit_window_are_preserved(self):
        window = SmallWindow(TopoContext(2027,480,40,0))
        def next_noon(body, observer, hour_angle, start_time, direction):
            return SimpleNamespace(time=astronomy.Time(start_time.ut+1))
        with patch.object(astronomy,"SearchRiseSet",return_value=None), \
             patch.object(astronomy,"SearchAltitude",return_value=None), \
             patch.object(astronomy,"SearchHourAngle",side_effect=next_noon):
            tables = compute_ephemeris(window)
        for table in tables.values():
            for column in table.colnames[1:]:
                self.assertTrue(np.all(table[column] == -999))

    def test_solar_altitudes_receive_context_times_and_original_flags(self):
        topo = TopoContext(2024,330,20,10)
        # Test one day via the same context-generated sample API; no full-year
        # astronomy run is needed to establish the input and reshape contract.
        window = SimpleNamespace(days_in_year=1,observer=topo.observer,
                                 time_at_minute=topo.time_at_minute)
        instants = []
        def equator(body, instant, observer, ofdate, aberration):
            self.assertEqual(body,astronomy.Body.Sun)
            self.assertEqual(observer.longitude,82.5)
            self.assertTrue(ofdate)
            self.assertFalse(aberration)
            instants.append(instant.ut)
            return SimpleNamespace(ra=1,dec=2)
        with patch.object(astronomy,"Equator",side_effect=equator), \
             patch.object(astronomy,"Horizon",return_value=SimpleNamespace(altitude=3.5)) as horizon:
            values = compute_sun_alts(window)
        self.assertEqual(values.shape,(1,1440))
        np.testing.assert_allclose(values, 3.5, atol=1e-12, rtol=0)
        self.assertEqual(len(instants), 481)
        self.assertEqual(instants[1], topo.time_at_minute(3).ut)
        self.assertEqual(instants[0],topo.origin_utc.ut)
        self.assertEqual(instants[-1],topo.time_at_minute(1439).ut)
        self.assertEqual(horizon.call_args.args[-1],astronomy.Refraction.Airless)

    def test_three_minute_sampler_against_exact_boundary_nights(self):
        # 0.01 arcsec is the new interpolation error budget, not a relaxation
        # of unchanged event-time regression tolerances. It is well below
        # the engine's arcminute design target and background display needs.
        from datetime import date

        cases = [(2028, 480, 40, 2, 29), (2028, 840, 40, 12, 31),
                 (2026, -330, -40, 1, 1), (2028, 345, 23.44, 6, 21),
                 (2028, 345, 0, 3, 20), (2028, 480, 80, 12, 21)]
        for year, offset, latitude, month, day in cases:
            with self.subTest(year=year, latitude=latitude, month=month, day=day):
                topo = TopoContext(year, offset, latitude)
                start = (date(year, month, day) - date(year, 1, 1)).days * 1440
                window = SimpleNamespace(days_in_year=1, observer=topo.observer,
                                         time_at_minute=lambda i: topo.time_at_minute(start + i))
                approximate = compute_sun_alts(window).ravel()
                exact = []
                for i in range(1440):
                    time = window.time_at_minute(i)
                    eq = astronomy.Equator(astronomy.Body.Sun, time, window.observer, True, False)
                    exact.append(astronomy.Horizon(time, window.observer, eq.ra, eq.dec,
                                                   astronomy.Refraction.Airless).altitude)
                exact = np.array(exact)
                self.assertLess(np.max(np.abs(approximate - exact)) * 3600, 0.01)
                np.testing.assert_array_equal(approximate[::3], exact[::3])
                self.assertEqual(approximate[-1], exact[-1])
                extreme = np.abs(approximate) > 85
                np.testing.assert_array_equal(approximate[extreme], exact[extreme])


if __name__ == "__main__":
    unittest.main()
