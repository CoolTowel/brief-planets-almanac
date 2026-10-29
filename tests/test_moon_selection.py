"""Select adjacent lunar events by circular phase error, not elapsed time."""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from almanac import render
from almanac._vendor import astronomy as astro
from tests.test_render import dataset_fixture


class PhaseMatchedMoonEventTests(unittest.TestCase):
    def setUp(self):
        self.data = dataset_fixture(2028)
        self.plot = render._AlmanacPlot(self.data, render.RenderOptions(moon=False))
        boundary_search = patch.object(render.astro, "SearchRiseSet", return_value=None)
        self.search = boundary_search.start()
        self.addCleanup(boundary_search.stop)

    def test_phase_matched_set_can_be_in_previous_night(self):
        topo = self.data.topo
        events = np.array([[59, 6.0], [60, 7.0], [61, 8.0]])
        phase = topo.time_from_plot(60, -11.8)
        with patch.object(render.astro, "MoonPhase", side_effect=[179, 184]):
            time, row = self.plot._phase_matched_moon_event(events, phase, astro.Direction.Set, 180)
        np.testing.assert_array_equal(row, events[0])
        self.assertEqual(time.ut, topo.time_from_plot(59, 6).ut)
        self.assertEqual(topo.dates[int(row[0]) - 1].isoformat(), "2028-02-28")

    def test_missing_values_are_not_candidates(self):
        events = np.array([[90, -999], [91, np.nan], [92, 6.0], [93, np.inf]])
        phase = self.data.topo.midnight(91)
        _, row = self.plot._phase_matched_moon_event(events, phase, astro.Direction.Rise, 90)
        np.testing.assert_array_equal(row, events[2])
        time, row = self.plot._phase_matched_moon_event(events[:2], phase, astro.Direction.Rise, 90)
        self.assertIsNone(time)
        self.assertIsNone(row)

    def test_equal_phase_errors_prefer_earlier_event_even_if_farther(self):
        events = np.array([[20, 6.0], [20, -6.0]])
        with patch.object(render.astro, "MoonPhase", side_effect=[87, 93]):
            _, row = self.plot._phase_matched_moon_event(
                events, self.data.topo.time_from_plot(20, 4), astro.Direction.Rise, 90)
        np.testing.assert_array_equal(row, events[1])

    def test_selection_precedes_visibility_filter(self):
        # At noon the nearer event is the daytime set, not the next visible one.
        events = np.array([[20, -11.0], [20, 6.0]])
        with patch.object(render.astro, "MoonPhase", return_value=0):
            _, row = self.plot._phase_matched_moon_event(
                events, self.data.topo.noon(20), astro.Direction.Set, 0)
        np.testing.assert_array_equal(row, events[0])

    def test_year_boundary_outside_winner_is_not_replaced(self):
        topo = self.data.topo
        cases = [
            (topo.origin_utc.ut + 0.01, [[1, -6]], topo.origin_utc.ut - 0.02, -1),
            (topo.end_utc.ut - 0.01, [[366, 6]], topo.end_utc.ut + 0.02, 1),
        ]
        for phase_ut, events, outside_ut, sign in cases:
            def phase_at_event(time):
                return 180 if time.ut == outside_ut else 185

            with self.subTest(sign=sign), patch.object(
                    render.astro, "SearchRiseSet", return_value=astro.Time(outside_ut)) as search, \
                    patch.object(render.astro, "MoonPhase", side_effect=phase_at_event):
                time, row = self.plot._phase_matched_moon_event(
                    np.array(events, dtype=float), astro.Time(phase_ut), astro.Direction.Set, 180)
                self.assertEqual(time.ut, outside_ut)
                self.assertIsNone(row)
                self.assertEqual(search.call_count, 1)
                self.assertEqual(np.sign(search.call_args.args[4]), sign)
                self.assertEqual(search.call_args.args[1].latitude, 40)

    def test_no_boundary_search_for_interior_phase(self):
        with patch.object(render.astro, "SearchRiseSet") as search:
            self.plot._phase_matched_moon_event(np.array([[49, 6.0], [50, 6.0]]),
                                                self.data.topo.midnight(50), astro.Direction.Rise, 180)
            search.assert_not_called()

    def test_phase_wins_over_time_distance_for_every_quarter(self):
        events = np.array([[20, -8.0], [20, 2.0]])
        for target in (0, 90, 180, 270):
            with self.subTest(target=target), patch.object(
                    render.astro, "MoonPhase", side_effect=[(target - 1) % 360, (target + 2) % 360]):
                _, row = self.plot._phase_matched_moon_event(
                    events, self.data.topo.midnight(20), astro.Direction.Rise, target)
                # Eight hours away, but only one degree away: before wins.
                np.testing.assert_array_equal(row, events[0])

    def test_new_moon_phase_distance_wraps_at_zero(self):
        events = np.array([[20, -2.0], [20, 8.0]])
        for angles, chosen in (([359.5, 1.0], 0), ([358.0, 1.0], 1)):
            with self.subTest(angles=angles), patch.object(
                    render.astro, "MoonPhase", side_effect=angles):
                _, row = self.plot._phase_matched_moon_event(
                    events, self.data.topo.midnight(20), astro.Direction.Set, 0)
            np.testing.assert_array_equal(row, events[chosen])

    def test_only_immediate_neighbours_are_compared(self):
        events = np.array([[21, 6.0], [20, 2.0], [19, -6.0], [20, -3.0]])
        with patch.object(render.astro, "MoonPhase", side_effect=[88, 91]) as phase:
            _, row = self.plot._phase_matched_moon_event(
                events, self.data.topo.midnight(20), astro.Direction.Rise, 90)
        np.testing.assert_array_equal(row, events[1])
        self.assertEqual([c.args[0].ut for c in phase.call_args_list],
                         [self.data.topo.time_from_plot(20, h).ut for h in (-3, 2)])

    def test_exact_event_is_compared_once(self):
        events = np.array([[20, -6.0], [20, 0.0], [20, 6.0]])
        with patch.object(render.astro, "MoonPhase", return_value=90) as phase:
            _, row = self.plot._phase_matched_moon_event(
                events, self.data.topo.midnight(20), astro.Direction.Rise, 90)
        np.testing.assert_array_equal(row, events[1])
        phase.assert_called_once()
        self.search.assert_not_called()


class MoonMarkerRuleTests(unittest.TestCase):
    def collect(self, phase, day=60, hour=-11.8):
        data = dataset_fixture(2028)
        plot = render._AlmanacPlot(data, render.RenderOptions(moon=False))
        quarter = SimpleNamespace(time=data.topo.time_from_plot(day, hour), quarter=phase)
        after_end = SimpleNamespace(time=astro.Time(data.topo.end_utc.ut + 1), quarter=1)
        with patch.object(render.astro, "SearchMoonQuarter", return_value=quarter), \
                patch.object(render.astro, "NextMoonQuarter", return_value=after_end), \
                patch.object(render.astro, "MoonPhase", side_effect=lambda t:
                             (90 * phase + 12 * (t.ut - quarter.time.ut)) % 360), \
                patch.object(plot, "moon_symbol") as symbol:
            plot.moons(Mock())
        specials = [(int(c.args[3]), float(c.args[4]), c.args[2])
                    for c in symbol.call_args_list if c.kwargs.get("quarter")]
        ordinary = {(int(c.args[3]), float(c.args[4]))
                    for c in symbol.call_args_list if not c.kwargs.get("quarter")}
        return data, quarter, specials, ordinary

    def test_full_moon_uses_phase_matched_pair_and_only_hides_requested_sides(self):
        # Leap-day noon selects the previous morning's rise and today's set.
        # Use deliberately fixed fixture times to check event type independently
        # of the physical usual full-moon evening-rise/morning-set locations.
        _, _, specials, ordinary = self.collect(2)
        self.assertEqual(specials, [(59, 6.0, 180), (60, -6.0, 180)])
        for day in (57, 58, 59):
            self.assertNotIn((day, 6.0), ordinary)
        for day in (60, 61, 62):
            self.assertNotIn((day, -6.0), ordinary)
        for key in ((56, 6.0), (60, 6.0), (59, -6.0), (63, -6.0)):
            self.assertIn(key, ordinary)

    def test_new_moon_hides_opposite_sides(self):
        _, _, specials, ordinary = self.collect(0)
        self.assertEqual(specials, [(59, 6.0, 0), (60, -6.0, 0)])
        for day in (59, 60, 61):
            self.assertNotIn((day, 6.0), ordinary)
        for day in (58, 59, 60):
            self.assertNotIn((day, -6.0), ordinary)
        for key in ((58, 6.0), (62, 6.0), (57, -6.0), (61, -6.0)):
            self.assertIn(key, ordinary)

    def test_quarters_use_independent_phase_matched_events_without_two_day_deletion(self):
        for phase in (1, 3):
            with self.subTest(phase=phase):
                _, _, specials, ordinary = self.collect(phase)
                self.assertEqual(specials, [(59, 6.0, phase * 90), (60, -6.0, phase * 90)])
                for day in (58, 59, 61, 62):
                    for hour in (-6.0, 6.0):
                        if (day, hour) == (59, 6.0):
                            continue
                        self.assertIn((day, hour), ordinary)
                self.assertNotIn((59, 6.0), ordinary)
                self.assertNotIn((60, -6.0), ordinary)
                self.assertIn((60, 6.0), ordinary)

    def test_calendar_days_include_second_previous_day_beyond_48_hours(self):
        data = dataset_fixture(2028)
        # New moon set at day 60 18:00 suppresses day 58 17:30 even though
        # their separation is 48.5 hours; day 57 remains outside the window.
        data.tables["set"]["Moon"][57] = -6.5
        plot = render._AlmanacPlot(data, render.RenderOptions(moon=False))
        quarter = SimpleNamespace(time=data.topo.time_from_plot(60, -6), quarter=0)
        after_end = SimpleNamespace(time=astro.Time(data.topo.end_utc.ut + 1), quarter=1)
        with patch.object(render.astro, "SearchMoonQuarter", return_value=quarter), \
                patch.object(render.astro, "NextMoonQuarter", return_value=after_end), \
                patch.object(render.astro, "MoonPhase", side_effect=lambda t:
                             (12 * (t.ut - quarter.time.ut)) % 360), \
                patch.object(plot, "moon_symbol") as symbol:
            plot.moons(Mock())
        ordinary = {(int(c.args[3]), float(c.args[4]))
                    for c in symbol.call_args_list if not c.kwargs.get("quarter")}
        self.assertNotIn((58, -6.5), ordinary)
        self.assertIn((57, -6), ordinary)


if __name__ == "__main__":
    unittest.main()
