"""Dataset integrity tests using cheap, deterministic synthetic calculations."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from astropy.table import Table

from almanac.calculations import TABLE_COLUMNS
from almanac.dataset import generate_dataset, load_dataset
from almanac.topo import TopoContext


def synthetic_ephemeris(topo):
    result = {}
    for name, columns in TABLE_COLUMNS.items():
        table = Table()
        table["Day"] = np.arange(1, topo.days_in_year + 1, dtype=float)
        for column in columns[1:]:
            table[column] = np.full(topo.days_in_year, -999.0 if column == "Moon" else 1.25)
        result[name] = table
    return result


class DatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.topo = TopoContext(2024, 480, 40, 40)
        self.patchers = [
            patch("almanac.dataset.compute_ephemeris", side_effect=synthetic_ephemeris),
            patch("almanac.dataset.compute_eot", side_effect=lambda t: np.full(t.days_in_year, t.year/10000)),
            patch("almanac.dataset.compute_sun_alts", side_effect=lambda t, progress=None: np.full((t.days_in_year,1440), t.latitude_deg)),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

    def make_dataset(self, topo=None):
        return generate_dataset(topo or self.topo, self.root)

    def edit_manifest(self, run, edit):
        path = run.run_dir / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        edit(manifest)
        path.write_text(json.dumps(manifest), encoding="utf-8")

    def test_roundtrip_and_unique_runs(self):
        first = self.make_dataset()
        second = self.make_dataset()
        self.assertNotEqual(first.run_dir, second.run_dir)
        self.assertIn("2024_UTCp0800_N40_E120_H40m_", first.run_dir.name)
        restored = load_dataset(first.run_dir)
        self.assertEqual(restored.topo, self.topo)
        self.assertEqual(restored.manifest["derived"]["days_in_year"], 366)
        self.assertEqual(restored.arrays["sun_alts"].shape, (366,1440))
        self.assertTrue(np.all(restored.tables["rise"]["Moon"] == -999))
        self.assertEqual(restored.manifest["status"], "complete")
        np.testing.assert_array_equal(restored.arrays["EOT"], first.arrays["EOT"])

    def test_different_configuration_does_not_leak(self):
        first = self.make_dataset()
        second_topo = TopoContext(2027, -300, -30, 0)
        second = self.make_dataset(second_topo)
        self.assertEqual(load_dataset(first.run_dir).topo, self.topo)
        loaded = load_dataset(second.run_dir)
        self.assertEqual(loaded.topo, second_topo)
        self.assertEqual(loaded.arrays["sun_alts"].shape, (365,1440))
        self.assertTrue(np.all(loaded.arrays["sun_alts"] == -30))

    def test_sampling_provenance_and_explicit_old_version_compatibility(self):
        run = self.make_dataset()
        algorithm = run.manifest["algorithm"]
        self.assertEqual(algorithm["solar_altitude"]["exact_sample_interval_minutes"], 3)
        self.assertEqual(algorithm["solar_altitude"]["output_interval_minutes"], 1)
        self.edit_manifest(run, lambda m: m.update(algorithm={
            "version": "1.0.0", "astronomy_sha256": algorithm["astronomy_sha256"]}))
        loaded = load_dataset(run.run_dir)
        self.assertEqual(loaded.manifest["algorithm"]["version"], "1.0.0")
        self.assertNotIn("solar_altitude", loaded.manifest["algorithm"])

    def test_unknown_sampling_method_is_rejected(self):
        run = self.make_dataset()
        self.edit_manifest(run, lambda m: m["algorithm"]["solar_altitude"].update(
            exact_sample_interval_minutes=30))
        with self.assertRaisesRegex(ValueError, "算法版本"):
            load_dataset(run.run_dir)

    def test_generation_failure_leaves_rejected_incomplete_run(self):
        with patch("almanac.dataset.compute_ephemeris", side_effect=RuntimeError("failure")):
            with self.assertRaisesRegex(RuntimeError, "failure"):
                self.make_dataset()
        run_dir, = list(self.root.iterdir())
        with self.assertRaisesRegex(ValueError, "尚未完成"):
            load_dataset(run_dir)

    def test_legacy_directory_without_manifest_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "不能推测旧数据"):
            load_dataset(self.root)

    def test_changed_topo_is_rejected(self):
        run = self.make_dataset()
        self.edit_manifest(run, lambda m: m["topo"].update(utc_offset_minutes=540))
        with self.assertRaisesRegex(ValueError, "派生时间或观测位置"):
            load_dataset(run.run_dir)

    def test_changed_catalogue_is_rejected(self):
        run = self.make_dataset()
        self.edit_manifest(run, lambda m: m["stars"][0].update(body="Star5"))
        with self.assertRaisesRegex(ValueError, "恒星编号或坐标"):
            load_dataset(run.run_dir)

    def test_missing_file_is_rejected(self):
        run = self.make_dataset()
        (run.run_dir / "EOT.npy").unlink()
        with self.assertRaisesRegex(ValueError, "数据文件缺失"):
            load_dataset(run.run_dir)

    def test_changed_file_checksum_is_rejected(self):
        run = self.make_dataset()
        with (run.run_dir / "EOT.npy").open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "校验失败"):
            load_dataset(run.run_dir)

    def test_escaped_filename_is_rejected(self):
        run = self.make_dataset()
        self.edit_manifest(run, lambda m: m["files"]["EOT"].update(name="../EOT.npy"))
        with self.assertRaisesRegex(ValueError, "文件名不符合"):
            load_dataset(run.run_dir)

    def test_wrong_array_shape_even_with_updated_checksum_is_rejected(self):
        run = self.make_dataset()
        path = run.run_dir / "EOT.npy"
        np.save(path, np.zeros(365), allow_pickle=False)
        self.edit_manifest(run, lambda m: m["files"]["EOT"].update(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), shape=[365]
        ))
        with self.assertRaisesRegex(ValueError, "数组形状"):
            load_dataset(run.run_dir)

    def test_wrong_day_sequence_even_with_updated_checksum_is_rejected(self):
        run = self.make_dataset()
        path = run.run_dir / "rise.fits"
        table = Table.read(path)
        table["Day"][0] = 2
        table.write(path, overwrite=True)
        self.edit_manifest(run, lambda m: m["files"]["rise"].update(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        ))
        with self.assertRaisesRegex(ValueError, "Day 必须"):
            load_dataset(run.run_dir)

    def test_bad_output_never_becomes_complete(self):
        with patch("almanac.dataset.compute_eot", return_value=np.zeros(365)):
            with self.assertRaisesRegex(ValueError, "数组形状"):
                self.make_dataset()
        run_dir, = list(self.root.iterdir())
        with self.assertRaisesRegex(ValueError, "尚未完成"):
            load_dataset(run_dir)


if __name__ == "__main__":
    unittest.main()
