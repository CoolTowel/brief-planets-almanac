"""Importing maintained modules must not start a run or create artifacts."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ImportTests(unittest.TestCase):
    def test_imports_have_no_generation_side_effects(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            env = os.environ.copy()
            env['PYTHONPATH'] = str(root)
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            env['MPLCONFIGDIR'] = str(Path(directory) / 'mpl')
            code = '''
import numpy as np
from astropy.table import Table
from almanac._vendor import astronomy
def forbidden(*args, **kwargs):
    raise AssertionError('Import attempted to load run data or compute astronomy')
np.load = forbidden
np.loadtxt = forbidden
Table.read = forbidden
astronomy.Time.Now = forbidden
astronomy.SearchRiseSet = forbidden
astronomy.SearchMoonQuarter = forbidden
astronomy.HourAngle = forbidden
import almanac.topo
import almanac.stars
import almanac.calculations
import almanac.dataset
import almanac.background
import almanac.moon_symbol
import almanac.render
import almanac.cli
import almanac.gui
import main
'''
            result = subprocess.run([sys.executable, '-B', '-c', code],
                                    cwd=directory, env=env, capture_output=True,
                                    text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, '')
            self.assertEqual(list(Path(directory).rglob('*')), [])


if __name__ == '__main__':
    unittest.main()
