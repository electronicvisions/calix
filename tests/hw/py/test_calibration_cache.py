from __future__ import annotations

from pathlib import Path
import unittest
import tempfile
from dataclasses import dataclass

from dlens_vx_v3 import logger
from calix import calibrate
from calix.common.base import CalibResult, TopLevelCalibTarget


class MockTarget(TopLevelCalibTarget):
    def calibrate(self, *_args, **_kwargs) -> MockResult:
        return MockResult(calibration_has_run=True, options=None, target=None)


@dataclass
class MockResult(CalibResult):
    calibration_has_run: bool = False

    def apply(self, *_args, **_kwargs):
        pass

    def __getstate__(self):
        return None  # don't pickle my state


class TestCalibCache(unittest.TestCase):
    def test_cache_enabled(self):
        log = logger.get("calix.calibrate")
        logger.set_loglevel(log, logger.LogLevel.INFO)

        result: CalibResult
        with tempfile.TemporaryDirectory() as temp_dir:
            result = calibrate(MockTarget(), cache_paths=[Path(temp_dir)])
            assert isinstance(result, MockResult)
            self.assertTrue(result.calibration_has_run)

            result = calibrate(MockTarget(), cache_paths=[Path(temp_dir)])
            assert isinstance(result, MockResult)
            self.assertFalse(result.calibration_has_run)

    def test_cache_disabled(self):
        log = logger.get("calix.calibrate")
        logger.set_loglevel(log, logger.LogLevel.INFO)

        result: CalibResult
        for _ in range(2):
            result = calibrate(MockTarget(), cache_paths=[])
            assert isinstance(result, MockResult)
            self.assertTrue(result.calibration_has_run)


if __name__ == "__main__":
    unittest.main()
