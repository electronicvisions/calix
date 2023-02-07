from pathlib import Path
import time
import unittest
import tempfile

from dlens_vx_v3 import logger
import calix.spiking
from calix import calibrate


class TestCalibCache(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # pylint: disable=consider-using-with
        cls.tmp_dir = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir.cleanup()

    def test_spiking(self):
        log = logger.get("calix.calibrate")
        logger.set_loglevel(log, logger.LogLevel.INFO)
        start = time.time()
        calibrate(
            calix.spiking.SpikingCalibTarget(),
            calix.spiking.SpikingCalibOptions(),
            cache_paths=[Path(self.tmp_dir.name)]
        )
        after_calib = time.time()
        self.assertTrue(after_calib - start > 100)
        calibrate(
            calix.spiking.SpikingCalibTarget(),
            calix.spiking.SpikingCalibOptions(),
            cache_paths=[Path(self.tmp_dir.name)]
        )
        self.assertTrue(time.time() - after_calib < 10)


if __name__ == "__main__":
    unittest.main()
