"""
Test the script that generates and serializes default calibrations.
"""

import unittest
import os
from pathlib import Path

from dlens_vx_v2 import logger
from calix.scripts import calix_generate_default_calibration


log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class TestGenerateDefaultCalibration(unittest.TestCase):
    """
    Run the default calibrations, assert they run and the generated files
    are not empty.
    """

    def test_run_and_save_all(self):
        calix_generate_default_calibration.run_and_save_all(Path("."))

        expected_prefixes = ["hagen", "hagen_synin", "spiking"]
        expected_suffixes = ["calix-native.pkl", "cocolist.pbin",
                             "cocolist.json.gz"]

        for expected_prefix in expected_prefixes:
            for expected_suffix in expected_suffixes:
                expected_filename = "_".join(
                    [expected_prefix, expected_suffix])

                self.assertGreater(
                    os.path.getsize(expected_filename), 0,
                    "File size of expected calibration result "
                    + f"{expected_filename} is zero.")


if __name__ == '__main__':
    unittest.main()
