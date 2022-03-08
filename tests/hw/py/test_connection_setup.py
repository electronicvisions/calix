"""
Assert that all default calibrations are available from the
apply_calibration method in connection_setup.

In turn, this tests most parts of the script that generates
and serializes default calibrations.
"""

import unittest
import os

from dlens_vx_v3 import logger

from connection_setup import ConnectionSetup


log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class TestConnectionSetup(ConnectionSetup):
    """
    Request the default calibrations, assert they are available on
    disk in all expected formats and are not empty.
    """

    def test_run_and_save_all(self):
        expected_prefixes = ["hagen", "hagen_synin", "spiking"]
        expected_suffixes = ["calix-native.pkl", "cocolist.pbin",
                             "cocolist.json.gz"]

        for expected_prefix in expected_prefixes:
            self.apply_calibration(expected_prefix)
            for expected_suffix in expected_suffixes:
                expected_filename = self.target_directory / "_".join(
                    [expected_prefix, expected_suffix])

                self.assertGreater(
                    os.path.getsize(expected_filename), 0,
                    "File size of expected calibration result "
                    + f"{expected_filename} is zero.")


if __name__ == '__main__':
    unittest.main()
