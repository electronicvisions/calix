"""
Tests functionality of CADC calibration, asserts that deviations after
calibration are significantly lower than one expects without calibration.
"""

import unittest
import numpy as np
from dlens_vx_v3 import halco, hal, logger

from connection_setup import ConnectionSetup

from calix.common import base, cadc, cadc_helpers, helpers
from calix import constants


log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class TestCADCCalib(ConnectionSetup):
    """
    Tests CADC calibration and ensures the results are ok.

    :cvar calibration_result: Result of CADC calibration.
    """

    calibration_result: cadc.CADCCalibResult

    def evaluate_results(
            self, calibrated_data: np.ndarray) -> None:
        """
        Checks whether the calibration was successful based on the
        calibrated parameters and the reads after calibration.

        :param calibrated_data: Reads of the CADC channels after calibration,
            as returned by the read_cadcs() method.
        """

        # Check for error flags of calibration
        self.assertTrue(
            np.all(self.__class__.calibration_result.success),
            "Calibrated parameters have reached range boundaries.")

        # Assert offsets are in an expected range
        offset_spread_threshold = 20
        self.assertLess(
            np.std(self.__class__.calibration_result.channel_offset),
            offset_spread_threshold,
            "Offsets show a higher than expected spread.")

        # Interpret results by quadrant, calculate mean of quadrant results
        calibrated_data = cadc_helpers.reshape_cadc_quadrants(
            calibrated_data)
        quadrant_means = np.mean(calibrated_data, axis=1)

        # Assert means of the quadrants are close to each other
        mean_spread_threshold = 3
        self.assertLess(
            np.max(quadrant_means) - np.min(quadrant_means),
            mean_spread_threshold,
            "Deviation in CADC reads betwen quadrants is high.")

        # Assert means are near the mid of the dynamic range of the CADCs
        dynamic_range = hal.CADCSampleQuad.Value.max - \
            hal.CADCSampleQuad.Value.min
        mean_upper_threshold = dynamic_range / 2 + 40
        self.assertLess(np.max(quadrant_means), mean_upper_threshold,
                        "Maximum read of CADC quadrant means is high.")
        mean_lower_threshold = dynamic_range / 2 - 40
        self.assertGreater(np.min(quadrant_means), mean_lower_threshold,
                           "Minimum read of CADC quadrant means is low.")

        # Judge success by standard deviation between all channels
        read_spread_threshold = 3
        mismatch = np.std(calibrated_data)
        self.assertLess(mismatch, read_spread_threshold,
                        "Mismatch between CADC channels is high.")

    def test_00_cadc_calibration(self):
        """
        Executes CADC calibration.
        Checks the results afterwards.
        """

        # Run CADC calibration
        self.__class__.calibration_result = \
            cadc.calibrate(self.__class__.connection)

        # Measure results after calibration
        calibrated_data = cadc_helpers.read_cadcs(self.__class__.connection)

        # Inspect results
        self.evaluate_results(calibrated_data)

    def test_01_overwrite(self):
        """
        Overwrite calibration, assert test fails.
        """

        # Overwrite calibration
        builder = base.WriteRecordingPlaybackProgramBuilder()
        capmem_config = {
            halco.CapMemCellOnCapMemBlock.cadc_v_ramp_offset: 50,
            halco.CapMemCellOnCapMemBlock.cadc_i_ramp_slope: 250
        }
        builder = helpers.capmem_set_quadrant_cells(builder, capmem_config)
        builder = helpers.wait(builder, constants.capmem_level_off_time)
        for coord in halco.iter_all(halco.CADCChannelConfigOnDLS):
            builder.write(coord, hal.CADCChannelConfig())

        # Measure results, assert calibration is overwritten
        uncalibrated_data = cadc_helpers.read_cadcs(
            self.__class__.connection, builder)
        self.assertRaises(
            AssertionError, self.evaluate_results,
            uncalibrated_data)

    def test_02_reapply(self):
        """
        Re-apply original calibration, assert tests work again.
        """

        # Apply result of previous calibration
        builder = base.WriteRecordingPlaybackProgramBuilder()
        self.__class__.calibration_result.apply(builder)
        builder = helpers.wait(builder, constants.capmem_level_off_time)

        # Measure results again, assert calibration is applied properly
        calibrated_data = cadc_helpers.read_cadcs(
            self.__class__.connection, builder)
        self.evaluate_results(calibrated_data)


if __name__ == "__main__":
    unittest.main()
