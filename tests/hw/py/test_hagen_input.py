"""
Send events to calibrated synapse drivers at different addresses in hagen
mode. Assert the amplitudes obtained at the neurons using the CADCs change
with the addresses.
"""

import unittest
from typing import Optional

import numpy as np

from dlens_vx_v3 import hal, halco, sta, logger

from connection_setup import ConnectionSetup

from calix.common import base, helpers
from calix.hagen import synapse_driver
import calix.hagen


log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class HagenInputTest(ConnectionSetup):
    """
    Provides methods for measuring the amplitudes sent by synapse drivers
    at the neurons. Tests functionality of hagen mode:

    Calibrate CADCs, neurons, synapse drivers. Send events at
    different addresses and verify the amplitudes change, as the
    hagen mode requires.

    :cvar vector_addresses: Addresses to be iterated during the test,
        i.e. addresses that change the amplitudes of events.
    :cvar log: Logger used for output.
    :cvar calib_result: Result of hagen-mode calibration, stored for
        re-applying.
    """

    # sweep vector address from low to high amplitudes
    vector_activations = np.arange(1, hal.PADIEvent.HagenActivation.size)

    log = logger.get("calix.tests.hw.test_hagen_inputs")
    calib_result: Optional[calix.hagen.HagenCalibrationResult] = None
    measurement = synapse_driver.SynapseDriverMeasurement(
        n_parallel_measurements=8)

    def measure_amplitudes(self) -> np.ndarray:
        """
        Sweep the vector_addresses and record the amplitudes of
        each synapse drivers. The median amplitude of all synapse columns
        acquired with the CADCs is used as synapse driver amplitude.

        :return: Array of synapse driver amplitudes per address.
        """

        amplitudes = self.measurement.measure_syndrv_amplitudes(
            self.connection, self.vector_activations)

        self.log.DEBUG(
            "Obtained amplitudes:\n",
            "\n".join([
                f"Activation {activation}: "
                + f"{amplitudes[index].mean():5.3f} +- "
                + f"{amplitudes[index].std():5.3f}"
                for index, activation in enumerate(self.vector_activations)]))

        return amplitudes

    def evaluate_amplitudes(self, amplitudes: np.ndarray) -> None:
        """
        Assert the amplitudes depend on the address as expected.

        :param amplitudes: Array of synapse driver amplitudes for
            an address block, where we expect linearity.
            One half of the returned data from the measure_amplitudes()
            function.
        """

        # calculate mean of drivers
        mean_amplitudes = np.mean(amplitudes, axis=1)

        # assert amplitudes increase on average
        self.assertGreater(np.mean(np.diff(mean_amplitudes)), 0.5,
                           "Amplitudes do not grow with activation.")

        # assert no saturation within 5 LSB on top and bottom
        self.assertGreater(np.mean(np.diff(mean_amplitudes[:5])), 0.2,
                           "Amplitudes do not increase in first 5 LSB "
                           + "of activations.")
        self.assertGreater(np.mean(np.diff(mean_amplitudes[-5:])), 0.2,
                           "Amplitudes do not increase in last 5 LSB "
                           + "of activations.")

        # assert baseline is higher than -1
        baseline = np.min(mean_amplitudes)
        self.assertGreater(baseline, -1, "Amplitudes read lower than -1.")

        # assert high amplitudes are higher than 20 LSB
        # (some 30 LSB are expected)
        maximum = np.max(mean_amplitudes)
        self.assertGreater(maximum, 20, "High amplitudes are low.")

        # assert dynamic range is 8 times higher than baseline
        dynamic_range = maximum - baseline
        self.assertGreater(dynamic_range / np.abs(baseline), 8,
                           "Dynamic range is small.")

        # assert std. dev. between drivers is at most 1/8 of dynamic range
        mismatch = np.mean(np.std(amplitudes, axis=1))
        self.assertLess(mismatch, dynamic_range / 8,
                        "Mismatch between drivers is high.")

        # assert mismatch is less than 4 LSB (some 2 LSB expected)
        self.assertLess(
            mismatch, 4, "Mismatch between drivers is more than 4 LSB.")

    def test_00_calibrate(self):
        """
        Apply calibration of CADCs, neurons and synapse drivers.
        """

        self.__class__.calib_result = self.apply_calibration("hagen_synin")

    def test_01_hagen_mode(self):
        """
        Send inputs in hagen mode, assert amplitudes are as expected.
        """

        # Sweep hagen activation and measure amplitudes per driver
        amplitudes = self.measure_amplitudes()
        self.evaluate_amplitudes(amplitudes)

    def test_02_overwrite(self):
        """
        Overwrite synapse driver calibration, assert it is gone.
        """

        # Overwrite synapse driver calibration
        builder = sta.PlaybackProgramBuilder()
        builder = helpers.capmem_set_quadrant_cells(
            builder,
            {halco.CapMemCellOnCapMemBlock.stp_i_ramp: 400})
        for coord in halco.iter_all(halco.SynapseDriverOnDLS):
            config = hal.SynapseDriverConfig()
            config.hagen_dac_offset = \
                hal.SynapseDriverConfig.HagenDACOffset.max // 2
            builder.write(coord, config)
        base.run(self.connection, builder)

        # Measure results, assert calibration is gone
        amplitudes = self.measure_amplitudes()
        self.assertRaises(
            AssertionError,
            self.evaluate_amplitudes, amplitudes)

    def test_03_reapply(self):
        """
        Apply synapse driver calibration again, assert test works.
        """

        # Apply calibration again
        builder = sta.PlaybackProgramBuilder()
        self.__class__.calib_result.synapse_driver_result.apply(builder)
        base.run(self.connection, builder)

        # Assert calibration works again
        amplitudes = self.measure_amplitudes()
        self.evaluate_amplitudes(amplitudes)


if __name__ == "__main__":
    unittest.main()
