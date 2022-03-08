import unittest
import numpy as np
from dlens_vx_v2 import halco, sta, logger, hxcomm

from calix.common import algorithms, base, cadc
from calix.hagen import neuron
import calix.hagen.synapse_driver as hagen_driver
from calix.spiking import synapse_driver

from connection_setup import ConnectionSetup


log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class STPCalibrationTest(ConnectionSetup):
    """
    Test STP calibration (for spiking mode).

    :cvar log: Logger for output.
    """

    log = logger.get("calix.tests.hw.test_stp_calibration")

    def measure_amplitudes(self, connection: hxcomm.ConnectionHandle
                           ) -> np.ndarray:
        """
        Measure the amplitudes of each synapse driver once.

        :param connection: Connection to the chip to run on.

        :return: Array of synapse driver amplitudes.
        """

        amplitudes = np.empty(halco.SynapseDriverOnDLS.size)
        address = 32

        builder = sta.PlaybackProgramBuilder()
        builder = hagen_driver.set_synapse_pattern(
            builder, address=address)
        amplitudes = hagen_driver.measure_syndrv_amplitudes(
            connection, builder, address=address, n_events=12)

        self.log.INFO(
            f"Amplitude at address {address}: "
            + f"{amplitudes.mean():5.3f} +- "
            + f"{amplitudes.std():5.3f}")

        return amplitudes

    def test_00_neuron_calibration(self):
        cadc.calibrate(
            self.connection, base.ParameterRange(100, 450))
        neuron.calibrate(self.connection)

    def test_01_stp_calibration(self):
        # Configure things for baseline read
        calib = synapse_driver.STPOffsetCalibration()
        calib.prelude(self.connection)

        # Baseline read
        uncalibrated_amplitudes = self.measure_amplitudes(self.connection)

        # Calibrate
        calib.run(self.connection,
                  algorithm=algorithms.BinarySearch())

        # Calibrated read
        calibrated_amplitudes = self.measure_amplitudes(self.connection)

        # Evaluate results
        reduction = uncalibrated_amplitudes.std() - calibrated_amplitudes.std()
        self.log.INFO("Calibration reduced std dev by ", reduction)
        self.assertGreater(
            reduction, 0.5, "Calibration lowered standard deviation "
            + "of amplitudes insignifantly.")

        block_results = \
            hagen_driver.STPRampCalibration.reshape_syndrv_amplitudes(
                calibrated_amplitudes)
        block_deviations = np.std(block_results, axis=1)
        self.log.INFO("Deviations per CapMem block: ", block_deviations)
        self.assertLess(np.max(block_deviations), 6,
                        "Amplitudes differ strongly within at least one "
                        + "CapMem block.")


if __name__ == "__main__":
    unittest.main()
