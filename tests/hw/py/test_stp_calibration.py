import unittest

import numpy as np

from dlens_vx_v2 import hal, halco, sta, logger, hxcomm

from calix.common import algorithms, base, cadc, synapse, helpers
from calix.hagen import neuron_helpers
import calix.hagen.synapse_driver as hagen_driver
from calix.spiking import synapse_driver
from calix import constants

from connection_setup import ConnectionSetup


log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class STPCalibrationTest(ConnectionSetup):
    """
    Test STP calibration (for spiking mode).

    :cvar log: Logger for output.
    :cvar measurement: Instance of synapse driver calib
        amplitude measurement.
    """

    log = logger.get("calix.tests.hw.test_stp_calibration")
    measurement = hagen_driver.SynapseDriverMeasurement()
    measurement.multiplication = synapse_driver.STPMultiplication(
        signed_mode=False)

    def measure_amplitudes(self, connection: hxcomm.ConnectionHandle
                           ) -> np.ndarray:
        """
        Measure the amplitudes of each synapse driver once.

        :param connection: Connection to the chip to run on.

        :return: Array of synapse driver amplitudes.
        """

        amplitudes = np.empty(halco.SynapseDriverOnDLS.size)
        address = 31
        activation = hal.PADIEvent.HagenActivation.size - address
        if activation == 0:
            raise AssertionError(
                "No event will be sent if activation is zero.")

        amplitudes = self.measurement.measure_syndrv_amplitudes(
            connection, activations=activation)

        self.log.INFO(
            f"Amplitude at address {address}: "
            + f"{amplitudes.mean():5.3f} +- "
            + f"{amplitudes.std():5.3f}")

        return amplitudes

    def test_00_preparations(self):
        # calibrate CADC
        cadc.calibrate(
            self.connection, base.ParameterRange(100, 450))

        # reconnect neuron readout to CADCs
        builder = sta.PlaybackProgramBuilder()
        neuron_helpers.configure_chip(builder)

        # set target synapse DAC bias current
        builder = helpers.capmem_set_quadrant_cells(
            builder,
            {halco.CapMemCellOnCapMemBlock.syn_i_bias_dac: 800})
        builder = helpers.wait(builder, constants.capmem_level_off_time)
        base.run(self.connection, builder)

        # Calibrate synapse DAC bias current
        calibration = synapse.DACBiasCalibCADC()
        calibration.run(
            self.connection, algorithm=algorithms.BinarySearch())

    def test_01_stp_calibration(self):
        """
        Measure amplitudes of synapse drivers before calibration of STP
        offsets and afterwards. Assert the standrad deviations of
        amplitudes within a quadrant decreases.
        """

        # Configure things for baseline read
        calib = synapse_driver.STPOffsetCalibration()
        calib.prelude(self.connection)

        # Baseline read
        uncalibrated_amplitudes = self.measure_amplitudes(self.connection)

        block_results = [
            list() for _ in halco.iter_all(halco.CapMemBlockOnDLS)]
        for coord, result in zip(
                halco.iter_all(halco.SynapseDriverOnDLS),
                uncalibrated_amplitudes):
            block_results[int(coord.toCapMemBlockOnDLS().toEnum())].append(
                result)

        uncalibrated_deviations = np.std(block_results, axis=1)
        self.log.INFO(
            "Uncalibrated deviations per CapMem block: ",
            uncalibrated_deviations)

        # Calibrate
        calib.run(self.connection,
                  algorithm=algorithms.BinarySearch())

        # Calibrated read
        calibrated_amplitudes = self.measure_amplitudes(self.connection)

        block_results = [
            list() for _ in halco.iter_all(halco.CapMemBlockOnDLS)]
        for coord, result in zip(
                halco.iter_all(halco.SynapseDriverOnDLS),
                calibrated_amplitudes):
            block_results[int(coord.toCapMemBlockOnDLS().toEnum())].append(
                result)

        # Evaluate results
        calibrated_deviations = np.std(block_results, axis=1)
        self.log.INFO("Deviations per CapMem block: ", calibrated_deviations)
        self.assertLess(np.max(calibrated_deviations), 2,
                        "Amplitudes differ strongly within at least one "
                        + "CapMem block.")

        reduction = uncalibrated_deviations - calibrated_deviations
        self.log.INFO("Calibration reduced std dev per quadrant by ",
                      reduction)
        self.assertGreater(
            np.min(reduction), 1, "Calibration lowered standard deviation "
            + "of amplitudes insignifantly in at least one quadrant.")


if __name__ == "__main__":
    unittest.main()
