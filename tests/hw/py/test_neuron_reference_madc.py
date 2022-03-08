"""
Tests a hagen-mode calibration with leak disabled.
Recalibrates the synaptic input reference potentials and ensures
the drift of the membrane potentials is not too large.
"""

import unittest
import numpy as np
from dlens_vx_v2 import halco, sta, logger

from calix.common import algorithms, base, helpers
import calix.hagen
from calix.hagen import neuron_synin
from calix import constants

from connection_setup import ConnectionSetup


log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class TestReferenceCalib(ConnectionSetup):
    """
    Runs a neuron calibration and ensures the results are ok.

    :cvar log: Logger used for output.
    :cvar calibration_result: Result of calibration, stored for re-applying.
    """

    log = logger.get("calix.tests.hw.test_neuron_integration")

    def measure_drift(self):
        """
        Measures the drift of the membrane potential following a reset.
        Asserts the amplitude is acceptable.
        """

        calibration = neuron_synin.InhSynReferenceCalibMADC()
        calibration.n_runs = 10
        calibration.prelude(self.connection)
        builder = sta.PlaybackProgramBuilder()
        drift = calibration.measure_results(self.connection, builder)
        self.log.DEBUG("Drift statistics: ([0, 10, 50, 90, 100] percentiles):",
                       np.percentile(np.abs(drift), [0, 10, 50, 90, 100]))

        self.assertLess(np.percentile(np.abs(drift), 90), 1.0,
                        "Drift of membrane potential is higher than expected.")

    def test_00_calibration(self):
        """
        Executes standard hagen-mode calibration.
        """

        calix.hagen.calibrate(self.connection)

    def test_01_overwrite(self):
        """
        Overwrites the reference calibration, asserts the test fails.
        """

        builder = helpers.capmem_set_neuron_cells(
            sta.PlaybackProgramBuilder(),
            {halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_shift: 310})
        builder = helpers.wait(builder, constants.capmem_level_off_time)
        base.run(self.connection, builder)

        # Measure results, drift assertion should fail
        self.assertRaises(AssertionError, self.measure_drift)

    def test_02_madc_calib(self):
        """
        Runs the MADC-based reference calibration, assert the test works.
        """

        calibration = neuron_synin.InhSynReferenceCalibMADC()
        calibration.run(
            self.connection, algorithm=algorithms.NoisyBinarySearch())

        self.measure_drift()


if __name__ == "__main__":
    unittest.main()
