"""
Tests the (debug only) MADC recording and plotting functionality.
"""

import unittest
import os

from dlens_vx_v3 import sta, logger

from connection_setup import ConnectionSetup

from calix.common import madc_base


log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class TestMADCRecorder(ConnectionSetup):
    """
    Tests MADC recording and plotting functionality.
    """

    def test_00_record_and_plot(self):
        """
        Acquires samples from the MADC, plots for the first few neurons.
        Asserts the expected plot files are generated.
        """

        n_neurons_to_plot = 4

        # recorder is intentionally marked private, as it is a debug feature.
        recorder = madc_base.MembraneRecorder()
        recorder.prepare_recording(self.connection)
        samples = recorder.record_traces(
            self.connection, builder=sta.PlaybackProgramBuilder())
        recorder.plot_traces(samples[:n_neurons_to_plot])

        for neuron_id in range(n_neurons_to_plot):
            expected_filename = f"trace_neuron{neuron_id}.png"

        self.assertGreater(
            os.path.getsize(expected_filename), 0,
            "File size of expected plotted figure "
            + f"{expected_filename} is zero.")


if __name__ == '__main__':
    unittest.main()
