"""
Tests the (debug only) MADC recording and plotting functionality.
"""

import unittest
import os

import numpy as np

from dlens_vx_v3 import halco, sta, logger

from connection_setup import ConnectionSetup

from calix.common import base, madc_base
from calix.hagen import neuron_helpers


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

        # Apply default configuration of neurons
        builder = sta.PlaybackProgramBuilder()
        builder, _ = neuron_helpers.configure_chip(builder)
        base.run(self.connection, builder)

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

        # assert the traces are sensible: There should not be major
        # edges in the trace after a few initial samples, that
        # are possibly assigned to the wrong neuron.
        has_edges = np.empty(halco.NeuronConfigOnDLS.size, dtype=bool)
        for neuron_id, neuron_samples in enumerate(samples):
            has_edges[neuron_id] = np.max(np.abs(np.diff(
                neuron_samples["value"]))) > 20
        self.assertLess(
            np.sum(has_edges), int(halco.NeuronConfigOnDLS.size) * 0.1,
            "Neuron MADC traces show unexpected edges in the trace.")


if __name__ == '__main__':
    unittest.main()
