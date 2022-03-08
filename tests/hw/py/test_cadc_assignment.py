"""
Tests the assignment of synapse switches to CADC samples by connecting one
CADC channel after another from a low debug voltage to a high neuron
leak potential.
In a second step, the assignment of neurons to CADC samples is checked, by
changing the leak potential in one neuron after another.
"""

import unittest
import numpy as np
from dlens_vx_v2 import sta, halco, hal, logger

from calix.common import base, cadc, helpers
from calix.hagen import neuron_helpers
from calix import constants

from connection_setup import ConnectionSetup


log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class TestCADCAssignment(ConnectionSetup):
    """
    Tests CADC channel assignment to synapses and neurons.

    :ivar column_switches: List of column switches for CADC connections.
    :ivar leak_voltages: Leak potential CapMem settings for all neurons.
    """

    @classmethod
    def preconfigure_chip(cls, builder: sta.PlaybackProgramBuilder
                          ) -> sta.PlaybackProgramBuilder:
        """
        Set the CADC debug voltage low and all neurons' leak voltage high.
        Also enable readout of the neuron membranes, but don't connect
        it to the CADCs.

        :param builder: Builder to append configuration to.

        :return: Builder with configurationa appended.
        """
        builder = helpers.capmem_set_quadrant_cells(
            builder, config={
                halco.CapMemCellOnCapMemBlock.
                neuron_i_bias_leak_source_follower: 100,
                halco.CapMemCellOnCapMemBlock.neuron_i_bias_readout_amp: 110,
                halco.CapMemCellOnCapMemBlock.neuron_v_bias_casc_n: 340,
                # cell at debug output:
                halco.CapMemCellOnCapMemBlock.stp_v_charge_0: 50})

        builder = helpers.capmem_set_neuron_cells(
            builder, config={
                halco.CapMemRowOnCapMemBlock.v_leak: 1000,
                halco.CapMemRowOnCapMemBlock.i_bias_leak: 1000})

        # Neuron Config: Enable readout and set leak strong
        neuron_config = hal.NeuronConfig()
        neuron_config.enable_readout_amplifier = True
        neuron_config.readout_source = hal.NeuronConfig.ReadoutSource.membrane
        neuron_config.enable_leak_multiplication = True

        for neuron_coord in halco.iter_all(halco.NeuronConfigOnDLS):
            builder.write(neuron_coord, neuron_config)

        return builder

    @classmethod
    def enable_synapse_switch(
            cls, builder: sta.PlaybackProgramBuilder,
            neuron_coord: halco.NeuronConfigOnDLS
    ) -> sta.PlaybackProgramBuilder:
        """
        Switch on the neuron readout for the given column correlation switch.
        All other switches are connected to the debug line.

        :param builder: Builder to append configuration to.
        :param neuron_coord: Coordinate of neuron in which column the switches
            will be set to read the neuron.

        :return: Builder with configuration appended.
        """

        # Connect all channels to debug line
        quad_config = hal.ColumnCorrelationQuad()
        switch_config = hal.ColumnCorrelationQuad.ColumnCorrelationSwitch()
        switch_config.enable_debug_causal = True
        switch_config.enable_debug_acausal = True

        for switch_coord in halco.iter_all(halco.EntryOnQuad):
            quad_config.set_switch(switch_coord, switch_config)

        for quad_coord in halco.iter_all(halco.ColumnCorrelationQuadOnDLS):
            builder.write(quad_coord, quad_config)

        # Switch one channel to the neurons
        switch_config = hal.ColumnCorrelationQuad.ColumnCorrelationSwitch()
        switch_config.enable_internal_causal = True
        switch_config.enable_internal_acausal = True
        switch_config.enable_cadc_neuron_readout_causal = True
        switch_config.enable_cadc_neuron_readout_acausal = True

        quad_config.set_switch(neuron_coord.toEntryOnQuad(), switch_config)
        builder.write(neuron_coord.toColumnCorrelationQuadOnDLS(), quad_config)

        return builder

    @classmethod
    def set_leak_pattern(cls, builder: sta.PlaybackProgramBuilder,
                         neuron_coord: halco.NeuronConfigOnDLS
                         ) -> sta.PlaybackProgramBuilder:
        """
        Set the leak potential for one neuron low. The other neurons are
        set to a value of 1000 (plus noise), which yields a high CADC read.

        :param builder: Builder to append configuration to.
        :param neuron_coord: Coordinate of neuron for which the leak voltage
            should be reduced.

        :return: Builder with configuration appended.
        """

        leak_potentials = 1000 * np.ones(
            halco.NeuronConfigOnDLS.size, dtype=int) + helpers.capmem_noise(
                size=halco.NeuronConfigOnDLS.size)
        leak_potentials[int(neuron_coord.toEnum())] = 0

        builder = helpers.capmem_set_neuron_cells(
            builder, config={
                halco.CapMemRowOnCapMemBlock.v_leak: leak_potentials})
        builder = helpers.wait(builder, constants.capmem_level_off_time)

        return builder

    def test_00_cadc_calibration(self):
        # Run CADC calibration
        result = cadc.calibrate(self.connection)

        self.assertTrue(np.all(result.success),
                        msg="CADC calibration failed for some channels.")

    def test_01_cadc_synapse(self):
        """
        Test CADC / synapse assignment.

        Initially connect all but two CADC channels to the shared debug
        line, where a low voltage is applied. The other two channels
        (causal and acausal channels of the same column) are connected
        to the membrane voltage, where a high leak potential is applied.
        The change in CADC reads is evaluated.

        It is asserted that in every run, 95% of the channels read low,
        and that in 95% of all runs the channels read high when being enabled.
        """

        # Preconfigure chip for reading synapse assignment
        builder = sta.PlaybackProgramBuilder()
        builder = self.preconfigure_chip(builder)
        base.run(self.connection, builder)

        success = np.zeros(halco.NeuronConfigOnDLS.size, dtype=bool)

        # For each synapse switch assert it affects the right channels
        for neuron_id, neuron_coord in enumerate(
                halco.iter_all(halco.NeuronConfigOnDLS)):
            builder = sta.PlaybackProgramBuilder()
            builder = self.enable_synapse_switch(builder, neuron_coord)
            results = neuron_helpers.cadc_read_neuron_potentials(
                self.connection, builder)

            success[neuron_id] = results[neuron_id] > 100

            other_mask = np.ones(halco.NeuronConfigOnDLS.size, dtype=bool)
            other_mask[neuron_id] = False
            other_success = results[other_mask] < 50
            self.assertGreater(
                np.sum(other_success),
                int(0.95 * halco.NeuronConfigOnDLS.size - 1),
                "More than 5% of disabled channels read high.")

        self.assertGreater(
            np.sum(success),
            int(0.95 * halco.NeuronConfigOnDLS.size),
            "More than 5% of enabled channels have read low.")

    def test_02_cadc_neuron(self):
        """
        Test CADC / neuron assignment.
        Initially connect all CADC channels to the neurons at a high leak
        potential, i.e. a high CADC read is expected.
        One neuron after another is set to a low leak potential, the change
        in CADC reads is checked to be at the correct index.

        It is asserted that 95% of neurons at high leak potentials read high
        in every run, and that 95% of the neurons read a low CADC value
        when their leak is set low.
        """

        # connect all CADC channels to the neurons
        builder = sta.PlaybackProgramBuilder()

        quad_config = hal.ColumnCorrelationQuad()
        switch_config = hal.ColumnCorrelationQuad.ColumnCorrelationSwitch()
        switch_config.enable_internal_causal = True
        switch_config.enable_internal_acausal = True
        switch_config.enable_cadc_neuron_readout_causal = True
        switch_config.enable_cadc_neuron_readout_acausal = True

        for switch_coord in halco.iter_all(halco.EntryOnQuad):
            quad_config.set_switch(switch_coord, switch_config)

        for quad_coord in halco.iter_all(halco.ColumnCorrelationQuadOnDLS):
            builder.write(quad_coord, quad_config)

        base.run(self.connection, builder)

        success = np.zeros(halco.NeuronConfigOnDLS.size, dtype=bool)

        # For each neuron assert the leak potential changes the CADC reads
        for neuron_id, neuron_coord in enumerate(
                halco.iter_all(halco.NeuronConfigOnDLS)):
            builder = sta.PlaybackProgramBuilder()
            builder = self.set_leak_pattern(builder, neuron_coord)
            results = neuron_helpers.cadc_read_neuron_potentials(
                self.connection, builder)

            success[neuron_id] = results[neuron_id] < 50

            other_mask = np.ones(halco.NeuronConfigOnDLS.size, dtype=bool)
            other_mask[neuron_id] = False
            other_success = results[other_mask] > 100
            self.assertGreater(
                np.sum(other_success),
                int(0.95 * halco.NeuronConfigOnDLS.size - 1),
                "More than 5% of high-leak neurons read low.")

        self.assertGreater(
            np.sum(success),
            int(0.95 * halco.NeuronConfigOnDLS.size),
            "More than 5% of low-leak neurons have read high.")


if __name__ == "__main__":
    unittest.main()
