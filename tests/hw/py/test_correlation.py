"""
Measures correlation of pre- and postsynaptic spikes with default
parameters (bias currents). We do not test the correct calibration
of correlation sensors but assert that they are functional.
"""


import unittest
from typing import List
import numpy as np
import quantities as pq

from dlens_vx_v3 import hal, halco, sta, lola, logger

from connection_setup import ConnectionSetup

from calix.hagen import base, cadc, helpers
from calix import constants


log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class TestCorrelation(ConnectionSetup):
    """
    Configures the chip such that correlation can be measured and read
    out via the CADC. Asserts the results change as expected with the
    delay between pre- and postsynaptic spikes, using default parameters.

    :cvar log: Logger used to log results.
    """

    log = logger.get("calix.tests.hw.test_correlation")

    @staticmethod
    def configure_synapses(builder: sta.PlaybackProgramBuilder,
                           quad: halco.SynapseQuadColumnOnDLS,
                           address: hal.SynapseQuad.Label,
                           amp_calib: int = 0, time_calib: int = 0):
        """
        Set the columns of synapses in the given column quad to the
        given address.

        The other synapses are set to address 0. All weights are set to 0.

        :param builder: Builder to append configuration instructions to.
        :param quad: Coordinate of synapse quad column to be set to address.
        :param address: Address to set target quad column to.
        :param amp_calib: Amplitude calibration for whole array.
        :param time_calib: Time calibration for whole array.
        """

        # Synapse connections
        synapses = lola.SynapseMatrix()
        weights = np.zeros((halco.SynapseRowOnSynram.size,
                            halco.SynapseOnSynapseRow.size), dtype=int)
        addresses = np.zeros(halco.SynapseOnSynapseRow.size, dtype=int)

        target_slice = []
        for entry in halco.iter_all(halco.EntryOnQuad):
            target_slice.append(
                int(quad.toNeuronColumnOnDLS()[entry]))

        addresses[target_slice] = address
        addresses = np.repeat(
            addresses[np.newaxis], halco.SynapseRowOnSynram.size, axis=0)
        amp_calib = np.ones_like(weights) * amp_calib
        time_calib = np.ones_like(weights) * time_calib

        synapses.weights.from_numpy(weights)
        synapses.labels.from_numpy(addresses)
        synapses.amp_calibs.from_numpy(amp_calib)
        synapses.time_calibs.from_numpy(time_calib)

        for synram in halco.iter_all(halco.SynramOnDLS):
            builder.write(synram, synapses)

    @staticmethod
    def configure_base(builder: sta.PlaybackProgramBuilder):
        """
        Basic config for reading correlation.

        Configure reference voltages, connect correlation readout
        to the CADC and configure forwarding of post-pulses to the
        synapses.

        :param builder: Builder to append configuration instructions to.
        """

        # apply V_reset and V_resmeas to the chip
        dac_config = lola.DACChannelBlock().default_ldo_2
        dac_config.set_voltage(halco.DACChannelOnBoard.v_res_meas, 0.95)
        dac_config.set_voltage(halco.DACChannelOnBoard.mux_dac_25, 1.85)
        builder.write(halco.DACChannelBlockOnBoard(), dac_config)

        # Column Switches: Connect correlation readout to CADC
        quad_config = hal.ColumnCorrelationQuad()
        switch_config = hal.ColumnCorrelationQuad.ColumnCorrelationSwitch()
        switch_config.enable_internal_causal = True
        switch_config.enable_internal_acausal = True

        for switch_coord in halco.iter_all(halco.EntryOnQuad):
            quad_config.set_switch(switch_coord, switch_config)

        for quad_coord in halco.iter_all(halco.ColumnCorrelationQuadOnDLS):
            builder.write(quad_coord, quad_config)

        # Neuron Config
        neuron_config = hal.NeuronConfig()
        neuron_config.enable_fire = True  # necessary for artificial resets

        for neuron_coord in halco.iter_all(halco.NeuronConfigOnDLS):
            builder.write(neuron_coord, neuron_config)

        # Common Neuron Backend Config
        neuron_common_config = hal.CommonNeuronBackendConfig()
        neuron_common_config.clock_scale_post_pulse = 2

        for coord in halco.iter_all(halco.CommonNeuronBackendConfigOnDLS):
            builder.write(coord, neuron_common_config)

        # Neuron Backend Config
        neuron_config = hal.NeuronBackendConfig()
        # Set roughly 2 us refractory time, the precise value does not
        # matter. But too low values will not work (issue 3741).
        neuron_config.refractory_time = \
            hal.NeuronBackendConfig.RefractoryTime.max
        neuron_config.enable_neuron_master = True

        for coord in halco.iter_all(halco.NeuronBackendConfigOnDLS):
            builder.write(coord, neuron_config)

        # Common Correlation Config
        config = hal.CommonCorrelationConfig()
        for coord in halco.iter_all(halco.CommonCorrelationConfigOnDLS):
            builder.write(coord, config)

    @staticmethod
    def configure_input(builder: sta.PlaybackProgramBuilder):
        """
        Configures PADI bus and synapse drivers such that
        pre-pulses can be sent to all synapse rows.

        :param builder: Builder to append configuration instructions to.
        """

        # PADI global config
        global_padi_config = hal.CommonPADIBusConfig()

        for padi_config_coord in halco.iter_all(
                halco.CommonPADIBusConfigOnDLS):
            builder.write(padi_config_coord, global_padi_config)

        # Synapse driver config
        synapse_driver_config = hal.SynapseDriverConfig()
        synapse_driver_config.enable_address_out = True
        synapse_driver_config.enable_receiver = True
        row_mode = hal.SynapseDriverConfig.RowMode.excitatory
        synapse_driver_config.row_mode_top = row_mode
        synapse_driver_config.row_mode_bottom = row_mode
        synapse_driver_config.row_address_compare_mask = 0b00000  # reach all

        for synapse_driver_coord in halco.iter_all(
                halco.SynapseDriverOnDLS):
            builder.write(synapse_driver_coord, synapse_driver_config)

    @staticmethod
    def configure_capmem(builder: sta.PlaybackProgramBuilder):
        """
        Configure synapse bias currents for the correlation sensors.
        Amplitudes of a single correlated event are rather large,
        time constants rather long, since store and ramp biases are low,
        respectively.

        :param builder: Builder to append the configuration to.
        """

        parameters = {
            halco.CapMemCellOnCapMemBlock.syn_i_bias_corout: 600,
            halco.CapMemCellOnCapMemBlock.syn_i_bias_ramp: 80,
            halco.CapMemCellOnCapMemBlock.syn_i_bias_store: 70}

        helpers.capmem_set_quadrant_cells(builder, parameters)
        helpers.wait(builder, 5 * constants.capmem_level_off_time)

    @classmethod
    def configure_chip(cls, builder: sta.PlaybackProgramBuilder):
        """
        Does all the necessary configurations on the chip.

        Configures neurons, synapse connections, global synapse driver
        and PADI settings, CapMem and readout settings in order to
        measure correlation.

        :param builder: Builder to append configuration to.
        """

        # Preconfigure chip:
        cls.configure_base(builder)

        # Preconfigure chip: Configure synapse drivers/PADI globally
        cls.configure_input(builder)

        # Preconfigure capmem for calibration
        cls.configure_capmem(builder)

    @staticmethod
    def reset_correlation(builder: sta.PlaybackProgramBuilder,
                          quad: halco.SynapseQuadColumnOnDLS,
                          row: halco.SynapseRowOnSynram,
                          synram: halco.SynramOnDLS):
        """
        Reset synapse correlations in given quad and row.

        :param builder: Builder to append instructions to.
        :param quad: Quad column to be reset.
        :param row: Synapse row to be reset.
        :param synram: Target synram coordinate.
        """

        coord = halco.CorrelationResetOnDLS(
            halco.SynapseQuadOnSynram(quad, row), synram)
        builder.write(coord, hal.CorrelationReset())

    @staticmethod
    def reset_neurons(builder: sta.PlaybackProgramBuilder,
                      quad: halco.SynapseQuadColumnOnDLS,
                      synram: halco.SynramOnDLS):
        """
        Reset the given quad of neurons.

        :param builder: Builder to append reset instructions to.
        :param quad: Quad column to reset neurons in.
        :param synram: Synram of neurons to reset.
        """

        coord = halco.NeuronResetQuadOnDLS(quad, synram)
        builder.write(coord, hal.NeuronResetQuad())

    @staticmethod
    def read_correlation(builder: sta.PlaybackProgramBuilder,
                         quad: halco.SynapseQuadColumnOnDLS,
                         synram: halco.SynramOnDLS
                         ) -> List[sta.ContainerTicket_CADCSampleQuad]:
        """
        Read CADCs in given quad column.
        Returns a list of tickets for each row.

        :param builder: Builder to append reads to.
        :param quad: Quad coordinate to be read.
        :param synram: Synram to be used.

        :return: List of read tickets, ordered
            [causal row 0, acausal row 0, causal row 1, ...]
        """

        tickets = []

        for row in halco.iter_all(halco.SynapseRowOnSynram):
            # causal read
            synapse_quad = halco.SynapseQuadOnSynram(quad, row)
            coord = halco.CADCSampleQuadOnSynram(
                synapse_quad, halco.CADCChannelType.causal,
                halco.CADCReadoutType.trigger_read)
            coord = halco.CADCSampleQuadOnDLS(coord, synram)

            tickets.append(builder.read(coord))

            # acausal read
            coord = halco.CADCSampleQuadOnSynram(
                synapse_quad, halco.CADCChannelType.acausal,
                halco.CADCReadoutType.buffered)
            coord = halco.CADCSampleQuadOnDLS(coord, synram)

            tickets.append(builder.read(coord))

        return tickets

    @staticmethod
    def evaluate_correlation(tickets: List[
            sta.ContainerTicket_CADCSampleQuad]) -> np.ndarray:
        """
        Evaluate list of CADC reads.

        :param tickets: List of tickets to evaluate, as returned by
            the read() function.

        :return: Numpy array containing all reads. It will be
            shaped (4, 256, 2) for the columns in a quad, the rows,
            and the causal/acausal correlation.
        """

        results = np.empty((
            halco.EntryOnQuad.size, halco.SynapseRowOnSynram.size,
            halco.CADCChannelType.size), dtype=int)

        channels_per_synapse = halco.CADCChannelType.size  # causal/acausal
        for ticket_id, ticket in enumerate(tickets):
            result = ticket.get()
            for entry_on_quad in halco.iter_all(halco.EntryOnQuad):
                results[int(entry_on_quad),
                        int(ticket_id / channels_per_synapse),
                        ticket_id % channels_per_synapse] = \
                    result.get_sample(entry_on_quad)

        return results

    def measure_quad(
            self, *,
            quad: halco.SynapseQuadColumnOnDLS, delay: pq.Quantity,
            n_events: int, address: hal.SynapseQuad.Label,
            synram: halco.SynramOnDLS,
            amp_calib: int = 0, time_calib: int = 0) -> np.ndarray:
        """
        Measure correlation in given quad.

        Configure given quad to receive prepulses at given address.
        Send prepulses to quad column, wait for given delay in us, send
        postpulses by resetting neurons below quad column.
        The difference between a baseline correlation read and another read
        after correlated events is returned.

        If the delay is negative, acausal correlation is measured, i.e.
        the postpulses are sent before the prepulses.

        :param quad: Synapse quad column coordinate to measure
            coorelation on.
        :param delay: Time delay between pre- and postpulse.
            If negative, post- are sent before prepulses.
        :param n_events: Number of pre/postpulse pairs to send before
            reading results.
        :param address: Address to use during measurements. Do not
            select 0.
        :param synram: Synram to use during measurements.
        :param amp_calib: Amplitude calibration for whole array.
        :param time_calib: Time calibration for whole array.

        :return: Array of correlation measurements, i.e.
            baseline - result reads. Shaped (4, 256, 2), for the columns
            in a quad, the rows, and the causal/acausal correlation.

        :raises ValueError: If delay is exactly 0, i.e. pre- and post
            pulses would be required to be sent at the same time.
        """

        # Inspect delay
        if delay == 0:
            raise ValueError(
                "Pre- and post pulse can not be sent concurrently.")

        builder = sta.PlaybackProgramBuilder()
        self.configure_synapses(
            builder, quad, address, amp_calib, time_calib)
        base.run(self.connection, builder)

        builder = sta.PlaybackProgramBuilder()

        # Reset correlations
        for coord in halco.iter_all(halco.SynapseRowOnSynram):
            self.reset_correlation(builder, quad, coord, synram)

        # Read baseline correlations
        baselines = self.read_correlation(builder, quad, synram)

        # Generate PADI event to stimulate all rows
        padi_event = hal.PADIEvent()
        for coord in halco.iter_all(halco.PADIBusOnPADIBusBlock):
            padi_event.fire_bus[coord] = True  # pylint: disable=unsupported-assignment-operation
        padi_event.event_address = address

        # Send correlated pulses
        builder.block_until(halco.BarrierOnFPGA(), hal.Barrier.omnibus)
        builder.write(halco.TimerOnDLS(), hal.Timer())  # reset timer
        for event_id in range(n_events):
            # initial wait
            time_offset = 500 * pq.us * (event_id + 1)
            builder.block_until(
                halco.TimerOnDLS(),
                int(time_offset.rescale(pq.us).magnitude * int(
                    hal.Timer.Value.fpga_clock_cycles_per_us)))

            if delay > 0:  # pre before post
                builder.write(synram.toPADIEventOnDLS(), padi_event)
            else:  # post before pre
                self.reset_neurons(builder, quad, synram)

            # wait for delay
            builder.block_until(
                halco.TimerOnDLS(),
                int((time_offset + np.abs(delay)
                     ).rescale(pq.us).magnitude
                    * int(hal.Timer.Value.fpga_clock_cycles_per_us)))

            if delay > 0:  # pre before post
                self.reset_neurons(builder, quad, synram)
            else:  # post before pre
                builder.write(synram.toPADIEventOnDLS(), padi_event)

        # Read correlation
        results = self.read_correlation(builder, quad, synram)

        # Run program, evaluate results
        base.run(self.connection, builder)
        baselines = self.evaluate_correlation(baselines)
        results = self.evaluate_correlation(results)

        self.log.debug(
            f"Results of synram {int(synram)} quad {int(quad)}:\n"
            + f"Baselines: {baselines.mean(axis=1)} "
            + f"+- {baselines.std(axis=1)},\n"
            + f"Amplitudes: {baselines.mean(1) - results.mean(1)}\n")

        # Assert the baseline reads are as expected. For some seupts,
        # the mux_dac_25 channel may not be routed to the V_reset
        # voltage. The fix is explained in [1].
        # [1]: https://brainscales-r.kip.uni-heidelberg.de/projects/symap2ic/wiki/xboard#putting-a-new-board-into-service  # pylint: disable=line-too-long
        self.assertGreater(
            np.min(baselines), 120,
            "Baseline read lower than expected. The setup may not "
            + "be equipped with the V_reset fix on the x-board.")

        return baselines - results

    def measure_correlation(
            self, delay: pq.Quantity, n_events: int, *,
            address: hal.SynapseQuad.Label = hal.SynapseQuad.Label(15),
            amp_calib: int = 0, time_calib: int = 0
    ) -> np.ndarray:
        """
        Measure correlation on the entire synapse array.

        :param delay: Delay between pre- and postpulse.
            If negative, post- are sent before prepulses.
        :param n_events: Number of pre/postpulse pairs to send before
            reading results.
        :param address: Address to use during measurements. Do not
            select 0.
        :param amp_calib: Amplitude calibration for whole array.
        :param time_calib: Time calibration for whole array.

        :return: Array of correlation measurements, i.e.
            baseline - result reads. Shaped (512, 256, 2), corresponding
            to the number of neurons, the number of synapse rows and
            causal/acausal correlation.
        """

        results = np.empty((
            halco.NeuronConfigOnDLS.size, halco.SynapseRowOnSynram.size,
            halco.CADCChannelType.size), dtype=int)

        for synram in halco.iter_all(halco.SynramOnDLS):
            for quad in halco.iter_all(halco.SynapseQuadColumnOnDLS):
                target_slice = []
                for entry in halco.iter_all(halco.EntryOnQuad):
                    target_slice.append(
                        int(quad.toNeuronColumnOnDLS()[entry])
                        + (halco.NeuronConfigOnDLS.size
                           // halco.SynramOnDLS.size) * int(synram))

                results[target_slice, :, :] = self.measure_quad(
                    quad=quad, delay=delay, n_events=n_events,
                    address=address, synram=synram,
                    amp_calib=amp_calib, time_calib=time_calib)

        return results

    def test_00_prepare(self):
        """
        Calibrate and preconfigure chip.
        """

        cadc.calibrate(self.connection)

        builder = sta.PlaybackProgramBuilder()
        self.configure_chip(builder)
        base.run(self.connection, builder)

    def test_01_correlation(self):
        """
        Measure correlation and assert results are as expected.
        """

        delays = np.array([-5, -1, 1, 5]) * pq.us  # need to be sorted!
        n_events = 20

        results = np.empty((len(delays), halco.NeuronConfigOnDLS.size,
                            halco.SynapseRowOnSynram.size,
                            halco.CADCChannelType.size), dtype=int)
        # shape: delays, neurons, rows, causal/acausal
        for delay_id, delay in enumerate(delays):
            self.log.INFO(f"Measuring at delay {delay}")
            results[delay_id] = self.measure_correlation(
                delay=delay, n_events=n_events)

        # assert causal > acausal for positive delay and vice versa
        is_causal = delays > 0

        is_ok = results[is_causal, :, :, 0] > results[is_causal, :, :, 1]
        self.assertGreater(
            np.sum(is_ok),
            halco.NeuronConfigOnDLS.size * halco.SynapseRowOnSynram.size
            * 0.8 * np.sum(is_causal),
            "Too few synapses respond to causal correlation.")

        is_ok = results[~is_causal, :, :, 0] < results[~is_causal, :, :, 1]
        self.assertGreater(
            np.sum(is_ok),
            halco.NeuronConfigOnDLS.size * halco.SynapseRowOnSynram.size
            * 0.8 * np.sum(~is_causal),
            "Too few synapses respond to acausal correlation.")

        # assert amplitudes decrease with time
        is_ok = results[is_causal, :, :, 0][0] > \
            results[is_causal, :, :, 0][-1]
        self.assertGreater(
            np.sum(is_ok),
            halco.NeuronConfigOnDLS.size * halco.SynapseRowOnSynram.size
            * 0.8,
            "Too few synapses show time dependency of causal correlation.")

        is_ok = results[~is_causal, :, :, 1][0] < \
            results[~is_causal, :, :, 1][-1]
        self.assertGreater(
            np.sum(is_ok),
            halco.NeuronConfigOnDLS.size * halco.SynapseRowOnSynram.size
            * 0.8,
            "Too few synapses show time dependency of acausal correlation.")


if __name__ == "__main__":
    unittest.main()
