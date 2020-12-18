from typing import List
import numpy as np
import quantities as pq
from dlens_vx_v2 import hal, sta, halco, hxcomm, logger

from calix.common import base, cadc_helpers, madc_base, helpers, exceptions
from calix.hagen import neuron_helpers
from calix import constants


class DACBiasCalibMADC(madc_base.Calibration):
    """
    Calibrates synapse DAC bias currents such that the amplitudes
    received at all neurons match those of the weakest quadrant.

    The voltage drop on the synapse lines is measured using the MADC.
    The synaptic inputs are disabled and the synaptic time constant
    is set maximum. The initial value of the synaptic time constant and input
    strength are not restored. The digital configuration of the neuron is
    restored at the end of the calibration.

    :ivar neuron_configs: List of neuron configs that will be used as
        basis for the `neuron_config_disabled` and `neuron_config_readout`
        functions.
    """

    def __init__(self):
        super().__init__(
            parameter_range=base.ParameterRange(
                hal.CapMemCell.Value.min, hal.CapMemCell.Value.max),
            inverted=False, n_instances=halco.CapMemBlockOnDLS.size)

        self.sampling_time = 50 * pq.us
        self.wait_between_neurons = 1000 * pq.us
        self._wait_before_stimulation = 20 * pq.us

        config = neuron_helpers.neuron_config_default()
        config.enable_synaptic_input_excitatory = False
        config.enable_synaptic_input_inhibitory = False
        self.neuron_configs = [
            hal.NeuronConfig(config) for _ in
            halco.iter_all(halco.NeuronConfigOnDLS)]

    def prelude(self, connection: hxcomm.ConnectionHandle):
        """
        Prepares chip for calibration.

        Disables synaptic inputs.
        Configures synapse drivers to stimulate the necessary input.

        Measures the amplitudes of synapses per quadrant once, calculates the
        median amplitude per quadrant, and sets the ivar `target` to the
        minimum of these median amplitudes (which ensures the target can
        be reached by all quadrants in case some are already configured
        to the maximum bias current).

        :param connection: Connection to the chip to calibrate.
        """

        # prepare MADC
        super().prelude(connection)

        builder = sta.PlaybackProgramBuilder()

        # Ensure synaptic inputs are disabled
        # Otherwise, high membrane potentials can leak through the
        # readout selection mux and affect synaptic input readout.
        # (The mux is not capable of disconnecting 1.2 V.)
        # Also, the synaptic time constant is chosen maximum, i.e. the
        # potential at the synaptic line should float.
        builder = helpers.capmem_set_neuron_cells(
            builder, {halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_gm: 0,
                      halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_gm: 0,
                      halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_tau: 0,
                      halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_tau: 0})
        builder = helpers.wait(builder, constants.capmem_level_off_time)

        # enable synapse drivers and configure synapses
        builder = neuron_helpers.enable_all_synapse_drivers(
            builder, row_mode=hal.SynapseDriverConfig.RowMode.excitatory)
        builder = neuron_helpers.configure_synapses(builder)

        # run program
        base.run(connection, builder)

        # set target
        self.target = np.min(self.measure_results(
            connection, builder=sta.PlaybackProgramBuilder()))

    def configure_parameters(self, builder: sta.PlaybackProgramBuilder,
                             parameters: np.ndarray
                             ) -> sta.PlaybackProgramBuilder:
        """
        Configures the given array of synapse DAC bias currents.

        :param builder: Builder to append configuration instructions to.
        :param parameters: Array of bias currents to set up.

        :return: Builder with configuration appended.
        """

        builder = helpers.capmem_set_quadrant_cells(
            builder,
            {halco.CapMemCellOnCapMemBlock.syn_i_bias_dac: parameters})

        builder = helpers.wait(builder, constants.capmem_level_off_time)
        return builder

    def stimulate(self, builder: sta.PlaybackProgramBuilder,
                  neuron_coord: halco.NeuronConfigOnDLS,
                  stimulation_time: hal.Timer.Value
                  ) -> sta.PlaybackProgramBuilder:
        """
        Send some PADI events to the synaptic input in order to
        drop the potential.

        :param builder: Builder to append PADI events to.
        :param neuron_coord: Coordinate of neuron which is currently recorded.
        :param stimulation_time: Timer value at beginning of stimulation.

        :return: Builder with PADI events appended.
        """

        padi_event = hal.PADIEvent()
        for bus in halco.iter_all(halco.PADIBusOnPADIBusBlock):
            padi_event.fire_bus[bus] = True  # pylint: disable=unsupported-assignment-operation

        for _ in range(3):
            builder.write(neuron_coord.toSynramOnDLS().toPADIEventOnDLS(),
                          padi_event)
        return builder

    def neuron_config_disabled(self, neuron_coord) -> hal.NeuronConfig:
        """
        Return a neuron config with readout disabled.

        :return: Neuron config with readout disabled.
        """

        config = self.neuron_configs[int(neuron_coord)]
        config.enable_readout = False
        return config

    def neuron_config_readout(self, neuron_coord) -> hal.NeuronConfig:
        """
        Return a neuron config with readout enabled.

        :return: Neuron config with readout enabled.
        """

        config = self.neuron_config_disabled(neuron_coord)
        config.readout_source = hal.NeuronConfig.ReadoutSource.exc_synin
        config.enable_readout = True
        return config

    def evaluate(self, samples: List[np.ndarray]) -> np.ndarray:
        """
        Evaluates the obtained MADC samples.

        Calculates for each trace the peak amplitude (maximum - minimum) and
        returns the median value of these amplitudes for each quadrant.

        :param samples: MADC samples obtained for each neuron.

        :return: Median peak amplitude per quadrant.
        """

        result_slice = slice(
            int(
                self.madc_config.calculate_sample_rate(
                    self.madc_input_frequency)
                * self._wait_before_stimulation.rescale(pq.s) / 2),
            self.madc_config.number_of_samples - 100)

        min_samples = np.array(
            [np.min(trace["value"][result_slice]) for trace in samples])
        max_samples = np.array(
            [np.max(trace["value"][result_slice]) for trace in samples])
        return np.median((max_samples - min_samples).reshape(
            halco.NeuronConfigBlockOnDLS.size,
            halco.NeuronConfigOnNeuronConfigBlock.size), axis=1)

    def postlude(self, connection: hxcomm.ConnectionHandle):
        super().postlude(connection)

        log = logger.get("calix.common.synapse.DACBiasCalibMADC")
        log.DEBUG("Calibrated synapse DAC bias current: "
                  + f"{self.result.calibrated_parameters}")


class DACBiasCalibCADC(base.Calibration):
    """
    Calibrates synapse DAC bias currents such that the amplitudes
    received at all neurons match those of the weakest quadrant.

    The voltage drop on the synapse lines is measured using the CADC.
    The synaptic inputs are disabled and the synaptic time constant
    is set maximum. The initial value of the synaptic time constant and input
    strength are not restored. The digital configuration of the neuron is
    restored at the end of the calibration.

    :ivar original_neuron_config: Neuron configs before calibration,
        read during prelude, restored in postlude.
    """

    def __init__(self):
        super().__init__(
            parameter_range=base.ParameterRange(
                hal.CapMemCell.Value.min, hal.CapMemCell.Value.max),
            n_instances=halco.NeuronConfigBlockOnDLS.size,
            inverted=False)
        self.original_neuron_config: List[hal.NeuronConfig] = list()

    def prelude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Prepares chip for calibration.

        Reads the current neuron configuration which will be restored at the
        end of the calibration. After that synaptic inputs are disabled.
        Configures synapse drivers to stimulate the necessary input.

        Measures the amplitudes of synapses per quadrant once, calculates the
        median amplitude per quadrant, and sets the ivar `target` to the
        minimum of these median amplitudes (which ensures the target can be
        reached, in case the maximum bias current of 1022 was used before).

        :param connection: Connection to the chip to calibrate.

        :raises CalibrationNotSuccessful: If amplitudes can not be brought
            into a reliable range by adjusting the number of enabled
            synapse rows.
        """

        builder = sta.PlaybackProgramBuilder()

        # read current neuron config
        neuron_tickets = list()
        for neuron_coord in halco.iter_all(halco.NeuronConfigOnDLS):
            neuron_tickets.append(builder.read(neuron_coord))

        # ensure synaptic inputs are disabled
        # otherwise, high membrane potentials can leak through the
        # readout selection mux and affect synaptic input readout.
        # (The mux is not capable of disconnecting 1.2 V.)
        # Also, the synaptic time constant is chosen maximum, i.e. the
        # potential at the synaptic line should float.
        builder = helpers.capmem_set_neuron_cells(
            builder, {halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_gm: 0,
                      halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_gm: 0,
                      halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_tau: 0,
                      halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_tau: 0})
        builder = helpers.wait(builder, constants.capmem_level_off_time)

        # enable synapse drivers
        builder = neuron_helpers.enable_all_synapse_drivers(
            builder, row_mode=hal.SynapseDriverConfig.RowMode.excitatory)

        # configure neurons for excitatory synaptic input readout
        config = neuron_helpers.neuron_config_default()
        config.enable_synaptic_input_excitatory = False
        config.enable_synaptic_input_inhibitory = False
        config.readout_source = hal.NeuronConfig.ReadoutSource.exc_synin

        for coord in halco.iter_all(halco.NeuronConfigOnDLS):
            builder.write(coord, config)

        # run program
        base.run(connection, builder)

        # inspect reads
        self.original_neuron_config = list()
        for ticket in neuron_tickets:
            self.original_neuron_config.append(ticket.get())

        # set target: iterate until target is in usable range.
        # The target amplitudes are quite high: The typical saturation
        # in hagen mode is not an issue here, since all synapses are
        # stimulated at once.
        target_range = range(40, 60)

        n_rows_enabled = halco.SynapseRowOnSynram.size // 4
        for _ in range(50):
            builder = sta.PlaybackProgramBuilder()

            # configure desired number of enabled rows
            builder = neuron_helpers.configure_synapses(
                builder, n_synapse_rows=n_rows_enabled,
                weight=hal.SynapseQuad.Weight.max // 2)

            # measure amplitudes
            self.target = np.min(self.measure_results(connection, builder))

            # inspect amplitudes
            if self.target in target_range:
                break
            if self.target > target_range.stop:
                n_rows_enabled = int(n_rows_enabled * 0.8)
            elif self.target < target_range.start:
                n_rows_enabled = int(n_rows_enabled * 1.3)
        if self.target not in target_range:
            raise exceptions.CalibrationNotSuccessful(
                "Optimal number of enabled synapse rows not found "
                + "during prelude of Synapse DAC bias calibration. "
                + f"last parameters: n_synapse_rows: {n_rows_enabled}, "
                + f"target: {self.target}. Please select a more moderate"
                + "target current for the synapse DAC bias.")

        log = logger.get("calix.common.synapse.DACBiasCalibCADC")
        log.DEBUG(f"Using {n_rows_enabled} synapse rows during DAC bias"
                  + f"calib. Target amplitude: {self.target}")

    def configure_parameters(self, builder: sta.PlaybackProgramBuilder,
                             parameters: np.ndarray
                             ) -> sta.PlaybackProgramBuilder:
        """
        Configures the given array of synapse DAC bias currents.

        :param builder: Builder to append configuration instructions to.
        :param parameters: Array of bias currents to set up.

        :return: Builder with configuration appended.
        """

        builder = helpers.capmem_set_quadrant_cells(
            builder,
            {halco.CapMemCellOnCapMemBlock.syn_i_bias_dac: parameters})

        builder = helpers.wait(builder, constants.capmem_level_off_time)
        return builder

    def measure_results(self, connection: hxcomm.ConnectionHandle,
                        builder: sta.PlaybackProgramBuilder
                        ) -> np.ndarray:
        """
        Measures the drop in synaptic input potentials for all
        synapse columns.

        :param connection: Connection to a chip.
        :param builder: Builder to append read instructions to.

        :return: Array containing the median drop in synaptic input
            potential per quadrant.
        """

        baseline_reads = list()
        result_reads = list()

        for synram in halco.iter_all(halco.SynramOnDLS):
            # read baseline
            builder, ticket = cadc_helpers.cadc_read_row(
                builder, synram)
            baseline_reads.append(ticket)

            # send event
            padi_event = hal.PADIEvent()
            for bus in halco.iter_all(halco.PADIBusOnPADIBusBlock):
                padi_event.fire_bus[bus] = True  # pylint: disable=unsupported-assignment-operation

            builder.write(synram.toPADIEventOnDLS(), padi_event)

            # read potential again
            builder, ticket = cadc_helpers.cadc_read_row(
                builder, synram)
            result_reads.append(ticket)

        base.run(connection, builder)

        # evaluate tickets
        baselines = neuron_helpers.inspect_read_tickets(baseline_reads)
        results = neuron_helpers.inspect_read_tickets(result_reads)
        amplitudes = (baselines - results).reshape(
            halco.NeuronConfigBlockOnDLS.size,
            halco.NeuronConfigOnNeuronConfigBlock.size)
        amplitudes = np.median(amplitudes, axis=1)

        return amplitudes

    def postlude(self, connection: hxcomm.ConnectionHandle):
        """
        Restore original neuron configuration.

        :param connection: Connection to the chip to calibrate.
        """

        log = logger.get("calix.common.synapse.DACBiasCalibCADC")
        log.DEBUG("Calibrated synapse DAC bias current: "
                  + f"{self.result.calibrated_parameters}")

        builder = sta.PlaybackProgramBuilder()

        # restore original neuron config
        for neuron_coord, neuron_config in zip(
                halco.iter_all(halco.NeuronConfigOnDLS),
                self.original_neuron_config):
            builder.write(neuron_coord, neuron_config)

        base.run(connection, builder)
