"""
Provides functions for reading neuron membrane potentials and configuring
them for integration.
"""

from __future__ import annotations
from typing import Tuple, Union, List, Optional
import numpy as np
import quantities as pq
from dlens_vx_v1 import hal, sta, halco, lola, hxcomm

from calix.common import cadc_helpers, helpers
from calix import constants


def inspect_read_tickets(
        read_tickets: Union[
            sta.ContainerTicket_CADCSampleRow,
            List[sta.ContainerTicket_CADCSampleRow]]
) -> np.ndarray:
    """
    Iterate the given read tickets and return the contained results as an
    array.

    The values are extracted from the tickets and for each column (and ticket)
    the average of the causal and acausal channel is calculated. Note
    that this only makes sense if both causal and acausal channels are
    connected to the same potential, here reading from the neurons.

    :param read_tickets: List of read tickets to be evaluated, or single
        ticket.

    :return: Array of CADC averaged reads. If a single ticket is provided,
        the array is one-dimensional, containing results for all reads
        in a row. If a list of tickets is provided, the array is
        two-dimensional, with the outer dimension matching the order of
        tickets in the list.
    """

    if not isinstance(read_tickets, list):
        read_tickets = [read_tickets]
        was_list = False
    else:
        was_list = True

    results = np.empty((len(read_tickets), halco.SynapseOnSynapseRow.size))

    for ticket_id, ticket in enumerate(read_tickets):
        data = ticket.get()
        results[ticket_id] = np.mean([
            data.causal.to_numpy(),
            data.acausal.to_numpy()], axis=0)

    if not was_list:
        results = results[0]

    return results


class CADCReadNeurons(sta.PlaybackGenerator):
    """
    Read CADC channels in both synrams, and return a Result object,
    which interprets the reads as neuron results. Note that the
    CADCs have to be connected to the neurons for this to make sense.
    After the builder has been run, the Result can be evaluated.
    """

    class Result:
        """
        Result object holding membrane potentials of all neurons.

        :ivar tickets: List of CADCSampleRow tickets containing the
            reads for top and bottom synram, respectively.
        """

        def __init__(self, tickets: List[
                sta.ContainerTicket_CADCSampleRow]):
            self.tickets = tickets

        def to_numpy(self) -> np.ndarray:
            """
            Return a numpy array containing CADC reads interpreted as
            neuron results. Note that the CADCs have to be connected
            to the neurons for this to make sense.

            :return: Potential acquired from all neurons.
            """

            return inspect_read_tickets(self.tickets).flatten()

    @classmethod
    def generate(cls) -> Tuple[sta.PlaybackProgramBuilder,
                               CADCReadNeurons.Result]:
        """
        Generate a builder with CADC read instructions for both synrams.

        :return: Tuple containing:
            * Builder containing the read instructions.
            * Result object that can be processed.
        """

        builder = sta.PlaybackProgramBuilder()
        tickets = list()
        for synram in halco.iter_all(halco.SynramOnDLS):
            coord = halco.CADCSampleRowOnDLS(
                block=halco.SynapseRowOnSynram(),
                synram=synram)
            tickets.append(builder.read(coord))

        return builder, cls.Result(tickets)


def cadc_read_neuron_potentials(connection: hxcomm.ConnectionHandle,
                                builder: sta.PlaybackProgramBuilder = None
                                ) -> np.ndarray:
    """
    Read from the CADCs and interpret the results as membrane potentials.

    Use acausal and causal channels for reading neuron output voltages.
    We expect the neurons' readout amplifiers to be enabled and connected
    to both acausal and causal channels. We return the mean of the two
    channels as a neuron membrane read.

    When supplying a builder, it is used to read.
    Note that between reading top and bottom neurons, some 100 us pass,
    i.e. this function is not suitable for reading integrated amplitudes.

    :param connection: Connection to chip to read membrane potentials.
    :param builder: Builder to append read instructions to before execution.

    :return: Numpy array containing the results for all neurons.
    """

    # Create read ticket
    read_builder, ticket = CADCReadNeurons().generate()
    read_builder = helpers.wait(read_builder, 100 * pq.us)

    # Merge with builder, if given
    if builder is not None:
        builder.merge_back(read_builder)
        sta.run(connection, builder.done())
    else:
        sta.run(connection, read_builder.done())

    # Evaluate result
    return ticket.to_numpy()


def reshape_neuron_quadrants(neuron_reads: np.ndarray) -> np.ndarray:
    """
    Reshape the flat reads for all neurons on the chip to a two-dimensional
    array containing the 4 quadrants in the first dimension and 128 neuron
    reads each in the second dimension.

    :param neuron_reads: Flat array of 512 neuron reads.

    :return: Reshaped array containing quadrant results.
    """

    output_array = np.empty((halco.NeuronConfigBlockOnDLS.size,
                             halco.NeuronConfigOnNeuronConfigBlock.size),
                            dtype=neuron_reads.dtype)

    for neuron_coord, result in zip(
            halco.iter_all(halco.NeuronConfigOnDLS), neuron_reads):
        output_array[
            int(neuron_coord.toNeuronConfigBlockOnDLS().toEnum()),
            int(neuron_coord.toNeuronConfigOnNeuronConfigBlock().toEnum())] = \
            result

    return output_array


# builders with stored neuron reset commands
# caching these saves 25 ms per call of reset_neurons().
neuron_reset_builders = {_coord: sta.PlaybackProgramBuilder() for _coord
                         in halco.iter_all(halco.SynramOnDLS)}
for _synram in halco.iter_all(halco.SynramOnDLS):
    for _quad_coord in halco.iter_all(halco.SynapseQuadColumnOnDLS):
        _coord = halco.NeuronResetQuadOnDLS(_quad_coord, _synram)
        neuron_reset_builders[_synram].write(_coord, hal.NeuronResetQuad())


def reset_neurons(builder: sta.PlaybackProgramBuilder,
                  synram: halco.SynramOnDLS = None
                  ) -> sta.PlaybackProgramBuilder:
    """
    Trigger an artificial reset in all neurons.

    The membrane potential gets pulled back to the reset potential.

    :param builder: Builder to append reset instructions to.
    :param synram: Synram of neurons to be reset. If None, all neurons on
        the chip are reset.

    :return: Builder with reset instructions appended.
    """

    # copy stored reset builders to given builder to save runtime
    if synram is None:
        for coord in halco.iter_all(halco.SynramOnDLS):
            builder.copy_back(neuron_reset_builders[coord])
    else:
        builder.copy_back(neuron_reset_builders[synram])

    return builder


def cadc_read_neurons_repetitive(
        connection: hxcomm.ConnectionHandle,
        builder: sta.PlaybackProgramBuilder, *,
        synram: halco.SynramOnDLS,
        n_reads: int = 50,
        wait_time: pq.quantity.Quantity = 1000 * pq.us,
        reset: bool = False) -> np.ndarray:
    """
    Read the potential from all neurons multiple times.

    Reads all CADC channels multiple times and calculates the mean of
    acausal and causal reads as neuron voltage. Note that the CADCs
    have to be connected to the neurons for this to make sense.
    Returns a 2-dimensional array of the results.

    :param connection: Connection to the chip to run on.
    :param builder: Builder to use for reads. Gets executed along the way.
    :param synram: Synram coordinate to read neuron potentials from.
    :param n_reads: Number of reads to execute for all channels.
    :param wait_time: Time to wait between two successive reads.
    :param reset: Select whether the neurons' membranes are connected
        to the reset potential some 30 us before measuring.

    :return: Array containing all obtained values. The first dimension is
        the number of reads, the second the number of neurons on synram.
    """

    read_tickets = list()

    for _ in range(n_reads):
        # Trigger neuron resets if desired
        if reset:
            builder = reset_neurons(builder, synram=synram)

            # wait the time a vector needs to be input
            builder = helpers.wait(builder, 30 * pq.us)

        # Read CADCs
        builder, ticket = cadc_helpers.cadc_read_row(builder, synram)
        read_tickets.append(ticket)

        # Wait before next measurement
        builder = helpers.wait(builder, wait_time)

    builder = helpers.wait(builder, 100 * pq.us)  # wait for transfers
    sta.run(connection, builder.done())

    # Inspect ticket results
    return inspect_read_tickets(read_tickets)


def neuron_config_default() -> hal.NeuronConfig:
    """
    Return a neuron configuration suitable for integration.

    Mainly, the synaptic inputs and membrane readout get enabled.
    Enabling leak divisions allows to achieve long membrane time constants,
    as required for integration.

    :return: NeuronConfig suitable for integration.
    """

    neuron_config = hal.NeuronConfig()
    neuron_config.enable_readout_amplifier = True
    neuron_config.membrane_capacitor_size = \
        hal.NeuronConfig.MembraneCapacitorSize.max
    neuron_config.enable_synaptic_input_excitatory = True
    neuron_config.enable_synaptic_input_inhibitory = True
    neuron_config.enable_leak_degeneration = True
    neuron_config.enable_leak_division = True
    neuron_config.enable_fire = True  # necessary for hal.NeuronResetQuad
    neuron_config.enable_reset_multiplication = True
    neuron_config.readout_source = hal.NeuronConfig.ReadoutSource.membrane

    return neuron_config


def configure_integration(builder: sta.PlaybackProgramBuilder
                          ) -> sta.PlaybackProgramBuilder:
    """
    Applies static configuration required for integrate-operation of
    the neurons in hagen mode.

    Configures the chip such that the CADCs can read out neuron membranes.
    This means setting the switches to connect the columns to the CADC and
    enabling the readout amplifiers in each neuron.

    :param builder: Builder to append configuration instructions to.

    :return: Builder with configuration.
    """

    # Column Switches at the edges of the synapse array
    # Connect the CADCs to the neurons
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

    # Connect the neurons' synaptic inputs to the synapse array
    quad_config = hal.ColumnCurrentQuad()
    switch_config = hal.ColumnCurrentQuad.ColumnCurrentSwitch()
    switch_config.enable_synaptic_current_excitatory = True
    switch_config.enable_synaptic_current_inhibitory = True

    for switch_coord in halco.iter_all(halco.EntryOnQuad):
        quad_config.set_switch(switch_coord, switch_config)

    for quad_coord in halco.iter_all(halco.ColumnCurrentQuadOnDLS):
        builder.write(quad_coord, quad_config)

    # Neuron Config
    neuron_config = neuron_config_default()
    for neuron_coord in halco.iter_all(halco.NeuronConfigOnDLS):
        builder.write(neuron_coord, neuron_config)

    # Configure refractory clocks:
    # slow clock to measure v_reset: roughly 160 us refractory
    # fast clock to reset membrane in usage: roughly 2 us refractory
    neuron_common_config = hal.CommonNeuronBackendConfig()
    neuron_common_config.clock_scale_slow = 9
    neuron_common_config.clock_scale_fast = 2

    for coord in halco.iter_all(halco.CommonNeuronBackendConfigOnDLS):
        builder.write(coord, neuron_common_config)

    # select fast clock for resetting integrated potentials
    neuron_config = hal.NeuronBackendConfig()
    neuron_config.refractory_time = 70
    neuron_config.select_input_clock = 1

    for coord in halco.iter_all(halco.NeuronBackendConfigOnDLS):
        builder.write(coord, neuron_config)

    # select timing of reset pulse for neurons
    config = hal.CommonCorrelationConfig()
    config.reset_duration = 5  # suffices for neuron reset

    for coord in halco.iter_all(halco.CommonCorrelationConfigOnDLS):
        builder.write(coord, config)

    return builder


def configure_synapses(
        builder: sta.PlaybackProgramBuilder,
        n_synapse_rows: int = 8, *,
        stimulation_address: hal.SynapseQuad.Label = hal.SynapseQuad.Label(0),
        weight: hal.SynapseQuad.Weight = hal.SynapseQuad.Weight(10)
) -> sta.PlaybackProgramBuilder:
    """
    Configures the synapses such that events can be sent to the neurons.
    The given number of synapse rows are enabled.
    This function does not configure synapse drivers, as they need to
    select excitatory/inhibitory row modes when events are sent.

    :param builder: Builder to append configuration instructions to.
    :param n_synapse_rows: Number of rows in which the synapses are enabled.
    :param stimulation_address: Address to use for inputs. Disabled synapses
        get set to (stimulation_address + 1) % address_range.
    :param weight: Weight of enabled synapses.

    :return: Builder with configuration.
    """

    # Synapse connections
    synapses = lola.SynapseMatrix()
    weights = np.zeros(halco.SynapseRowOnSynram.size, dtype=int)
    weights[:n_synapse_rows] = int(weight)
    weights = np.repeat(
        weights[:, np.newaxis], halco.SynapseOnSynapseRow.size, axis=1)
    disabled_address = \
        (int(stimulation_address) + 1) % hal.SynapseQuad.Label.max
    addresses = np.ones(halco.SynapseRowOnSynram.size, dtype=int) \
        * disabled_address
    addresses[:n_synapse_rows] = int(stimulation_address)
    addresses = np.repeat(
        addresses[:, np.newaxis], halco.SynapseOnSynapseRow.size, axis=1)

    synapses.weights.from_numpy(weights)
    synapses.labels.from_numpy(addresses)

    for synram in halco.iter_all(halco.SynramOnDLS):
        builder.write(synram, synapses)

    return builder


def configure_stp_and_padi(builder: sta.PlaybackProgramBuilder
                           ) -> sta.PlaybackProgramBuilder:
    """
    Configure global STP config and PADI bus config to default.

    :param builder: Builder to append instructions to.

    :return: Builder with instructions appended.
    """

    # STP global config
    global_stp_config = hal.CommonSTPConfig()

    for stp_config_coord in halco.iter_all(halco.CommonSTPConfigOnDLS):
        builder.write(stp_config_coord, global_stp_config)

    # PADI global config
    global_padi_config = hal.CommonPADIBusConfig()

    for padi_config_coord in halco.iter_all(halco.CommonPADIBusConfigOnDLS):
        builder.write(padi_config_coord, global_padi_config)

    return builder


def enable_all_synapse_drivers(builder: sta.PlaybackProgramBuilder,
                               row_mode: hal.SynapseDriverConfig.RowMode
                               ) -> sta.PlaybackProgramBuilder:
    """
    Configure synapse drivers to drive both connected rows of
    synapses either excitatory or inhibitory.
    All drivers listen to all row select addresses of PADI events.

    :param builder: Builder to append configuration to.
    :param row_mode: Row mode (excitatory/inhibitory) to use for both rows.

    :return: Builder with configuration appended.
    """

    synapse_driver_config = hal.SynapseDriverConfig()
    synapse_driver_config.enable_address_out = True
    synapse_driver_config.enable_receiver = True
    synapse_driver_config.row_mode_top = row_mode
    synapse_driver_config.row_mode_bottom = row_mode
    synapse_driver_config.row_address_compare_mask = 0b00000

    for synapse_driver_coord in halco.iter_all(
            halco.SynapseDriverOnDLS):
        builder.write(synapse_driver_coord, synapse_driver_config)

    return builder


def set_analog_neuron_config(builder: sta.PlaybackProgramBuilder,
                             v_leak: Union[int, np.ndarray], i_leak: int
                             ) -> sta.PlaybackProgramBuilder:
    """
    Configure all neurons' CapMem cells to the given v_leak potential and
    the given i_leak bias current. Also turn off the synaptic input bias
    currents, since the synaptic inputs can not be used without
    calibration. The reset bias current is set maximum.
    Furthermore, static required biases are set.

    :param builder: Builder to append the configuration to.
    :param v_leak: CapMem setting of the leak potential for all neurons.
        If an array is given, it is written to the neurons following
        enumeration of `halco.NeuronConfig`s. If a single integer is
        given, some noise (+- 5) is added before writing the values.
        The reset voltage is set as well, always equal to the leak voltage.
    :param i_leak: CapMem setting of the leak bias current for all
        neurons, noise will be added as for v_leak.

    :return: Builder with configuration instructions appended.
    """

    # set individual neuron parameters to sensible defaults
    parameters = {
        halco.CapMemRowOnCapMemBlock.i_bias_syn_exc_gm: 0,
        halco.CapMemRowOnCapMemBlock.i_bias_syn_inh_gm: 0,
        halco.CapMemRowOnCapMemBlock.i_bias_leak: i_leak,
        halco.CapMemRowOnCapMemBlock.v_leak: v_leak,
        halco.CapMemRowOnCapMemBlock.i_bias_reset: 1015,
        halco.CapMemRowOnCapMemBlock.v_reset: v_leak,
        halco.CapMemRowOnCapMemBlock.v_syn_exc: 655,
        halco.CapMemRowOnCapMemBlock.v_syn_inh: 645,
        halco.CapMemRowOnCapMemBlock.i_bias_syn_exc_res: 1021,
        halco.CapMemRowOnCapMemBlock.i_bias_syn_inh_res: 1020,
        halco.CapMemRowOnCapMemBlock.i_bias_source_follower: 490,
        halco.CapMemRowOnCapMemBlock.i_bias_readout: 1000
    }
    builder = helpers.capmem_set_neuron_cells(builder, parameters)

    # set per-quadrant parameters
    parameters = {
        halco.CapMemCellOnCapMemBlock.neuron_i_bias_synin_sd_exc: 1022,
        halco.CapMemCellOnCapMemBlock.neuron_i_bias_synin_sd_inh: 1022,
        halco.CapMemCellOnCapMemBlock.syn_i_bias_dac: 1022,
        halco.CapMemCellOnCapMemBlock.neuron_i_bias_threshold_comparator: 200
    }
    builder = helpers.capmem_set_quadrant_cells(builder, parameters)

    # enable readout buffers
    builder.write(halco.CapMemCellOnDLS.readout_out_amp_i_bias_0,
                  hal.CapMemCell(1022))
    builder.write(halco.CapMemCellOnDLS.readout_out_amp_i_bias_1,
                  hal.CapMemCell(1022))

    builder = helpers.wait(builder, 5 * constants.capmem_level_off_time)

    return builder


# pylint: disable=unsupported-assignment-operation
def configure_readout_neurons(
        builder: sta.PlaybackProgramBuilder,
        readout_neuron: halco.AtomicNeuronOnDLS
) -> sta.PlaybackProgramBuilder:
    """
    Configure the readout such that the membrane potential of one neuron
    and the CADC ramp is available at the pads.

    When using an oscilloscope, you may set the trigger to the CADC
    ramp in order to observe CADC measurements. Note, however,
    that connecting a simple cable to the CADC ramp output will
    not suffice, a low-capacitance probe is required to read the
    ramp correctly.

    :param builder: Builder to append configuration to.
    :param readout_neuron: Coordinate of the neuron to be connected to
        a readout pad, i.e. can be observed using an oscilloscope.

    :return: Builder with configuration instructions.
    """

    # Reconfigure readout_neuron to connect to readout lines
    neuron_config = neuron_config_default()
    neuron_config.enable_readout = True
    builder.write(readout_neuron.toNeuronConfigOnDLS(), neuron_config)

    # Enable readout of neuron on buffer 0
    mux_config = hal.ReadoutSourceSelection.SourceMultiplexer()
    is_odd = bool(readout_neuron.x() % 2)
    if is_odd:
        mux_config.neuron_odd[
            readout_neuron.y().toHemisphereOnDLS()] = True
    else:
        mux_config.neuron_even[
            readout_neuron.y().toHemisphereOnDLS()] = True

    block_config = hal.ReadoutSourceSelection()
    block_config.set_buffer(
        halco.SourceMultiplexerOnReadoutSourceSelection(0),
        mux_config)
    block_config.enable_buffer_to_pad[
        halco.SourceMultiplexerOnReadoutSourceSelection(0)] = True
    builder.write(halco.ReadoutSourceSelectionOnDLS(), block_config)

    # Enable source multiplexer 0 to top pad
    mux_config = hal.PadMultiplexerConfig()
    mux_config.buffer_to_pad[
        halco.SourceMultiplexerOnReadoutSourceSelection(0)] = True
    builder.write(halco.PadMultiplexerConfigOnDLS(0), mux_config)

    # Enable CADC ramp readout on lower pad
    mux_config = hal.PadMultiplexerConfig()
    mux_config.cadc_v_ramp_mux[halco.CapMemBlockOnDLS(0)] = True
    mux_config.cadc_v_ramp_mux_to_pad = True
    builder.write(halco.PadMultiplexerConfigOnDLS(1), mux_config)

    return builder


def configure_chip(builder: sta.PlaybackProgramBuilder,
                   v_leak: int = 700, n_synapse_rows: int = 8,
                   readout_neuron: Optional[halco.AtomicNeuronOnDLS] = None
                   ) -> sta.PlaybackProgramBuilder:
    """
    Does all the necessary configurations on the chip to start calibrating
    the neurons for usage in hagen mode.
    Configures neurons statically for integration, sets synapse connections,
    global synapse driver and PADI settings, CapMem and readout settings.

    :param builder: Builder to append configuration to.
    :param v_leak: Leak potential CapMem setting to start calibrations with.
    :param n_synapse_rows: Number of synapse rows to enable.
    :param readout_neuron: Coordinate of the neuron to be connected to
        a readout pad, i.e. can be observed using an oscilloscope.
        or the MADC. The neuron is connected to the upper pad via
        readout mux 0, CADC ramp is connected to the lower pad
        via readout mux 1.
        If None, neither the CADC ramp nor the neuron is connected
        to the pads, the readout chain configuration is untouched.

    :return: Builder with configuration instructions appended.
    """

    # Preconfigure chip: Configure Neuron readout
    builder = configure_integration(builder)

    # Preconfigure chip: Configure synapses and drivers/PADI globally
    builder = configure_synapses(builder, n_synapse_rows)
    builder = configure_stp_and_padi(builder)

    # Preconfigure capmem for calibration
    builder = set_analog_neuron_config(builder, v_leak, i_leak=200)

    # Preconfigure readout (MADC register)
    if readout_neuron:
        builder = configure_readout_neurons(builder, readout_neuron)

    return builder


def reconfigure_synaptic_input(
        connection: hxcomm.ConnectionHandle,
        excitatory_biases: Optional[Union[int, np.ndarray]] = None,
        inhibitory_biases: Optional[Union[int, np.ndarray]] = None
) -> None:
    """
    Reconfigures the excitatory and inhibitory synaptic inputs of all neurons.

    If excitatory_biases are given, they are written to the excitatory
    synaptic input bias current cells. If the bias current is zero,
    it is disabled in the neuron config, if another number is given,
    it is enabled. If a single integer is given, all neurons are configured
    to this value plus some noise. If an array of values is provided,
    each neuron's synaptic input OTA bias is set to the given values.

    If excitatory_biases are None, the excitatory synaptic input is not
    reconfigured at all.
    Configuring the inhibitory synaptic input works similarly.

    This function reads the current neuron config, turns on/off the desired
    synaptic inputs as described above, and writes the config back to the
    neurons.

    :param connection: Connection to chip to run on.
    :param excitatory_biases: Neuron excitatory synaptic input OTA
        bias current setting to configure.
    :param inhibitory_biases: Neuron excitatory synaptic input OTA
        bias current setting to configure.
    """

    if excitatory_biases is None and inhibitory_biases is None:
        return

    # Read current neuron configurations
    builder = sta.PlaybackProgramBuilder()
    read_tickets = list()

    for neuron_coord in halco.iter_all(halco.NeuronConfigOnDLS):
        read_tickets.append(builder.read(neuron_coord))
    builder = helpers.wait(builder, 50 * pq.us)  # wait for transfers
    sta.run(connection, builder.done())

    # Reconfigure CapMem bias currents
    builder = sta.PlaybackProgramBuilder()
    config = dict()
    if excitatory_biases is not None:
        config.update({
            halco.CapMemRowOnCapMemBlock.i_bias_syn_exc_gm:
            excitatory_biases})
    if inhibitory_biases is not None:
        config.update({
            halco.CapMemRowOnCapMemBlock.i_bias_syn_inh_gm:
            inhibitory_biases})
    builder = helpers.capmem_set_neuron_cells(builder, config)

    # inspect given biases more closely to decide per-neuron
    # which synaptic inputs to reconfigure
    if isinstance(excitatory_biases, int):
        excitatory_biases = np.ones(
            halco.NeuronConfigOnDLS.size, dtype=int) * excitatory_biases
    if isinstance(inhibitory_biases, int):
        inhibitory_biases = np.ones(
            halco.NeuronConfigOnDLS.size, dtype=int) * inhibitory_biases

    # Reconfigure inputs in neuron configurations
    neuron_configs = [ticket.get() for ticket in read_tickets]

    if excitatory_biases is not None:
        for neuron_id, neuron_config in enumerate(neuron_configs):
            neuron_config.enable_synaptic_input_excitatory = bool(
                excitatory_biases[neuron_id])
    if inhibitory_biases is not None:
        for neuron_id, neuron_config in enumerate(neuron_configs):
            neuron_config.enable_synaptic_input_inhibitory = bool(
                inhibitory_biases[neuron_id])

    # write neuron configs
    for neuron_coord, neuron_config in zip(
            halco.iter_all(halco.NeuronConfigOnDLS), neuron_configs):
        builder.write(neuron_coord, neuron_config)

    sta.run(connection, builder.done())
