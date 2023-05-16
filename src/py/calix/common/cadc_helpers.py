"""
Provides functions for reading the CADCs and configuring for
calibration purposes.
"""

from typing import Tuple
import numpy as np
from dlens_vx_v3 import halco, hal, sta, hxcomm

from calix.common import base, helpers


def configure_readout_cadc_debug(
        builder: base.WriteRecordingPlaybackProgramBuilder) \
        -> base.WriteRecordingPlaybackProgramBuilder:
    """
    Writes the readout chain configuration in a way that allows
    reading the capmem debug cell output voltage at the
    CADC debug line.
    Enables the capmem debug output on stp_v_charge_0.

    :param builder: Builder to append the bit-configuration to.
    :return: Builder with configuration appended to.
    """

    mux_config = hal.PadMultiplexerConfig()

    # connect CapMem debug output of one quadrant to pad
    mux_config.capmem_v_out_mux[halco.CapMemBlockOnDLS()] = True  # pylint: disable=unsupported-assignment-operation
    mux_config.capmem_v_out_mux_to_capmem_intermediate_mux = True
    mux_config.capmem_intermediate_mux_to_pad = True

    # connect CADC debug lines to the same pad
    mux_config.cadc_debug_acausal_to_synapse_intermediate_mux = True
    mux_config.cadc_debug_causal_to_synapse_intermediate_mux = True
    mux_config.synapse_intermediate_mux_to_pad = True

    builder.write(halco.PadMultiplexerConfigOnDLS(), mux_config)

    # Enable capmem debug output in one quadrant
    # This should be replaced with an external DAC as voltage source:
    # cf. issue 3953
    capmem_config = hal.CapMemBlockConfig()
    capmem_config.debug_readout_enable = True
    capmem_config.debug_capmem_coord = \
        halco.CapMemCellOnCapMemBlock.stp_v_charge_0

    builder.write(halco.CapMemBlockConfigOnDLS(), capmem_config)

    # set shift register default: otherwise, the pad may be connected
    # to something on the board.
    builder.write(halco.ShiftRegisterOnBoard(), hal.ShiftRegister())

    return builder


def configure_chip(builder: base.WriteRecordingPlaybackProgramBuilder
                   ) -> base.WriteRecordingPlaybackProgramBuilder:
    """
    Configures the chip from an arbitrary state to run CADC calibration.

    Set relevant CapMem values of the CADC, connect debug lines to the CADC and
    set all CADCChannels to default (with zero offsets).

    :param builder: Builder to append chip configuration commands to.
    :return: Builder with configuring commands appended.
    """

    # static capmem config
    parameters = {
        halco.CapMemCellOnCapMemBlock.cadc_v_bias_ramp_buf: 400,
        halco.CapMemCellOnCapMemBlock.cadc_i_bias_comp: 1022,
        halco.CapMemCellOnCapMemBlock.cadc_i_bias_vreset_buf: 1022
    }
    builder = helpers.capmem_set_quadrant_cells(builder, parameters)

    # global digital CADC block config
    cadc_config = hal.CADCConfig()
    cadc_config.enable = True

    for cadc_coord in halco.iter_all(halco.CADCConfigOnDLS):
        builder.write(cadc_coord, cadc_config)

    # Column Switch (top synapse row): Connect CADCs to debug line
    quad_config = hal.ColumnCorrelationQuad()
    switch_config = hal.ColumnCorrelationQuad.ColumnCorrelationSwitch()
    switch_config.enable_debug_causal = True
    switch_config.enable_debug_acausal = True

    for switch_coord in halco.iter_all(halco.EntryOnQuad):
        quad_config.set_switch(switch_coord, switch_config)

    for quad_coord in halco.iter_all(halco.ColumnCorrelationQuadOnDLS):
        builder.write(quad_coord, quad_config)

    # Configure zero offsets for all CADC channels
    for channel_coord in halco.iter_all(halco.CADCChannelConfigOnDLS):
        builder.write(channel_coord, hal.CADCChannelConfig())

    return builder


def cadc_read_row(
        builder: base.WriteRecordingPlaybackProgramBuilder,
        synram: halco.SynramOnDLS
) -> Tuple[base.WriteRecordingPlaybackProgramBuilder, sta.ContainerTicket]:
    """
    Read one row of CADCs and return the read ticket.

    :param builder: Builder to append the read instruction to.
    :param synram: Synram coordinate of row to read.

    :return: Tuple containing:
        * Builder with read instruction appended
        * CADC read ticket
    """

    coord = halco.CADCSampleRowOnDLS(
        block=halco.SynapseRowOnSynram(),
        synram=synram)
    ticket = builder.read(coord)

    return builder, ticket


def read_cadcs(
        connection: hxcomm.ConnectionHandle,
        builder: base.WriteRecordingPlaybackProgramBuilder = None) \
        -> np.ndarray:
    """
    Read all CADC channels in order top causal, top acausal,
    bottom causal, bottom acausal. Dump results into a numpy array.

    The readout of the top row is triggered before the bottom row,
    such that there is a delay of some 100 us between the measurements.

    :param connection: Connection to a chip to read CADC channels from.
    :param builder: Optional builder to be run before reading the CADCs.
        This allows reading fast after doing other operations.

    :return: Array containing integer results for each channel.
        The order is top causal, top acausal, bottom causal, bottom acausal;
        within these blocks the channels are ordered from left to right.
        The order of returned results corresponds to the enums of
        halco.CADCChannelConfigOnDLS.
    """

    # Construct read builder
    read_tickets = []
    read_builder = base.WriteRecordingPlaybackProgramBuilder()

    for synram in halco.iter_all(halco.SynramOnDLS):
        read_builder, ticket = cadc_read_row(read_builder, synram)
        read_tickets.append(ticket)

    # Run builder
    if builder is not None:
        builder.merge_back(read_builder)
        base.run(connection, builder)
    else:
        base.run(connection, read_builder)

    # Extract results
    results = np.empty(halco.CADCChannelConfigOnDLS.size, dtype=int)

    for ticket_id, ticket in enumerate(read_tickets):
        # causal channels
        results[halco.CADCChannelType.size * ticket_id
                * halco.SynapseOnSynapseRow.size:
                (halco.CADCChannelType.size * ticket_id + 1)
                * halco.SynapseOnSynapseRow.size] = \
            ticket.get().causal.to_numpy()

        # acausal channels
        results[(halco.CADCChannelType.size * ticket_id + 1)
                * halco.SynapseOnSynapseRow.size:
                (halco.CADCChannelType.size * ticket_id + 2)
                * halco.SynapseOnSynapseRow.size] = \
            ticket.get().acausal.to_numpy()

    return results


def reshape_cadc_quadrants(cadc_results: np.ndarray) -> np.ndarray:
    """
    Reshapes an array containing all CADC samples as obtained with the
    read_cadcs() function (shape: (1024,)) into a two-dimensional array
    that holds the CADC results per quadrant (shape: (4, 256)).

    :param cadc_results: Flat array of CADC results.

    :return: Two-dimensional array of CADC results with the first index
        being the index of the Quadrant.
    """

    cadc_channels = list(halco.iter_all(halco.CADCChannelConfigOnDLS))
    quadrant_reads = []

    for quadrant_coord in halco.iter_all(halco.NeuronConfigBlockOnDLS):
        quadrant_reads.append(
            [read for coord, read in zip(cadc_channels, cadc_results)
             if coord.toNeuronConfigBlockOnDLS() == quadrant_coord])

    return np.array(quadrant_reads)


def convert_success_masks(quadrant_mask: np.ndarray) -> np.ndarray:
    """
    Converts a mask showing the calibration success of quadrants to a
    mask showing the success of each channel. If a quadrant shows False,
    all channels on that quadrants are set to False.

    :param quadrant_mask: Boolean array of length 4, i.e. matching the
        quadrants on chip.

    :return: Boolean array of length 1024, i.e. matching all CADC channels.
    """

    channel_mask = np.empty(halco.CADCChannelConfigOnDLS.size, dtype=bool)

    # CADCChannelConfigOnDLS are not ordered by quadrant. Therefore, we
    # iterate over all channels and choose for each channel the result of the
    # correct quadrant
    for channel_id, channel_coord in enumerate(
            halco.iter_all(halco.CADCChannelConfigOnDLS)):
        channel_mask[channel_id] = quadrant_mask[
            int(channel_coord.toNeuronConfigBlockOnDLS().toEnum())]

    return channel_mask
