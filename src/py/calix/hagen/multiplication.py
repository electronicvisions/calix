"""
Provides a class which handles multiplication of vectors with
matrices on the synaptic input line.
"""

from typing import Tuple, List, Union
import time
import numpy as np
import quantities as pq

from dlens_vx_v3 import lola, halco, hal, sta, logger, hxcomm

from calix.common import base, helpers
from calix.hagen import neuron_helpers
from calix import constants


class Multiplication:
    """
    Handles multiplication of vector and matrix with integration
    on the synaptic input lines.

    Requirements:
    * CADCs are calibrated and connected to the neuron readout. You can
      use `calix.common.cadc.calibrate()` and
      `calix.hagen.neuron_helpers.configure_chip()` to achieve this.
    * Synapse DAC bias is calibrated.
    * Synapse drivers are calibrated.

    :cvar synram_selection_bit: Position of bit in event label that
        selects whether an SPL1 event reaches the top or bottom synram.
        You can choose one of bits 11 to 13 (default).

    :ivar _synram_coord: Coordinate of synapse array to use.
    :ivar num_sends: Number of sends of the vector, values greater
        than one indicate repetitions of the input events.
    :ivar wait_period: Number of clock cycles to wait between events.
    :ivar signed_mode: Decide whether the multiplication is using signed
        weights. This affects the shape of the matrix: in unsigned mode,
        i.e. signed_mode = False, weights are shaped (256, 256).
        If using signed weights, the shape is reduced to (128, 256),
        i.e. 256 inputs are mapped to 128 results.
    :ivar cached_reset_synin: Cached builder containing instructions
        to reset synaptic input lines. Contains a call to reset_synin().
    """

    synram_selection_bit = 13

    def __init__(self, synram: halco.SynramOnDLS = halco.SynramOnDLS(), *,
                 num_sends: int = 1, wait_period: int = 1,
                 signed_mode: bool = True):
        self._synram_coord = synram
        self.num_sends = num_sends
        if num_sends < 1:
            raise ValueError("No events are sent if number of sends is < 1.")
        self.wait_period = wait_period
        self.signed_mode = signed_mode

        # store cached reset_synin builder
        self.cached_reset_synin = sta.PlaybackProgramBuilder()
        self.reset_synin(self.cached_reset_synin)

    @property
    def synram_coord(self):
        """
        Get the synram to perform MAC operations on.
        """

        return self._synram_coord

    @synram_coord.setter
    def synram_coord(self, value):
        """
        Update the synram to perform MAC operations on.
        """

        self._synram_coord = value

        # update cached reset_synin builder
        self.cached_reset_synin = sta.PlaybackProgramBuilder()
        self.reset_synin(self.cached_reset_synin)

    @staticmethod
    def configure_for_integration(config: lola.AtomicNeuron) -> None:
        """
        Set static configuration for integration on excitatory synaptic
        input lines.

        :param config: Configuration container to be altered.
        """

        # Disable pullup of synaptic input lines (tau_syn -> inf) and
        # select the capacitor to be connected to the line.
        config.excitatory_input.enable = False
        config.excitatory_input.enable_small_capacitance = False
        config.excitatory_input.enable_high_resistance = True
        config.excitatory_input.i_bias_tau = 0
        config.inhibitory_input.enable = False
        config.inhibitory_input.enable_small_capacitance = False
        config.inhibitory_input.enable_high_resistance = True
        config.inhibitory_input.i_bias_tau = 0

        # We need the source follower before the synaptic input OTA biased
        # (i_drop_input, which is a global bias current) in order to see
        # the potential at the readout.

        # We choose the excitatory synaptic input line for the integration,
        # therefore we select it as readout source.
        config.readout.source = lola.AtomicNeuron.Readout.Source.exc_synin
        config.readout.enable_amplifier = True
        config.readout.enable_buffered_access = False

    @classmethod
    def preconfigure_crossbar(
            cls, builder: sta.PlaybackProgramBuilder) -> None:
        """
        Configure the crossbar such that the upper bits in an event
        label select target hemisphere and PADI bus. Also enable
        SPL1 events in the PADI buses.

        :param builder: Builder to append instructions to.
        """

        width_of_label = int(np.log2(halco.NeuronLabel.size))
        if width_of_label != 14:
            raise AssertionError(
                "Crossbar configuration is intended for 14-bit spike labels "
                + "reaching the crossbar.")

        # Each SPL1 input can be connected to all available PADI buses.
        # Here, we want to distribute the events of each of the
        # SPL1 inputs onto two different PADI buses, respectively,
        # one in the top hemisphere and one in the bottom hemisphere.
        # Therefore, we use a 1:1 mapping from SPL1Adress and
        # PADIBusOnPADIBusBlock, while at the same time using bit 13
        # of the neuron label to select to which hemisphere
        # (PADIBusBlockOnDLS) the event is forwarded.
        for input_coord in halco.iter_all(halco.SPL1Address):
            for output_coord in halco.iter_all(halco.PADIBusOnDLS):
                config = hal.CrossbarNode()
                if int(output_coord.toPADIBusOnPADIBusBlock().toEnum()) \
                        == int(input_coord.toEnum()):
                    # enable diagonal entries, filtering for their synram
                    config.mask = 1 << cls.synram_selection_bit
                    config.target = int(
                        output_coord.toPADIBusBlockOnDLS().toEnum()) \
                        << cls.synram_selection_bit
                else:  # disable non-diagonal entries
                    config.mask = 0
                    config.target = 1

                builder.write(halco.CrossbarNodeOnDLS(
                    y=input_coord.toCrossbarInputOnDLS(),
                    x=output_coord.toCrossbarOutputOnDLS()), config)

        # configure PADI busses to receive SPL1 events from crossbar
        padi_config = hal.CommonPADIBusConfig()
        enable_spl1 = padi_config.enable_spl1
        for padi_bus in halco.iter_all(halco.PADIBusOnPADIBusBlock):
            enable_spl1[padi_bus] = True
        padi_config.enable_spl1 = enable_spl1
        for coord in halco.iter_all(halco.CommonPADIBusConfigOnDLS):
            builder.write(coord, padi_config)

    @staticmethod
    def preconfigure_dac(builder: sta.PlaybackProgramBuilder) -> None:
        """
        Connect the external DAC to the synapse debug lines, supplying
        a voltage of 1.2 V.

        The potential will be used for resetting potentials on the
        synapse lines before integration.

        :param builder: Builder to append instructions to.
        """

        mux_config = hal.PadMultiplexerConfig()
        mux_config.synin_debug_excitatory_to_synapse_intermediate_mux = True
        mux_config.synin_debug_inhibitory_to_synapse_intermediate_mux = True
        mux_config.synapse_intermediate_mux_to_pad = True
        builder.write(halco.PadMultiplexerConfigOnDLS(0), mux_config)
        builder.write(halco.PadMultiplexerConfigOnDLS(1),
                      hal.PadMultiplexerConfig())

        # connect the channel normally used for V_reset to pad 0
        config = hal.ShiftRegister()
        config.select_analog_readout_mux_1_input = \
            config.AnalogReadoutMux1Input.readout_chain_0
        config.select_analog_readout_mux_2_input = \
            config.AnalogReadoutMux2Input.v_reset
        builder.write(halco.ShiftRegisterOnBoard(), config)

        config = lola.DACChannelBlock().default_ldo_2
        config.set_voltage(halco.DACChannelOnBoard.mux_dac_25, 1.2)
        builder.write(halco.DACChannelBlockOnBoard(), config)

    def preconfigure(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Configure synapse drivers, neurons, crossbar and external DAC.

        The previously configured hagen dac offsets and STP ramp offsets
        are kept unchanged.

        :param connection: Connection to the chip to run on.
        """

        # read current synapse driver and neuron config (to keep calibration
        # of synapse drivers)
        syndrv_tickets = list()
        neuron_tickets = list()
        builder = sta.PlaybackProgramBuilder()
        for coord in halco.iter_all(halco.SynapseDriverOnSynapseDriverBlock):
            syndrv_tickets.append(builder.read(
                halco.SynapseDriverOnDLS(
                    coord,
                    block=self._synram_coord.toSynapseDriverBlockOnDLS())))
        for coord in halco.iter_all(halco.NeuronColumnOnDLS):
            neuron_tickets.append(builder.read(
                halco.AtomicNeuronOnDLS(
                    x=coord, y=self._synram_coord.toNeuronRowOnDLS())))
        base.run(connection, builder)

        # configure synapse drivers
        builder = sta.PlaybackProgramBuilder()
        for coord in halco.iter_all(halco.SynapseDriverOnSynapseDriverBlock):
            config = syndrv_tickets[coord.toEnum()].get()
            config.enable_address_out = False
            config.enable_receiver = True
            config.row_address_compare_mask = 0b11111
            config.enable_stp = True
            config.enable_hagen_modulation = True
            config.enable_hagen_dac = True
            config.row_mode_top = hal.SynapseDriverConfig.RowMode.excitatory
            config.row_mode_bottom = hal.SynapseDriverConfig.RowMode.excitatory

            builder.write(
                halco.SynapseDriverOnDLS(
                    coord,
                    block=self._synram_coord.toSynapseDriverBlockOnDLS()),
                config)

        # configure neurons
        for coord in halco.iter_all(halco.NeuronColumnOnDLS):
            config = neuron_tickets[coord.toEnum()].get()
            self.configure_for_integration(config)
            builder.write(halco.AtomicNeuronOnDLS(
                x=coord, y=self._synram_coord.toNeuronRowOnDLS()), config)

        self.preconfigure_crossbar(builder)
        self.preconfigure_dac(builder)

        # wait for CapMem, run
        helpers.wait(builder, constants.capmem_level_off_time)
        base.run(connection, builder)

    def get_synapse_matrix(self, matrix: np.ndarray) -> lola.SynapseMatrix:
        """
        Return a suitable lola SynapseMatrix depending on the
        requested weights.

        :param matrix: Weight matrix as a numpy array.
        :return: `lola.SynapseMatrix` with addresses and weights set.
        """

        synapses = lola.SynapseMatrix()

        if self.signed_mode:
            # split weights for every second column positive/negative
            weights = np.empty((halco.SynapseOnSynapseRow.size,
                                halco.SynapseRowOnSynram.size), dtype=int)
            excitatory_matrix = np.ma.masked_less(matrix.T, 0)
            excitatory_matrix.fill_value = 0
            weights[:, ::2] = excitatory_matrix.filled()  # pylint: disable=no-member

            inhibitory_matrix = np.ma.masked_greater(matrix.T, 0)
            inhibitory_matrix.fill_value = 0
            weights[:, 1::2] = np.abs(inhibitory_matrix.filled())  # pylint: disable=no-member
        else:
            weights = matrix.T

        # we want to send independent inputs to both rows of a synapse
        # driver. Only the uppermost address bit is sent to the
        # synapses, therefore we use 0 and 32.
        addresses = np.zeros(halco.SynapseRowOnSynram.size, dtype=int)
        addresses[1::2] = 32
        addresses = np.repeat(
            addresses[:, np.newaxis], halco.SynapseOnSynapseRow.size, axis=1)

        synapses.weights.from_numpy(weights)
        synapses.labels.from_numpy(addresses)

        return synapses

    def prepare_event(
            self, address: hal.SynapseQuad.Label,
            target_driver: halco.SynapseDriverOnSynapseDriverBlock
    ) -> hal.SpikePack1ToChip:
        """
        Return a spike pack to chip, containing an event reaching
        the desired synapse driver on the synram in use.

        :param address: Address that is sent to the driver. The
            MSB reaches the synapses, the lower 5 bit encode the desired
            activation.
        :param target_driver: Coordinate of the targeted driver.

        :return: Spike packet to chip.
        """

        label = halco.SpikeLabel()

        # select target PADI bus
        label.spl1_address = int(
            target_driver.toPADIBusOnPADIBusBlock().toEnum())

        # select target synram
        label.neuron_label = (int(
            self._synram_coord.toEnum()) << self.synram_selection_bit)

        # select target driver on the PADI bus
        label.row_select_address = int(
            target_driver.toSynapseDriverOnPADIBus().toEnum())

        # set address sent to the driver (MSB + hagen activation)
        label.synapse_label = address

        return hal.SpikePack1ToChip([label])

    def reset_synin(self,
                    builder: sta.PlaybackProgramBuilder):
        """
        Connect synapse lines to the debug lines shortly.

        The switches for excitatory and inhibitory synaptic currents
        are enabled.

        :param builder: Builder to append instructions to.
        """

        # Additional wait time for charging the synapse line:
        # Currently, we don't need more wait time since enabling all
        # switches takes long enough that by the time we start
        # disabling switches, the lines have charged long enough.
        # However, this may change if faster or more parallel writes
        # are used.
        charge_time = 0 * pq.us

        # enable all reset connections
        config = hal.ColumnCurrentQuad.ColumnCurrentSwitch()
        config.enable_synaptic_current_excitatory = True
        config.enable_synaptic_current_inhibitory = True
        config.enable_debug_excitatory = True
        config.enable_debug_inhibitory = True

        quad_config = hal.ColumnCurrentQuad()
        for coord in halco.iter_all(halco.EntryOnQuad):
            quad_config.set_switch(coord, config)

        for coord in halco.iter_all(halco.ColumnCurrentQuadOnSynram):
            builder.write(halco.ColumnCurrentQuadOnDLS(
                coord, self._synram_coord), quad_config)

        # wait an additional time
        builder = helpers.wait(builder, charge_time)

        # disable all reset connections
        config.enable_debug_excitatory = False
        config.enable_debug_inhibitory = False

        quad_config = hal.ColumnCurrentQuad()
        for coord in halco.iter_all(halco.EntryOnQuad):
            quad_config.set_switch(coord, config)

        for coord in halco.iter_all(halco.ColumnCurrentQuadOnSynram):
            builder.write(halco.ColumnCurrentQuadOnDLS(
                coord, self._synram_coord), quad_config)

    def _send_vectors(self, vectors: np.ndarray
                      ) -> Tuple[sta.PlaybackProgramBuilder, List]:
        """
        Send given vectors and multiply them with the previously
        configured synapse matrix.
        Return a list of CADC sample rows containing accumulated results.

        :param vectors: Array of vectors to be multiplied. We expect
            row vectors, with the inner dimension (columns) holding the
            individual entries of each vector.
        :return: Tuple containing:
            * Program builder with instructions to send vectors and
              read results
            * List of CADC sample row tickets, contianing results
        """

        read_tickets = list()
        builder = sta.PlaybackProgramBuilder()

        for vector in vectors.copy():
            # convert vector to address: high amplitude = low address
            vector[vector != 0] = 32 - vector[vector != 0]

            # Reset synapse lines
            builder.copy_back(self.cached_reset_synin)
            helpers.wait(builder, 10 * pq.us)

            # Optimize runtime if wait_period is 1 and multiple sends
            # are requested by copying a vector_builder for each send
            if self.num_sends > 1 and self.wait_period <= 1:  # pylint: disable=chained-comparison
                num_loops = 1
                num_copies = self.num_sends
            else:
                num_loops = self.num_sends
                num_copies = 1

            entry_counter = 0
            builder.write(halco.TimerOnDLS(), hal.Timer())

            vector_builder = sta.PlaybackProgramBuilder()
            for _ in range(num_loops):
                for row, entry in enumerate(vector):
                    if entry == 0:
                        continue

                    # send event on a different address in order to
                    # select one of the two rows connected to a driver
                    entry += 32 if row % 2 == 1 else 0
                    vector_builder.write(
                        halco.SpikePack1ToChipOnDLS(),
                        self.prepare_event(
                            halco.SynapseLabel(entry),
                            halco.SynapseDriverOnSynapseDriverBlock(row // 2)))

                    # wait only if needed:
                    if self.wait_period > 1:
                        vector_builder.wait_until(
                            halco.TimerOnDLS(), hal.Timer.Value(
                                int(self.wait_period * entry_counter)))
                    entry_counter += 1

            for _ in range(num_copies):
                builder.copy_back(vector_builder)

            # Read amplitudes
            coord = halco.CADCSampleRowOnDLS(
                block=halco.SynapseRowOnSynram(),
                synram=self._synram_coord)
            read_tickets.append(builder.read(coord))

        return builder, read_tickets

    def _check_shape(self, vectors: np.ndarray, matrix: np.ndarray):
        """
        Asserts the vecotrs and matrix fit on a synapse array.

        :raises ValueError: If the vector or matrix shapres are bad or
            contain entries outside the feasible range.
        """

        if np.any(vectors < hal.PADIEvent.HagenActivation.min) or \
                np.any(vectors > hal.PADIEvent.HagenActivation.max):
            raise ValueError(
                "Vector entries have to be "
                + f"{hal.PADIEvent.HagenActivation.min}..."
                + f"{hal.PADIEvent.HagenActivation.max}.")

        if self.signed_mode:
            if np.any(matrix < -hal.SynapseQuad.Weight.max) \
                    or np.any(matrix > hal.SynapseQuad.Weight.max):
                raise ValueError(
                    f"Matrix entries have to be {-hal.SynapseQuad.Weight.max}"
                    + f"...{hal.SynapseQuad.Weight.max}.")
        else:
            if np.any(matrix < hal.SynapseQuad.Weight.min) \
                    or np.any(matrix > hal.SynapseQuad.Weight.max):
                raise ValueError(
                    f"Matrix entries have to be {hal.SynapseQuad.Weight.min}"
                    + f"...{hal.SynapseQuad.Weight.max}.")

        available_width = halco.SynapseOnSynapseRow.size
        if self.signed_mode:
            available_width //= 2  # signed mode requires 2 columns per weight
        if matrix.shape != (available_width, halco.SynapseRowOnSynram.size):
            raise ValueError(
                f"Input matrix shaped {matrix.shape} doesn't fit on synram. "
                + "Expected shape: "
                + f"{(available_width, halco.SynapseRowOnSynram.size)}")

        if vectors.shape[1] != halco.SynapseRowOnSynram.size:
            raise ValueError(
                f"Input vectors shaped {vectors.shape} don't correspond to "
                + "the height of the synapse array, "
                + f"{halco.SynapseRowOnSynram.size}.")

    # pylint: disable=too-many-locals
    def multiply(self, connection: hxcomm.ConnectionHandle,
                 vectors: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """
        Multiply given vectors with the given matrix, return results.

        The matrix shape has to match the synapse array exactly, there
        is no splitting or merging of matrix shapes.

        :param connection: Connection to the chip to run on.
        :param vectors: Array of vectors to be multiplied. We expect
            row vectors, with the inner dimension (columns) holding the
            individual entries of each vector.
        :param matrix: Array of weights.

        :return: Array of results of the MAC operation.
        """

        self._check_shape(vectors, matrix)

        # set up synapse matrix
        builder = sta.PlaybackProgramBuilder()
        synapses = self.get_synapse_matrix(matrix)
        builder.write(self._synram_coord, synapses)

        # Read synaptic input baseline potentials (after reset)
        builder.copy_back(self.cached_reset_synin)
        helpers.wait(builder, 2 * pq.us)
        read_tickets = builder.read(halco.CADCSampleRowOnDLS(
            block=halco.SynapseRowOnSynram(), synram=self._synram_coord))
        base.run(connection, builder)
        baselines = neuron_helpers.inspect_read_tickets(read_tickets)

        # Send vectors, create tickets
        start_time = time.time()
        builder, read_tickets = self._send_vectors(vectors)

        base.run(connection, builder)

        logger.get("calix.hagen.multiplication").TRACE(
            ("Multiplying {0} vectors shaped {1} took {2:3d} ms, "
             + "{3} non-zero entries in last vector").format(
                 len(vectors), vectors[0].shape,
                 int(np.around((time.time() - start_time) * 1000)),
                 np.sum(vectors[-1] != 0)))

        # Get results from tickets
        results = baselines - neuron_helpers.inspect_read_tickets(read_tickets)
        if self.signed_mode:
            exc_results = results[:, ::2]
            inh_results = results[:, 1::2]
            results = exc_results - inh_results

        return results

    # pylint: disable=too-many-locals
    def auto_multiply(self, connection: hxcomm.ConnectionHandle,
                      vectors: np.ndarray, matrix: np.ndarray, *,
                      n_row_repeats: Union[int, str] = 1,
                      ) -> np.ndarray:
        """
        Multiply given vectors with the given matrix.

        Handle input shape of the matrix automatically, i.e. split
        large matrices (containing many inputs) up into multiple runs
        or repeat small matrices multiple times in order to increase
        signal. The outer dimension (i.e. number of outputs) has to fit
        on the width of the synapse array, it is not handled here.

        :param connection: Connection to the chip to run on.
        :param vectors: Array of vectors to be multiplied. We expect
            row vectors, with the inner dimension (columns) holding the
            individual entries of each vector.
        :param matrix: Weight matrix to be multiplied.
        :param n_row_repeats: Number of repetitions of the weight
            matrix. In contrast to vector resends, the matrix rows
            are repeated, which also compensates fixed-pattern noise.
            Defaults to 1. Select "auto" to fill up the synapse array
            with a small matrix.

        :return: Numpy array with the result of multiplication.
        """

        # Check shapes
        if len(vectors[0]) != len(matrix[0]):
            raise ValueError("Shape of vector and matrix is incompatible.")
        input_vector_size = len(matrix[0])

        available_width = halco.SynapseOnSynapseRow.size
        if self.signed_mode:
            available_width //= 2  # signed mode requires 2 columns per weight
        if len(matrix) > available_width:
            raise ValueError("Matrix is too wide to fit on synram.")

        # Repeat rows if desired
        if n_row_repeats == 'auto':
            n_row_repeats = max(
                int(halco.SynapseRowOnSynram.size / input_vector_size), 1)
        if n_row_repeats < 1:
            raise ValueError("Number of repeats is smaller than one.")
        if n_row_repeats > 1:
            # Rows are repeated in order: first all events are sent,
            # only then they are all sent again. This avoids saturation
            # effects of sending one (possibly large) activation over
            # and over again before moving to the next activation.
            # Simply repeating along axis 1 would result in this different
            # order.
            vectors = np.repeat(vectors, n_row_repeats, axis=0).reshape(
                len(vectors), input_vector_size * n_row_repeats)
            matrix = np.repeat(matrix, n_row_repeats, axis=0).reshape(
                len(matrix), input_vector_size * n_row_repeats)

        # Split to multiple runs
        n_entries_per_run = halco.SynapseRowOnSynram.size
        n_runs = int((len(vectors[0]) - 1) / n_entries_per_run) + 1
        results = np.empty((n_runs, len(vectors), len(matrix)))

        for run in range(n_runs):
            run_slice = slice(
                run * n_entries_per_run,
                min(len(vectors[0]), (run + 1) * n_entries_per_run))
            n_entries = run_slice.stop - run_slice.start

            if n_entries == 0:
                raise AssertionError(
                    "0 entries found in vector slice: from entry "
                    + f"{run_slice.start} to {run_slice.stop}")

            # Slice vector and matrix to hardware size
            run_vectors = np.zeros(
                (len(vectors), n_entries_per_run), dtype=int)
            run_vectors[:, :n_entries] = vectors[:, run_slice]

            run_matrix = np.zeros(
                (available_width, n_entries_per_run), dtype=int)
            run_matrix[:len(matrix), :n_entries] = matrix[:, run_slice]

            # Call multiplication
            run_results = self.multiply(
                connection, vectors=run_vectors, matrix=run_matrix)
            results[run, :, :] = run_results[:, :len(matrix)]

        return np.sum(results, axis=0)
