"""
Provides a function to calibrate the synapse drivers for hagen-mode
input activation encoding. This boils down to calibrating the STP
ramp current such that the emitted amplitudes utilize the available
dynamic range, and calibrating the comparator offsets such that
different drivers yield the same amplitudes.
"""

from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
import copy
import numpy as np
from dlens_vx_v1 import hal, sta, halco, logger, lola, hxcomm

from calix.common import algorithms, base, cadc_helpers, helpers
from calix.hagen import neuron_helpers
from calix import constants


@dataclass
class SynapseDriverCalibResult:
    """
    Result object of a synapse driver calibration.

    Holds CapMem cells and synapse driver configs that result from
    calibration, as well as a success flag for each driver.
    """

    capmem_cells: Dict[halco.CapMemCellOnDLS, hal.CapMemCell]
    synapse_driver_configs: Dict[
        halco.SynapseDriverOnDLS, hal.SynapseDriverConfig]
    success: Dict[halco.SynapseDriverOnDLS, bool]

    def apply(self, builder: sta.PlaybackProgramBuilder
              ) -> sta.PlaybackProgramBuilder:
        """
        Apply the calibration result in the given builder.

        :param builder: Builder to append instructions to.

        :return: Builder with instructions appended.
        """

        builder = preconfigure_capmem(builder)
        for coord, config in self.capmem_cells.items():
            builder.write(coord, config)

        for coord, config in self.synapse_driver_configs.items():
            builder.write(coord, config)

        return builder


@dataclass
class _SynapseDriverResultInternal:
    """
    Internal result object of a synapse driver calibration.

    Holds numpy arrays containing STP ramp currents for each CapMem block,
    comparator offsets for every driver and their calibration success.
    """

    ramp_current: np.ndarray = np.empty(
        halco.NeuronConfigBlockOnDLS.size, dtype=int)
    offset: np.ndarray = np.empty(
        halco.SynapseDriverOnDLS.size, dtype=int)
    success: np.ndarray = np.ones(halco.SynapseDriverOnDLS.size, dtype=bool)

    def to_synapse_driver_calib_result(self) -> SynapseDriverCalibResult:
        """
        Conversion to SynapseDriverCalibResult.
        The numpy arrays get transformed to dicts.

        :return: Equivalent SynapseDriverCalibResult.
        """

        result = SynapseDriverCalibResult(dict(), dict(), dict())

        for capmem_block, ramp_current in zip(
                halco.iter_all(halco.CapMemBlockOnDLS), self.ramp_current):
            coord = halco.CapMemCellOnDLS(
                halco.CapMemCellOnCapMemBlock.stp_i_ramp, capmem_block)
            config = hal.CapMemCell(hal.CapMemCell.Value(ramp_current))
            result.capmem_cells.update({coord: config})

            coord = halco.CapMemCellOnDLS(
                halco.CapMemCellOnCapMemBlock.stp_i_calib, capmem_block)
            config = hal.CapMemCell(hal.CapMemCell.Value(ramp_current))
            result.capmem_cells.update({coord: config})

        for coord, offset in zip(
                halco.iter_all(halco.SynapseDriverOnDLS), self.offset):
            config = syndrv_config_enabled()
            config.offset = hal.SynapseDriverConfig.Offset(offset)
            result.synapse_driver_configs.update({coord: config})

        for coord, success in zip(
                halco.iter_all(halco.SynapseDriverOnDLS), self.success):
            result.success.update({coord: success})

        return result


# default address to use for events:
# This results in medium amplitudes as the activations range
# from 0 to 31 and are encoded via the addresses.
# Due to a bug on HICANN-DLS v1, we need to set this address
# correctly in the synapses.
_DEFAULT_STIMULATION_ADDRESS = hal.SynapseQuad.Label(15)


def preconfigure_capmem(builder: sta.PlaybackProgramBuilder
                        ) -> sta.PlaybackProgramBuilder:
    """
    Set necessary static biases required for hagen mode.

    :param builder: Builder to append configuration instructions to.

    :return: Builder with configuration instructions appended.
    """

    parameters = {
        halco.CapMemCellOnCapMemBlock.stp_i_bias_comparator: 1022}
    builder = helpers.capmem_set_quadrant_cells(builder, parameters)

    builder.write(halco.CapMemCellOnDLS.hagen_ibias_dac_top,
                  hal.CapMemCell(1022))
    builder.write(halco.CapMemCellOnDLS.hagen_ibias_dac_bottom,
                  hal.CapMemCell(1022))

    return builder


def set_synapses_diagonal(builder: sta.PlaybackProgramBuilder,
                          address: hal.SynapseQuad.Label =
                          _DEFAULT_STIMULATION_ADDRESS,
                          weight: hal.SynapseQuad.Weight =
                          hal.SynapseQuad.Weight.max
                          ) -> sta.PlaybackProgramBuilder:
    """
    Configure a diagonal matrix with the given address.

    The activation is encoded in the lower 5 address bits. Due to a
    bug on HICANN-X v1 the whole address is forwarded to the synapses.
    As a result one has to set the correct address in the synapses
    to forward the events to the neurons.

    Blocks of synapses with a size of 8x8 are placed diagonally in each
    synapse array. As a result, each synapse driver drives 8 neighboring
    neurons and each neuron can receive input from 4 synapse drivers,
    connected to different PADI buses. By iterating over the 4 different
    PADI buses, the amplitude of each single driver can be determined.

    :param builder: Builder to append configuration instructions to.
    :param address: Address to configure the enabled synapses to. Other
        synapses are set to zero. If the given address is zero, other
        synapses are set to one.
    :param weight: Weight to configure the enabled synapses to.

    :return: Builder with configuration appended.
    """

    neurons_per_driver = 8
    n_blocks = int(halco.SynapseOnSynapseRow.size / neurons_per_driver)

    weights = np.identity(n_blocks, dtype=int) * int(weight)
    if address != 0:
        addresses = np.identity(n_blocks, dtype=int) * int(address)
    else:
        addresses = np.ones((n_blocks, n_blocks), dtype=int)
        addresses[np.identity(n_blocks, dtype=np.bool)] = int(address)

    weights = np.repeat(weights, neurons_per_driver, axis=1)
    weights = np.repeat(weights, neurons_per_driver, axis=0)
    addresses = np.repeat(addresses, neurons_per_driver, axis=1)
    addresses = np.repeat(addresses, neurons_per_driver, axis=0)

    synapse_matrix = lola.SynapseMatrix()
    synapse_matrix.weights.from_numpy(weights)
    synapse_matrix.labels.from_numpy(addresses)

    for synram in halco.iter_all(halco.SynramOnDLS):
        builder.write(synram, synapse_matrix)
    return builder


def syndrv_config_enabled() -> hal.SynapseDriverConfig:
    """
    Return synapse driver config container instance
    with both rows excitatory in hagen mode.

    All drivers listen to all events on the PADI bus, since
    the row address compare mask is set to all zeros.

    :returns: config container for enabled synapse driver.
    """
    synapse_driver_config = hal.SynapseDriverConfig()
    synapse_driver_config.enable_address_out = True
    synapse_driver_config.enable_receiver = True
    synapse_driver_config.row_mode_top = \
        hal.SynapseDriverConfig.RowMode.excitatory
    synapse_driver_config.row_mode_bottom = \
        hal.SynapseDriverConfig.RowMode.excitatory
    synapse_driver_config.enable_stp = True
    synapse_driver_config.enable_hagen_modulation = True
    synapse_driver_config.row_address_compare_mask = 0b00000

    return synapse_driver_config


# pylint: disable=too-many-locals
def measure_syndrv_amplitudes(
        connection: hxcomm.ConnectionHandle,
        builder: sta.PlaybackProgramBuilder = None, *,
        address: hal.SynapseQuad.Label =
        _DEFAULT_STIMULATION_ADDRESS,
        n_runs: int = 20, n_events: int = 10,
        wait_time: float = 2.) -> np.ndarray:
    """
    Measure membrane potentials before and after PADI events are
    injected in one PADI bus after another.

    Send events on one PADI bus after another. When the synapse array
    is configured with the function `set_synapses_diagonal()` this allows
    to read the activation of one synapse driver after another as each
    driver on the same PADI bus is connected to eight different neurons.

    The CADC results are analyzed accordingly and an array of amplitudes
    of all drivers, acquired as median from 8 neurons, is returned.

    :param connection: Connection to the chip to run on.
    :param builder: Builder to send events and read on.
    :param address: Address to send events on.
    :param n_runs: Number of experiments to take mean from.
    :param n_events: Number of events to send before each measurement.
    :param wait_time: Wait time between two events in us.

    :return: Array of amplitudes observed at neurons after integration
        of events, mean of n_runs.
    """

    baselines = list()
    results = list()

    # Read baseline potentials:
    for _ in range(n_runs):
        for synram in halco.iter_all(halco.SynramOnDLS):
            builder = neuron_helpers.reset_neurons(builder, synram)
            builder = helpers.wait_for_us(builder, 30)
            builder, ticket = cadc_helpers.cadc_read_row(builder, synram)
            baselines.append(ticket)

    # set up PADI event, iterate 4 busses as the drivers with identical row
    # addresses on different PADI busses each map to the same group of neurons.
    for padi_bus in halco.iter_all(halco.PADIBusOnPADIBusBlock):
        padi_event = hal.PADIEvent()
        padi_event.fire_bus[padi_bus] = True  # pylint: disable=unsupported-assignment-operation
        padi_event.event_address = hal.SynapseQuad.Label(address)

        for _ in range(n_runs):
            for synram in halco.iter_all(halco.SynramOnDLS):
                # Reset neurons and timer
                time_offset = 5  # us
                builder = neuron_helpers.reset_neurons(builder, synram)
                builder = helpers.wait_for_us(builder, time_offset)

                # Send events
                for event in range(n_events):
                    builder.wait_until(
                        halco.TimerOnDLS(),
                        int(((wait_time * event) + time_offset)
                            * int(hal.Timer.Value.fpga_clock_cycles_per_us)))
                    builder.write(synram.toPADIEventOnDLS(), padi_event)

                # Read CADCs after integration in the same builder
                builder, ticket = cadc_helpers.cadc_read_row(
                    builder, synram)
                results.append(ticket)

    # Wait for transfers, execute
    builder = helpers.wait_for_us(builder, 100)
    sta.run(connection, builder.done())

    # Inspect tickets
    baselines = np.mean(
        neuron_helpers.inspect_read_tickets(baselines).reshape(
            (n_runs, halco.NeuronConfigOnDLS.size)),
        axis=0)
    results = neuron_helpers.inspect_read_tickets(results).reshape(
        (halco.PADIBusOnPADIBusBlock.size, n_runs,
         halco.NeuronConfigOnDLS.size))

    # take mean of n_runs, subtract baseline reads
    results = np.mean(results, axis=1) - baselines

    # take median of 8 neurons per driver
    results = np.median(results.reshape(
        (len(results), int(
            halco.NeuronConfigOnDLS.size / 8), 8)), axis=2)

    # arrange in order of drivers: iterate drivers per group first
    results = results.flatten(order='F')
    assert len(results) == halco.SynapseDriverOnDLS.size
    return results


class STPRampCalibration(base.Calibration):
    """
    Search synapse driver STP ramp currents such that the mean amplitudes
    of drivers on different CapMem blocks are the same.

    This is necessary as 64 drivers each, which are connected to the same
    CapMem block, receive the same bias current, but different instances
    of the CapMem can drive different currents at the same setting.
    This calibration counters systematic deviations in amplitudes between
    synapse drivers on different quadrants.

    The amplitudes are measured with the neurons in integration mode.
    In order to obtain a representative result for a quadrant's
    amplitudes with the STP offsets not calibrated yet, we select
    an "average" driver for each CapMem block during the prelude.
    The median of their amplitudes is used as calibration target.

    Requirements:
    * Neurons are calibrated for integration mode.
    * Static synapse driver bias currents are set. This can be achieved
      using `preconfigure_capmem()`.
    * Synaptic events can reach the neurons, i.e. the synapse DAC bias
      is set and the `hal.ColumnCurrentSwitch`es allow currents from
      the synapses through.
    * Neuron membrane readout is connected to the CADCs (causal and acausal).

    :ivar test_address: Address at which amplitudes of the quadrants will
        be matched. Determines the point of the ramp where the calibration
        runs, since the hagen-mode activation encoding is used.
    :ivar target_drivers: Coordinates of drivers to use, one per CapMem
        block.
    """

    def __init__(self, test_address: hal.SynapseQuad.Label =
                 _DEFAULT_STIMULATION_ADDRESS):
        super().__init__(
            parameter_range=base.ParameterRange(200, 711),
            n_instances=halco.CapMemBlockOnDLS.size,
            inverted=False,
            errors=["STP ramp current for quadrants {0} has reached {1}"] * 2)
        self.test_address = test_address
        self.target_drivers: Optional[List[halco.SynapseDriverOnDLS]] = None

    def set_synapses_row(self, builder: sta.PlaybackProgramBuilder,
                         address: hal.SynapseQuad.Label =
                         _DEFAULT_STIMULATION_ADDRESS
                         ) -> sta.PlaybackProgramBuilder:
        """
        Configure synapses such that every second neuron receives input
        from the same driver.

        On every synapse matrix, the input from two different synapse
        drivers shall be recorded, one for each CapMem block (saved
        in `self.target_drivers`). Every second neuron should receive
        input from the same synapse driver. For that purpose, we enable
        every second synapse in the two respective rows connected to
        the desired synapse driver. For the first driver we start with
        synapse 0 and use all even synapses, for the second driver we
        start with synapse 1 and use all odd synapses. The process is
        repeated for both synapse arrays.

        :param builder: Builder to append configuration to.
        :param address: Address to configure synapses to.

        :return: Builder with configuration appended.
        """

        # Set all synapses to zero
        for synram in halco.iter_all(halco.SynramOnDLS):
            builder.write(synram, lola.SynapseMatrix())

        # Set every second neuron for the two drivers per synram
        capmems_on_hemisphere = halco.CapMemBlockOnHemisphere.size
        for driver_coord in self.target_drivers:
            addresses = np.zeros(halco.SynapseOnSynapseRow.size, dtype=int)
            weights = np.zeros_like(addresses)

            capmem_block_id = int(driver_coord.toCapMemBlockOnDLS().toEnum())
            addresses[capmem_block_id % capmems_on_hemisphere::
                      capmems_on_hemisphere] = address
            weights[capmem_block_id % capmems_on_hemisphere::
                    capmems_on_hemisphere] = hal.SynapseQuad.Weight.max

            synapse_row = lola.SynapseRow()
            synapse_row.labels.from_numpy(addresses)
            synapse_row.weights.from_numpy(weights)

            for coord in driver_coord.toSynapseRowOnDLS():
                builder.write(coord, synapse_row)

        return builder

    @staticmethod
    def reshape_syndrv_amplitudes(amplitudes: np.ndarray) -> np.ndarray:
        """
        Reshape an array that contains amplitudes of all synapse drivers
        to an array that lists those amplitudes by CapMem quadrant the synapse
        drivers are connected to.

        :param amplitudes: Flat array containing 256 synapse driver results.

        :return: Two-dimensional array that contains 4 quadrants and 64 driver
            results per quadrant.
        """

        output_data = np.empty(
            (halco.CapMemBlockOnDLS.size, int(
                halco.SynapseDriverOnDLS.size / halco.CapMemBlockOnDLS.size)),
            dtype=amplitudes.dtype)

        capmem_assignment = list()
        for driver_coord in halco.iter_all(halco.SynapseDriverOnDLS):
            capmem_assignment.append(driver_coord.toCapMemBlockOnDLS())

        capmem_assignment = np.array(capmem_assignment)
        for capmem_block in halco.iter_all(halco.CapMemBlockOnDLS):
            mask = capmem_assignment == capmem_block
            output_data[int(capmem_block.toEnum()), :] = amplitudes[mask]

        return output_data

    @staticmethod
    def preconfigure_syndrvs(builder: sta.PlaybackProgramBuilder,
                             offset: hal.SynapseDriverConfig.Offset
                             ) -> sta.PlaybackProgramBuilder:
        """
        Enable synapse drivers and set offset.

        :param builder: Builder to append configuration.
        :param offset: Offset for the synapse drivers.

        :return: Builder with configuration appended.
        """
        synapse_driver_config = syndrv_config_enabled()
        synapse_driver_config.offset = offset

        for coord in halco.iter_all(halco.SynapseDriverOnDLS):
            builder.write(coord, synapse_driver_config)

        return builder

    @staticmethod
    def measure_amplitudes(
            builder: sta.PlaybackProgramBuilder,
            address: hal.SynapseQuad.Label, n_runs: int = 20,
            n_events: int = 10, wait_time: float = 2.0
    ) -> Tuple[sta.PlaybackProgramBuilder,
               List[sta.ContainerTicket_CADCSampleRow],
               List[sta.ContainerTicket_CADCSampleRow]]:
        """
        Create a builder which reads baseline neuron membrane potentials
        and potentials after sending events to synapse drivers.

        :param builder: Builder to append instructions to.
        :param address: Address to send PADI events on.
        :param n_runs: Number of times to measure baseline and
            event amplitudes.
        :param n_events: Number of events to send for integration.
        :param wait_time: Time in us to wait between two successive events.

        :return: Tuple containing:
            * Playback program builder with instructions appended.
            * List of baseline read tickets.
            * List of read tickets after events were sent. Two tickets
                per run are returned, one for each synram.
        """

        baseline_tickets = list()
        result_tickets = list()

        # set up PADI event, send on all busses
        padi_event = hal.PADIEvent()
        padi_event.event_address = address
        for padi_bus in halco.iter_all(halco.PADIBusOnPADIBusBlock):
            padi_event.fire_bus[padi_bus] = True  # pylint: disable=unsupported-assignment-operation

        for _ in range(n_runs):
            # Read baseline potentials:
            for synram in halco.iter_all(halco.SynramOnDLS):
                builder = neuron_helpers.reset_neurons(builder, synram)
                builder = helpers.wait_for_us(builder, 30)
                builder, ticket = cadc_helpers.cadc_read_row(builder, synram)
                baseline_tickets.append(ticket)

                # Reset neurons and timer
                time_offset = 5  # us
                builder = neuron_helpers.reset_neurons(builder, synram)
                builder = helpers.wait_for_us(builder, time_offset)

                # Send events
                for event in range(n_events):
                    builder.wait_until(
                        halco.TimerOnDLS(),
                        int(((wait_time * event) + time_offset)
                            * int(hal.Timer.Value.fpga_clock_cycles_per_us)))
                    builder.write(synram.toPADIEventOnDLS(), padi_event)

                # Read CADCs after integration in the same builder
                builder, ticket = cadc_helpers.cadc_read_row(builder, synram)
                result_tickets.append(ticket)

        return builder, baseline_tickets, result_tickets

    @staticmethod
    def evaluate_amplitudes(
            baseline_tickets: List[sta.ContainerTicket_CADCSampleRow],
            result_tickets: List[sta.ContainerTicket_CADCSampleRow]
    ) -> np.ndarray:
        """
        Evaluate the given lists of tickets, as returned by the function
        `self.measure_amplitudes`.

        We take the mean of all neurons' amplitudes that have received
        inputs from a driver on the respective CapMem block. During
        measurement, we iterate the synapse arrays, but different neurons
        still contain results from different drivers.

        Therefore, we firstly split the results by synapse arrays, and
        secondly split the neuron results depending on the CapMem block
        of the driver they were connected to.

        :param baseline_tickets: List of baseline read tickets, as
            returned by `self.measure_amplitudes`.
        :param result_tickets: List of read tickets after events were
            sent, as returned by `self.measure_amplitudes`.

        :return: Array of amplitudes, one per CapMem block.
        """

        baselines = neuron_helpers.inspect_read_tickets(baseline_tickets)
        results = neuron_helpers.inspect_read_tickets(result_tickets)

        # number of runs (for averaging only):
        n_runs = len(baselines) // halco.SynramOnDLS.size
        # number of neurons connected to each driver (CapMem block):
        n_neurons_per_driver = halco.SynapseOnSynapseRow.size \
            // halco.CapMemBlockOnHemisphere.size

        # Split baseline and result arrays by synram and CapMem block
        baselines = baselines.reshape(
            n_runs, halco.SynramOnDLS.size, n_neurons_per_driver,
            halco.CapMemBlockOnHemisphere.size)
        results = results.reshape(
            n_runs, halco.SynramOnDLS.size, n_neurons_per_driver,
            halco.CapMemBlockOnHemisphere.size)

        # Calculate mean of all neurons connected to the same CapMem block
        # and mean of all measurement runs
        baselines = np.mean(baselines, axis=(0, 2))
        results = np.mean(results, axis=(0, 2))

        # Calculate amplitudes as differences, flatten into one result
        # per CapMem block
        amplitudes = results.flatten() - baselines.flatten()
        return amplitudes

    def measure_and_evaluate(self, connection: hxcomm.ConnectionHandle,
                             builder: sta.PlaybackProgramBuilder,
                             address: hal.SynapseQuad.Label =
                             _DEFAULT_STIMULATION_ADDRESS
                             ) -> np.ndarray:
        """
        Initiate the measurement of amplitudes of one driver per CapMem
        block and evaluate the result.

        Call the measurement function to obtain amplitudes of only four
        drivers, one per CapMem block, with synapses configured in an
        alternating pattern. The results are evaluated and the obtained
        amplitudes are returned.

        Before measuring, the synapses are configured to the
        given address.

        :param connection: Connection to the chip to run on.
        :param builder: Builder to append read instructions to.
        :param address: Address to send events to.

        :return: Array of amplitudes of the four drivers.
        """

        builder = self.set_synapses_row(builder, address=address)

        # Create program: read baseline, stimulate, read
        builder, baselines, results = self.measure_amplitudes(
            builder, address)

        # Wait for transfers, execute
        builder = helpers.wait_for_us(builder, 100)
        sta.run(connection, builder.done())

        # Interpret results
        amplitudes = self.evaluate_amplitudes(baselines, results)
        return amplitudes

    def configure_parameters(self, builder: sta.PlaybackProgramBuilder,
                             parameters: np.ndarray
                             ) -> sta.PlaybackProgramBuilder:
        """
        Configure the given STP ramp currents to the CapMem quadrants.

        The STP calibration current is set equal to this ramp current:
        The calibration works by sinking part of the ramp current within
        the first 2 ns of the ramp generation phase. The calibration
        current controls the sinked current, therefore it is sensible
        to scale it with the sourced (ramp) current.

        :param builder: Builder to append configuration to.
        :param parameters: STP ramp currents to set.

        :return: Builder with configuration appended.
        """

        builder = helpers.capmem_set_quadrant_cells(
            builder,
            {halco.CapMemCellOnCapMemBlock.stp_i_ramp: parameters,
             halco.CapMemCellOnCapMemBlock.stp_i_calib: parameters})

        builder = helpers.wait_for_us(builder, constants.capmem_level_off_time)
        return builder

    def prelude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Find target drivers and target amplitudes for the ramp current
        calibration.

        First, we measure suitable drivers for the calibration: The
        median driver (yielding median amplitudes within a group of
        drivers connected to the same CapMem block) is selected as a
        representative driver.

        Second, we measure the maximum achievable amplitudes. This is
        done using a low address (i.e. a high vector value), a high ramp
        current and medium STP offsets (which will be calibrated only
        later).

        Finally, we calculate the target amplitude by scaling the
        maximum amplitude down using the rule of three.

        :param connection: Connection to the chip to run on.
        """

        log = logger.get("calix.hagen.synapse_driver"
                         + ".STPRampCalibration.prelude")

        # Find one "good" driver per block:
        builder = sta.PlaybackProgramBuilder()
        builder = self.preconfigure_syndrvs(
            builder, hal.SynapseDriverConfig.Offset(int(
                hal.SynapseDriverConfig.Offset.max / 2)))
        builder = self.configure_parameters(
            builder, 300 * np.ones(halco.CapMemBlockOnDLS.size, dtype=int))
        builder = set_synapses_diagonal(builder, address=10)
        results = measure_syndrv_amplitudes(
            connection, builder, address=10, n_runs=100)

        quadrant_results = self.reshape_syndrv_amplitudes(results)
        target_drivers = np.argsort(  # obtain _index_ of median driver
            quadrant_results, axis=1)[:, quadrant_results.shape[1] // 2]
        target_driver_amplitudes = quadrant_results[
            np.arange(halco.CapMemBlockOnDLS.size), target_drivers]
        log.DEBUG("Using drivers for ramp current calibration: "
                  + f"{target_drivers}, amplitudes were: "
                  + f"{target_driver_amplitudes}")
        self.target_drivers = list()
        for capmem_block_id, driver_id in enumerate(target_drivers):
            self.target_drivers.append(halco.SynapseDriverOnDLS(
                halco.SynapseDriverOnSynapseDriverBlock(
                    (halco.CapMemBlockOnHemisphere.size * driver_id)
                    + (capmem_block_id % halco.CapMemBlockOnHemisphere.size)),
                halco.SynapseDriverBlockOnDLS(int(
                    capmem_block_id / halco.CapMemBlockOnHemisphere.size))))

        # Set synapses to read good drivers
        builder = sta.PlaybackProgramBuilder()

        # Find maximum amplitudes of good drivers (at high activation,
        # i.e. low address)
        builder = self.configure_parameters(
            builder, 700 * np.ones(halco.CapMemBlockOnDLS.size, dtype=int))
        amplitude_target = self.measure_and_evaluate(
            connection, builder, address=2)

        # Calculate amplitude target by rule of three and magic factor
        # Assuming the `amplitude_target` is the expected amplitude
        # at maximum activation, we scale the target amplitude down
        # to the value expected at the `self.test_address`.
        # The magic factor accounts for non-linearity of the mapping.
        self.target = amplitude_target / hal.PADIEvent.HagenActivation.max \
            * (hal.PADIEvent.HagenActivation.max
               - int(self.test_address)) \
            * 0.85  # magic factor

        log.DEBUG(f"Set target high amplitudes: {amplitude_target}")
        log.DEBUG(f"Set read target at address {self.test_address}: "
                  + f"{self.target}")

    def measure_results(self, connection: hxcomm.ConnectionHandle,
                        builder: sta.PlaybackProgramBuilder
                        ) -> np.ndarray:
        """
        Read the amplitudes at medium address of the previously found
        good drivers, one per CapMem block.

        :param connection: Connection to the chip to run on.
        :param builder: Builder to read CADCs with.

        :return: Array of STP amplitudes per quadrant.
        """

        amplitudes = self.measure_and_evaluate(
            connection, builder, address=self.test_address)

        return amplitudes

    def postlude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Print the calibrated ramp currents.

        :param connection: Connection to the chip to run on.
        """
        logger.get(
            "calix.hagen.synapse_driver.STPRampCalibration.postlude"
        ).INFO("Calibrated STP ramp currents, values: "
               + f"{self.result.calibrated_parameters}")


class STPOffsetCalibration(base.Calibration):
    """
    Search STP offset settings such that drivers yield the same amplitudes.

    Can configure offsets to the drivers and measure their amplitudes with
    STP in hagen mode enabled.
    Initially, amplitudes at half the maximum offset are measured and
    the median is taken as target for calibration.

    Requirements:
    * Neurons are calibrated for integration mode.
    * STP ramp currents are calibrated.
    * Synaptic events can reach the neurons, i.e. the synapse DAC bias
      is set and the `hal.ColumnCurrentSwitch`es allow currents from
      the synapses through.
    * Neuron membrane readout is connected to the CADCs (causal and acausal).

    :ivar default_syndrv_config: Synapse driver config to use during
        calibration, with everything but the offset configured.
        Used in configure_parameters, only the offset is set there.
    """

    def __init__(self):
        super().__init__(
            parameter_range=base.ParameterRange(
                hal.SynapseDriverConfig.Offset.min,
                hal.SynapseDriverConfig.Offset.max),
            n_instances=halco.SynapseDriverOnDLS.size,
            inverted=True)
        self.default_syndrv_config = syndrv_config_enabled()

    def configure_parameters(self, builder: sta.PlaybackProgramBuilder,
                             parameters: np.ndarray
                             ) -> sta.PlaybackProgramBuilder:
        """
        Configure the synapse drivers to the given offsets.

        :param builder: Builder to append configuration to.
        :param parameters: Synapse driver offset settings to configure.

        :return: Builder with configuration instructions appended.
        """

        for coord in halco.iter_all(halco.SynapseDriverOnDLS):
            synapse_driver_config = copy.deepcopy(self.default_syndrv_config)
            synapse_driver_config.offset = hal.SynapseDriverConfig.Offset(
                parameters[int(coord.toEnum())])

            builder.write(coord, synapse_driver_config)

        return builder

    def measure_results(self, connection: hxcomm.ConnectionHandle,
                        builder: sta.PlaybackProgramBuilder
                        ) -> np.ndarray:
        """
        Read output amplitudes of synapse drivers.

        :param connection: Connection to the chip to run on.
        :param builder: Builder to send events and read on.

        :return: Array of synapse drivers' output amplitudes.
        """

        return measure_syndrv_amplitudes(connection, builder)

    def prelude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Configure synapse array and measure calibration target.

        Measure target: median amplitude of all drivers.
        Log standard deviation of amplitudes before calibration.

        :param connection: Connection to the chip to run on.
        """

        builder = sta.PlaybackProgramBuilder()
        builder = set_synapses_diagonal(builder)  # set to medium activation
        builder = self.configure_parameters(
            builder, np.ones(halco.SynapseDriverOnDLS.size, dtype=int)
            * int(hal.SynapseDriverConfig.Offset.max / 2))
        results = self.measure_results(connection, builder)
        self.target = np.median(results)

        logger.get(
            "calix.hagen.synapse_driver.STPOffsetCalibration.prelude"
        ).DEBUG(
            "Deviation of synapse driver amplitudes before offset calib: "
            + f"{np.std(results):4.2f}")

    def postlude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Log standard deviation of amplitudes after calibration.

        :param connection: Connection to the chip to run on.
        """

        builder = sta.PlaybackProgramBuilder()
        results = self.measure_results(connection, builder)
        logger.get(
            "calix.hagen.synapse_driver.STPOffsetCalibration.postlude"
        ).INFO(
            "Deviation of synapse driver amplitudes after offset calib: "
            + f"{np.std(results):4.2f}")


def calibrate(connection: hxcomm.ConnectionHandle
              ) -> SynapseDriverCalibResult:
    """
    Calibrate the synapse drivers' STP offsets such that the amplitudes
    match when using hagen mode.

    The STP ramp currents are calibrated such that drivers connected
    to each CapMem Block use the available dynamic range. To achieve
    that, we (1) select the drivers with median amplitudes per quadrant,
    (2) measure the maximum amplitudes generated by these drivers,
    (3) set the ramp current such that the activation scales amplitudes
    sensibly.

    The STP offsets are calibrated for each driver afterwards.

    Requirements:
    - Chip is initialized and CADCs as well as neurons have been
      calibrated for integration mode. You can use
      `calix.common.cadc.calibrate()` as well as
      `calix.hagen.neuron.calibrate()` to achieve that.

    :param connection: Connection to the chip to run on.

    :return: SynapseDriverCalibResult containing STP ramp currents for
        the CapMem blocks, offsets for the drivers, and a success mask.
    """

    # preconfigure chip
    builder = sta.PlaybackProgramBuilder()
    builder = preconfigure_capmem(builder)
    sta.run(connection, builder.done())

    # prepare result object
    calib_result = _SynapseDriverResultInternal()

    # Calibrate STP ramps
    calibration = STPRampCalibration()
    result = calibration.run(connection, algorithm=algorithms.BinarySearch())
    calib_result.ramp_current = result.calibrated_parameters

    # set success of drivers using the respective ramp currents
    for coord in halco.iter_all(halco.SynapseDriverOnDLS):
        calib_result.success[int(coord.toEnum())] = \
            result.success[int(coord.toCapMemBlockOnDLS().toEnum())]

    # Calibrate STP offsets
    calibration = STPOffsetCalibration()
    calib_result.offset = calibration.run(
        connection, algorithm=algorithms.BinarySearch()).calibrated_parameters
    # success not checked since hitting range boundaries is expected

    return calib_result.to_synapse_driver_calib_result()
