"""
Provides a function to calibrate the synapse drivers for hagen-mode
input activation encoding. This boils down to calibrating the STP
ramp current such that the emitted amplitudes utilize the available
dynamic range, and calibrating the hagen-mode DAC offsets such that
different drivers yield the same amplitudes.

These calibrations are performed on the synaptic input lines,
therefore we require calibrated CADCs, but the neurons don't need to
be calibrated.
"""

from typing import Optional, Dict, Union
from dataclasses import dataclass
import numpy as np
from dlens_vx_v2 import hal, sta, halco, logger, hxcomm

from calix.common import algorithms, base, helpers
from calix.hagen import multiplication
from calix import constants


# Limit for dynamic range of synapse drivers' output amplitudes:
# We measure the maximum amplitude of each synapse driver and take the
# mean of all measurements times this constant as a target for all
# synapse drivers. This allows synapse drivers with a less steep
# ramp to reach the same value as other drivers.
RANGE_LIMIT = 0.9

# Default STP ramp offset
DEFAULT_STP_OFFSET = hal.SynapseDriverConfig.Offset.max // 2


@dataclass
class SynapseDriverCalibResult(base.CalibrationResult):
    """
    Result object of a synapse driver calibration.

    Holds CapMem cells and synapse driver configs that result from
    calibration, as well as a success flag for each driver.
    """

    capmem_cells: Dict[halco.CapMemCellOnDLS, hal.CapMemCell]
    synapse_driver_configs: Dict[
        halco.SynapseDriverOnDLS, hal.SynapseDriverConfig]
    success: Dict[halco.SynapseDriverOnDLS, bool]

    def apply(self, builder: Union[sta.PlaybackProgramBuilder,
                                   sta.PlaybackProgramBuilderDumper]):
        """
        Apply the calibration result in the given builder.

        :param builder: Builder or dumper to append instructions to.
        """

        builder = preconfigure_capmem(builder)
        for coord, config in self.capmem_cells.items():
            builder.write(coord, config)

        for coord, config in self.synapse_driver_configs.items():
            builder.write(coord, config)


@dataclass
class _SynapseDriverResultInternal:
    """
    Internal result object of a synapse driver calibration.

    Holds numpy arrays containing STP ramp currents for each CapMem block,
    comparator offsets for every driver and their calibration success.
    """

    ramp_current: np.ndarray = np.empty(
        halco.NeuronConfigBlockOnDLS.size, dtype=int)
    hagen_dac_offset: np.ndarray = np.empty(
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
                halco.iter_all(halco.SynapseDriverOnDLS),
                self.hagen_dac_offset):
            config = hal.SynapseDriverConfig()
            config.enable_address_out = False
            config.enable_hagen_dac = True
            config.enable_stp = True
            config.enable_hagen_modulation = True
            config.hagen_dac_offset = hal.SynapseDriverConfig.HagenDACOffset(
                offset)
            config.offset = DEFAULT_STP_OFFSET
            result.synapse_driver_configs.update({coord: config})

        for coord, success in zip(
                halco.iter_all(halco.SynapseDriverOnDLS), self.success):
            result.success.update({coord: success})

        return result


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
                  hal.CapMemCell(920))
    builder.write(halco.CapMemCellOnDLS.hagen_ibias_dac_bottom,
                  hal.CapMemCell(920))

    return builder


class SynapseDriverMeasurement:
    """
    Provides a function to measure output amplitudes of each synapse
    driver.

    Each synapse driver is connected to at least 8 synapse columns,
    their median result is taken as the driver's output amplitude.
    The requested number of parallel measurements determines how many
    synapse columns are used per driver: If only one driver is measured
    at a time, all 256 synapse columns are used.

    When several synapse drivers are measured in parallel, events are
    sent to one synapse driver after another. Due to analog effects in
    the synapse array, earlier events will result in a lower activation
    than later events. Do not use more than one parallel measurement
    if you cannot cope with this effect.

    :ivar n_parallel_measurements: Number of synapse drivers to be
        measured in one run. Only powers of 2 are supported, i.e.
        possible values are 1, 2, 4, 8, 16, 32. We do not support
        more than 32 parallel measurements since we require at least
        8 synapse columns per driver, in order to take the median
        read from those as the driver's output.
    :ivar multiplication: Multiplication class used for measurements.
        Each measurement run is a vector-matrix multiplication.
    """

    def __init__(self, n_parallel_measurements: int = 1):
        if n_parallel_measurements not in [1, 2, 4, 8, 16, 32]:
            raise ValueError("Illegal number of parallel measurements given.")
        self.n_parallel_measurements = n_parallel_measurements
        self.multiplication = multiplication.Multiplication(signed_mode=False)

    def get_synapse_mapping(self,
                            driver: halco.SynapseDriverOnSynapseDriverBlock
                            ) -> np.ndarray:
        """
        Return mask of synapses to enable in a row.

        This function can be called for all synapse drivers and will
        return a synapse matrix that allows measuring amplitudes from
        a block of `self.n_parallel_measurements` drivers in parallel.
        The synapse matrix is configured such that the drivers which
        are measured in parallel do not use the same synapse columns.
        For the next block of synapse drivers which are measured in
        parallel, the same synapse columns as for the previous block
        can be used.

        :param driver: Coordinate of the synapse driver which shall be
            connected to neurons.

        :return: Boolean mask of enabled synapses within the synapse driver's
            row.
        """

        if not isinstance(driver, halco.SynapseDriverOnSynapseDriverBlock):
            raise ValueError(
                "'driver' has to be a "
                + "`halco.SynapseDriverOnSynapseDriverBlock` "
                + "coordinate. Note: synapse mapping is identical on both "
                + "hemispheres.")

        # The synapses are chosen such that at least 8 synapses are enabled
        # per row, taken from a distribution over the entire row, avoiding
        # asymmetry of amplitudes due to location of the synapses.
        n_blocks_per_row = 8

        # Generate 8 blocks of synapses, drivers will be assigned at least
        # one synapse in each of these blocks. Half of the blocks contain
        # the synapses in reverse order to ensure an equal assignment for
        # all drivers.
        available_columns = np.arange(halco.SynapseOnSynapseRow.size).reshape(
            n_blocks_per_row,
            halco.SynapseOnSynapseRow.size // n_blocks_per_row)
        for block in range(n_blocks_per_row):
            if (block < n_blocks_per_row // 2 and block % 2 == 1) or \
                    (block >= n_blocks_per_row // 2 and block % 2 == 0):
                available_columns[block] = available_columns[block, ::-1]

        # pick an appropriate number of synapses from each block
        offset = int(driver.toEnum()) % self.n_parallel_measurements
        selected_columns = available_columns[
            :, offset::self.n_parallel_measurements]

        mask = np.zeros(halco.SynapseOnSynapseRow.size, dtype=bool)
        mask[selected_columns] = True

        return mask

    def get_synapse_matrix(self, weight: hal.SynapseQuad.Weight =
                           hal.SynapseQuad.Weight.max) -> np.ndarray:
        """
        Return a mapping matrix using the given weight.

        For the mapping, we use the function `get_synapse_mapping`,
        which yields a connection matrix between the drivers and neurons.
        This mapping is designed to create a non-overlapping synapse matrix
        that allows parallel measurement of multiple drivers.

        :param weight: Weight to configure the enabled synapses to.

        :return: Numpy array of weight matrix.
        """

        weights = np.zeros(
            (halco.SynapseRowOnSynram.size, halco.SynapseOnSynapseRow.size),
            dtype=int)

        for driver in halco.iter_all(halco.SynapseDriverOnSynapseDriverBlock):
            target_mask = self.get_synapse_mapping(driver)
            for synapse_row in driver.toSynapseRowOnSynram():
                weights[int(synapse_row), target_mask] = weight

        return weights

    def get_input_vectors(self, activations: np.ndarray) -> np.ndarray:
        """
        Create vectors that are multiplied with the synapse matrix in
        order to characterize all drivers' outputs at the given
        activations.

        In each vector, we send activations to a subset of drivers, that
        has different columns for their results. Hence, the unique
        activations of each driver can be obtained. Only in the next
        vector, we reuse the same columns in order to characterize
        each synapse driver.

        :param activations: Array of activations, each is tested.

        :return: Array of vectors, to be used as input for multiplication.
        """

        # In order to measure all drivers with self.n_parallel_measurements
        # in parallel, we need n_measurements iterations.
        n_measurements = halco.SynapseDriverOnSynapseDriverBlock.size \
            // self.n_parallel_measurements
        vectors = np.zeros(
            (n_measurements,
             halco.SynapseDriverOnSynapseDriverBlock.size),
            dtype=int)

        # For each iteration, we send activations only to drivers that
        # do not yet re-use the same columns for readout. This boils down
        # to creating vectors with batches of n_parallel_measurements
        # activations each.
        for measurement_id in range(n_measurements):
            vectors[measurement_id,
                    measurement_id * self.n_parallel_measurements:
                    (measurement_id + 1) * self.n_parallel_measurements:] = 1
        vectors = np.repeat(vectors, halco.SynapseRowOnSynapseDriver.size,
                            axis=1)
        vectors = np.repeat(vectors, len(activations), axis=0)

        # So far, we created "boolean" vectors with 1 marking that an
        # activation should be sent. Now, we fill them with the actually
        # desired activations.
        for activation_id, activation in enumerate(activations):
            vectors[activation_id * n_measurements:
                    (activation_id + 1) * n_measurements:] *= activation

        return vectors

    # pylint: disable=too-many-locals
    def measure_syndrv_amplitudes(
            self, connection: hxcomm.ConnectionHandle,
            activations: Union[hal.PADIEvent.HagenActivation, np.ndarray], *,
            weight: hal.SynapseQuad.Weight = hal.SynapseQuad.Weight.max,
            n_runs: int = 5, num_sends: int = 3) -> np.ndarray:
        """
        Multiply the given activation with a suitable weight matrix to
        determine the output amplitudes of each driver.

        Use the multiplication function with integration on the synaptic
        input lines. The weight matrix is configured using the function
        `get_synapse_matrix()`, and vectors equaling the positioning of
        drivers on PADI busses are used to characterize each drivers'
        outputs. The parameter num_sends repeats the vector, i.e. sends
        multiple events per driver for getting suitable amplitudes.
        The experiment is repeated n_runs times.

        For readout, we use at least 8 synapse columns per driver,
        determined by the number of parallel measurements, and return the
        median of each of those synapse blocks. The returned array of
        amplitudes has one entry for each synapse driver.

        :param connection: Connection to the chip to run on.
        :param activations: Activation to use in input vector. If an array is
            given, each of the contained activations is tested. In this case,
            the returned result will have a second, outer dimension
            corresponding to the different activations.
        :param weight: Weight to use in synapse matrix.
        :param n_runs: Number of experiments to take mean from.
        :param num_sends: Number of events to send per driver in each
            measurement.

        :return: Array of amplitudes of each synapse driver.
        """

        self.multiplication.num_sends = num_sends

        # handle option for multiple activations
        if not isinstance(activations, np.ndarray):
            activations = np.array([int(activations)])
            squeeze_result = True
        else:
            squeeze_result = False

        # Create vectors iterating all drivers
        vectors = self.get_input_vectors(activations)

        # multiply vectors with mapping matrix
        results = np.empty(
            (halco.SynramOnDLS.size, n_runs,
             halco.SynapseDriverOnSynapseDriverBlock.size, len(activations)),
            dtype=int)
        for synram in halco.iter_all(halco.SynramOnDLS):
            self.multiplication.synram_coord = synram
            self.multiplication.preconfigure(connection)
            for run in range(n_runs):
                # matrix is transposed: multiply function takes matrix in
                # mathematical sense, but matrix here is synapse weights
                run_results = self.multiplication.multiply(
                    connection, vectors, self.get_synapse_matrix(weight).T)

                # calculate median of columns per driver
                n_measurements_per_activation = \
                    len(vectors) // len(activations)
                for activation_id in range(len(activations)):
                    for driver in halco.iter_all(
                            halco.SynapseDriverOnSynapseDriverBlock):
                        data_slice = (
                            (int(driver.toEnum())
                             // self.n_parallel_measurements)
                            + (activation_id * n_measurements_per_activation),
                            self.get_synapse_mapping(driver))
                        results[
                            int(synram.toEnum()), run,
                            int(driver.toEnum()), activation_id] = \
                            np.median(run_results[data_slice])

        # take mean of n_runs, adjust result shape
        results = np.mean(results.astype(float), axis=1)
        results = results.reshape((halco.SynapseDriverOnDLS.size,
                                   len(activations)))
        if squeeze_result:
            results = np.squeeze(results, axis=1)
        else:
            results = np.swapaxes(results, 0, 1)
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
    * Synapse DAC bias is calibrated.
    * Static synapse driver bias currents are set. This can be achieved
      using `preconfigure_capmem()`.

    :ivar test_activation: Hagen-mode activation to use when
        determining the ramp slope. Should be low, close but not
        equal to zero. Make sure to change the target amplitude
        accordingly when changing the test_activation.
    :ivar n_parallel_measurements: Number of parallel measurements
        to execute when measuring results.
    :ivar offset_calib: Instance of calibration used for Hagen DAC
        offset. This calibration is called each time after changing
        the STP ramp currents, before measuring results.
    :ivar log: Logger used to log outputs.
    """

    def __init__(self):
        super().__init__(
            parameter_range=base.ParameterRange(
                hal.CapMemCell.Value.min, hal.CapMemCell.Value.max),
            n_instances=halco.CapMemBlockOnDLS.size,
            inverted=False,
            errors=["STP ramp current for quadrants {0} has reached {1}"] * 2)
        self.test_activation = hal.PADIEvent.HagenActivation(2)
        self.n_parallel_measurements = 8
        self.offset_calib = HagenDACOffsetCalibration()
        self.offset_calib.n_parallel_measurements = 8
        self.log = logger.get(
            "calix.hagen.synapse_driver.STPRampCalibration")

    def configure_parameters(self, builder: sta.PlaybackProgramBuilder,
                             parameters: np.ndarray
                             ) -> sta.PlaybackProgramBuilder:
        """
        Configure the given STP ramp currents to the CapMem quadrants.

        The STP calibration current is set equal to this ramp current:
        The ramp offset calibration works by sinking part of the
        ramp current within the first 2 ns of the ramp generation phase.
        The calibration current controls the sinked current, therefore it
        is sensible to scale it with the sourced (ramp) current.

        :param builder: Builder to append configuration to.
        :param parameters: STP ramp currents to set.

        :return: Builder with configuration appended.
        """

        self.log.DEBUG("Configuring STP ramp currents:", parameters)
        builder = helpers.capmem_set_quadrant_cells(
            builder,
            {halco.CapMemCellOnCapMemBlock.stp_i_ramp: parameters,
             halco.CapMemCellOnCapMemBlock.stp_i_calib: parameters})

        builder = helpers.wait(builder, constants.capmem_level_off_time)
        return builder

    @staticmethod
    def calculate_quadrant_means(results: np.ndarray) -> np.ndarray:
        """
        Calculate mean amplitudes per quadrant from an array of
        all drivers' amplitudes.

        :param results: Results from all synapse drivers on chip.

        :return: Mean results of all drivers on a quadrant.
        """

        quadrant_results = [
            list() for _ in halco.iter_all(halco.CapMemBlockOnDLS)]
        for coord in halco.iter_all(halco.SynapseDriverOnDLS):
            quadrant_results[int(coord.toCapMemBlockOnDLS().toEnum())].append(
                results[int(coord.toEnum())])

        return np.mean(quadrant_results, axis=1)

    def prelude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Measure maximum amplitude per driver, using a high ramp bias
        current and a high activation.

        :param connection: Connection to the chip to run on.
        """

        # measure maximum amplitudes per driver
        maximum_amplitudes = self.offset_calib.measure_maximum_amplitudes(
            connection)
        self.offset_calib.maximum_amplitudes = maximum_amplitudes

        # calculate target by rule of three and range limit
        all_targets = maximum_amplitudes \
            / int(hal.PADIEvent.HagenActivation.max) \
            * int(self.test_activation) * RANGE_LIMIT

        self.target = self.calculate_quadrant_means(all_targets)

        self.log.DEBUG("STP ramp calib target amplitudes:", self.target)

    def measure_results(self, connection: hxcomm.ConnectionHandle,
                        builder: sta.PlaybackProgramBuilder
                        ) -> np.ndarray:
        """
        Measure the amplitudes of all drivers at a low activation.

        This corresponds to a high DAC voltage, i.e. the end of
        the ramp. We return the mean amplitude per quadrant.

        :param connection: Connection to the chip to run on.
        :param builder: Builder to measure results with.

        :return: Array of STP amplitudes per quadrant.
        """

        base.run(connection, builder)

        # calibrate Hagen-mode DAC offset
        self.offset_calib.run(
            connection, algorithm=algorithms.BinarySearch())

        # measure amplitude at low activation
        measurement = SynapseDriverMeasurement(
            n_parallel_measurements=self.n_parallel_measurements)
        results = measurement.measure_syndrv_amplitudes(
            connection, activations=self.test_activation)

        quadrant_results = self.calculate_quadrant_means(results)

        self.log.DEBUG(
            f"Mean amplitude at activation {self.test_activation}:",
            quadrant_results)

        return quadrant_results

    def postlude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Print the calibrated ramp currents.

        :param connection: Connection to the chip to run on.
        """

        self.log.INFO(
            "Calibrated STP ramp currents, values:\n"
            + f"{self.result.calibrated_parameters}")


class HagenDACOffsetCalibration(base.Calibration):
    """
    Search Hagen-mode DAC offset settings such that drivers
    yield the same pulse lengths.

    Initially, amplitudes at a high STP ramp current and a low hagen-mode
    DAC potential are measured, corresponding to the maximum available
    pulse length. We select the DAC offset such that with the desired
    ramp current, the amplitude starts decreasing from this maximum
    at a low, but non-zero DAC value (which equals a high, but
    non-maximum activation). Hence, this alignes the start points
    of the STP ramps to the start point of the DAC's dynamic range.

    Requirements:
    * Static synapse driver bias currents are set. This can be achieved
      using `preconfigure_capmem()`.

    :ivar test_activation: Hagen-mode activation to use when
        determining the ramp start. Should be high, close but not
        equal to the maximum activation.
    :ivar n_parallel_measurements: Number of parallel measurements
        to execute when measuring results.
    :ivar maximum_amplitudes: Maximum amplitudes per driver, measured
        at a high ramp current and high activation. If not supplied
        otherwise, it will be measured during the prelude.
    :ivar log: Logger used for printing outputs.
    """

    def __init__(self):
        super().__init__(
            parameter_range=base.ParameterRange(
                hal.SynapseDriverConfig.HagenDACOffset.min,
                hal.SynapseDriverConfig.HagenDACOffset.max),
            n_instances=halco.SynapseDriverOnDLS.size,
            inverted=True)
        self.test_activation = hal.PADIEvent.HagenActivation(29)
        self.n_parallel_measurements = 1
        self.maximum_amplitudes: Optional[np.ndarray] = None
        self.log = logger.get(
            "calix.hagen.synapse_driver.HagenDACOffsetCalibration")

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
            synapse_driver_config = hal.SynapseDriverConfig()
            synapse_driver_config.hagen_dac_offset = int(
                parameters[int(coord.toEnum())])
            synapse_driver_config.offset = DEFAULT_STP_OFFSET

            builder.write(coord, synapse_driver_config)

        return builder

    def measure_results(self, connection: hxcomm.ConnectionHandle,
                        builder: sta.PlaybackProgramBuilder
                        ) -> np.ndarray:
        """
        Read output amplitudes of synapse drivers.

        :param connection: Connection to the chip to run on.
        :param builder: Builder that is run before measuring.

        :return: Array of synapse drivers' output amplitudes.
        """

        base.run(connection, builder)
        measurement = SynapseDriverMeasurement(
            n_parallel_measurements=self.n_parallel_measurements)
        results = measurement.measure_syndrv_amplitudes(
            connection, activations=self.test_activation)
        return results

    def measure_maximum_amplitudes(self, connection: hxcomm.ConnectionHandle
                                   ) -> np.ndarray:
        """
        Measure the maximum output amplitudes available per driver.

        We configure a high ramp current and measure amplitudes at a
        low address in order to achieve the full pulse length,
        i.e. the maximum amplitude for each driver.

        :param connection: Connection to the chip to be measured.

        :return: Array of maximum amplitudes per driver.
        """

        iramp_tickets = list()
        previous_test_activation = self.test_activation
        self.test_activation = hal.PADIEvent.HagenActivation.max

        builder = sta.PlaybackProgramBuilder()

        for capmem_block in halco.iter_all(halco.CapMemBlockOnDLS):
            coord = halco.CapMemCellOnDLS(
                halco.CapMemCellOnCapMemBlock.stp_i_ramp, capmem_block)
            iramp_tickets.append(builder.read(coord))
            builder.write(coord, hal.CapMemCell(hal.CapMemCell.Value.max))
        helpers.wait(builder, constants.capmem_level_off_time)

        builder = self.configure_parameters(
            builder, np.ones(halco.SynapseDriverOnDLS.size, dtype=int)
            * hal.SynapseDriverConfig.HagenDACOffset.min)

        maximum_amplitudes = self.measure_results(connection, builder)
        self.log.DEBUG("Maximum amplitudes per driver:",
                       self.maximum_amplitudes)

        # restore previous i_ramp and test activation
        builder = sta.PlaybackProgramBuilder()
        for capmem_block in halco.iter_all(halco.CapMemBlockOnDLS):
            coord = halco.CapMemCellOnDLS(
                halco.CapMemCellOnCapMemBlock.stp_i_ramp, capmem_block)
            builder.write(
                coord, iramp_tickets[int(capmem_block.toEnum())].get())
        self.test_activation = previous_test_activation
        base.run(connection, builder)

        return maximum_amplitudes

    def prelude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Configure synapse array and measure calibration target.

        Measure target: median amplitude of all drivers.

        :param connection: Connection to the chip to run on.
        """

        if self.maximum_amplitudes is None:
            self.maximum_amplitudes = self.measure_maximum_amplitudes(
                connection)

        # calculate target by rule of three while limiting range:
        self.target = self.maximum_amplitudes \
            / int(hal.PADIEvent.HagenActivation.max) \
            * int(self.test_activation) * RANGE_LIMIT

        self.log.TRACE("DAC offset calib target amplitudes:", self.target)

    def postlude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Print the calibrated DAC offsets.

        :param connection: Connection to the chip to run on.
        """

        self.log.TRACE("Calibrated DAC offsets:\n"
                       + f"{self.result.calibrated_parameters}")

        builder = sta.PlaybackProgramBuilder()
        results = self.measure_results(connection, builder)
        self.log.DEBUG(
            "Deviation of synapse driver amplitudes after offset calib: "
            + f"{np.std(results):4.2f}")


def calibrate(connection: hxcomm.ConnectionHandle,
              offset_test_activation: hal.SynapseQuad.Label
              = hal.PADIEvent.HagenActivation(3)
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

    The hagen-mode DAC offsets are calibrated for each driver afterwards.

    Requirements:
    * Chip is initialized.
    * CADCs are calibrated. You can use `calix.common.cadc.calibrate()`
        to achieve this.
    * Neuron readout is connected to the CADC lines and global bias
        currents are set. You can use
        `calix.hagen.neuron_helpers.configure_chip()` to achieve this.

    :param connection: Connection to the chip to run on.
    :param offset_test_address: Address to align the hagen-mode DAC
        offsets on. Note that the address is the inverted activation,
        i.e. 0 yields a large value and 31 a small value.

    :return: SynapseDriverCalibResult containing STP ramp currents for
        the CapMem blocks, offsets for the drivers, and a success mask.
    """

    # preconfigure chip
    builder = sta.PlaybackProgramBuilder()
    builder = preconfigure_capmem(builder)
    base.run(connection, builder)

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

    # calibrate Hagen-mode DAC offset
    calibration = HagenDACOffsetCalibration()
    calibration.test_activation = offset_test_activation
    result = calibration.run(connection, algorithm=algorithms.BinarySearch())
    calib_result.hagen_dac_offset = result.calibrated_parameters
    calib_result.success = np.all(
        [calib_result.success, result.success], axis=0)

    return calib_result.to_synapse_driver_calib_result()
