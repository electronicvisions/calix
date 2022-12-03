"""
Calibrates all CADC channels on Hicann-X for a given dynamic range.
"""

from typing import Optional, Union
import os
from dataclasses import dataclass
from warnings import warn

import numpy as np

from dlens_vx_v3 import halco, hal, sta, logger, hxcomm

from calix.common import helpers, algorithms, base, \
    cadc_helpers, cadc_evaluation
from calix import constants


@dataclass
class CADCCalibTarget(base.CalibrationTarget):
    """
    Target parameters for the CADC calibration.

    :ivar dynamic_range: CapMem settings (LSB) at the minimum and maximum
        of the desired dynamic range. By default, the full dynamic range
        of the CADC is used, which corresponds to some 0.15 to 1.05 V.
        The voltages are configured as `stp_v_charge_0`, which gets
        connected to the CADCs via the CapMem debug readout.
    :ivar read_range: Target CADC reads at the lower and upper end of
        the dynamic range.
    """

    dynamic_range: base.ParameterRange = base.ParameterRange(70, 550)
    read_range: base.ParameterRange = base.ParameterRange(20, 220)
    feasible_ranges = {
        "dynamic_range": base.ParameterRange(70, 550),
        "read_range": base.ParameterRange(20, 220)
    }

    def check_values(self):
        """
        Check whether the given parameters are possible within the
        CapMem and CADC value ranges.
        """

        super().check_values()

        if np.any(np.array(self.dynamic_range) < hal.CapMemCell.Value.min):
            raise ValueError("CADC dynamic_range is below CapMem range.")
        if np.any(np.array(self.dynamic_range) > hal.CapMemCell.Value.max):
            raise ValueError("CADC dynamic_range is above CapMem range.")
        if np.any(np.array(self.read_range) < hal.CADCSampleQuad.Value.min):
            raise ValueError("CADC read_range is below CADC value range.")
        if np.any(np.array(self.read_range) > hal.CADCSampleQuad.Value.max):
            raise ValueError("CADC read_range is above CADC value range.")


@dataclass
class CADCCalibOptions(base.CalibrationOptions):
    """
    Further configuration parameters for the CADC calibration, that are
    not directly calibration targets.

    :ivar calibrate_offsets: Decide whether the individual channel
        offsets are calibrated. For standard usecases, including
        neuron and correlation measurements, this should be enabled
        (default). Only in case the auto-calibrating correlation
        reset (cf. hal.CommonCorrelationConfig) is used, this
        should be disabled.
    """

    calibrate_offsets: bool = True


@dataclass
class CADCCalibResult(base.CalibrationResult):
    """
    Result object for the CADC calibration.

    Contains:
    * Calibrated values for cadc_v_ramp_offset. One per quadrant.
    * Calibrated values for cadc_i_ramp_slope. One per quadrant.
    * Calibrated digital channel offsets. One per channel.
    * Success mask for calibration. Successfully calibrated channels
    are indicated by True, for any failure the channels will be flagged
    False. Failing quadrants result in all channels on that quadrant
    being flagged False.
    """

    v_ramp_offset: np.ndarray = np.empty(
        halco.NeuronConfigBlockOnDLS.size, dtype=int)
    i_ramp_slope: np.ndarray = np.empty(
        halco.NeuronConfigBlockOnDLS.size, dtype=int)
    channel_offset: np.ndarray = np.zeros(
        halco.CADCChannelConfigOnDLS.size, dtype=int)
    success: Optional[np.ndarray] = None

    def apply(self, builder: Union[sta.PlaybackProgramBuilder,
                                   sta.PlaybackProgramBuilderDumper]):
        """
        Apply the calibration into the given builder.

        Applying configures and enables the CADC so that it is
        ready for usage, just like after running the calibration.

        :param builder: Builder or dumper to append configuration
            instructions to.
        """

        # global digital CADC block config
        cadc_config = hal.CADCConfig()
        cadc_config.enable = True

        for cadc_coord in halco.iter_all(halco.CADCConfigOnDLS):
            builder.write(cadc_coord, cadc_config)

        # Set quadrant CapMem cells
        builder = helpers.capmem_set_quadrant_cells(
            builder, {
                halco.CapMemCellOnCapMemBlock.cadc_v_bias_ramp_buf: 400,
                halco.CapMemCellOnCapMemBlock.cadc_i_bias_comp: 1022,
                halco.CapMemCellOnCapMemBlock.cadc_i_bias_vreset_buf: 1022,
                halco.CapMemCellOnCapMemBlock.cadc_v_ramp_offset:
                self.v_ramp_offset,
                halco.CapMemCellOnCapMemBlock.cadc_i_ramp_slope:
                self.i_ramp_slope})

        # Set channel offsets
        for coord in halco.iter_all(halco.CADCChannelConfigOnDLS):
            config = hal.CADCChannelConfig()
            config.offset = self.channel_offset[int(coord)]
            builder.write(coord, config)


class RampOffsetCalibration(base.Calibration):
    """
    CADC Calibration Part 1: Calibrate the ramp reset/start voltage.

    This is done by setting the current onto the ramp high and searching
    the ramp start voltage such that the given minimum of the dynamic
    range reads the given target value at the CADCs.

    This class implements configure and measure functions required by a
    calibration algorithm to run. Calling run on the calibration does
    the necessary preconfiguring and uses the given algorithm to
    find suitable parameters.

    Requirements:
    * All CADC channels are connected to the CADC debug line (can be
      achieved with `cadc_helpers.configure_chip()`).
    * CADC debug line connected to CapMem cell `stp_v_charge_0`
      (can be archived with `cadc_helpers.configure_readout_cadc_debug()`)

    :ivar dynamic_range_min: Lower end of the desired dynamic range
        of the CADCs. Given in CapMem LSB.
        The voltage is configured as `stp_v_charge_0`, which is
        connected to the CADCs via the CapMem debug readout.
    """

    def __init__(self, target: int = 20, dynamic_range_min: int = 70):
        """
        :param dynamic_range_min: Lower end of the desired dynamic range
            of the CADCs. Given in LSB of a global CapMem cell.
        """
        self.dynamic_range_min = dynamic_range_min

        allowed_range = base.ParameterRange(hal.CapMemCell.Value.min,
                                            hal.CapMemCell.Value.max)
        super().__init__(
            parameter_range=allowed_range,
            n_instances=halco.CapMemBlockOnDLS.size,
            inverted=True,
            errors=["Ramp start voltage for quadrants {0} has reached {1}."
                    + os.linesep + "Please choose a greater lower end of the "
                    + "dynamic range. 70 seems like a feasible value.",
                    "Ramp start voltage for quadrants {0} has reached {1}."
                    + os.linesep + "Please check your dynamic range "
                    + "lower boundary."])
        self.target = target

    def prelude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Function to preconfigure CapMem parameters for the calibration runs.
        Sets the ramp slope current high and the common voltage applied to
        all CADC channels to the lower end of the specified dynamic range.
        This reference voltage is set via the cell `stp_v_charge_0` that
        has to be connected to the CapMem debug output.

        :param connection: Connection to the chip to calibrate.
        """

        builder = helpers.capmem_set_quadrant_cells(
            sta.PlaybackProgramBuilder(),
            {halco.CapMemCellOnCapMemBlock.cadc_i_ramp_slope: 350,
             halco.CapMemCellOnCapMemBlock.stp_v_charge_0:
             self.dynamic_range_min})

        base.run(connection, builder)

    def configure_parameters(self, builder: sta.PlaybackProgramBuilder,
                             parameters: np.ndarray
                             ) -> sta.PlaybackProgramBuilder:
        """
        Set CapMem cells cadc_v_ramp_offset of each quadrant to the
        values given in the array.
        :param builder: Builder to append configuring instructions.
        :param parameters: Array of ramp start voltages to be configured.

        :return: Builder with configuration instructions appended.
        """

        builder = helpers.capmem_set_quadrant_cells(
            builder,
            {halco.CapMemCellOnCapMemBlock.cadc_v_ramp_offset: parameters})

        builder = helpers.wait(builder, constants.capmem_level_off_time)
        return builder

    def measure_results(self, connection: hxcomm.ConnectionHandle,
                        builder: sta.PlaybackProgramBuilder) -> np.ndarray:
        """
        Read all CADC channels. Compute means of quadrants and return those.

        :param connection: Connection to read CADCs with.
        :param builder: Builder to append read instructions to before
            executing it.

        :return: Array with mean CADC reads of each quadrant.
        """

        cadc_data = cadc_helpers.read_cadcs(connection, builder)
        cadc_data = cadc_helpers.reshape_cadc_quadrants(cadc_data)

        quadrant_results = np.mean(cadc_data, axis=1)
        return quadrant_results


class RampSlopeCalibration(base.Calibration):
    """
    CADC Calibration part 2: Calibrate the steepness of the ramp,
    i.e. the current onto the ramp capacitor.

    This is done by adjusting the value until the upper end of the
    given dynamic range reads a target value at the CADCs.

    This class implements functions to configure the capmem cell that controls
    the ramp current and to measure the results. It also contains necessary
    preconfiguration of the CapMem.

    Requirements:
    * All CADC channels are connected to the CADC debug line (can be
      achieved with `cadc_helpers.configure_chip()`).
    * CADC debug line connected to CapMem cell `stp_v_charge_0`
      (can be archived with `cadc_helpers.configure_readout_cadc_debug()`)

    :ivar dynamic_range_max: Upper end of the desired dynamic range
        of the CADCs. Given in LSB of a global CapMem cell.
        The voltage is configured as `stp_v_charge_0`, which has to be
        connected to the CADCs via the CapMem debug readout.
    """

    def __init__(self, target: int = 220, dynamic_range_max: int = 550):
        allowed_range = base.ParameterRange(hal.CapMemCell.Value.min,
                                            hal.CapMemCell.Value.max)
        super().__init__(
            parameter_range=allowed_range,
            n_instances=halco.CapMemBlockOnDLS.size,
            inverted=True,
            errors=["Ramp current for quadrants {0} has reached {1}."
                    + os.linesep + "Check if the upper boundary of the "
                    + "dynamic range is sufficiently larger than the "
                    + "lower boundary.",
                    "Ramp current for quadrants {0} has reached {1}."
                    + os.linesep + "Please check your dynamic range "
                    + "upper boundary. Feasible values range up to roughly "
                    + "600."])

        self.dynamic_range_max = dynamic_range_max
        self.target = target

    def prelude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Function to preconfigure capmem parameters for the calibration runs.
        Sets the the common voltage applied to all CADC channels to the upper
        end of the specified dynamic range.

        :param connection: Connection to the chip to be calibrated.
        """

        builder = helpers.capmem_set_quadrant_cells(
            sta.PlaybackProgramBuilder(),
            {halco.CapMemCellOnCapMemBlock.stp_v_charge_0:
             self.dynamic_range_max})
        base.run(connection, builder)

    def configure_parameters(self, builder: sta.PlaybackProgramBuilder,
                             parameters: np.ndarray
                             ) -> sta.PlaybackProgramBuilder:
        """
        Set CapMem cells cadc_i_ramp_slope of each quadrant to the
        values given in the array.
        :param builder: Builder to append configuring instructions.
        :param parameters: Array of ramp currents to be configured.

        :return: Builder with configuration instructions appended.
        """

        builder = helpers.capmem_set_quadrant_cells(
            builder,
            {halco.CapMemCellOnCapMemBlock.cadc_i_ramp_slope: parameters})

        builder = helpers.wait(builder, constants.capmem_level_off_time)
        return builder

    def measure_results(self, connection: hxcomm.ConnectionHandle,
                        builder: sta.PlaybackProgramBuilder) -> np.ndarray:
        """
        Read all CADC channels. Compute means of quadrants and return those.

        :param connection: Connection to read CADCs with.
        :param builder: Builder to append read instructions to.

        :return: Array with mean CADC reads of each quadrant.
        """

        cadc_data = cadc_helpers.read_cadcs(connection, builder)
        cadc_data = cadc_helpers.reshape_cadc_quadrants(cadc_data)

        quadrant_results = np.mean(cadc_data, axis=1)
        return quadrant_results


class ChannelOffsetCalibration(base.Calibration):
    """
    CADC Calibration part 3: Calibrating digital offsets of individual
    channels. This is done by setting an intermediate voltage at the
    CADC debug line and setting the offsets such that all channels
    read the same result.

    As one measurement at constant voltage for all channels suffices, the
    LinearPrediction algorithm can be used for calibration.

    Requirements:
    * All CADC channels are connected to the CADC debug line (can be
      achieved with `cadc_helpers.configure_chip()`).
    * CADC debug line connected to CapMem cell `stp_v_charge_0`
      (can be archived with `cadc_helpers.configure_readout_cadc_debug()`)

    :ivar dynamic_range_mid: Mean of dynamic range min and max. The CADC
        channels should read medium results (around 120-128) there, if the
        range and read targets in the previous parts are set up sensible.
        The voltage is configured as `stp_v_charge_0`, which has to be
        connected to the CADCs via the CapMem debug readout.
    :ivar initial_results: CADC reads before calibrating the offsets.
        Stored to print statistics after calibration.
    """

    def __init__(self, dynamic_range_mid: hal.CapMemCell.Value
                 = hal.CapMemCell.Value(310)):
        super().__init__(
            parameter_range=base.ParameterRange(
                hal.CADCChannelConfig.Offset.min,
                hal.CADCChannelConfig.Offset.max),
            n_instances=halco.CADCChannelConfigOnDLS.size,
            inverted=False,
            errors=["Offsets for channels {0} have reached {1}."] * 2)
        self.dynamic_range_mid = dynamic_range_mid
        self.initial_results: Optional[np.ndarray] = None

    @staticmethod
    def find_target_read(reads: np.ndarray) -> int:
        """
        Inspect a given array of CADC Samples and return the median
        read of all CADC channels. This will be used as target read
        when calculating the individual channel offsets.

        If the median of all CADC channels is far off the expected value,
        a warning is logged.

        :param reads: Array of CADC Samples for both synrams obtained at
            a constant common voltage.

        :return: Calibration target for individual CADC channel offsets.
        """

        read_target = int(np.median(reads))

        allowed_deviation = 50
        middle_of_cadc_range = int(hal.CADCSampleQuad.Value.end / 2)
        if abs(read_target - middle_of_cadc_range) > allowed_deviation:
            log = logger.get(
                "calix.common.cadc.ChannelOffsetCalibration.find_target_read")
            log.WARN(f"Median read has been {read_target} "
                     + f"while a value around {middle_of_cadc_range} is "
                     + "expected." + os.linesep
                     + "Check if the first parts of the calibration worked "
                     + "properly and all CADC channels read proper results.")

        return read_target

    def prelude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Configure debug readout cell to roughly the middle of the
        desired dynamic CADC range.
        Measure the target read to aim for when calculating the offsets.

        :param connection: Connection to the chip to calibrate.
        """

        builder = sta.PlaybackProgramBuilder()
        builder = helpers.capmem_set_quadrant_cells(
            builder,
            {halco.CapMemCellOnCapMemBlock.stp_v_charge_0:
             self.dynamic_range_mid})

        # longer wait here to ensure stable voltages (c.f. issue 3512)
        builder = helpers.wait(builder, 5 * constants.capmem_level_off_time)

        results = self.measure_results(connection, builder)
        self.initial_results = results
        self.target = self.find_target_read(results)

    def configure_parameters(self, builder: sta.PlaybackProgramBuilder,
                             parameters: np.ndarray
                             ) -> sta.PlaybackProgramBuilder:
        """
        Set up the given CADC channel offsets.
        Expects the given parameter array to be ordered like the iter_all
        of CADCChannelConfigOnDLS.

        :param builder: Builder to append configuration instructions to.
        :param parameters: CADC channel offsets to set.

        :return: Builder with configuration instructions appended.
        """

        for channel_id, channel_coord in enumerate(
                halco.iter_all(halco.CADCChannelConfigOnDLS)):
            channel_config = hal.CADCChannelConfig()
            channel_config.offset = parameters[channel_id]
            builder.write(channel_coord, channel_config)

        return builder

    def measure_results(self, connection: hxcomm.ConnectionHandle,
                        builder: sta.PlaybackProgramBuilder) -> np.ndarray:
        """
        Read all CADC channels on chip.

        :param connection: Connection to the chip to calibrate.
        :param builder: Builder to append read instructions to.

        :return: Array containing integer results for each channel.
            The order is top causal, top acausal,
            bottom causal, bottom acausal;
            within these blocks the channels are ordered from left to right.
            The order of returned results corresponds to the enums of
            halco.CADCChannelConfigOnDLS.
        """

        return cadc_helpers.read_cadcs(connection, builder)

    def postlude(self, connection: hxcomm.ConnectionHandle):
        """
        Print statistics about the CADC reads before and after calibration.
        Data after calibration is measured using the connection.

        :param connection: Connection to the chip to run on.
        """

        cadc_evaluation.check_calibration_success(
            connection, builder=sta.PlaybackProgramBuilder(),
            read_data=self.initial_results)


def calibrate(
        connection: hxcomm.ConnectionHandle,
        target: Optional[CADCCalibTarget] = None,
        options: Optional[CADCCalibOptions] = None, *,
        dynamic_range: Optional[base.ParameterRange] = None,
        read_range: Optional[base.ParameterRange] = None,
        calibrate_offsets: Optional[bool] = None
) -> CADCCalibResult:
    """
    Calibrates all the CADCs (top, bottom, causal, acausal) to work
    in a given dynamic range (given as CapMem LSB voltage settings).
    After calling this function, the CADC is left in a calibrated state,
    ready for usage.

    Part 1 sets the reset voltage of the CADC ramp depending on the
    given minimum of the dynamic range for readout.
    The reset voltage is set slightly below this desired readout,
    such that the CADC reads read_range.lower (default: 20) there
    (to avoid clipping of channels that get high offsets later).

    Part 2 sets the ramp current (steepness) depending on the given
    maximum of the dynamic range for readout.
    The current is set such that the ramp crosses the upper end of the
    dynamic readout range when the counter reads read_range.upper
    (default: 220). This is partly to avoid the last part of the dynamic
    range as the ramp will become less linear, and again to avoid clipping
    of channels with higher offsets.

    Part 3 applies a constant calibration voltage to all CADC channels,
    chosen in the middle of the dynamic range, and requires all channels
    to read the same result. The digital offsets are set to compensate
    the observed differences in reads.

    :param connection: Connection to a chip that this calibration will
        run on.
    :param target: Target parameters for calibration, given as an
        instance of CADCCalibTarget. Refer there for the individual
        parameters.
    :param options: Further options for calibration, given as an
        instance of CADCCalibOptions. Refer there for the individual
        parameters.

    :returns: CADCCalibResult, containing the settings for the CADC ramps
        of each quadrant and the digital offsets and calibration success
        of each channel.
    """

    if target is None:
        target = CADCCalibTarget()
    if options is None:
        options = CADCCalibOptions()

    # pylint: disable=too-many-branches
    used_deprecated_parameters = False
    if dynamic_range is not None:
        target.dynamic_range = dynamic_range
        used_deprecated_parameters = True
    if read_range is not None:
        target.read_range = read_range
        used_deprecated_parameters = True
    if calibrate_offsets is not None:
        options.calibrate_offsets = calibrate_offsets
        used_deprecated_parameters = True

    # delete deprecated arguments, to ensure the correct ones are used
    # in the following code
    del dynamic_range
    del read_range
    del calibrate_offsets

    if used_deprecated_parameters:
        warn(
            "Passing arguments directly to calibrate() functions is "
            "deprecated. Please now use the target and option classes.",
            DeprecationWarning, stacklevel=2)

    target.check()

    log = logger.get("calix.common.cadc_calibration.calibrate")

    # Configure switches for CADC debug line.
    builder = sta.PlaybackProgramBuilder()
    builder = cadc_helpers.configure_chip(builder)

    # configure readout/voltage feedback:
    builder = cadc_helpers.configure_readout_cadc_debug(builder)

    # run program for configuration
    base.run(connection, builder)

    # Initialize results to be returned as list
    calib_result = CADCCalibResult()

    # Part 1: Ramp offset
    calibration = RampOffsetCalibration(
        dynamic_range_min=target.dynamic_range.lower)
    result = calibration.run(
        connection, algorithms.BinarySearch(), target=target.read_range.lower)
    calib_result.v_ramp_offset = result.calibrated_parameters
    calib_result.success = cadc_helpers.convert_success_masks(result.success)
    log.INFO(f"Calibrated v_ramp_start, values: {calib_result.v_ramp_offset}")

    # Part 2: Ramp slope
    calibration = RampSlopeCalibration(
        dynamic_range_max=target.dynamic_range.upper)
    result = calibration.run(
        connection, algorithms.BinarySearch(), target=target.read_range.upper)
    calib_result.i_ramp_slope = result.calibrated_parameters
    calib_result.success = np.all([
        calib_result.success,
        cadc_helpers.convert_success_masks(result.success)], axis=0)
    log.INFO(f"Calibrated i_ramp, values: {calib_result.i_ramp_slope}")

    if options.calibrate_offsets:
        # Part 3: Channel offsets
        calibration = ChannelOffsetCalibration(
            int(np.mean(target.dynamic_range)))
        result = calibration.run(
            connection, algorithms.LinearPrediction(
                probe_parameters=0,
                offset=-int(hal.CADCSampleQuad.Value.max / 2), slope=1))
        calib_result.channel_offset = result.calibrated_parameters
        calib_result.success = np.all([
            calib_result.success, result.success], axis=0)
        log.INFO("Calibrated digital offsets, values: "
                 + f"{calib_result.channel_offset}")

    return calib_result
