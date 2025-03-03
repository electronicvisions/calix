"""
Provides classes for calibration of the correlation
characteristics, i.e. time constants and amplitudes.
"""

from typing import ClassVar, Dict, Optional, Union
from enum import Enum, auto
from dataclasses import dataclass, field
import copy

import numpy as np
import quantities as pq

from dlens_vx_v3 import hal, halco, lola, hxcomm, logger

from calix.common import algorithms, base, exceptions
from calix.hagen import helpers
from calix.spiking import correlation_measurement
from calix import constants


# Select a few quads where correlation is measured to gain an estimate
# of the characteristics in a whole quadrant.
# The quads should be spread in the physical dimensions of the quadrant,
# as we observe an asymmetry between left to right and top to bottom.
# All quads are located on the left (quadrant 0), they will be shifted
# to the other quadrants as required during the measurement.
REPRESENTATIVE_QUADS = [
    halco.SynapseQuadColumnOnDLS(col) for col in range(0, 16, 4)]


class CorrelationBranches(Enum):
    """
    Select branches of correlation: causal, acausal or both.
    """

    CAUSAL = auto()
    ACAUSAL = auto()
    BOTH = auto()

    @property
    def axes(self) -> slice:
        """
        Return the appropriate slice to apply to the result data in order
        to select the respective correlation branches.

        :return: Slice to apply along correlation branch dimension of
            result array.
        """

        if self == CorrelationBranches.CAUSAL:
            return slice(0, 1, 1)
        if self == CorrelationBranches.ACAUSAL:
            return slice(1, 2, 1)
        return slice(0, 2, 1)


class TimeConstantCalib(base.Calib):
    """
    Calibrate the correlation time constant.

    Correlation traces are measured in a few quads in all CapMem
    quadrants. The median time constant is estimated and the CapMem
    bias currents are tweaked accordingly.

    Requirements:
    * CADCs are enabled and calibrated.
    * External correlation voltages (v_res_meas, v_reset) are set.

    :ivar measurement: Instance of the correlation measurement class,
        used for measuring results.
    :ivar quads: List of synapse quad column coordinates. Measurements
        from all rows in those quads are taken as an estimate for the
        characteristics of the whole quadrant. All quads in the list
        are located in quadrant 0, the measurement is repeated in the
        other quadrants using corresponding offsets to the coordinates.
    :ivar log: Logger used to log outputs.
    :ivar branches: Correlation branch to consider. Choose from causal,
        acausal or both, using the Enum CorrelationBranches.
    """

    def __init__(self, target: Optional[pq.Quantity] = 5 * pq.us, *,
                 branches: CorrelationBranches = CorrelationBranches.CAUSAL,
                 amp_calib: int = 0, time_calib: int = 0):
        """
        :param target: Target time constant for calibration.
        :param amp_calib: Amplitude calibration setting to use in all
            synapses during measurement.
        :param time_calib: Time constant calibration setting to use in all
            synapses during measurement.
        """

        super().__init__(
            parameter_range=base.ParameterRange(
                hal.CapMemCell.Value.min, hal.CapMemCell.Value.max),
            n_instances=halco.CapMemBlockOnDLS.size,
            inverted=True)

        self.target = target
        self.measurement = correlation_measurement.CorrelationMeasurement(
            delays=[-50, -30, -20, -10, -5, -2.5, -1, -0.5,
                    0.5, 1, 2.5, 5, 10, 20, 30, 50] * pq.us,
            amp_calib=amp_calib, time_calib=time_calib)
        self.quads = copy.deepcopy(REPRESENTATIVE_QUADS)
        self.log = logger.get(
            "calix.spiking.correlation.TimeConstantCalib")
        self.branches = branches

    def prelude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Preconfigure the chip for correlation measurements.

        :param connection: Connection to the chip to run on.
        """

        self.measurement.prelude(connection)

    def configure_parameters(
            self, builder: base.WriteRecordingPlaybackProgramBuilder,
            parameters: np.ndarray) \
            -> base.WriteRecordingPlaybackProgramBuilder:
        """
        Configure the given ramp bias currents in the given builder.

        :param builder: Builder to append configuration instructions to.
        :param parameters: Correlation ramp bias current for each
            quadrant.

        :return: Builder with configuration appended.
        """

        config = {
            halco.CapMemCellOnCapMemBlock.syn_i_bias_ramp: parameters}
        self.log.DEBUG("Configuring ramp bias:", parameters)

        helpers.capmem_set_quadrant_cells(builder, config)
        helpers.wait(builder, constants.capmem_level_off_time)
        return builder

    def measure_results(self, connection: hxcomm.ConnectionHandle,
                        builder: base.WriteRecordingPlaybackProgramBuilder
                        ) -> np.ndarray:
        """
        Estimate correlation parameters of each quadrant.

        :param connection: Connection to the chip to run on.
        :param builder: Builder to run before measurements.

        :return: Array of each quadrant's time constants.
        """

        base.run(connection, builder)

        results = np.empty(self.n_instances) * pq.us
        for quadrant in halco.iter_all(halco.CapMemBlockOnDLS):
            quadrant_results = []
            for quad in self.quads:
                if int(quadrant.toCapMemBlockOnHemisphere().toEnum()) == 1:
                    quad = quad + 16  # move quad coord to east quadrant
                quadrant_results.append(self.measurement.measure_quad(
                    connection, quad=quad,
                    synram=quadrant.toHemisphereOnDLS().toSynramOnDLS()))
            quadrant_results = np.concatenate(quadrant_results, axis=1)
            _, taus = self.measurement.estimate_fit(quadrant_results)

            # select smaller time constant in case it could not be determined
            taus[np.isnan(taus)] = np.inf * pq.us

            # take median as result for the quadrant
            results[int(quadrant.toEnum())] = np.median(
                taus[:, :, self.branches.axes])

        self.log.DEBUG("Obtained time constants:", results)
        return results


class AmplitudeCalib(base.Calib):
    """
    Calibrate the correlation amplitude at delay zero.

    Correlation traces are measured in a few quads in all CapMem
    quadrants. The median amplitude, extrapolated at a delay of zero,
    is estimated and the CapMem bias currents are tweaked accordingly.

    Requirements:
    * CADCs are enabled and calibrated.
    * External correlation voltages (v_res_meas, v_reset) are set.

    :ivar measurement: Instance of the correlation measurement class,
        used for measuring results.
    :ivar quads: List of synapse quad column coordinates. Measurements
        from all rows in those quads are taken as an estimate for the
        characteristics of the whole quadrant. All quads in the list
        are located in quadrant 0, the measurement is repeated in the
        other quadrants using corresponding offsets to the coordinates.
    :ivar log: Logger used to log outputs.
    :ivar branches: Correlation branch to consider. Choose from causal,
        acausal or both, using the Enum CorrelationBranches.
    :ivar parameters: Last configured CapMem parameters.
    """

    def __init__(self, target: Optional[Union[float, np.ndarray]] = 0.5, *,
                 branches: CorrelationBranches = CorrelationBranches.CAUSAL,
                 amp_calib: int = 0, time_calib: int = 0):
        """
        :param target: Target amplitude per correlated event.
        :param amp_calib: Amplitude calibration setting to use in all
            synapses during measurement.
        :param time_calib: Time constant calibration setting to use in all
            synapses during measurement.
        :param capmem_current_low: Threshold to consider a current generated
            by the CapMem low. During calibration, the current will be
            increased in case amplitudes cannot be measured as expected.
            If the result contains low currents, a warning will be logged.
        """

        super().__init__(
            parameter_range=base.ParameterRange(
                0, hal.CapMemCell.Value.max),
            n_instances=halco.CapMemBlockOnDLS.size,
            inverted=True)

        self.target = target
        self.measurement = correlation_measurement.CorrelationMeasurement(
            delays=[-10, -5, -2.5, -1, 1, 2.5, 5, 10] * pq.us,
            amp_calib=amp_calib, time_calib=time_calib)
        self.quads = copy.deepcopy(REPRESENTATIVE_QUADS)
        self.log = logger.get(
            "calix.spiking.correlation.AmplitudeCalib")
        self.branches = branches
        self.parameters = np.zeros(halco.CapMemBlockOnDLS.size, dtype=int)
        self.capmem_current_low = 50

    def prelude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Preconfigure the chip for correlation measurements.

        :param connection: Connection to the chip to run on.
        """

        self.measurement.prelude(connection)

    def configure_parameters(
            self, builder: base.WriteRecordingPlaybackProgramBuilder,
            parameters: np.ndarray) \
            -> base.WriteRecordingPlaybackProgramBuilder:
        """
        Configure the given store bias currents in the given builder.

        :param builder: Builder to append configuration instructions to.
        :param parameters: Correlation store bias current for each
            quadrant.

        :return: Builder with configuration appended.
        """

        self.parameters = parameters
        config = {
            halco.CapMemCellOnCapMemBlock.syn_i_bias_store: parameters}
        self.log.DEBUG("Configuring store bias:", parameters)

        helpers.capmem_set_quadrant_cells(builder, config)
        helpers.wait(builder, constants.capmem_level_off_time)
        return builder

    def measure_results(self, connection: hxcomm.ConnectionHandle,
                        builder: base.WriteRecordingPlaybackProgramBuilder
                        ) -> np.ndarray:
        """
        Estimate correlation parameters of each quadrant.

        :param connection: Connection to the chip to run on.
        :param builder: Builder to run before measurements.

        :return: Array of each quadrant's amplitudes.
        """

        base.run(connection, builder)

        results = np.empty(self.n_instances)
        for quadrant in halco.iter_all(halco.CapMemBlockOnDLS):
            # set number of events depending on target
            try:
                self.measurement.n_events = \
                    int(70 / self.target[int(quadrant.toEnum())])
            except TypeError:
                self.measurement.n_events = int(70 / self.target)

            quadrant_results = []
            baselines_invalid = np.zeros(
                halco.CapMemBlockOnDLS.size, dtype=bool)
            for quad in self.quads:
                if int(quadrant.toCapMemBlockOnHemisphere().toEnum()) == 1:
                    quad = quad + 16  # move quad coord to east quadrant
                try:
                    quad_result = self.measurement.measure_quad(
                        connection, quad=quad,
                        synram=quadrant.toHemisphereOnDLS().toSynramOnDLS())
                    quadrant_results.append(quad_result)
                except exceptions.HardwareError as error:
                    if self.parameters[int(quadrant.toEnum())] \
                            < self.capmem_current_low:
                        baselines_invalid[int(quadrant.toEnum())] = True
                        expected_result_shape = (
                            len(self.measurement.delays),
                            halco.EntryOnQuad.size,
                            halco.SynapseRowOnSynram.size,
                            halco.SynapticInputOnNeuron.size)
                        quadrant_results.append(
                            np.ones(expected_result_shape) * np.nan)
                    else:
                        raise exceptions.CalibNotSuccessful(error)

            quadrant_results = np.concatenate(quadrant_results, axis=1)
            amps, _ = self.measurement.estimate_fit(quadrant_results)

            # select larger amplitude in case it could not be determined
            amps[np.isnan(amps)] = -np.inf

            # take median as result for the quadrant
            results[int(quadrant.toEnum())] = np.median(
                amps[:, :, self.branches.axes])

            # select smaller amplitude in case baseline read in quadrant is low
            results[baselines_invalid] = np.inf

            # select smaller amplitude in case amplitude could not be
            # determined (for whole quadrant) and CapMem bias is low
            results[np.all([results == -np.inf,
                            self.parameters < self.capmem_current_low],
                           axis=0)] = np.inf

        self.log.DEBUG("Obtained amplitudes:", results)
        return results

    def postlude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Log a warning in case resulting CapMem currents are low.

        :param connection: Connection to the chip to be calibrated.
        """

        if np.any(self.result.calibrated_parameters < self.capmem_current_low):
            self.log.WARN(
                "Correlation store bias currents were obtained at "
                + f"{self.result.calibrated_parameters}, which is low. "
                + "Consider increasing v_res_meas in case the calibration "
                + "matches your targets badly.")


@dataclass
class CorrelationCalibTarget(base.CalibTarget):
    """
    Target parameters for correlation calibration.

    :ivar amplitude: Target correlation amplitude (at delay 0) for
        all synapses, per correlated event. Feasible targets range from
        some 0.2 to 2.0, higher amplitudes will likely require adjusting
        v_res_meas.
    :ivar time_constant: Target correlation time constant for all
        synapses. Feasible targets range from some 2 to 30 us.
    """

    amplitude: float = 0.5
    time_constant: pq.Quantity = field(
        default_factory=lambda: 5 * pq.us)

    feasible_ranges: ClassVar[Dict[str, base.ParameterRange]] = \
        {"amplitude": base.ParameterRange(0.2, 2),
         "time_constant": base.ParameterRange(2 * pq.us, 30 * pq.us)}


@dataclass
class CorrelationCalibOptions(base.CalibOptions):
    """
    Further options for correlation calibration.

    :ivar calibrate_synapses: Decide whether individual synapses'
        calibration bits shall be calibrated. This requires hours
        of runtime and may not improve the usability singificantly.
    :ivar time_constant_priority: Priority given to time constant during
        individual calibration of synapses. Has to be in the range from
        0 to 1, the remaining priority is given to the amplitude.
    :ivar branches: Correlation traces to consider during calibration.
        Use the Enum type CorrelationBranches to select from causal,
        acausal or both.
    :ivar v_res_meas: Reset voltage for the measurement capacitors.
        Affects the achievable amplitudes: In case your desired amplitudes
        cannot be reached at sensible CapMem currents, consider increasing
        v_res_meas. However, this also increases problems observed on some
        synapses, like traces no longer behaving exponentially. Generally,
        you should set this as low as possible - we recommend some 0.9 V.
    :ivar v_reset: Reset voltage for the accumulation capacitors in
        each synapse. Controls the baseline measurement when the sensors'
        accumulation capacitors were reset and no correlated events
        were recorded. The baseline read should be near the upper end
        of the reliable range of the CADC, which is the case when
        v_reset is set to some 1.85 V. (There's a source follower circuit
        in the correlation readout.)
    :ivar default_amp_calib: Amplitude calibration setting used for
        all synapses when calibrating CapMem bias currents. Should
        allow headroom for adjusting amplitudes in both directions
        if individual synapses are to be calibrated. Otherwise, a
        value of 0 is recommended. Set sensibly by default.
    :ivar default_time_calib: Time constant calibration setting used
        for all synapses when calibrating CapMem bias currents. Should
        allow headroom for adjusting amplitudes in both directions
        if individual synapses are to be calibrated. Set sensibly
        by default.
    """

    branches: CorrelationBranches = CorrelationBranches.BOTH
    v_res_meas: pq.Quantity = field(
        default_factory=lambda: 0.9 * pq.V)
    v_reset: pq.Quantity = field(
        default_factory=lambda: 1.85 * pq.V)
    calibrate_synapses: bool = False
    time_constant_priority: float = 0.3
    default_amp_calib: Optional[int] = None
    default_time_calib: Optional[int] = None

    def check(self) -> None:
        """
        Check if given parameters are in a valid range.
        """

        if not 0 <= self.time_constant_priority <= 1:
            raise ValueError(
                f"Time constant priority {self.time_constant_priority} is "
                + "not in the range from 0 to 1.")

    def __post_init__(self) -> None:
        if self.default_amp_calib is None:
            self.default_amp_calib = 1 if self.calibrate_synapses else 0
        if self.default_time_calib is None:
            self.default_time_calib = 1 if self.calibrate_synapses else 0


@dataclass
class CorrelationCalibResult(base.CalibResult):
    """
    Result of a synapse correlation sensor calibration.

    Holds CapMem bias currents and individual calibration bits.
    """

    i_bias_ramp: np.ndarray = field(
        default_factory=lambda: np.ones(
            halco.CapMemBlockOnDLS.size, dtype=int) * 80)
    i_bias_store: np.ndarray = field(
        default_factory=lambda: np.ones(
            halco.CapMemBlockOnDLS.size, dtype=int) * 70)
    amp_calib: np.ndarray = field(
        default_factory=lambda: np.empty(
            (halco.NeuronConfigOnDLS.size, halco.SynapseRowOnSynram.size),
            dtype=int))
    time_calib: np.ndarray = field(
        default_factory=lambda: np.empty(
            (halco.NeuronConfigOnDLS.size, halco.SynapseRowOnSynram.size),
            dtype=int))

    def __post_init__(self) -> None:
        self.amp_calib[:] = self.options.default_amp_calib
        self.time_calib[:] = self.options.default_time_calib

    def apply(self, builder: base.WriteRecordingPlaybackProgramBuilder) \
            -> None:
        """
        Apply the calibration in the given builder.

        :param builder: Builder or dumper to append configuration
            instructions to.
        """

        dac_config = lola.DACChannelBlock().default_ldo_2
        dac_config.set_voltage(
            halco.DACChannelOnBoard.v_res_meas,
            self.options.v_res_meas.rescale(pq.V).magnitude)
        dac_config.set_voltage(
            halco.DACChannelOnBoard.mux_dac_25,
            self.options.v_reset.rescale(pq.V).magnitude)
        builder.write(halco.DACChannelBlockOnBoard(), dac_config)

        measurement = correlation_measurement.CorrelationMeasurement(
            delays=1 * pq.us, amp_calib=self.amp_calib,
            time_calib=self.time_calib, i_ramp=self.i_bias_ramp,
            i_store=self.i_bias_store)
        measurement.configure_all(builder)


def calibrate_synapses(connection: hxcomm.ConnectionHandle,
                       calib_result: CorrelationCalibResult) -> None:
    """
    Set the individual synapses' calibration bits.

    The bits are selected such that the deviation from the targets
    is smallest. To calculate the deviation, the relative difference
    in amplitude and time constant is weighted according to the
    time_constant_priority option and then added squared.

    This function will be called automatically as part of the `calibrate`
    function, if individual synapses are to be calibrated.

    Requirements:
    * CADCs are enabled and calibrated.
    * Correlation sensors are enabled and configured with CapMem
      biases suitable for the given targets. This can be achieved
      using the `calibrate` function.

    :param connection: Connection to the chip to calibrate.
    :param calib_result: Correlation calib result, with CapMem biases
        already calibrated.
    """

    amplitude_score = np.empty(
        (hal.SynapseQuad.AmpCalib.size, hal.SynapseQuad.TimeCalib.size,
         halco.NeuronConfigOnDLS.size, halco.SynapseRowOnSynram.size))
    time_constant_score = np.empty_like(amplitude_score)

    for amp_calib in range(hal.SynapseQuad.AmpCalib.size):
        for time_calib in range(hal.SynapseQuad.TimeCalib.size):
            measurement = correlation_measurement.CorrelationMeasurement(
                delays=[-20, -10, -5, -2.5, -1, -0.5, 0.5, 1, 2.5, 5,
                        10, 20] * pq.us,
                amp_calib=amp_calib, time_calib=time_calib)
            correlations = measurement.measure_chip(connection)
            amplitudes, time_constants = \
                measurement.estimate_fit(correlations)
            amplitudes = np.mean(
                amplitudes[:, :, calib_result.options.branches.axes], axis=2)
            time_constants = np.mean(
                time_constants[:, :, calib_result.options.branches.axes],
                axis=2)

            amplitude_score[amp_calib, time_calib, :, :] = \
                (amplitudes - calib_result.target.amplitude) \
                / calib_result.target.amplitude
            time_constant_score[amp_calib, time_calib, :, :] = \
                (time_constants - calib_result.target.time_constant) \
                / calib_result.target.time_constant

    # calculate best calibration bits: least squared deviation
    deviations = \
        (np.square(amplitude_score)
         * (1 - calib_result.options.time_constant_priority)) \
        + (np.square(time_constant_score)
           * calib_result.options.time_constant_priority)
    deviations[np.isnan(deviations)] = np.inf
    calib_result.amp_calib = np.argmin(np.min(deviations, axis=1), axis=0)
    calib_result.time_calib = np.argmin(np.min(deviations, axis=0), axis=0)


def calibrate(connection: hxcomm.ConnectionHandle, *,
              target: Optional[CorrelationCalibTarget] = None,
              options: Optional[CorrelationCalibOptions] = None
              ) -> CorrelationCalibResult:
    """
    Calibrate all synapses' correlation parameters.

    The CapMem bias currents are set such that the target amplitude
    and time constant are reached in the median of the measured synapses.

    If desired, the individual synapses' calibration bits are then
    used to shrink the deviations between synapses. This takes some
    two hours of runtime, and can be enabled via the `calibrate_synapses`
    option.

    Note that a strong asymmetry between causal and acausal traces
    is present on many synapses. This cannot be countered by calibration,
    as calibration parameters apply to both branches equally.
    In case your experiment uses only one branch, you can set the
    `branches` option accordingly, so that only one is considered
    during calibration.

    Requirements:
    * CADCs are enabled and calibrated.

    :param connection: Connection to the chip to calibrate.
    :param target: Calibration targets, given as an instance of
        CorrelationCalibTargets.
    :param options: Further options for calibration, given as an
        instance of CorrelationCalibOptions.

    :return: Correlation calib result.
    """

    if target is None:
        target = CorrelationCalibTarget()
    if options is None:
        options = CorrelationCalibOptions()

    # check input parameters
    target.check()

    calib_result = CorrelationCalibResult(target=target, options=options)

    # set necessary voltages
    builder = base.WriteRecordingPlaybackProgramBuilder()
    calib_result.apply(builder)
    base.run(connection, builder)

    # calibrate CapMem amplitude
    calibration = AmplitudeCalib(
        target=target.amplitude, branches=options.branches,
        amp_calib=options.default_amp_calib,
        time_calib=options.default_time_calib)
    calib_result.i_bias_store = \
        calibration.run(
            connection, algorithm=algorithms.BinarySearch()
        ).calibrated_parameters

    # calibrate CapMem time constant
    calibration = TimeConstantCalib(
        target=target.time_constant, branches=options.branches,
        amp_calib=options.default_amp_calib,
        time_calib=options.default_time_calib)
    calib_result.i_bias_ramp = \
        calibration.run(
            connection, algorithm=algorithms.BinarySearch()
        ).calibrated_parameters

    # re-apply amplitude calibration
    calibration = AmplitudeCalib()
    builder = base.WriteRecordingPlaybackProgramBuilder()
    builder = calibration.configure_parameters(
        builder, calib_result.i_bias_store)
    base.run(connection, builder)

    if options.calibrate_synapses:
        calibrate_synapses(connection, calib_result)

    return calib_result
