"""
Provides calibration classes for conductance-based (COBA) synaptic
input parameters.
"""


from typing import Optional, Type, Union
from abc import abstractmethod

import numpy as np

from dlens_vx_v3 import halco, hal, hxcomm, logger

from calix import constants
from calix.common import algorithms, base, boundary_check, helpers
from calix.hagen import neuron_synin, neuron_potentials, neuron_helpers


class COBAReferenceCalib(base.Calib):
    """
    Calibrates the COBA reference potential.

    The COBA OTA compares the membrane potential with its reference
    potential and modulates the CUBA bias current accordingly. We
    calibrate the reference potential such that the COBA OTA does not
    generate an output current and therefore the original CUBA synaptic
    amplitude is not modulated.

    This calibration needs to be supplied with the original CUBA amplitude
    as a target and the corresponding parameters how the amplitude
    was measured. It then adjusts the COBA reference potential such that
    the original CUBA amplitude is observed. Since the COBA OTA has this
    reference potential and the membrane potential (here: leak potential)
    as inputs, the reference potential equals the leak potential after
    successful calibration.

    Requirements:
    * CADC is calibrated and neuron membrane potential is connected to
      the CADCs.
    * Leak potential is calibrated at the potential where the original
      CUBA amplitudes should be unchanged. Note that if you use more
      than one event (`n_events` > 1) to measure amplitudes, consider
      setting the leak potential slightly below/above the potential
      where you want the same strength for a single synaptic event,
      such that the potential is roughly at half the amplitude of
      the synaptic events. This will counter the effect of the
      amplitude modulation due to the COBA circuit.
    * The original CUBA amplitude is supplied as target.

    :ivar cuba_bias_calib: Instance of the calibration for CUBA synaptic
        input bias current. Used for measuring synaptic input amplitudes.
    :ivar target_potential: Target reference potential in CADC units.
    :ivar expected_leak_parameters: Leak parameters found at target
        reference potential with COBA modulation disabled.
    :ivar allowed_deviation: Allowed deviation of resting potential from
        given target_potential after COBA modulation is enabled. A high
        deviation indicates a strong offset current generated from the
        CUBA synaptic input OTA, which in turn means a strong change in
        its bias current, i.e. a COBA modulation. Since the COBA modulation
        should be zero at the reference potential, we counter this state
        by adjusting the reference parameters. We try to measure the
        correct direction of parameter update by testing higher and lower
        parameters. In case the change in resting potential is not
        significant, which can happen if the reference potential is far
        off the target, we update the parameters in the direction of the
        expected leak parameters - which match the expected reference
        potentials apart from the individual OTA's offsets.
    :ivar parameters: Currently configured reference potential parameters.
        Saved to correct them towards the expected leak parameters if
        necessary, cf. allowed_deviation.
    :ivar log: Logger used to log outputs.
    """

    def __init__(self, n_events: int, target_potential: Union[int, np.ndarray],
                 expected_leak_parameters: Union[int, np.ndarray]):
        """
        :param n_events: Number of events to use during measurement of
            synaptic input amplitudes. Has to correspond to how the given
            target was measured.
        """

        super().__init__(
            parameter_range=base.ParameterRange(
                hal.CapMemCell.Value.min, hal.CapMemCell.Value.max),
            n_instances=halco.NeuronConfigOnDLS.size, inverted=self._inverted)

        self.target_potential = target_potential
        self.expected_leak_parameters = expected_leak_parameters
        self.allowed_deviation = 15
        self.parameters: Optional[np.ndarray] = None

        self.cuba_bias_calib = self._cuba_bias_calib_type(
            recalibrate_reference=False)
        self.cuba_bias_calib.n_events = n_events

        # supply a random target to CUBA bias calib, so that it doesn't
        # try to find it during prelude
        self.cuba_bias_calib.target = 0
        self.log = logger.get(
            "calix.hagen.neuron_synin_coba.COBAReferenceCalib")

    @property
    @abstractmethod
    def _inverted(self) -> halco.CapMemRowOnCapMemBlock:
        """
        Select whether the parameter change needs to be inverted.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def _cuba_bias_calib_type(self) -> Type[neuron_synin.SynBiasCalib]:
        """
        Type of the calibration to use for measuring the synaptic
        input amplitudes.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def _capmem_coord(self) -> halco.CapMemRowOnCapMemBlock:
        """
        Coordinate of the capmem reference voltage to calibrate.
        """

        raise NotImplementedError

    def prelude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Calls prelude of CUBA bias calib, which is used for measurements.

        :param connection: Connection to the chip to run on.
        """

        self.cuba_bias_calib.prelude(connection)

    def configure_parameters(
            self, builder: base.WriteRecordingPlaybackProgramBuilder,
            parameters: np.ndarray) \
            -> base.WriteRecordingPlaybackProgramBuilder:
        """
        Configure the given COBA reference potentials.

        :param builder: Builder to append instructions to.
        :param parameters: Array of CapMem parameters to configure.

        :return: Builder with instructions appended.
        """

        self.parameters = parameters
        self.log.TRACE("COBA reference parameters:", parameters)

        helpers.capmem_set_neuron_cells(
            builder, {self._capmem_coord: parameters})
        builder = helpers.wait(builder, constants.capmem_level_off_time)
        return builder

    def measure_slope(self, connection: hxcomm.ConnectionHandle,
                      slope_estimation_distance: int = 80) -> np.ndarray:
        """
        Measure the change in resting potential with COBA modulation
        enabled, depending on the COBA reference potential setting.

        For two different reference potential settings (one above and
        one below the currently set reference potentials), the resting
        potentials are measured.
        The slope is calculated as the change in resting potential
        divided by the difference in reference potential settings.

        :param connection: Connection to the chip to run on.
        :param slope_estimation_distance: Amount to decrease (increase)
            COBA reference potential below (above) the previously
            configured parameters, in order to estimate the slope.
            Given in CapMem LSB, used for the lower (upper) measurement.
        """

        builder = base.WriteRecordingPlaybackProgramBuilder()
        reference_low = boundary_check.check_range_boundaries(
            self.parameters - slope_estimation_distance,
            self.parameter_range).parameters
        helpers.capmem_set_neuron_cells(
            builder, {self._capmem_coord: reference_low})
        helpers.wait(builder, constants.capmem_level_off_time)
        resting_potential_low = neuron_helpers.cadc_read_neuron_potentials(
            connection, builder)

        builder = base.WriteRecordingPlaybackProgramBuilder()
        reference_high = boundary_check.check_range_boundaries(
            self.parameters + slope_estimation_distance,
            self.parameter_range).parameters
        helpers.capmem_set_neuron_cells(
            builder, {self._capmem_coord: reference_high})
        helpers.wait(builder, constants.capmem_level_off_time)
        resting_potential_high = neuron_helpers.cadc_read_neuron_potentials(
            connection, builder)

        slope = (resting_potential_high - resting_potential_low) \
            / (reference_high - reference_low)

        return slope

    def measure_results(
            self, connection: hxcomm.ConnectionHandle,
            builder: base.WriteRecordingPlaybackProgramBuilder) -> np.ndarray:
        """
        Measure synaptic input amplitudes.

        We reuse the measure_amplitudes function of the CUBA bias calib
        for this purpose, but skip recalibrating the CUBA OTA's reference
        potentials, as its bias is unchanged.

        :param connection: Connection to the chip to run on.
        :param builder: Builder to execute before measuring amplitudes.

        :return: Array of measured synaptic input amplitudes.
        """

        # Measure resting potential to notice large deviations from target
        resting_potential = neuron_helpers.cadc_read_neuron_potentials(
            connection, builder)
        deviating_mask = np.abs(resting_potential - self.target_potential) \
            > self.allowed_deviation

        # Measure amplitudes
        results = self.cuba_bias_calib.measure_amplitudes(
            connection, builder=base.WriteRecordingPlaybackProgramBuilder())

        # If the difference between leak (resting) potential without and with
        # COBA is high, decide in which direction the reference potential
        # should be changed based on two factors:
        # - decide based on change in resting potential based on updated
        # reference potential.
        # - if this dependency is weak, solely take a decision based on
        # the deviation between reference and leak parameters.

        # measure change in resting potential depending on COBA reference
        slope = self.measure_slope(connection)

        # threshold to consider slope significant, in CADC LSB per CapMem LSB
        significant_slope = 0.1

        # store significant slopes in masks
        high_mask = np.any(
            [np.all([deviating_mask, slope > significant_slope,
                     resting_potential > self.target_potential], axis=0),
             np.all([deviating_mask, slope < -significant_slope,
                     resting_potential < self.target_potential], axis=0)],
            axis=0)
        low_mask = np.any(
            [np.all([deviating_mask, slope > significant_slope,
                     resting_potential < self.target_potential], axis=0),
             np.all([deviating_mask, slope < -significant_slope,
                     resting_potential > self.target_potential], axis=0)],
            axis=0)

        # update results according to masks
        results[high_mask] = -np.inf if self.inverted else np.inf
        results[low_mask] = np.inf if self.inverted else -np.inf

        # If slope of dependency of resting potential on reference potential
        # is low, correct parameters based on expected leak parameters
        deviating_mask[np.any([high_mask, low_mask], axis=0)] = False
        high_mask = np.all(
            [deviating_mask, self.parameters > self.expected_leak_parameters],
            axis=0)
        low_mask = np.all(
            [deviating_mask, self.parameters < self.expected_leak_parameters],
            axis=0)
        results[high_mask] = -np.inf if self.inverted else np.inf
        results[low_mask] = np.inf if self.inverted else -np.inf

        self.log.TRACE("Amplitude at reference potential:", results)
        return results


class ExcCOBAReferenceCalib(COBAReferenceCalib):
    _cuba_bias_calib_type = neuron_synin.ExcSynBiasCalib
    _capmem_coord = halco.CapMemRowOnCapMemBlock.e_synin_exc_rev
    _inverted = False


class InhCOBAReferenceCalib(COBAReferenceCalib):
    _cuba_bias_calib_type = neuron_synin.InhSynBiasCalib
    _capmem_coord = halco.CapMemRowOnCapMemBlock.e_synin_inh_rev
    _inverted = True


class COBABiasCalib(base.Calib):
    """
    Calibration for the COBA synaptic input modulation strength.

    To perform this calibration, we first calibrate the leak potential
    at the target reference potential and measure the CUBA amplitudes.
    We then calibrate the COBA bias current such that the amplitudes
    reach zero at the given reversal potential, and remain unchanged
    at the given reference potential.

    To determine the size of the amplitudes at the reversal potential,
    we need a second voltage where we measure the amplitudes of the
    synaptic input. We then estimate the amplitude at the reversal
    potential by a linear extrapolation. This second voltage is
    normally chosen in between the reference voltage and the reversal
    potential, close to the latter. In case the distance between
    reference and this second voltage is too small, or the measurement
    would fall outside the reliable CADC range, it is chosen differently,
    compare the doc-string of `too_little_separation_mask`.

    During the calibration, the COBA OTA's reference potential is
    re-calibrated each time its bias current was changed. A member
    coba_reference_calib is used to handle this sub-calibration.

    Requirements:
    * CADCs are calibrated and neuron membranes are connected to the CADCs.
    * Synaptic inputs are enabled and calibrated (in CUBA mode)
      to the paramters desired at e_coba_reference.

    :ivar e_coba_reference: Reference potential for COBA mode, where
        there is no modulation, i.e. the original CUBA amplitude is
        present. Given in CADC units. May be different than the neurons'
        desired leak potential, but must be reachable via the leak term,
        and allow for some headroom in the reliable CADC range in order
        to measure synaptic input amplitudes.
    :ivar e_coba_reversal: Desired COBA reversal potential, i.e. the
        potential where the amplitude has decreased to 0. Given in CADC
        units. May exceed the valid range of leak or CADC.
    :ivar leak_parameters_reference: Calibrated parameters for achieving
        a leak potential at the target e_coba_reference.
    :ivar cuba_bias_calib: Instance of the calibration for CUBA synaptic
        input bias current. Used for measuring synaptic input amplitudes.
    :ivar coba_reference_calib: Instance of the calibration for the COBA
        reference potential. Used to re-calibrate the reference each time
        when touching the bias current.
    :ivar too_little_separation_mask: Mask containing instances where
        the measurement that is normally taken near the reversal
        potential is taken on the other side of the reference
        potential due to insufficient space in the leak potential range.
    :ivar max_leak_target: Maximum value of feasible leak potential where
        synaptic input amplitudes can be measured, i.e. limited by the
        CADC's dynamic range. Used as the start point for interpolation
        of e_coba_reversal.
    :ivar min_leak_target: Minimum value of feasible leak potential where
        synaptic input amplitudes can be measured, i.e. limited by the
        CADC's dynamic range. Used as the start point for interpolation
        of e_coba_reversal.
    :ivar leak_target_reversal: Target for resting potential close to
        the reversal potential. If there is not enough separation to
        the reference potential, this target may be at the other side
        of the reference potential: cf. too_little_separation_mask.
    :ivar reliable_amplitudes: Target amplitude for CADC-based measurement
        of syn. input amplitude at COBA reference potential. Also affects
        the measurement point for the reversal potential as we ensure
        enough separation.
    :ivar allowed_deviation: Maximum deviation from resting potential
        near reversal potential from the intended target. If the
        deviation is exceeded, the COBA bias current is reduced.
    :ivar log: Logger used for output.
    """

    def __init__(self, e_coba_reference: Union[int, np.ndarray],
                 e_coba_reversal: Union[int, np.ndarray]):
        super().__init__(
            # Limit parameter range to 1/4 of the maximum:
            # Without this limit, a few neurons get larger COBA biases
            # than desired. The limited range is sufficient for biologically
            # plausible targets.
            parameter_range=base.ParameterRange(
                hal.CapMemCell.Value.min, hal.CapMemCell.Value.max // 4),
            n_instances=halco.NeuronConfigOnDLS.size, inverted=True)

        self.e_coba_reference = e_coba_reference
        if np.ndim(self.e_coba_reference) == 0:
            self.e_coba_reference = self.e_coba_reference \
                * np.ones(halco.NeuronConfigOnDLS.size, dtype=int)
        self.e_coba_reversal = e_coba_reversal
        if np.ndim(self.e_coba_reversal) == 0:
            self.e_coba_reversal = self.e_coba_reversal \
                * np.ones(halco.NeuronConfigOnDLS.size, dtype=int)

        self.max_leak_target = 165
        self.min_leak_target = 60
        self.reliable_amplitudes = 30
        self.allowed_deviation = 15

        # Sanity check:
        # COBA reference potential must not exceeed the range limits
        if np.any(self.e_coba_reference < self.min_leak_target):
            raise ValueError(
                "COBA reference potential is given lower than the minimum "
                + f"leak target. Choose at least {self.min_leak_target}.")
        if np.any(self.e_coba_reference > self.max_leak_target):
            raise ValueError(
                "COBA reference potential is given higher than the maximum "
                + f"leak target. Choose at most {self.max_leak_target}.")

        self.cuba_amplitudes: Optional[np.ndarray] = None
        self.leak_parameters_reference: Optional[np.ndarray] = None
        self.leak_target_reversal: Optional[np.ndarray] = None
        self.too_little_separation_mask: Optional[np.ndarray] = None

        self.cuba_bias_calib = self._cuba_bias_calib_type(
            recalibrate_reference=False)
        self.cuba_bias_calib.reliable_amplitudes = self.reliable_amplitudes

        self.coba_reference_calib: Optional[
            self._coba_reference_calib_type] = None
        self.log = logger.get("calix.hagen.neuron_synin_coba.CobaBiasCalib")

    @property
    @abstractmethod
    def _sign(self) -> int:
        """
        Expected sign of amplitudes, +1 or -1 for excitatory and inhibitory
        case, respectively.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def _cuba_bias_calib_type(self) -> Type[neuron_synin.SynBiasCalib]:
        """
        Type of the calibration to use for measuring the synaptic
        input amplitudes.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def _coba_reference_calib_type(self) -> Type[COBAReferenceCalib]:
        """
        Type of the calibration to use for the COBA reference potential.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def _capmem_coord(self) -> halco.CapMemRowOnCapMemBlock:
        """
        Coordinate of the capmem COBA bias current to calibrate.
        """

        raise NotImplementedError

    def prelude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Interpolate given potentials to set calibration targets.

        Since several events are used to determine the strength of the
        synaptic inputs, the COBA modulation has to be taken into
        account when setting the reference potential (compare the
        documentation of COBAReferenceCalib). We set the integration
        start potential such that the desired reference potential is
        at roughly half the expected integrated amplitudes.
        The obtained leak parameters are saved in order to later
        switch between this setting and a measurement close to
        the reversal potential quickly.

        We calibrate leak potentials shortly below (above) the desired
        reference potentials for excitatory (inhibitory) mode. This
        ensures that the changed membrane potential by integrating
        amplitudes does not cause a modulating effect already. The desired
        reference potential is centered within the expected range of
        integrated amplitudes. The obtained leak parameters are saved
        in order to later switch between this setting and a measurement
        close to the reversal potential quickly.

        The currently visible (CUBA) amplitude is measured and used for
        interpolation of the target amplitude at the maximum (or minimum)
        reachable leak potentials.

        :param connection: Connection to the chip to run on.
        """

        # disable COBA modulation for leak calibration and CUBA measurement
        builder = base.WriteRecordingPlaybackProgramBuilder()
        tickets = []
        for coord in halco.iter_all(halco.NeuronConfigOnDLS):
            tickets.append(builder.read(coord))
        base.run(connection, builder)

        builder = base.WriteRecordingPlaybackProgramBuilder()
        for coord, ticket in zip(
                halco.iter_all(halco.NeuronConfigOnDLS), tickets):
            config = ticket.get()
            config.enable_synaptic_input_excitatory_coba_mode = False
            config.enable_synaptic_input_inhibitory_coba_mode = False
            builder.write(coord, config)
        base.run(connection, builder)

        # Calibrate leak around e_coba_reference:
        # We calibrate the leak potential such that the reference
        # potential is at half the expected amplitude.
        leak_target = self.e_coba_reference - self.reliable_amplitudes * 0.5 \
            * self._sign

        calibration = neuron_potentials.LeakPotentialCalib(leak_target)
        self.leak_parameters_reference = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch()
        ).calibrated_parameters

        # Measure (original CUBA) amplitude
        self.cuba_bias_calib.prelude(connection)
        cuba_amplitudes = self.cuba_bias_calib.measure_amplitudes(
            connection, builder=base.WriteRecordingPlaybackProgramBuilder())
        self.log.DEBUG(
            "Target amplitude near reference potential:", cuba_amplitudes)

        # Configure COBA reference calib with measured CUBA amplitudes
        self.coba_reference_calib = self._coba_reference_calib_type(
            n_events=self.cuba_bias_calib.n_events,
            target_potential=leak_target,
            expected_leak_parameters=self.leak_parameters_reference)
        self.coba_reference_calib.target = cuba_amplitudes

        # Calibrate leak shortly before e_coba_reversal (may be unreachable).
        # Even if it was reachable, we cannot calibrate at the reversal
        # potential, since the target would be zero and a high bias
        # current would always satisfy this.
        leak_target = (self.e_coba_reversal - self.e_coba_reference) * 0.8 \
            + self.e_coba_reference
        leak_target = np.min(
            [leak_target, self.max_leak_target * np.ones_like(leak_target)],
            axis=0)
        leak_target = np.max(
            [leak_target, self.min_leak_target * np.ones_like(leak_target)],
            axis=0)

        # if leak_target is too close to the reference potential, choose
        # a different target (in the other direction).
        minimum_separation = 2.0 * self.reliable_amplitudes
        self.too_little_separation_mask = \
            np.abs(self.e_coba_reference - leak_target) < minimum_separation
        leak_target[self.too_little_separation_mask] = \
            (self.e_coba_reference - minimum_separation * self._sign
             )[self.too_little_separation_mask]

        if not np.all(
                [self.min_leak_target <= leak_target,
                 leak_target <= self.max_leak_target]):
            raise ValueError(
                "Insufficient range for COBA reference and reversal "
                "potential in the allowed leak range.")

        self.leak_target_reversal = leak_target

        # The measure_results() function returns the difference between
        # the (moving) target and the measurement. We want a difference
        # of zero -> fixed target should be zero
        self.target = 0

        # restore original COBA modulation enables
        builder = base.WriteRecordingPlaybackProgramBuilder()
        for coord, ticket in zip(
                halco.iter_all(halco.NeuronConfigOnDLS), tickets):
            config = ticket.get()
            builder.write(coord, config)
        base.run(connection, builder)

    def configure_parameters(
            self, builder: base.WriteRecordingPlaybackProgramBuilder,
            parameters: np.ndarray) \
            -> base.WriteRecordingPlaybackProgramBuilder:
        """
        Configure the given bias currents on the chip.

        :param builder: Builder to append instructions to.
        :param parameters: Array of CapMem parameters to configure.

        :return: Builder with instructions appended.
        """

        self.log.TRACE(
            "COBA bias:", parameters,
            f"statistics: {parameters.mean():4.2f} +- {parameters.std():4.2f}")

        helpers.capmem_set_neuron_cells(
            builder, {self._capmem_coord: parameters})
        builder = helpers.wait(builder, constants.capmem_level_off_time)
        return builder

    def measure_results(
            self, connection: hxcomm.ConnectionHandle,
            builder: base.WriteRecordingPlaybackProgramBuilder) \
            -> np.ndarray:
        """
        Measure the synaptic input amplitudes close to the
        reversal potential.

        Before measurement, the reference potential is re-calibrated.

        :param connection: Connection to the chip to run on.
        :param builder: Builder to be run before measurement.

        :return: Array of synaptic input amplitudes when measured
            near the target reversal potential.
        """

        # set up leak at e_coba_reference (using given builder)
        helpers.capmem_set_neuron_cells(
            builder, {halco.CapMemRowOnCapMemBlock.v_leak:
                      self.leak_parameters_reference})
        builder = helpers.wait(builder, constants.capmem_level_off_time)
        base.run(connection, builder)

        # re-calibrate e_synin_rev for original CUBA amplitudes
        self.coba_reference_calib.run(
            connection, algorithm=algorithms.NoisyBinarySearch())

        # calibrate leak near e_coba_reversal
        calibration = neuron_potentials.LeakPotentialCalib(
            self.leak_target_reversal)
        # we expect that some neurons do not reach the target and
        # handle them explicitly below -> disable error messages here
        calibration.errors = None
        calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch())

        # Measure reached leak potential
        reached_potentials = neuron_helpers.cadc_read_neuron_potentials(
            connection)
        deviating_mask = np.abs(
            reached_potentials - self.leak_target_reversal) \
            > self.allowed_deviation
        self.log.TRACE(
            "Resting potential before measurement for reversal potential:\n",
            reached_potentials)

        # Calculate target amplitude: linearly interpolated at
        # the reached leak potential
        distance_fraction = (reached_potentials - self.e_coba_reference) / (
            self.e_coba_reversal - self.e_coba_reference)
        cuba_amplitudes = self.coba_reference_calib.target
        target = cuba_amplitudes * (1 - distance_fraction)

        # measure amplitude at e_coba_reversal (or maximum leak potential)
        results = self.cuba_bias_calib.measure_amplitudes(
            connection, builder=base.WriteRecordingPlaybackProgramBuilder())

        # Invert result for instances that get measured on the other
        # side of the reference potential
        results -= target
        results[self.too_little_separation_mask] *= -1
        self.log.DEBUG(
            "Statistics of amplitude deviation:",
            f"{results.mean():4.2f} +- {results.std():4.2f}")

        # reduce COBA bias current in case deviation from target
        # resting potential is high
        results[deviating_mask] = -np.inf
        self.log.TRACE("Deviation of amplitude and target:", results)

        return results

    def postlude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Re-calibrate the reference potentials again after
        the bias currents have reached their final parameters.

        :param connection: Connection to the chip to run on.
        """

        # set up leak at e_coba_reference
        builder = base.WriteRecordingPlaybackProgramBuilder()
        helpers.capmem_set_neuron_cells(
            builder, {halco.CapMemRowOnCapMemBlock.v_leak:
                      self.leak_parameters_reference})
        builder = helpers.wait(builder, constants.capmem_level_off_time)
        base.run(connection, builder)

        # re-calibrate e_synin_rev for original CUBA amplitudes
        self.coba_reference_calib.run(
            connection, algorithm=algorithms.NoisyBinarySearch())


class ExcCOBABiasCalib(COBABiasCalib):
    _sign = 1
    _cuba_bias_calib_type = neuron_synin.ExcSynBiasCalib
    _coba_reference_calib_type = ExcCOBAReferenceCalib
    _capmem_coord = halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_coba


class InhCOBABiasCalib(COBABiasCalib):
    _sign = -1
    _cuba_bias_calib_type = neuron_synin.InhSynBiasCalib
    _coba_reference_calib_type = InhCOBAReferenceCalib
    _capmem_coord = halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_coba
