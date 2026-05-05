"""
Provides an interface for calibrating LIF neurons.
"""

from typing import Optional, Union, List
from dataclasses import dataclass, field

import numpy as np
import quantities as pq

from dlens_vx_v3 import sta, halco, hal, hxcomm, lola, logger

from pyccalix import NeuronCalibOptions, NeuronCalibTarget
from calix.common import algorithms, base
from calix.hagen import neuron_helpers, neuron_potentials, \
    neuron_dataclasses
from calix.hagen.neuron_dataclasses import NeuronCalibResult
from calix.spiking import neuron_threshold, refractory_period, \
    neuron_calib_parts
from calix import constants


@dataclass
class _CalibResultInternal(neuron_dataclasses.CalibResultInternal):
    """
    Class providing array-like access to calibrated parameters.
    Used internally during calibration.
    """

    v_threshold: np.ndarray = field(
        default_factory=lambda: np.empty(
            halco.NeuronConfigOnDLS.size, dtype=int))
    i_syn_exc_coba: np.ndarray = field(
        default_factory=lambda: np.zeros(
            halco.NeuronConfigOnDLS.size, dtype=int))
    i_syn_inh_coba: np.ndarray = field(
        default_factory=lambda: np.zeros(
            halco.NeuronConfigOnDLS.size, dtype=int))
    e_syn_exc_rev: np.ndarray = field(
        default_factory=lambda: np.zeros(
            halco.NeuronConfigOnDLS.size, dtype=int))
    e_syn_inh_rev: np.ndarray = field(
        default_factory=lambda: np.zeros(
            halco.NeuronConfigOnDLS.size, dtype=int))
    i_bias_nmda: np.ndarray = field(
        default_factory=lambda: np.zeros(
            halco.NeuronConfigOnDLS.size, dtype=int))
    clock_settings: Optional[refractory_period.Settings] = None
    neuron_configs: Optional[List[hal.NeuronConfig]] = None
    use_synin_small_capacitance: bool = False

    def set_neuron_configs_default(
            self, membrane_capacitance: Union[
                hal.NeuronConfig.MembraneCapacitorSize, np.ndarray],
            tau_syn: pq.Quantity,
            readout_neuron: Optional[halco.AtomicNeuronOnDLS] = None,
            e_coba_reversal: Optional[np.ndarray] = None):
        """
        Fill neuron configs with the given membrane capacitances, but
        otherwise default values.

        :param membrane_capacitance: Desired membrane capacitance (in LSB).
        :param tau_syn: Synaptic input time constant. Used to
            decide whether the high resistance mode is enabled.
        :param readout_neuron: Neuron to enable readout for.
        :param e_coba_reversal: COBA-mode reversal potential. Used to
            decide whether to enable COBA mode on a per-neuron basis:
            If the reversal potential is [+ infinite, - infinite], for
            excitatory and inhibitory input respectively, CUBA mode will
            be used.
            The array can be of shape (2,) providing global values for
            inhibitory/excitatory synapses or of shape (2, 512) setting
            the reversal potential for each neuron individually.
        """

        if e_coba_reversal is None:
            e_coba_reversal = np.array(
                [[None, None]] * halco.AtomicNeuronOnDLS.size)

        self.neuron_configs = []

        for atomic_neuron in halco.iter_all(halco.AtomicNeuronOnDLS):
            config = neuron_helpers.neuron_config_default()
            config.enable_threshold_comparator = True
            config.enable_synaptic_input_excitatory_coba_mode = \
                e_coba_reversal[atomic_neuron][0] is not None
            config.enable_synaptic_input_inhibitory_coba_mode = \
                e_coba_reversal[atomic_neuron][1] is not None

            if readout_neuron is not None:
                if readout_neuron == atomic_neuron:
                    config.enable_readout = True

            tau_exc = tau_syn[atomic_neuron][0]
            tau_inh = tau_syn[atomic_neuron][1]
            c_mem = membrane_capacitance[atomic_neuron]
            config.membrane_capacitor_size = \
                hal.NeuronConfig.MembraneCapacitorSize(c_mem)

            # min. tau_syn with high resistance mode: some 20 us
            if tau_exc < 20 * pq.us:
                config.enable_synaptic_input_excitatory_high_resistance = False
            else:
                config.enable_synaptic_input_excitatory_high_resistance = True
            if tau_inh < 20 * pq.us:
                config.enable_synaptic_input_inhibitory_high_resistance = False
            else:
                config.enable_synaptic_input_inhibitory_high_resistance = True

            self.neuron_configs.append(config)

    def to_atomic_neuron(self,
                         neuron_coord: halco.AtomicNeuronOnDLS
                         ) -> lola.AtomicNeuron:
        """
        Returns an AtomicNeuron with calibration applied.

        :param neuron_coord: Coordinate of requirested neuron.

        :return: Complete AtomicNeuron configuration.
        """

        atomic_neuron = super().to_atomic_neuron(neuron_coord)
        neuron_id = neuron_coord.toEnum().value()

        atomic_neuron.set_from(self.neuron_configs[neuron_id])

        atomic_neuron.threshold.v_threshold = hal.CapMemCell.Value(
            self.v_threshold[neuron_id])

        anref = atomic_neuron.refractory_period
        anref.refractory_time = hal.NeuronBackendConfig.RefractoryTime(
            self.clock_settings.refractory_counters[neuron_id])
        anref.reset_holdoff = hal.NeuronBackendConfig.ResetHoldoff(
            self.clock_settings.reset_holdoff[neuron_id])
        anref.input_clock = hal.NeuronBackendConfig.InputClock(
            self.clock_settings.input_clock[neuron_id])

        atomic_neuron.excitatory_input.i_bias_coba = hal.CapMemCell.Value(
            self.i_syn_exc_coba[neuron_id])
        atomic_neuron.excitatory_input.v_rev_coba = hal.CapMemCell.Value(
            self.e_syn_exc_rev[neuron_id])
        atomic_neuron.inhibitory_input.i_bias_coba = hal.CapMemCell.Value(
            self.i_syn_inh_coba[neuron_id])
        atomic_neuron.inhibitory_input.v_rev_coba = hal.CapMemCell.Value(
            self.e_syn_inh_rev[neuron_id])

        atomic_neuron.multicompartment.i_bias_nmda = hal.CapMemCell.Value(
            self.i_bias_nmda[neuron_id])

        return atomic_neuron

    def to_neuron_calib_result(
            self, target: NeuronCalibTarget, options: NeuronCalibOptions
    ) -> NeuronCalibResult:
        """
        Conversion to NeuronCalibResult.
        The numpy arrays get merged into lola AtomicNeurons.

        :param target: Target parameters for calibration.
        :param options: Further options for calibration.

        :return: Equivalent NeuronCalibResult.
        """

        result = super().to_neuron_calib_result(target, options)

        # set common correlation config
        # Restore default, which is configured differently in
        # the hagen-mode base class via
        # `neuron_helpers.configure_integration()`.
        dumper = sta.PlaybackProgramBuilderDumper()
        config = hal.CommonCorrelationConfig()
        for coord in halco.iter_all(halco.CommonCorrelationConfigOnDLS):
            dumper.write(coord, config)

        cocolist = dumper.done().tolist()
        for coord, config in cocolist:
            if coord == halco.TimerOnDLS():
                continue
            result.cocos[coord] = config

        # set common neuron backend config
        config = hal.CommonNeuronBackendConfig()
        config.clock_scale_slow = \
            config.ClockScale(self.clock_settings.slow_clock)
        config.clock_scale_fast = \
            config.ClockScale(self.clock_settings.fast_clock)

        for coord in halco.iter_all(halco.CommonNeuronBackendConfigOnDLS):
            result.cocos[coord] = config

        return result


def check_target(target: NeuronCalibTarget):
    if target.holdoff_time.size not in [1, halco.NeuronConfigOnDLS.size]:
        raise ValueError("Holdoff time needs to have size 1 or "
                         f"{halco.NeuronConfigOnDLS.size}.")

    base.check_values(
        "leak",
        target.leak.to_numpy(),
        base.ParameterRange(50, 160))
    base.check_values(
        "reset",
        target.reset.to_numpy(),
        base.ParameterRange(50, 160))
    base.check_values(
        "threshold",
        target.threshold.to_numpy(),
        base.ParameterRange(50, 220))
    base.check_values(
        "tau_mem",
        target.tau_mem.as_quantity(),
        base.ParameterRange(0.5 * pq.us, 60 * pq.us))
    base.check_values(
        "tau_syn",
        target.tau_syn.as_quantity(),
        base.ParameterRange(0.3 * pq.us, 30 * pq.us))
    base.check_values(
        "i_synin_gm",
        target.cuba_synin.i_synin_gm.value() if isinstance(
            target.cuba_synin,
            NeuronCalibTarget.CalibratedMatchingCubaSynapticInput)
        else target.cuba_synin.i_synin_gm.to_numpy(),
        base.ParameterRange(30, 800))
    base.check_values(
        "e_coba_reversal",
        np.array([list(e) for e in target.coba_synin.e_coba_reversal]),
        base.ParameterRange(-np.inf, np.inf))
    base.check_values(
        "e_coba_reference",
        np.array([list(e) for e in target.coba_synin.e_coba_reference]),
        base.ParameterRange(60, 160))
    base.check_values(
        "membrane_capacitance",
        target.membrane_capacitance.to_numpy(),
        base.ParameterRange(
            hal.NeuronConfig.MembraneCapacitorSize.min,
            hal.NeuronConfig.MembraneCapacitorSize.max))
    base.check_values(
        "refractory_time",
        target.refractory_time.as_quantity(),
        base.ParameterRange(40 * pq.ns, 32 * pq.us))
    base.check_values(
        "synapse_dac_bias",
        target.synapse_dac_bias.value(),
        base.ParameterRange(30, hal.CapMemCell.Value.max))
    base.check_values(
        "holdoff_time",
        target.holdoff_time.as_quantity(),
        base.ParameterRange(0 * pq.ns, 4 * pq.us))
    if target.tau_icc is not None:
        base.check_values(
            "tau_icc",
            target.tau_icc.as_quantity(),
            base.ParameterRange(0.1 * pq.us, 30 * pq.us))

    if np.any([target.tau_mem.as_quantity() < constants.tau_mem_range.lower,
               target.tau_mem.as_quantity() > constants.tau_mem_range.upper]):
        raise ValueError(
            "Target membrane time constant is out of allowed range "
            + "in the respective fit function.")
    if np.any([target.tau_syn.as_quantity() < constants.tau_syn_range.lower,
               target.tau_syn.as_quantity() > constants.tau_syn_range.upper]):
        raise ValueError(
            "Target synaptic time constant is out of allowed range "
            + "in the respective fit function.")


def calibrate(
        connection: hxcomm.ConnectionHandle,
        target: Optional[NeuronCalibTarget] = None,
        options: Optional[NeuronCalibOptions] = None
) -> NeuronCalibResult:
    """
    Calibrate neurons for spiking operation in the LIF model.

    Parts of the calibration use the CADC, thus it has to be
    calibrated beforehand. All parameters are given in a "technical"
    domain. Parameters can be single values or numpy arrays shaped (512,),
    matching the number of neurons.

    Requirements:
    - The CADCs are enabled and calibrated. You can achieve this using
      the function `calix.common.cadc.calibrate()`.

    :param connection: Connection to the chip to calibrate.
    :param target: Calib target, given as an instance of
        NeuronCalibTarget. Refer there for the individual parameters.
    :param options: Further options for neuron calibration.

    :raises TypeError: If time constants are not given with a unit
        from the `quantities` package.
    :raises ValueError: If parameters are badly shaped
        or out of feasible range.
    """

    if target is None:
        target = NeuronCalibTarget()
    if options is None:
        options = NeuronCalibOptions()

    # process target
    check_target(target)

    # create result object
    calib_result = _CalibResultInternal()
    calib_result.set_neuron_configs_default(
        target.membrane_capacitance, target.tau_syn, options.readout_neuron,
        target.coba_synin.e_coba_reversal)

    # Calculate refractory time
    calib_result.clock_settings = refractory_period.calculate_settings(
        tau_ref=target.refractory_time.as_quantity(),
        holdoff_time=target.holdoff_time.as_quantity())

    # Configure chip for calibration
    builder = base.WriteRecordingPlaybackProgramBuilder()
    builder, _ = neuron_helpers.configure_chip(
        builder, readout_neuron=options.readout_neuron)
    base.run(connection, builder)

    neuron_calib_parts.disable_synin_and_threshold(connection, calib_result)

    neuron_calib_parts.calibrate_tau_syn(
        connection, target.tau_syn.as_quantity().T, calib_result)

    neuron_calib_parts.calibrate_synapse_dac_bias(
        connection, target.synapse_dac_bias.value(), calib_result)

    # calibrate threshold
    calibration = neuron_threshold.ThresholdCalibCADC()
    calib_result.v_threshold = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch(),
        target=target.threshold.to_numpy()
    ).calibrated_parameters

    # bring chip into a state for synin calibration
    target_cadc_reads = neuron_calib_parts.prepare_for_synin_calib(
        connection, options, calib_result)

    # calibrate or configure CUBA synaptic input OTA biases
    neuron_calib_parts.calibrate_synaptic_input(
        connection, target, calib_result, target_cadc_reads)

    # calibrate syn. input references at final biases
    neuron_calib_parts.calibrate_synin_references(
        connection, target_cadc_reads, calib_result)

    neuron_calib_parts.calibrate_synin_coba(connection, target, calib_result)

    # leave state with synin-calib configuration
    neuron_calib_parts.finalize_synin_calib(connection, calib_result)

    # set desired neuron configs, disable syn. input and spikes again
    neuron_calib_parts.disable_synin_and_threshold(connection, calib_result)

    # calibrate leak
    calibration = neuron_potentials.LeakPotentialCalib(target.leak.to_numpy())
    calibration.run(connection, algorithm=algorithms.NoisyBinarySearch())

    neuron_calib_parts.calibrate_tau_mem(
        connection, target.tau_mem.as_quantity(), calib_result)

    # calibrate reset
    calibration = neuron_potentials.ResetPotentialCalib(
        target.reset.to_numpy())
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.v_reset = result.calibrated_parameters
    calib_result.success = np.all([
        calib_result.success, result.success], axis=0)

    # calibrate leak
    calibration = neuron_potentials.LeakPotentialCalib(target.leak.to_numpy())
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.v_leak = result.calibrated_parameters
    calib_result.success = np.all([
        calib_result.success, result.success], axis=0)

    # calibrate inter-compartment conductance
    if target.tau_icc is not None:
        neuron_calib_parts.calibrate_tau_icc(
            connection, target.tau_icc.as_quantity(), calib_result)

    # print warning in case of failed neurons
    n_neurons_failed = np.sum(np.invert(calib_result.success))
    if n_neurons_failed > 0:
        logger.get("calix.spiking.neuron.calibrate").WARN(
            f"Calib failed for {n_neurons_failed} neurons: ",
            np.arange(halco.NeuronConfigOnDLS.size)[
                np.invert(calib_result.success)])

    result = calib_result.to_neuron_calib_result(target, options)
    builder = base.WriteRecordingPlaybackProgramBuilder()
    result.apply(builder)
    base.run(connection, builder)

    return result


def refine_potentials(connection: hxcomm.ConnectionHandle,
                      result: NeuronCalibResult,
                      target: Optional[NeuronCalibTarget] = None
                      ) -> None:
    """
    Re-calibrate the leak, reset and threshold potentials.

    This can be useful to minimize the effects of CapMem crosstalk
    on the end calibration. For best results, first run/apply a neuron
    calibration and then re-run the CADC calibration, as it is also
    affected by the neuron calibration. Call this function last.

    :param connection: Connection to the chip to re-calibrate.
    :param result: Result of the previous neuron calibration.
        The potentials will be overwritten by the refinement.
    :param target: Calib target parameters. Only the potentials
        (leak, reset, threshold) will be used and re-calibrated.
    """

    if target is None:
        target = NeuronCalibTarget()

    check_target(target)

    # apply given calibration result
    builder = base.WriteRecordingPlaybackProgramBuilder()
    result.apply(builder)
    base.run(connection, builder)

    # calibrate threshold
    calibration = neuron_threshold.ThresholdCalibCADC()
    v_threshold = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch(),
        target=target.threshold.to_numpy()
    ).calibrated_parameters

    # disable threshold (necessary before calibrating leak and reset)
    builder = base.WriteRecordingPlaybackProgramBuilder()
    for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
        config = result.neurons[coord]
        config.threshold.enable = False
        builder.write(coord, config)
    base.run(connection, builder)

    # calibrate reset
    calibration = neuron_potentials.ResetPotentialCalib(
        target.reset.to_numpy())
    v_reset = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch()
    ).calibrated_parameters

    # calibrate leak
    calibration = neuron_potentials.LeakPotentialCalib(
        target.leak.to_numpy())
    v_leak = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch()
    ).calibrated_parameters

    # set new parameters in calib result
    result.target.leak = target.leak
    result.target.reset = target.reset
    result.target.threshold = target.threshold
    for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
        result.neurons[coord].leak.v_leak = \
            hal.CapMemCell.Value(v_leak[int(coord.toEnum())])
        result.neurons[coord].reset.v_reset = \
            hal.CapMemCell.Value(v_reset[int(coord.toEnum())])
        result.neurons[coord].threshold.v_threshold = \
            hal.CapMemCell.Value(v_threshold[int(coord.toEnum())])

    # re-enable threshold
    builder = base.WriteRecordingPlaybackProgramBuilder()
    for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
        config = result.neurons[coord]
        config.threshold.enable = True
        builder.write(coord, config)
    base.run(connection, builder)


def calibrate_leak_over_threshold(
        connection: hxcomm.ConnectionHandle,
        result: NeuronCalibResult,
        leak_over_threshold_rate: pq.Quantity = 100 * pq.kHz):
    """
    Calibrate neurons to a specific firing rate in leak over threshold
    setup, by adjusting the threshold.

    The previously chosen threshold is disregarded here, it is not necessary
    to choose a leak over threshold setup beforehand. There has to be
    sufficient range between the reset and leak potential, though, in order
    to calibrate the threshold in between the two.

    :param connection: Connection to the chip to calibrate.
    :param result: Result of the previous neuron calibration.
        The spike thresholds will be adjusted.
    :param leak_over_threshold_rate: Target spike rate, given with a
        unit in the hardware time domain (so usual values
        are 10 to 100 kHz). The threshold is calibrated such that
        the neuron spikes with the given rate without any synaptic input.
        The previously calibrated threshold is threfore overwritten.
        If configuring an array in order to set individual targets
        per neuron, use a rate of zero to leave the threshold untouched.
    """

    # apply calib result (in case chip is configured differently)
    builder = base.WriteRecordingPlaybackProgramBuilder()
    result.apply(builder)
    base.run(connection, builder)

    calibration = neuron_threshold.LeakOverThresholdCalib(
        target=leak_over_threshold_rate)
    calibration_result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())

    # set new parameters in calib result
    individual_targets = np.ndim(leak_over_threshold_rate) > 0
    for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
        if individual_targets and \
                leak_over_threshold_rate[int(coord.toEnum())] == 0:
            continue
        result.neurons[coord].threshold.v_threshold = \
            hal.CapMemCell.Value(
                calibration_result.calibrated_parameters[int(coord.toEnum())])

    # apply calib result (restores config of neurons that are
    # not to be calibrated)
    builder = base.WriteRecordingPlaybackProgramBuilder()
    result.apply(builder)
    base.run(connection, builder)
