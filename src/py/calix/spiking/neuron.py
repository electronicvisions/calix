"""
Provides an interface for calibrating LIF neurons.
"""

import numbers
from typing import Optional, Union, List
from dataclasses import dataclass
from warnings import warn
from copy import deepcopy

import numpy as np
import quantities as pq

from dlens_vx_v3 import sta, halco, hal, hxcomm, lola, logger

from calix.common import algorithms, base, helpers
from calix.hagen import neuron_helpers, neuron_potentials
import calix.hagen.neuron as hagen_neuron
from calix.hagen.neuron import NeuronCalibResult
from calix.spiking import neuron_threshold, refractory_period, \
    neuron_calib_parts
from calix import constants


@dataclass
class NeuronCalibTarget(base.CalibrationTarget):
    """
    Target parameters for the neuron calibration.

    Calibration target parameters:
    :ivar leak: Target CADC read at leak (resting) potential.
    :ivar reset: Target CADC read at reset potential.
    :ivar threshold: Target CADC read near spike threshold.
    :ivar tau_mem: Membrane time constant.
    :ivar tau_syn: Synaptic input time constant.
        If a single float value is given, it is used for all synaptic
        inputs on all neurons, excitatory and inhibitory.
        If a numpy array of floats is given, it can be shaped
        (2, 512) for the excitatory and inhibitory synaptic input
        of each neuron. It can also be shaped (2,) for the excitatory
        and inhibitory synaptic input of all neurons, or shaped (512,)
        for both inputs per neuron.
    :ivar i_synin_gm: Synaptic input strength as CapMem bias current.
        Here, if a single value is given, both excitatory and inhibitory
        synaptic inputs are calibrated to a target measured as the median
        of all neurons at this setting.
        If an array shaped (2,) is given, the values are used as excitatory
        and inhibitory target values, respectively.
        If an array (512,) or (2, 512) is given, the synaptic input
        strength is NOT calibrated, as this array would already be
        the result of calibration. Instead, the values are set up
        per neuron and used during the later parts, i.e. synaptic
        input reference calibration.
    :ivar membrane_capacitance: Selected membrane capacitance.
        The available range is 0 to approximately 2.2 pF, represented
        as 0 to 63 LSB.
    :ivar refractory_time: Refractory time, given with a unit as
        `quantities.quantity.Quantity`.
    :ivar synapse_dac_bias: Synapse DAC bias current that is desired.
        Can be lowered in order to reduce the amplitude of a spike
        at the input of the synaptic input OTA. This can be useful
        to avoid saturation when using larger synaptic time constants.
    :ivar holdoff_time: Target length of the holdoff period. The holdoff
        period is the time at the end of the refractory period in which the
        clamping to the reset voltage is already released but new spikes can
        still not be generated.
    """

    leak: Union[int, np.ndarray] = 80
    reset: Union[int, np.ndarray] = 70
    threshold: Union[int, np.ndarray] = 125
    tau_mem: pq.quantity.Quantity = 10. * pq.us
    tau_syn: pq.quantity.Quantity = 10. * pq.us
    i_synin_gm: Union[int, np.ndarray] = 500
    membrane_capacitance: Union[int, np.ndarray] = 63
    refractory_time: pq.quantity.Quantity = 2. * pq.us
    synapse_dac_bias: int = 600
    holdoff_time: pq.Quantity = 0 * pq.us

    feasible_ranges = {
        "leak": base.ParameterRange(50, 160),
        "reset": base.ParameterRange(50, 160),
        "threshold": base.ParameterRange(50, 220),
        "tau_mem": base.ParameterRange(0.5 * pq.us, 60 * pq.us),
        "tau_syn": base.ParameterRange(0.3 * pq.us, 30 * pq.us),
        "i_synin_gm": base.ParameterRange(30, 800),
        "membrane_capacitance": base.ParameterRange(
            hal.NeuronConfig.MembraneCapacitorSize.min,
            hal.NeuronConfig.MembraneCapacitorSize.max),
        "refractory_time": base.ParameterRange(
            40 * pq.ns, 32 * pq.us),
        "synapse_dac_bias": base.ParameterRange(
            30, hal.CapMemCell.Value.max),
        "holdoff_time": base.ParameterRange(
            0 * pq.ns, 4 * pq.us)
    }

    @dataclass
    class SyninParameters:
        """
        Collection of parameters for synaptic input calibration.

        Contains boolean decisions that are set automatically based on
        the shape of given targets:

        :ivar i_synin_gm: Target bias currents for synaptic input OTAs,
            with shapes modified to match the needs of the calibration
            routines.
        :ivar calibrate_synin: Decide whether synaptic input OTA
            strengths are calibrated.
        :ivar equalize_synin: Decide whether excitatory and inhibitory
          synaptic input strengths are equalized.
        """

        i_synin_gm: np.ndarray
        calibrate_synin: bool
        equalize_synin: bool

    def check_types(self):
        """
        Check whether parameters have the right types and shapes.

        :raises TypeError: If time constants are not given with a unit
            from the `quantities` package.
        :raises ValueError: If shape of parameters is bad.
        """

        super().check_types()

        if not isinstance(self.tau_mem, pq.Quantity):
            raise TypeError(
                "Membrane time constant is not given as a "
                "`quantities.Quantity`.")
        if not isinstance(self.tau_syn, pq.Quantity):
            raise TypeError(
                "Synaptic time constant is not given as a "
                "`quantities.Quantity`.")
        if not isinstance(self.refractory_time, pq.Quantity):
            raise TypeError(
                "Refractory time is not given as a "
                "`quantities.Quantity`.")
        if not isinstance(self.holdoff_time, pq.Quantity):
            raise TypeError(
                "Holdoff time is not given as a "
                "`quantities.Quantity`.")
        if self.holdoff_time.size not in [1, halco.NeuronConfigOnDLS.size]:
            raise ValueError("Holdoff time needs to have size 1 or "
                             f"{halco.NeuronConfigOnDLS.size}.")
        if self.holdoff_time.size == 1:
            self.holdoff_time = \
                np.ones(halco.NeuronConfigOnDLS.size) * self.holdoff_time

    def check_values(self):
        """
        Check whether calibration targets are feasible.

        Log warnings if the parameters are out of the typical range
        which can be calibrated and raise an error if the time constants
        exceed the range which can be handled by the calibration routine.

        :raises ValueError: If target parameters are outside the allowed
            range for spiking neuron calibration.
        """

        super().check_values()

        if np.any([self.tau_mem < constants.tau_mem_range.lower,
                   self.tau_mem > constants.tau_mem_range.upper]):
            raise ValueError(
                "Target membrane time constant is out of allowed range "
                + "in the respective fit function.")
        if np.any([self.tau_syn < constants.tau_syn_range.lower,
                   self.tau_syn > constants.tau_syn_range.upper]):
            raise ValueError(
                "Target synaptic time constant is out of allowed range "
                + "in the respective fit function.")

    def prepare_synin(self) -> SyninParameters:
        """
        Does preparations for synaptic input calibration.

        :return: SyninParameters class, containing desicions and
            targets for synaptic input calibration.
        """

        i_synin_gm = deepcopy(self.i_synin_gm)

        if not isinstance(i_synin_gm, np.ndarray) \
                and np.ndim(i_synin_gm) > 0:
            i_synin_gm = np.array(i_synin_gm)
        if np.ndim(i_synin_gm) > 0 \
                and i_synin_gm.shape[-1] == halco.NeuronConfigOnDLS.size:
            calibrate_synin = False
        else:
            calibrate_synin = True
        if np.ndim(i_synin_gm) > 0 \
                and i_synin_gm.shape[0] \
                == halco.SynapticInputOnNeuron.size:
            equalize_synin = False
        else:
            equalize_synin = True
            i_synin_gm = np.array(
                [i_synin_gm] * halco.SynapticInputOnNeuron.size)

        return self.SyninParameters(
            i_synin_gm, calibrate_synin, equalize_synin)


NeuronCalibTarget.DenseDefault = NeuronCalibTarget(
    leak=np.ones(
        halco.AtomicNeuronOnDLS.size, dtype=int) * NeuronCalibTarget.leak,
    reset=np.ones(
        halco.AtomicNeuronOnDLS.size, dtype=int) * NeuronCalibTarget.reset,
    threshold=np.ones(
        halco.AtomicNeuronOnDLS.size, dtype=int) * NeuronCalibTarget.threshold,
    tau_mem=np.ones(halco.AtomicNeuronOnDLS.size) * NeuronCalibTarget.tau_mem,
    tau_syn=np.ones((
        halco.SynapticInputOnNeuron.size,
        halco.AtomicNeuronOnDLS.size)) * NeuronCalibTarget.tau_syn,
    i_synin_gm=np.ones(
        halco.SynapticInputOnNeuron.size,
        dtype=int) * NeuronCalibTarget.i_synin_gm,
    membrane_capacitance=np.ones(
        halco.AtomicNeuronOnDLS.size,
        dtype=int) * NeuronCalibTarget.membrane_capacitance,
    refractory_time=np.ones(
        halco.AtomicNeuronOnDLS.size) * NeuronCalibTarget.refractory_time,
    synapse_dac_bias=NeuronCalibTarget.synapse_dac_bias,
    holdoff_time=np.ones(
        halco.AtomicNeuronOnDLS.size) * NeuronCalibTarget.holdoff_time
)


@dataclass
class NeuronCalibOptions(base.CalibrationOptions):
    """
    Further configuration parameters for neuron calibration.

    :ivar readout_neuron: Coordinate of the neuron to be connected to
        a readout pad, i.e. can be observed using an oscilloscope.
        The selected neuron is connected to the upper pad (channel 0),
        the lower pad (channel 1) always shows the CADC ramp of quadrant 0.
        When using the MADC, select
        halco.SourceMultiplexerOnReadoutSourceSelection(0) for the neuron
        and mux 1 for the CADC ramps.
        If None is given, the readout is not configured.
    """

    readout_neuron: Optional[halco.AtomicNeuronOnDLS] = None


@dataclass
class _CalibrationResultInternal(hagen_neuron.CalibrationResultInternal):
    """
    Class providing array-like access to calibrated parameters.
    Used internally during calibration.
    """

    v_threshold: np.ndarray = np.empty(
        halco.NeuronConfigOnDLS.size, dtype=int)
    i_syn_exc_coba: np.ndarray = np.zeros(
        halco.NeuronConfigOnDLS.size, dtype=int)
    i_syn_inh_coba: np.ndarray = np.zeros(
        halco.NeuronConfigOnDLS.size, dtype=int)
    syn_bias_dac: np.ndarray = np.empty(
        halco.NeuronConfigBlockOnDLS.size, dtype=int)
    clock_settings: Optional[refractory_period.Settings] = None
    neuron_configs: Optional[List[hal.NeuronConfig]] = None
    use_synin_small_capacitance: bool = False

    def set_neuron_configs_default(
            self, membrane_capacitance: Union[
                hal.NeuronConfig.MembraneCapacitorSize, np.ndarray],
            tau_syn: pq.Quantity,
            readout_neuron: Optional[halco.AtomicNeuronOnDLS] = None):
        """
        Fill neuron configs with the given membrane capacitances, but
        otherwise default values.

        :param membrane_capacitance: Desired membrane capacitance (in LSB).
        :param tau_syn: Synaptic input time constant. Used to
            decide whether the high resistance mode is enabled.
        :param readout_neuron: Neuron to enable readout for.
        """

        def find_neuron_value(array: Union[numbers.Integral, np.ndarray],
                              index: int) -> numbers.Integral:
            """
            Extract the value for a given (neuron) index from either
            an array of values or return the input number, if the
            input is not an array.

            :param array: Array or single number. We try to index the
                array, or directly return the number if it's not an array.

            :return: Array value at the given (neuron) index.
            """

            try:
                return array[index]
            except IndexError:
                return array[()]
            except TypeError:
                return array

        self.neuron_configs = []
        if np.ndim(tau_syn) > 0 \
                and tau_syn.shape[0] == halco.SynapticInputOnNeuron.size:
            tau_syn_exc = tau_syn[0]
            tau_syn_inh = tau_syn[1]
        else:
            tau_syn_exc = tau_syn
            tau_syn_inh = tau_syn

        for neuron_id in range(halco.NeuronConfigOnDLS.size):
            config = neuron_helpers.neuron_config_default()
            config.enable_threshold_comparator = True

            if readout_neuron is not None:
                if int(readout_neuron.toNeuronConfigOnDLS().toEnum()
                       ) == neuron_id:
                    config.enable_readout = True

            tau_exc = find_neuron_value(tau_syn_exc, neuron_id)
            tau_inh = find_neuron_value(tau_syn_inh, neuron_id)
            c_mem = find_neuron_value(membrane_capacitance, neuron_id)
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
        atomic_neuron.inhibitory_input.i_bias_coba = hal.CapMemCell.Value(
            self.i_syn_inh_coba[neuron_id])

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

        # set synapse DAC bias current
        dumper = helpers.capmem_set_quadrant_cells(
            dumper,
            {halco.CapMemCellOnCapMemBlock.syn_i_bias_dac: self.syn_bias_dac})

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


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def calibrate(
        connection: hxcomm.ConnectionHandle,
        target: Optional[NeuronCalibTarget] = None,
        options: Optional[NeuronCalibOptions] = None, *,
        leak: Optional[Union[int, np.ndarray]] = None,
        reset: Optional[Union[int, np.ndarray]] = None,
        threshold: Optional[Union[int, np.ndarray]] = None,
        tau_mem: Optional[pq.Quantity] = None,
        tau_syn: Optional[pq.Quantity] = None,
        i_synin_gm: Optional[Union[int, np.ndarray]] = None,
        membrane_capacitance: Optional[Union[int, np.ndarray]] = None,
        refractory_time: Optional[pq.Quantity] = None,
        synapse_dac_bias: Optional[int] = None,
        readout_neuron: Optional[halco.AtomicNeuronOnDLS] = None,
        holdoff_time: Optional[pq.Quantity] = None
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
    :param target: Calibration target, given as an instance of
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

    used_deprecated_parameters = False
    if leak is not None:
        target.leak = leak
        used_deprecated_parameters = True
    if reset is not None:
        target.reset = reset
        used_deprecated_parameters = True
    if threshold is not None:
        target.threshold = threshold
        used_deprecated_parameters = True
    if tau_mem is not None:
        target.tau_mem = tau_mem
        used_deprecated_parameters = True
    if tau_syn is not None:
        target.tau_syn = tau_syn
        used_deprecated_parameters = True
    if i_synin_gm is not None:
        target.i_synin_gm = i_synin_gm
        used_deprecated_parameters = True
    if membrane_capacitance is not None:
        target.membrane_capacitance = membrane_capacitance
        used_deprecated_parameters = True
    if refractory_time is not None:
        target.refractory_time = refractory_time
        used_deprecated_parameters = True
    if synapse_dac_bias is not None:
        target.synapse_dac_bias = synapse_dac_bias
        used_deprecated_parameters = True
    if readout_neuron is not None:
        options.readout_neuron = readout_neuron
        used_deprecated_parameters = True
    if holdoff_time is not None:
        target.holdoff_time = holdoff_time
        used_deprecated_parameters = True

    # delete deprecated arguments, to ensure the correct ones are used
    # in the following code
    del leak
    del reset
    del threshold
    del tau_mem
    del tau_syn
    del i_synin_gm
    del membrane_capacitance
    del refractory_time
    del synapse_dac_bias
    del readout_neuron
    del holdoff_time

    if used_deprecated_parameters:
        warn(
            "Passing arguments directly to calibrate() functions is "
            "deprecated. Please now use the target and options classes.",
            DeprecationWarning, stacklevel=2)

    # process target
    target.check()
    synin_parameters = target.prepare_synin()

    # create result object
    calib_result = _CalibrationResultInternal()
    calib_result.set_neuron_configs_default(
        target.membrane_capacitance, target.tau_syn, options.readout_neuron)

    # Calculate refractory time
    calib_result.clock_settings = refractory_period.calculate_settings(
        tau_ref=target.refractory_time, holdoff_time=target.holdoff_time)

    # Configure chip for calibration
    builder = sta.PlaybackProgramBuilder()
    builder, _ = neuron_helpers.configure_chip(
        builder, readout_neuron=options.readout_neuron)
    base.run(connection, builder)

    neuron_calib_parts.disable_synin_and_threshold(connection, calib_result)

    neuron_calib_parts.calibrate_tau_syn(
        connection, target.tau_syn, calib_result)

    neuron_calib_parts.calibrate_synapse_dac_bias(
        connection, target.synapse_dac_bias, calib_result)

    # calibrate threshold
    calibration = neuron_threshold.ThresholdCalibCADC()
    calib_result.v_threshold = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch(),
        target=target.threshold
    ).calibrated_parameters

    # bring chip into a state for synin calibration
    target_cadc_reads = neuron_calib_parts.prepare_for_synin_calib(
        connection, options, calib_result)

    # calibrate or configure CUBA synaptic input OTA biases
    if synin_parameters.calibrate_synin:
        neuron_calib_parts.calibrate_synaptic_input(
            connection, synin_parameters, calib_result, target_cadc_reads)
    else:
        calib_result.i_syn_exc_gm = synin_parameters.i_synin_gm[0]
        calib_result.i_syn_inh_gm = synin_parameters.i_synin_gm[1]

    # calibrate syn. input references at final biases
    neuron_calib_parts.calibrate_synin_references(
        connection, target_cadc_reads, calib_result)

    # leave state with synin-calib configuration
    neuron_calib_parts.finalize_synin_calib(connection, calib_result)

    # set desired neuron configs, disable syn. input and spikes again
    neuron_calib_parts.disable_synin_and_threshold(connection, calib_result)

    # calibrate leak
    calibration = neuron_potentials.LeakPotentialCalibration(target.leak)
    calibration.run(connection, algorithm=algorithms.NoisyBinarySearch())

    neuron_calib_parts.calibrate_tau_mem(
        connection, target.tau_mem, calib_result)

    # calibrate reset
    calibration = neuron_potentials.ResetPotentialCalibration(target.reset)
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.v_reset = result.calibrated_parameters
    calib_result.success = np.all([
        calib_result.success, result.success], axis=0)

    # calibrate leak
    calibration = neuron_potentials.LeakPotentialCalibration(target.leak)
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.v_leak = result.calibrated_parameters
    calib_result.success = np.all([
        calib_result.success, result.success], axis=0)

    # print warning in case of failed neurons
    n_neurons_failed = np.sum(~calib_result.success)
    if n_neurons_failed > 0:
        logger.get("calix.spiking.neuron.calibrate").WARN(
            f"Calibration failed for {n_neurons_failed} neurons: ",
            np.arange(halco.NeuronConfigOnDLS.size)[~calib_result.success])

    result = calib_result.to_neuron_calib_result(target, options)
    builder = sta.PlaybackProgramBuilder()
    result.apply(builder)
    base.run(connection, builder)

    return result


def refine_potentials(connection: hxcomm.ConnectionHandle,
                      result: NeuronCalibResult,
                      target: Optional[NeuronCalibTarget] = None, *,
                      leak: Optional[Union[int, np.ndarray]] = None,
                      reset: Optional[Union[int, np.ndarray]] = None,
                      threshold: Optional[Union[int, np.ndarray]] = None
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
    :param target: Calibration target parameters. Only the potentials
        (leak, reset, threshold) will be used and re-calibrated.
    """

    if target is None:
        target = NeuronCalibTarget()

    used_deprecated_parameters = False
    if leak is not None:
        target.leak = leak
        used_deprecated_parameters = True
    if reset is not None:
        target.reset = reset
        used_deprecated_parameters = True
    if threshold is not None:
        target.threshold = threshold
        used_deprecated_parameters = True

    del leak
    del reset
    del threshold

    if used_deprecated_parameters:
        warn(
            "Passing arguments directly to calibrate() functions is "
            "deprecated. Please now use the target parameter class.",
            DeprecationWarning, stacklevel=2)

    target.check()

    # apply given calibration result
    builder = sta.PlaybackProgramBuilder()
    result.apply(builder)
    base.run(connection, builder)

    # calibrate threshold
    calibration = neuron_threshold.ThresholdCalibCADC()
    v_threshold = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch(),
        target=target.threshold
    ).calibrated_parameters

    # disable threshold (necessary before calibrating leak and reset)
    builder = sta.PlaybackProgramBuilder()
    for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
        config = result.neurons[coord]
        config.threshold.enable = False
        builder.write(coord, config)
    base.run(connection, builder)

    # calibrate reset
    calibration = neuron_potentials.ResetPotentialCalibration(target.reset)
    v_reset = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch()
    ).calibrated_parameters

    # calibrate leak
    calibration = neuron_potentials.LeakPotentialCalibration(target.leak)
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
    builder = sta.PlaybackProgramBuilder()
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
    builder = sta.PlaybackProgramBuilder()
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
    builder = sta.PlaybackProgramBuilder()
    result.apply(builder)
    base.run(connection, builder)
