"""
Provides an interface for calibrating LIF neurons.
"""

import numbers
from typing import Optional, Union, List
from dataclasses import dataclass

import numpy as np
import quantities as pq

from dlens_vx_v3 import sta, halco, hal, hxcomm, lola, logger

from calix.common import algorithms, base, synapse, helpers
from calix.hagen import neuron_helpers, neuron_leak_bias, neuron_synin, \
    neuron_potentials
import calix.hagen.neuron as hagen_neuron
from calix.hagen.neuron import NeuronCalibResult
from calix.spiking import neuron_threshold, refractory_period
from calix import constants


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
            self, membrane_capacitance: Union[int, np.ndarray],
            tau_syn: pq.quantity.Quantity,
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

        self.neuron_configs = list()
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
            config.membrane_capacitor_size = c_mem

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

    def to_neuron_calib_result(self) -> NeuronCalibResult:
        """
        Conversion to NeuronCalibResult.
        The numpy arrays get merged into lola AtomicNeurons.

        :return: Equivalent NeuronCalibResult.
        """

        result = super().to_neuron_calib_result()

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
        connection: hxcomm.ConnectionHandle, *,
        leak: Union[int, np.ndarray] = 80,
        reset: Union[int, np.ndarray] = 70,
        threshold: Union[int, np.ndarray] = 125,
        tau_mem: pq.quantity.Quantity = 10. * pq.us,
        tau_syn: pq.quantity.Quantity = 10. * pq.us,
        i_synin_gm: Union[int, np.ndarray] = 500,
        membrane_capacitance: Union[int, np.ndarray] = 63,
        refractory_time: pq.quantity.Quantity = 2. * pq.us,
        synapse_dac_bias: int = 600,
        readout_neuron: Optional[halco.AtomicNeuronOnDLS] = None,
        holdoff_time: pq.Quantity = 0 * pq.us
) -> NeuronCalibResult:
    """
    Calibrate neurons for spiking operation in the LIF model.

    Parts of the calibration use the CADC, thus it has to be
    calibrated beforehand. All parameters are given in a "technical"
    domain. Parameters can be single values or numpy arrays shaped (512,),
    matching the number of neurons.

    :param connection: Connection to the chip to calibrate.
    :param leak: Target CADC read at leak (resting) potential.
    :param reset: Target CADC read at reset potential.
    :param threshold: Target CADC read near spike threshold.
    :param tau_mem: Membrane time constant.
    :param tau_syn: Synaptic input time constant.
        If a single float value is given, it is used for all synaptic
        inputs on all neurons, excitatory and inhibitory.
        If a numpy array of floats is given, it can be shaped
        (2, 512) for the excitatory and inhibitory synaptic input
        of each neuron. It can also be shaped (2,) for the excitatory
        and inhibitory synaptic input of all neurons, or shaped (512,)
        for both inputs per neuron.
    :param i_synin_gm: Synaptic input strength as CapMem bias current.
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
    :param membrane_capacitance: Selected membrane capacitance.
        The available range is 0 to approximately 2.2 pF, represented
        as 0 to 63 LSB.
    :param refractory_time: Refractory time, given with a unit as
        `quantities.quantity.Quantity`.
    :param synapse_dac_bias: Synapse DAC bias current that is desired.
        Can be lowered in order to reduce the amplitude of a spike
        at the input of the synaptic input OTA. This can be useful
        to avoid saturation when using larger synaptic time constants.
    :param readout_neuron: Coordinate of the neuron to be connected to
        a readout pad, i.e. can be observed using an oscilloscope.
        The selected neuron is connected to the upper pad (channel 0),
        the lower pad (channel 1) always shows the CADC ramp of quadrant 0.
        When using the MADC, select
        halco.SourceMultiplexerOnReadoutSourceSelection(0) for the neuron
        and mux 1 for the CADC ramps.
        If None is given, the readout is not configured.
    :param holdoff_time: Target length of the holdoff period. The holdoff
        period is the time at the end of the refractory period in which the
        clamping to the reset voltage is already released but new spikes can
        still not be generated.

    :raises TypeError: If time constants are not given with a unit
        from the `quantities` package.
    :raises ValueError: If parameters are badly shaped
        or out of feasible range.
    """

    if not isinstance(tau_mem, pq.quantity.Quantity):
        raise TypeError(
            "Membrane time constant is not given as a "
            "`quantities.quantity.Quantity`.")
    if not isinstance(tau_syn, pq.quantity.Quantity):
        raise TypeError(
            "Synaptic time constant is not given as a "
            "`quantities.quantity.Quantity`.")
    if not isinstance(refractory_time, pq.quantity.Quantity):
        raise TypeError(
            "Refractory time is not given as a "
            "`quantities.quantity.Quantity`.")
    if not isinstance(holdoff_time, pq.quantity.Quantity):
        raise TypeError(
            "Holdoff time is not given as a `quantities.quantity.Quantity`.")

    if np.any([tau_mem < 0.1 * pq.us, tau_mem > 200. * pq.us]):
        raise ValueError(
            "Target membrane time constant is out of feasible range.")
    if np.any([tau_syn < 0.1 * pq.us, tau_syn > 50. * pq.us]):
        raise ValueError(
            "Target synaptic time constant is out of feasible range.")
    if np.any(refractory_time < 0.04 * pq.us):
        raise ValueError(
            "Target refractory time is out of feasible range.")

    calib_result = _CalibrationResultInternal()
    calib_result.set_neuron_configs_default(
        membrane_capacitance, tau_syn, readout_neuron)

    # calculate refractory time. Resize holdoff_time to number of neurons
    # to get separate results for each neuron circuit
    if holdoff_time.size == 1:
        holdoff_time = np.ones(halco.NeuronConfigOnDLS.size) * holdoff_time
    elif holdoff_time.size != halco.NeuronConfigOnDLS.size:
        raise ValueError("Holdoff time needs to have size 1 or "
                         f"{halco.NeuronConfigOnDLS.size}.")
    calib_result.clock_settings = refractory_period.calculate_settings(
        tau_ref=refractory_time, holdoff_time=holdoff_time)

    # Configure chip for calibration
    # We start using a hagen-mode-like setup until the synaptic input is
    # calibrated. Afterwards we calibrate parameters like the spike threshold.
    builder = sta.PlaybackProgramBuilder()
    builder, _ = neuron_helpers.configure_chip(
        builder, readout_neuron=readout_neuron)
    base.run(connection, builder)

    # calibrate synaptic input time constant to given target
    if np.ndim(tau_syn) > 0 \
            and tau_syn.shape[0] == halco.SynapticInputOnNeuron.size:
        calibration = neuron_synin.ExcSynTimeConstantCalibration(
            neuron_configs=calib_result.neuron_configs,
            target=tau_syn[0])
    else:
        calibration = neuron_synin.ExcSynTimeConstantCalibration(
            neuron_configs=calib_result.neuron_configs,
            target=tau_syn)
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.i_syn_exc_tau = result.calibrated_parameters
    calib_result.success = np.all([
        calib_result.success, result.success], axis=0)

    if np.ndim(tau_syn) > 0 \
            and tau_syn.shape[0] == halco.SynapticInputOnNeuron.size:
        calibration = neuron_synin.InhSynTimeConstantCalibration(
            neuron_configs=calib_result.neuron_configs,
            target=tau_syn[1])
    else:
        calibration = neuron_synin.InhSynTimeConstantCalibration(
            neuron_configs=calib_result.neuron_configs,
            target=tau_syn)
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.i_syn_inh_tau = result.calibrated_parameters
    calib_result.success = np.all([
        calib_result.success, result.success], axis=0)

    # Configure neurons without syn. input and threshold disabled
    builder = sta.PlaybackProgramBuilder()
    for neuron_coord, neuron_config in zip(
            halco.iter_all(halco.NeuronConfigOnDLS),
            calib_result.neuron_configs):
        neuron_config = hal.NeuronConfig(neuron_config)  # copy
        neuron_config.enable_threshold_comparator = False
        builder.write(neuron_coord, neuron_config)
    base.run(connection, builder)

    neuron_helpers.reconfigure_synaptic_input(
        connection, excitatory_biases=0, inhibitory_biases=0)

    # set synapse DAC bias current
    builder = sta.PlaybackProgramBuilder()
    builder = helpers.capmem_set_quadrant_cells(
        builder,
        {halco.CapMemCellOnCapMemBlock.syn_i_bias_dac: synapse_dac_bias})
    builder = helpers.wait(builder, constants.capmem_level_off_time)
    base.run(connection, builder)

    # Calibrate synapse DAC bias current
    # The CADC-based calibration is faster and provides sufficient accuracy.
    calibration = synapse.DACBiasCalibCADC()
    calib_result.syn_bias_dac = calibration.run(
        connection, algorithm=algorithms.BinarySearch()).calibrated_parameters

    # calibrate threshold
    calibration = neuron_threshold.ThresholdCalibCADC()
    calib_result.v_threshold = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch(), target=threshold
    ).calibrated_parameters

    # Configure chip for synin calibration:
    # The synaptic input calibration needs to run in a hagen-mode-like setup,
    # therefore we overwrite all previous calibrations (like spike threshold)
    # and cherry-pick the desired ones (like synaptic time constant).
    builder = sta.PlaybackProgramBuilder()
    builder, initial_config = neuron_helpers.configure_chip(
        builder, readout_neuron=readout_neuron)
    calib_result.i_bias_reset = initial_config[
        halco.CapMemRowOnCapMemBlock.i_bias_reset]

    # re-apply syn. input time constant calib which we need,
    # re-apply spike threshold which may affect CapMem crosstalk,
    # re-apply synapse DAC bias calib
    builder = helpers.capmem_set_neuron_cells(
        builder, {halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_tau:
                  calib_result.i_syn_exc_tau,
                  halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_tau:
                  calib_result.i_syn_inh_tau,
                  halco.CapMemRowOnCapMemBlock.v_threshold:
                  calib_result.v_threshold})
    builder = helpers.capmem_set_quadrant_cells(
        builder, {halco.CapMemCellOnCapMemBlock.syn_i_bias_dac:
                  calib_result.syn_bias_dac})
    builder = helpers.wait(builder, constants.capmem_level_off_time)
    base.run(connection, builder)

    # disable synaptic inputs initially
    neuron_helpers.reconfigure_synaptic_input(
        connection, excitatory_biases=0, inhibitory_biases=0)

    # calibrate leak near middle of CADC range
    calibration = neuron_potentials.LeakPotentialCalibration(120)
    calibration.run(connection, algorithm=algorithms.NoisyBinarySearch())

    # decide how to execute synin calib
    if np.ndim(i_synin_gm) > 0 \
            and i_synin_gm.shape[-1] == halco.NeuronConfigOnDLS.size:
        calibrate_synin = False
    else:
        calibrate_synin = True
    if np.ndim(i_synin_gm) > 0 \
            and i_synin_gm.shape[0] == halco.SynapticInputOnNeuron.size:
        equalize_synin = False
    else:
        equalize_synin = True

    if calibrate_synin:
        # ensure syn. input high resistance mode is off
        neuron_configs_synin_calib = list()
        for neuron_coord, neuron_config in zip(
                halco.iter_all(halco.NeuronConfigOnDLS),
                calib_result.neuron_configs):
            neuron_config = hal.NeuronConfig(neuron_config)  # copy
            neuron_config.enable_threshold_comparator = False
            neuron_config.enable_synaptic_input_excitatory_high_resistance = \
                False
            neuron_config.enable_synaptic_input_inhibitory_high_resistance = \
                False
            neuron_config.enable_synaptic_input_excitatory = False
            neuron_config.enable_synaptic_input_inhibitory = False
            builder.write(neuron_coord, neuron_config)
            neuron_configs_synin_calib.append(neuron_config)
        base.run(connection, builder)

        # calibrate synaptic input time constant to a low value as a
        # single event could charge the membrane too much at high synaptic time
        # constants. The real synaptic time constant is reapplied later.
        small_tau_syn = 1.5 * pq.us
        calibration = neuron_synin.ExcSynTimeConstantCalibration(
            neuron_configs=neuron_configs_synin_calib,
            target=small_tau_syn)
        result = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch())

        calibration = neuron_synin.InhSynTimeConstantCalibration(
            neuron_configs=neuron_configs_synin_calib,
            target=small_tau_syn)
        result = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch())

        # Equalize membrane time constant for synin calib
        calibration = neuron_leak_bias.MembraneTimeConstCalibCADC(
            target_time_const=60 * pq.us)
        calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch())

        # Enable and calibrate excitatory synaptic input amplitudes to median
        target_cadc_reads = neuron_helpers.cadc_read_neuron_potentials(
            connection)
        neuron_helpers.reconfigure_synaptic_input(
            connection, excitatory_biases=i_synin_gm
            if equalize_synin else i_synin_gm[0])

        exc_synin_calibration = neuron_synin.ExcSynBiasCalibration(
            target_leak_read=target_cadc_reads,
            parameter_range=base.ParameterRange(hal.CapMemCell.Value.min, min(
                # the upper boundary is restricted to avoid starting in a
                # very noisy environment, which may not work for low targets.
                (i_synin_gm * 1.8) + 100, hal.CapMemCell.Value.max)))

        result = exc_synin_calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch())
        calib_result.i_syn_exc_gm = result.calibrated_parameters
        calib_result.success = np.all(
            [calib_result.success, result.success], axis=0)

        # Disable exc. synaptic input, enable and calibrate inhibitory
        neuron_helpers.reconfigure_synaptic_input(
            connection, excitatory_biases=0,
            inhibitory_biases=i_synin_gm if equalize_synin else i_synin_gm[1])

        calibration = neuron_synin.InhSynBiasCalibration(
            target_leak_read=target_cadc_reads,
            parameter_range=base.ParameterRange(0, min(
                # the upper boundary is restricted to avoid starting in a
                # very noisy environment, which may not work for low targets.
                (i_synin_gm * 1.8) + 100, hal.CapMemCell.Value.max)),
            target=exc_synin_calibration.target if equalize_synin else None)
        if equalize_synin:
            # match number of input events to the one found during the
            # excitatory calibration, where it was dynamically adjusted.
            calibration.n_events = exc_synin_calibration.n_events

        result = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch())
        calib_result.i_syn_inh_gm = result.calibrated_parameters
        calib_result.success = np.all(
            [calib_result.success, result.success], axis=0)

        # re-apply synaptic input time constant
        builder = sta.PlaybackProgramBuilder()
        builder = helpers.capmem_set_neuron_cells(
            builder, {halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_tau:
                      calib_result.i_syn_exc_tau,
                      halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_tau:
                      calib_result.i_syn_inh_tau})
        builder = helpers.wait(builder, constants.capmem_level_off_time)
        base.run(connection, builder)
    else:
        calib_result.i_syn_exc_gm = i_synin_gm if equalize_synin \
            else i_synin_gm[0]
        calib_result.i_syn_inh_gm = i_synin_gm if equalize_synin \
            else i_synin_gm[1]

        # Set suitable membrane time constant for leak potential calib
        calibration = neuron_leak_bias.MembraneTimeConstCalibCADC(
            target_time_const=30 * pq.us)
        calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch())

    # set desired neuron configs, disable syn. input and spikes again
    builder = sta.PlaybackProgramBuilder()
    for neuron_coord, neuron_config in zip(
            halco.iter_all(halco.NeuronConfigOnDLS),
            calib_result.neuron_configs):
        neuron_config = hal.NeuronConfig(neuron_config)  # copy
        neuron_config.enable_threshold_comparator = False
        builder.write(neuron_coord, neuron_config)
    base.run(connection, builder)

    neuron_helpers.reconfigure_synaptic_input(
        connection, excitatory_biases=0, inhibitory_biases=0)

    # calibrate leak
    calibration = neuron_potentials.LeakPotentialCalibration(leak)
    calibration.run(connection, algorithm=algorithms.NoisyBinarySearch())

    # Re-enable synaptic inputs and calibrate reference
    target_cadc_reads = neuron_helpers.cadc_read_neuron_potentials(
        connection)
    neuron_helpers.reconfigure_synaptic_input(
        connection, excitatory_biases=calib_result.i_syn_exc_gm)
    calibration = neuron_synin.ExcSynReferenceCalibration(
        target=target_cadc_reads)
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.i_syn_exc_shift = result.calibrated_parameters
    calib_result.success = np.all([
        calib_result.success, result.success], axis=0)

    neuron_helpers.reconfigure_synaptic_input(
        connection, inhibitory_biases=calib_result.i_syn_inh_gm)
    calibration = neuron_synin.InhSynReferenceCalibration(
        target=target_cadc_reads)
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.i_syn_inh_shift = result.calibrated_parameters
    calib_result.success = np.all([
        calib_result.success, result.success], axis=0)

    # calibrate membrane time constant
    if np.min(tau_mem) >= 3 * pq.us:
        # if all target time constants are at least 3 us, use offset currents
        calibration = neuron_leak_bias.MembraneTimeConstCalibOffset(
            neuron_configs=calib_result.neuron_configs,
            target=tau_mem)
    else:
        # if at least one neuron targets a faster membrane time constant,
        # use resets in order to achieve enough amplitude for the fits.
        calibration = neuron_leak_bias.MembraneTimeConstCalibReset(
            neuron_configs=calib_result.neuron_configs,
            target=tau_mem)

    calib_result.i_bias_leak = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch()
    ).calibrated_parameters
    calib_result.neuron_configs = calibration.neuron_configs

    # calibrate reset
    calibration = neuron_potentials.ResetPotentialCalibration(reset)
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.v_reset = result.calibrated_parameters
    calib_result.success = np.all([
        calib_result.success, result.success], axis=0)

    # calibrate leak
    calibration = neuron_potentials.LeakPotentialCalibration(leak)
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.v_leak = result.calibrated_parameters
    calib_result.success = np.all([
        calib_result.success, result.success], axis=0)

    # re-enable spike threshold
    for neuron_coord, neuron_config in zip(
            halco.iter_all(halco.NeuronConfigOnDLS),
            calib_result.neuron_configs):
        neuron_config.enable_threshold_comparator = True

    # print warning in case of failed neurons
    n_neurons_failed = np.sum(~calib_result.success)
    if n_neurons_failed > 0:
        logger.get("calix.spiking.neuron.calibrate").WARN(
            f"Calibration failed for {n_neurons_failed} neurons: ",
            np.arange(halco.NeuronConfigOnDLS.size)[~calib_result.success])

    result = calib_result.to_neuron_calib_result()
    builder = sta.PlaybackProgramBuilder()
    result.apply(builder)
    base.run(connection, builder)

    return result


def refine_potentials(connection: hxcomm.ConnectionHandle,
                      result: NeuronCalibResult,
                      leak: Union[int, np.ndarray] = 80,
                      reset: Union[int, np.ndarray] = 70,
                      threshold: Union[int, np.ndarray] = 125
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
    :param leak: Target CADC read at leak (resting) potential.
    :param reset: Target CADC read at reset potential.
    :param threshold: Target CADC read near spike threshold.

    """

    # apply given calibration result
    builder = sta.PlaybackProgramBuilder()
    result.apply(builder)
    base.run(connection, builder)

    # calibrate threshold
    calibration = neuron_threshold.ThresholdCalibCADC()
    v_threshold = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch(), target=threshold
    ).calibrated_parameters

    # disable threshold (necessary before calibrating leak and reset)
    builder = sta.PlaybackProgramBuilder()
    for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
        config = result.neurons[coord]
        config.threshold.enable = False
        builder.write(coord, config)
    base.run(connection, builder)

    # calibrate reset
    calibration = neuron_potentials.ResetPotentialCalibration(reset)
    v_reset = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch()
    ).calibrated_parameters

    # calibrate leak
    calibration = neuron_potentials.LeakPotentialCalibration(leak)
    v_leak = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch()
    ).calibrated_parameters

    # set new parameters in calib result
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
