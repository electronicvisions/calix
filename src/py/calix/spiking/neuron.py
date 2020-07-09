"""
Provides an interface for calibrating LIF neurons.
"""

import numbers
from typing import Dict, Optional, Union, List
from dataclasses import dataclass
import numpy as np
import quantities as pq
from dlens_vx_v2 import sta, halco, hal, hxcomm

from calix.common import algorithms, base, helpers
from calix.hagen import neuron_helpers, neuron_leak_bias, neuron_synin, \
    neuron_potentials
from calix.spiking import neuron_threshold
from calix import constants


@dataclass
class NeuronCalibResult:
    """
    Result object of a neuron calibration.
    Holds calibrated parameters for all neurons and their calibration success.
    """

    capmem_parameters: Dict[halco.AtomicNeuronOnDLS, Dict[
        halco.CapMemRowOnCapMemBlock, hal.CapMemCell.Value]]
    neuron_configs: Dict[halco.NeuronConfigOnDLS, hal.NeuronConfig]
    refractory_counters: Dict[halco.NeuronBackendConfigOnDLS,
                              hal.NeuronBackendConfig.RefractoryTime]
    refractory_clock: hal.CommonNeuronBackendConfig.ClockScale
    success: Dict[halco.AtomicNeuronOnDLS, bool]

    def apply(self, builder: sta.PlaybackProgramBuilder
              ) -> sta.PlaybackProgramBuilder:
        """
        Apply the neuron calibration.

        Configures neurons in a "default-working" state with calibration
        applied, just like after the calibration.

        :param builder: Builder to append configuration instructions to.

        :return: Builder with configuration instructions appended.
        """

        # write static CapMem config
        config = {
            halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_drop: 300,
            halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_drop: 300,
            halco.CapMemRowOnCapMemBlock.i_bias_reset: 1015}
        builder = helpers.capmem_set_neuron_cells(builder, config)

        # set per-quadrant parameters
        config = {
            halco.CapMemCellOnCapMemBlock.neuron_v_bias_casc_n: 250,
            halco.CapMemCellOnCapMemBlock.neuron_i_bias_readout_amp: 110,
            halco.CapMemCellOnCapMemBlock.neuron_i_bias_leak_source_follower:
            100,
            halco.CapMemCellOnCapMemBlock.syn_i_bias_dac: 1022,
            halco.CapMemCellOnCapMemBlock.neuron_i_bias_spike_comparator: 500}
        builder = helpers.capmem_set_quadrant_cells(builder, config)

        # enable readout buffers
        builder.write(halco.CapMemCellOnDLS.readout_out_amp_i_bias_0,
                      hal.CapMemCell(1022))
        builder.write(halco.CapMemCellOnDLS.readout_out_amp_i_bias_1,
                      hal.CapMemCell(1022))

        # write neuron CapMem cells
        for neuron_coord, params in \
                self.capmem_parameters.items():
            for capmem_row, value in params.items():
                cell_coord = halco.CapMemCellOnDLS(
                    cell=halco.CapMemCellOnCapMemBlock(
                        x=neuron_coord.toCapMemColumnOnCapMemBlock(),
                        y=capmem_row),
                    block=neuron_coord.toCapMemBlockOnDLS())
                builder.write(cell_coord, hal.CapMemCell(value))

        # write configs
        for coord, config in self.neuron_configs.items():
            builder.write(coord, config)

        for coord, counter in self.refractory_counters.items():
            config = hal.NeuronBackendConfig()
            config.refractory_time = counter
            config.select_input_clock = 1
            builder.write(coord, config)

        config = hal.CommonNeuronBackendConfig()
        config.enable_clocks = True
        config.enable_event_registers = True
        config.clock_scale_slow = 9
        config.clock_scale_fast = self.refractory_clock

        for coord in halco.iter_all(halco.CommonNeuronBackendConfigOnDLS):
            builder.write(coord, config)
        builder = helpers.wait(builder, constants.capmem_level_off_time)

        return builder


@dataclass
class _CalibrationResultInternal:
    """
    Class providing array-like access to calibrated parameters.
    Used internally during calibration.
    """

    v_leak: np.ndarray = np.empty(halco.NeuronConfigOnDLS.size, dtype=int)
    v_reset: np.ndarray = np.empty(halco.NeuronConfigOnDLS.size, dtype=int)
    v_threshold: np.ndarray = np.empty(
        halco.NeuronConfigOnDLS.size, dtype=int)
    i_syn_exc_shift: np.ndarray = np.empty(
        halco.NeuronConfigOnDLS.size, dtype=int)
    i_syn_inh_shift: np.ndarray = np.empty(
        halco.NeuronConfigOnDLS.size, dtype=int)
    i_bias_leak: np.ndarray = np.empty(
        halco.NeuronConfigOnDLS.size, dtype=int)
    i_syn_exc_gm: np.ndarray = np.empty(
        halco.NeuronConfigOnDLS.size, dtype=int)
    i_syn_inh_gm: np.ndarray = np.empty(
        halco.NeuronConfigOnDLS.size, dtype=int)
    i_syn_exc_tau: np.ndarray = np.empty(
        halco.NeuronConfigOnDLS.size, dtype=int)
    i_syn_inh_tau: np.ndarray = np.empty(
        halco.NeuronConfigOnDLS.size, dtype=int)
    refractory_counters: np.ndarray = np.empty(
        halco.NeuronConfigOnDLS.size, dtype=int)
    refractory_clock: int = 0
    neuron_configs: Optional[List[hal.NeuronConfig]] = None
    success: np.ndarray = np.ones(
        halco.NeuronConfigOnDLS.size, dtype=np.bool)

    def set_neuron_configs_default(
            self, membrane_capacitance: Union[int, np.ndarray],
            tau_mem: pq.quantity.Quantity,
            tau_syn: pq.quantity.Quantity,
            readout_neuron: Optional[halco.AtomicNeuronOnDLS] = None):
        """
        Fill neuron configs with the given membrane capacitances, but
        otherwise default values.
        Decide on leak multiplication/division based on target
        membrane time constant.

        :param membrane_capacitance: Desired membrane capacitance (in LSB).
        :param tau_mem: Desired membrane time constant.
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
            tau = find_neuron_value(tau_mem, neuron_id)
            c_mem = find_neuron_value(membrane_capacitance, neuron_id)
            config.membrane_capacitor_size = c_mem

            # min. tau with division at C = 63: some 20 us
            if float(tau.rescale(pq.us)) > 20 * c_mem / 63:
                config.enable_leak_division = True
            # min. tau in "normal mode" at C = 63: some 2 us
            if float(tau.rescale(pq.us)) < 2 * c_mem / 63:
                config.enable_leak_multiplication = True

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

    def to_neuron_calib_result(self) -> NeuronCalibResult:
        """
        Conversion to NeuronCalibResult.
        The numpy arrays get transformed to dicts.

        :return: Equivalent NeuronCalibResult.
        """

        result = NeuronCalibResult(
            dict(), dict(), dict(), hal.CommonNeuronBackendConfig.ClockScale(),
            dict())
        conversion = {
            halco.CapMemRowOnCapMemBlock.v_leak: self.v_leak,
            halco.CapMemRowOnCapMemBlock.v_reset: self.v_reset,
            halco.CapMemRowOnCapMemBlock.v_threshold: self.v_threshold,
            halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_shift:
            self.i_syn_exc_shift,
            halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_shift:
            self.i_syn_inh_shift,
            halco.CapMemRowOnCapMemBlock.i_bias_leak: self.i_bias_leak,
            halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_gm:
            self.i_syn_exc_gm,
            halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_gm:
            self.i_syn_inh_gm,
            halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_tau:
            self.i_syn_exc_tau,
            halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_tau:
            self.i_syn_inh_tau
        }

        # convert CapMem parameters
        for neuron_id, neuron_coord in enumerate(
                halco.iter_all(halco.AtomicNeuronOnDLS)):
            neuron_params = dict()
            for coord, param in conversion.items():
                neuron_params.update({
                    coord: hal.CapMemCell.Value(param[neuron_id])
                })
            result.capmem_parameters.update({neuron_coord: neuron_params})

        # convert configs
        for neuron_id, neuron_coord in enumerate(
                halco.iter_all(halco.NeuronConfigOnDLS)):
            result.neuron_configs.update(
                {neuron_coord: self.neuron_configs[neuron_id]})
            result.refractory_counters.update(
                {neuron_coord.toNeuronBackendConfigOnDLS():
                 hal.NeuronBackendConfig.RefractoryTime(
                     self.refractory_counters[neuron_id])})
        result.refractory_clock = hal.CommonNeuronBackendConfig.ClockScale(
            self.refractory_clock)

        # convert success
        if self.success is not None:
            result.success = dict()
            for neuron_id, neuron_coord in enumerate(
                    halco.iter_all(halco.AtomicNeuronOnDLS)):
                result.success.update({neuron_coord: self.success[neuron_id]})

        return result


# pylint: disable=too-many-locals, too-many-branches, too-many-statements
def calibrate(
        connection: hxcomm.ConnectionHandle, *,
        leak: Union[int, np.ndarray] = 80,
        reset: Union[int, np.ndarray] = 70,
        threshold: Union[int, np.ndarray] = 125,
        tau_mem: pq.quantity.Quantity = 6. * pq.us,
        tau_syn: pq.quantity.Quantity = 2. * pq.us,
        i_synin_gm: Union[int, np.ndarray] = 200,
        membrane_capacitance: Union[int, np.ndarray] = 63,
        refractory_time: pq.quantity.Quantity = 2. * pq.us,
        readout_neuron: Optional[halco.AtomicNeuronOnDLS] = None
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
    :param readout_neuron: Coordinate of the neuron to be connected to
        a readout pad, i.e. can be observed using an oscilloscope.
        The selected neuron is connected to the upper pad (channel 0),
        the lower pad (channel 1) always shows the CADC ramp of quadrant 0.
        When using the MADC, select
        halco.SourceMultiplexerOnReadoutSourceSelection(0) for the neuron
        and mux 1 for the CADC ramps.
        If None is given, the readout is not configured.

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
        membrane_capacitance, tau_mem, tau_syn, readout_neuron)

    # calculate refractory time
    # clock scaler 0 means 125 MHz refractory clock, i.e. 8 ns per cycle
    fastest_clock = 125 * pq.MHz
    if np.ndim(refractory_time) == 0:
        refractory_time = refractory_time.repeat(halco.NeuronConfigOnDLS.size)
    elif refractory_time.shape != (halco.NeuronConfigOnDLS.size,):
        raise ValueError("Refractory times need to match the neurons.")

    # calculate refractory clock scaler to use
    calib_result.refractory_clock = max(int(np.ceil(np.log2(
        ((np.max(refractory_time) * fastest_clock).simplified
         / hal.NeuronBackendConfig.RefractoryTime.max)))), 0)
    if calib_result.refractory_clock \
            > hal.CommonNeuronBackendConfig.ClockScale.max:
        raise ValueError("Refractory times are larger than feasible.")
    # Calculate the refractory counter settings per neuron.
    # The counter setting is rounded down to the next-lower one.
    calib_result.refractory_counters = (
        (refractory_time * fastest_clock).simplified.magnitude
        / (2 ** calib_result.refractory_clock)).astype(int)

    # Configure chip for calibration
    # We start using a hagen-mode-like setup until the synaptic input is
    # calibrated. Afterwards we calibrate parameters like the spike threshold.
    builder = sta.PlaybackProgramBuilder()
    builder = neuron_helpers.configure_chip(
        builder, readout_neuron=readout_neuron)
    sta.run(connection, builder.done())

    # calibrate synaptic input time constant to given target
    calibration = neuron_synin.ExcSynTimeConstantCalibration(
        neuron_configs=calib_result.neuron_configs)
    if np.ndim(tau_syn) > 0 \
            and tau_syn.shape[0] == halco.SynapticInputOnNeuron.size:
        result = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch(),
            target=tau_syn[0])
    else:
        result = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch(),
            target=tau_syn)
    calib_result.i_syn_exc_tau = result.calibrated_parameters
    calib_result.success = np.all([
        calib_result.success, result.success], axis=0)

    calibration = neuron_synin.InhSynTimeConstantCalibration(
        neuron_configs=calib_result.neuron_configs)
    if np.ndim(tau_syn) > 0 \
            and tau_syn.shape[0] == halco.SynapticInputOnNeuron.size:
        result = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch(),
            target=tau_syn[1])
    else:
        result = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch(),
            target=tau_syn)
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
    sta.run(connection, builder.done())

    neuron_helpers.reconfigure_synaptic_input(
        connection, excitatory_biases=0, inhibitory_biases=0)

    # calibrate leak at threshold (in order to calibrate threshold at
    # leak afterwards):
    calibration = neuron_potentials.LeakPotentialCalibration(threshold)
    calibration.run(connection, algorithm=algorithms.NoisyBinarySearch())

    # calibrate threshold
    calibration = neuron_threshold.NeuronThresholdCalibration(safe_margin=0)
    calib_result.v_threshold = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch()
    ).calibrated_parameters

    # Configure chip for synin calibration:
    # The synaptic input calibration needs to run in a hagen-mode-like setup,
    # therefore we overwrite all previous calibrations (like spike threshold)
    # and cherry-pick the desired ones (like synaptic time constant).
    builder = sta.PlaybackProgramBuilder()
    builder = neuron_helpers.configure_chip(
        builder, readout_neuron=readout_neuron)

    # re-apply syn. input time constant calib which we need,
    # and re-apply spike threshold which may affect CapMem crosstalk
    builder = helpers.capmem_set_neuron_cells(
        builder, {halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_tau:
                  calib_result.i_syn_exc_tau,
                  halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_tau:
                  calib_result.i_syn_inh_tau,
                  halco.CapMemRowOnCapMemBlock.v_threshold:
                  calib_result.v_threshold})
    builder = helpers.wait(builder, constants.capmem_level_off_time)
    sta.run(connection, builder.done())

    # disable synaptic inputs initially
    neuron_helpers.reconfigure_synaptic_input(
        connection, excitatory_biases=0, inhibitory_biases=0)

    # calibrate leak near middle of CADC range
    # choose 100 since some neurons can not reach higher leak potentials
    calibration = neuron_potentials.LeakPotentialCalibration(100)
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
                i_synin_gm + 250, hal.CapMemCell.Value.max)))
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
            parameter_range=base.ParameterRange(hal.CapMemCell.Value.min, min(
                # the upper boundary is restricted to avoid starting in a
                # very noisy environment, which may not work for low targets.
                i_synin_gm + 250, hal.CapMemCell.Value.max)),
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
    else:
        calib_result.i_syn_exc_gm = i_synin_gm if equalize_synin \
            else i_synin_gm[0]
        calib_result.i_syn_inh_gm = i_synin_gm if equalize_synin \
            else i_synin_gm[1]

    # set desired neuron configs, disable syn. input and spikes again
    builder = sta.PlaybackProgramBuilder()
    for neuron_coord, neuron_config in zip(
            halco.iter_all(halco.NeuronConfigOnDLS),
            calib_result.neuron_configs):
        neuron_config = hal.NeuronConfig(neuron_config)  # copy
        neuron_config.enable_threshold_comparator = False
        builder.write(neuron_coord, neuron_config)
    sta.run(connection, builder.done())

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
    calibration = neuron_leak_bias.MembraneTimeConstCalibMADC(
        neuron_configs=calib_result.neuron_configs, target=tau_mem)
    calib_result.i_bias_leak = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch()
    ).calibrated_parameters

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

    result = calib_result.to_neuron_calib_result()
    builder = sta.PlaybackProgramBuilder()
    result.apply(builder)
    sta.run(connection, builder.done())

    return result
