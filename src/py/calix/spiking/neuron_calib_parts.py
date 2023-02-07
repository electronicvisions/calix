"""
Helper functions representing parts of the spiking neuron calibration.

These functions are intended to be called in a specific order, as it is
done in the spiking neuron calibration - please use caution in case
you call them directly.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np
import quantities as pq

from dlens_vx_v3 import sta, halco, hal, hxcomm

from calix.common import algorithms, base, synapse, helpers
from calix.hagen import neuron_helpers, neuron_leak_bias, neuron_synin, \
    neuron_potentials
from calix import constants

if TYPE_CHECKING:
    from calix.spiking import neuron


class SyninParameters:
    """
    Collection of parameters for synaptic input calibration.

    Contains decisions that are set automatically based on
    the shape of targets for the calibration:

    :ivar i_synin_gm: Target bias currents for synaptic input OTAs,
        with shapes modified to match the needs of the calibration
        routines.
    :ivar calibrate_synin: Decide whether synaptic input OTA
        strengths are calibrated.
    :ivar equalize_synin: Decide whether excitatory and inhibitory
      synaptic input strengths are equalized.
    """

    def __init__(self, target: neuron.NeuronCalibTarget):
        """
        :param target: Target parameters for neuron calib.
        """

        self.i_synin_gm = deepcopy(target.i_synin_gm)

        if not isinstance(self.i_synin_gm, np.ndarray) \
                and np.ndim(self.i_synin_gm) > 0:
            self.i_synin_gm = np.array(self.i_synin_gm)
        if np.ndim(self.i_synin_gm) > 0 \
                and self.i_synin_gm.shape[-1] == halco.NeuronConfigOnDLS.size:
            self.calibrate_synin = False
        else:
            self.calibrate_synin = True
        if np.ndim(self.i_synin_gm) > 0 \
                and self.i_synin_gm.shape[0] \
                == halco.SynapticInputOnNeuron.size:
            self.equalize_synin = False
        else:
            self.equalize_synin = True
            self.i_synin_gm = np.array(
                [self.i_synin_gm] * halco.SynapticInputOnNeuron.size)


def calibrate_tau_syn(
        connection: hxcomm.ConnectionHandle,
        tau_syn: np.ndarray,
        calib_result: neuron._CalibResultInternal):
    """
    Calibrate synaptic input time constant to given target.

    :param connection: Connection to chip to run on.
    :param tau_syn: Target synaptic input time constant.
    :param calib_result: Calib result to store parameters in.
    """

    if np.ndim(tau_syn) > 0 \
            and tau_syn.shape[0] == halco.SynapticInputOnNeuron.size:
        calibration = neuron_synin.ExcSynTimeConstantCalib(
            neuron_configs=calib_result.neuron_configs,
            target=tau_syn[0])
    else:
        calibration = neuron_synin.ExcSynTimeConstantCalib(
            neuron_configs=calib_result.neuron_configs,
            target=tau_syn)
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.i_syn_exc_tau = result.calibrated_parameters
    calib_result.success = np.all([
        calib_result.success, result.success], axis=0)

    if np.ndim(tau_syn) > 0 \
            and tau_syn.shape[0] == halco.SynapticInputOnNeuron.size:
        calibration = neuron_synin.InhSynTimeConstantCalib(
            neuron_configs=calib_result.neuron_configs,
            target=tau_syn[1])
    else:
        calibration = neuron_synin.InhSynTimeConstantCalib(
            neuron_configs=calib_result.neuron_configs,
            target=tau_syn)
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.i_syn_inh_tau = result.calibrated_parameters
    calib_result.success = np.all([
        calib_result.success, result.success], axis=0)


def calibrate_synapse_dac_bias(
        connection: hxcomm.ConnectionHandle,
        synapse_dac_bias: int,
        calib_result: neuron._CalibResultInternal):
    """
    Calibrate the synapse DAC bias.

    First, configure the target parameters on each quadrant.
    Then, calibrate each quadrant's parameters such that the minimum
    amplitude of those measured previously on the quadrants is reached.

    :param connection: Connection to chip to run on.
    :param synapse_dac_bias: Target CapMem value for calibration.
    :param calib_result: Calib result.
    """

    # set synapse DAC bias current
    builder = sta.PlaybackProgramBuilder()
    builder = helpers.capmem_set_quadrant_cells(
        builder,
        {halco.CapMemCellOnCapMemBlock.syn_i_bias_dac:
         synapse_dac_bias})
    builder = helpers.wait(builder, constants.capmem_level_off_time)
    base.run(connection, builder)

    # Calibrate synapse DAC bias current
    # The CADC-based calibration is faster than using the MADC
    # and provides sufficient accuracy.
    calibration = synapse.DACBiasCalibCADC()
    calib_result.syn_bias_dac = calibration.run(
        connection, algorithm=algorithms.BinarySearch()
    ).calibrated_parameters


def prepare_for_synin_calib(
        connection: hxcomm.ConnectionHandle,
        options: neuron.NeuronCalibOptions,
        calib_result: neuron._CalibResultInternal
) -> np.ndarray:
    """
    Preconfigure the chip for synaptic input calibration.

    Select a hagen-mode-like setup, i.e. calibrate membrane time
    constant to a high value and synaptic input time constant to
    a low value.

    Note that there is a corresponding function `finalize_synin_calib`
    that undoes some of these changes, i.e. re-applies the synaptic
    input time constant to the user-selected value.

    :param connection: Connection to chip to run on.
    :param options: Further options for calibration.
    :param calib_result: Calib result to store parameters in.

    :return: Array of target CADC reads at resting potential, to be
        used as reference point during synaptic input calib.
    """

    # Configure chip for synin calibration:
    builder = sta.PlaybackProgramBuilder()
    builder, initial_config = neuron_helpers.configure_chip(
        builder, readout_neuron=options.readout_neuron)
    calib_result.i_bias_reset = initial_config[
        halco.CapMemRowOnCapMemBlock.i_bias_reset]

    # re-apply spike threshold which may affect CapMem crosstalk,
    # re-apply synapse DAC bias calib
    builder = helpers.capmem_set_neuron_cells(
        builder, {halco.CapMemRowOnCapMemBlock.v_threshold:
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
    calibration = neuron_potentials.LeakPotentialCalib(120)
    calibration.run(connection, algorithm=algorithms.NoisyBinarySearch())

    # ensure syn. input high resistance mode is off
    neuron_configs_synin_calib = []
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
    calibration = neuron_synin.ExcSynTimeConstantCalib(
        neuron_configs=neuron_configs_synin_calib,
        target=small_tau_syn)
    calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())

    calibration = neuron_synin.InhSynTimeConstantCalib(
        neuron_configs=neuron_configs_synin_calib,
        target=small_tau_syn)
    calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())

    # Equalize membrane time constant for synin calib
    calibration = neuron_leak_bias.MembraneTimeConstCalibCADC(
        target_time_const=60 * pq.us)
    calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())

    # re-calibrate leak potential after touching leak bias
    calibration = neuron_potentials.LeakPotentialCalib(120)
    calibration.run(connection, algorithm=algorithms.NoisyBinarySearch())
    target_cadc_reads = neuron_helpers.cadc_read_neuron_potentials(
        connection)

    return target_cadc_reads


def finalize_synin_calib(
        connection: hxcomm.ConnectionHandle,
        calib_result: neuron._CalibResultInternal):
    """
    Un-set some of the hagen-mode specific parameters that were set
    in `prepare_for_synin_calib`.

    The synaptic input time constant is re-set to the user-selected
    value, that was calibrated previously. We don't change the membrane
    time constant here, as it will be calibrated only afterwards.

    :param connection: Connection to chip to run on.
    :param calib_result: Calib result to store parameters in.
    """

    # re-apply synaptic input time constant
    builder = sta.PlaybackProgramBuilder()
    builder = helpers.capmem_set_neuron_cells(
        builder, {halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_tau:
                  calib_result.i_syn_exc_tau,
                  halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_tau:
                  calib_result.i_syn_inh_tau})
    builder = helpers.wait(builder, constants.capmem_level_off_time)
    base.run(connection, builder)


def calibrate_synaptic_input(
        connection: hxcomm.ConnectionHandle,
        target: neuron.NeuronCalibTarget,
        calib_result: neuron._CalibResultInternal,
        target_cadc_reads: np.ndarray):
    """
    Run calibration of (current-based) synaptic inputs.

    :param connection: Connection to chip to run on.
    :param target: Target parameters for neuron calibration.
    :param calib_result: Calib result to store parameters in.
    :param target_cadc_reads: CADC samples at resting potential, with
        synaptic input disabled.
    """

    synin_parameters = SyninParameters(target)

    # return early if synaptic inputs are not to be calibrated
    if not synin_parameters.calibrate_synin:
        calib_result.i_syn_exc_gm = synin_parameters.i_synin_gm[0]
        calib_result.i_syn_inh_gm = synin_parameters.i_synin_gm[1]
        return

    # Enable and calibrate excitatory synaptic input amplitudes to median
    neuron_helpers.reconfigure_synaptic_input(
        connection, excitatory_biases=synin_parameters.i_synin_gm[0])

    exc_synin_calibration = neuron_synin.ExcSynBiasCalib(
        target_leak_read=target_cadc_reads,
        parameter_range=base.ParameterRange(hal.CapMemCell.Value.min, min(
            # the upper boundary is restricted to avoid starting in a
            # very noisy environment, which may not work for low targets.
            (synin_parameters.i_synin_gm[0] * 1.8) + 100,
            hal.CapMemCell.Value.max)))

    result = exc_synin_calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.i_syn_exc_gm = result.calibrated_parameters
    calib_result.success = np.all(
        [calib_result.success, result.success], axis=0)

    # Disable exc. synaptic input, enable and calibrate inhibitory
    neuron_helpers.reconfigure_synaptic_input(
        connection, excitatory_biases=0,
        inhibitory_biases=synin_parameters.i_synin_gm[1])

    calibration = neuron_synin.InhSynBiasCalib(
        target_leak_read=target_cadc_reads,
        parameter_range=base.ParameterRange(0, min(
            # the upper boundary is restricted to avoid starting in a
            # very noisy environment, which may not work for low targets.
            (synin_parameters.i_synin_gm[1] * 1.8) + 100,
            hal.CapMemCell.Value.max)),
        target=exc_synin_calibration.target
        if synin_parameters.equalize_synin else None)
    if synin_parameters.equalize_synin:
        # match number of input events to the one found during the
        # excitatory calibration, where it was dynamically adjusted.
        calibration.n_events = exc_synin_calibration.n_events

    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.i_syn_inh_gm = result.calibrated_parameters
    calib_result.success = np.all(
        [calib_result.success, result.success], axis=0)


def calibrate_synin_references(
        connection: hxcomm.ConnectionHandle,
        target_cadc_reads: np.ndarray,
        calib_result: neuron._CalibResultInternal):
    """
    Calibrate synaptic input OTA reference potentials such that
    the given target CADC reads are reached with synaptic inputs
    enabled, i.e., set to the bias currents in the calib result.

    :param connection: Connection to chip to run on.
    :param target_cadc_reads: Target CADC reads, obtained at leak
        potential with synaptic inputs disabled.
    :param calib_result: Result of neuron calibration.
    """

    neuron_helpers.reconfigure_synaptic_input(
        connection, inhibitory_biases=calib_result.i_syn_inh_gm)
    calibration = neuron_synin.InhSynReferenceCalib(
        target=target_cadc_reads)
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.i_syn_inh_shift = result.calibrated_parameters
    calib_result.success = np.all([
        calib_result.success, result.success], axis=0)

    neuron_helpers.reconfigure_synaptic_input(
        connection, excitatory_biases=calib_result.i_syn_exc_gm)
    calibration = neuron_synin.ExcSynReferenceCalib(
        target=target_cadc_reads)
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.i_syn_exc_shift = result.calibrated_parameters
    calib_result.success = np.all([
        calib_result.success, result.success], axis=0)


def calibrate_tau_mem(
        connection: hxcomm.ConnectionHandle,
        tau_mem: pq.Quantity,
        calib_result: neuron._CalibResultInternal):
    """
    Calibrate membrane time constant to given target.

    :param connection: Connection to chip to run on.
    :param tau_mem: Target membrane time constant.
    :param calib_result: Neuron calibration result.
    """

    neuron_configs = deepcopy(calib_result.neuron_configs)

    if np.min(tau_mem) >= 3 * pq.us:
        # if all target time constants are at least 3 us, use
        # offset currents
        calibration = neuron_leak_bias.MembraneTimeConstCalibOffset(
            neuron_configs=neuron_configs, target=tau_mem)
    else:
        # if at least one neuron targets a faster membrane time constant,
        # use resets in order to achieve enough amplitude for the fits.
        calibration = neuron_leak_bias.MembraneTimeConstCalibReset(
            neuron_configs=neuron_configs, target=tau_mem)

    calib_result.i_bias_leak = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch()
    ).calibrated_parameters

    # set possibly modified leak division/multiplication in calib result
    for neuron_id, neuron_config in enumerate(calibration.neuron_configs):
        calib_result.neuron_configs[neuron_id].enable_leak_division = \
            neuron_config.enable_leak_division
        calib_result.neuron_configs[neuron_id] \
            .enable_leak_multiplication = \
            neuron_config.enable_leak_multiplication


def disable_synin_and_threshold(
        connection: hxcomm.ConnectionHandle,
        calib_result: neuron._CalibResultInternal):
    """
    Configure neurons with synaptic input OTAs and spike threshold
    disabled.

    This is a benign state in which many other parameters can be
    calibrated more easily.

    :param connection: Connection to chip to run on.
    :param calib_result: Neuron calibration result.
    """

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
