"""
Calibrate neurons for integrating synaptic inputs, as it is desired
using the hagen mode. Call the function calibrate(connection)
to run the calibration.
"""

from typing import Optional

import numpy as np
import quantities as pq

from dlens_vx_v3 import halco, hal, hxcomm

from calix.common import algorithms, base
from calix.hagen import neuron_helpers, neuron_evaluation, \
    neuron_leak_bias, neuron_synin, neuron_potentials
from calix.hagen.neuron_dataclasses import NeuronCalibTarget, \
    NeuronCalibOptions, NeuronCalibResult, CalibResultInternal
from calix.spiking import neuron_calib_parts
from calix import constants


# pylint: disable=too-many-statements,too-many-branches
def calibrate(
        connection: hxcomm.ConnectionHandle,
        target: Optional[NeuronCalibTarget] = None,
        options: Optional[NeuronCalibOptions] = None
) -> NeuronCalibResult:
    """
    Set up the neurons for integration.

    This calibration aims to equalize the neurons' leak and reset
    potentials as well as the synaptic input strength and time
    constants.

    Initially, the leak bias currents are calibrated such that the membrane
    time constants are equal for all neurons. This has to be done
    before matching the synaptic input strengths, as a neuron with
    higher leakage would show smaller integrated amplitudes even if
    the current inputs were matched. For this calibration, the neurons are
    reset to a potential below the leak, and the rise of the voltage is
    captured using the CADCs. This allows to roughly match the membrane
    time constant to the given value.

    The synaptic input OTA bias currents are calibrated such that they yield
    the same charge output to the membrane for each synaptic input.
    The synaptic input reference potentials are recalibrated in every run
    of the search for the bias currents, as the required voltage offset
    between the two inputs of the OTA changes with the bias current.

    Afterwards, the leak OTA bias current is reduced while keeping the
    noise in CADC readouts after the integration near the target.
    Reducing the target noise results in higher leak bias currents and thus
    a lower membrane time constant, which means fast decay of the synaptic
    inputs. Increasing the target noise means the neuron does a better job
    at integrating the inputs, but at the cost of reproducibility of
    the results, since noise of the synaptic input is more amplified.
    Apart from the pure statistical noise, systematic offsets are
    considered in this part of the calibration. The leak potential was
    calibrated before adjusting the leak biases. If a neuron drifts
    away from the expected leak potential, the leak bias current will
    be increased as well.

    The whole leak bias current readjustment can be disabled by setting
    the parameter target_noise to None.

    After setting the leak bias currents low, the resting potential of the
    neurons is prone to floating away from the leak potential.
    Thus, the synaptic input reference potentials are recalibrated, which
    can compensate systematic effects like a constant leakage current onto
    the membrane. Eventually, leak and reset potentials are calibrated.

    Requirements:
    - The CADCs are enabled and calibrated. You can achieve this using
      the function `calix.common.cadc.calibrate()`.

    :param connection: Connection to the chip to calibrate.
    :param target: Calib target, given as an instance of
        NeuronCalibTarget. Refer there for the individual parameters.
    :param options: Further calibration options, given as an instance of
        NeuronCalibOptions. Refer there for the individual parameters.

    :return: NeuronCalibResult, containing all calibrated parameters.

    :raises ValueError: If target parameters are not in a feasible range.
    :raises ValueError: If target parameters are not shaped as specified.
    :raises TypeError: If time constants are not given with a unit
        from the `quantities` package.
    """

    if target is None:
        target = NeuronCalibTarget()
    if options is None:
        options = NeuronCalibOptions()

    if not isinstance(target.tau_mem, pq.Quantity):
        raise TypeError(
            "Membrane time constant is not given as a "
            "`quantities.quantity.Quantity`.")
    if not isinstance(target.tau_syn, pq.Quantity):
        raise TypeError(
            "Synaptic time constant is not given as a "
            "`quantities.quantity.Quantity`.")

    if np.any([target.tau_mem < constants.tau_mem_range.lower,
               target.tau_mem > constants.tau_mem_range.upper]):
        raise ValueError(
            "Target membrane time constant is out of allowed range "
            + "in the respective fit function.")
    if np.any([target.tau_syn < constants.tau_syn_range.lower,
               target.tau_syn > constants.tau_syn_range.upper]):
        raise ValueError(
            "Target synaptic time constant is out of allowed range "
            + "in the respective fit function.")

    base.check_values(
        "target_leak_read",
        target.target_leak_read,
        base.ParameterRange(100, 140))
    base.check_values(
        "tau_syn",
        target.tau_syn,
        base.ParameterRange(0.3 * pq.us, 20 * pq.us))
    base.check_values(
        "tau_mem",
        target.tau_mem,
        base.ParameterRange(20 * pq.us, 100 * pq.us))
    base.check_values(
        "i_synin_gm",
        target.i_synin_gm,
        base.ParameterRange(30, 600))
    base.check_values(
        "target_noise",
        target.target_noise,
        base.ParameterRange(1.0, 2.5))
    base.check_values(
        "synapse_dac_bias",
        target.synapse_dac_bias,
        base.ParameterRange(30, hal.CapMemCell.Value.max))

    # Configure chip for calibration
    builder = base.WriteRecordingPlaybackProgramBuilder()
    builder, initial_config = neuron_helpers.configure_chip(
        builder, readout_neuron=options.readout_neuron)
    base.run(connection, builder)

    # call optional program for further initial configuration
    if options.initial_configuration is not None:
        options.initial_configuration(connection)

    # Initialize return object
    calib_result = CalibResultInternal()
    calib_result.i_bias_reset = initial_config[
        halco.CapMemRowOnCapMemBlock.i_bias_reset]
    calib_result.i_syn_exc_tau = initial_config[
        halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_tau]
    calib_result.i_syn_inh_tau = initial_config[
        halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_tau]

    # disable synaptic inputs initially
    neuron_helpers.reconfigure_synaptic_input(
        connection, excitatory_biases=0, inhibitory_biases=0)

    neuron_calib_parts.calibrate_synapse_dac_bias(
        connection, target.synapse_dac_bias, calib_result)

    # select small capacitance mode for syn. input lines
    tickets = []
    builder = base.WriteRecordingPlaybackProgramBuilder()
    for coord in halco.iter_all(halco.NeuronConfigOnDLS):
        tickets.append(builder.read(coord))
    base.run(connection, builder)

    builder = base.WriteRecordingPlaybackProgramBuilder()
    for coord, ticket in zip(halco.iter_all(halco.NeuronConfigOnDLS), tickets):
        config = ticket.get()
        config.enable_synaptic_input_excitatory_small_capacitance = True
        config.enable_synaptic_input_inhibitory_small_capacitance = True
        builder.write(coord, config)
    base.run(connection, builder)

    # Calibrate synaptic input time constant using MADC
    if np.all(target.tau_syn == 0 * pq.us):
        pass
    elif np.ndim(target.tau_syn) > 0 \
            and target.tau_syn.shape[0] == halco.SynapticInputOnNeuron.size:
        calibration = neuron_synin.ExcSynTimeConstantCalib(
            target=target.tau_syn[0])
        calib_result.i_syn_exc_tau = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch()
        ).calibrated_parameters
        calibration = neuron_synin.InhSynTimeConstantCalib(
            target=target.tau_syn[1])
        calib_result.i_syn_inh_tau = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch()
        ).calibrated_parameters
    else:
        calibration = neuron_synin.ExcSynTimeConstantCalib(
            target=target.tau_syn)
        calib_result.i_syn_exc_tau = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch()
        ).calibrated_parameters
        calibration = neuron_synin.InhSynTimeConstantCalib(
            target=target.tau_syn)
        calib_result.i_syn_inh_tau = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch(),
        ).calibrated_parameters

    # Calibrate leak potential at target leak read
    calibration = neuron_potentials.LeakPotentialCalib(
        target.target_leak_read)
    calibration.run(connection, algorithm=algorithms.NoisyBinarySearch())

    # Calibrate membrane time constant using reset
    calibration = neuron_leak_bias.MembraneTimeConstCalibCADC(
        target_time_const=target.tau_mem, target_amplitude=50)
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.i_bias_leak = result.calibrated_parameters
    calib_result.success = result.success

    # Read current resting potentials.
    # These will be used as target values when calibrating the synaptic
    # input reference potentials. We need to use these instead of the
    # target leak values in case the targets could not be reached
    # for some neurons.
    target_cadc_reads = neuron_helpers.cadc_read_neuron_potentials(
        connection)

    # Enable and calibrate excitatory synaptic input amplitudes to median
    neuron_helpers.reconfigure_synaptic_input(
        connection, excitatory_biases=target.i_synin_gm)

    exc_synin_calibration = neuron_synin.ExcSynBiasCalib(
        target_leak_read=target_cadc_reads,
        parameter_range=base.ParameterRange(0, min(
            target.i_synin_gm + 250, hal.CapMemCell.Value.max)))
    result = exc_synin_calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.i_syn_exc_gm = result.calibrated_parameters
    calib_result.success = np.all(
        [calib_result.success, result.success], axis=0)

    # Disable exc. synaptic input, enable and calibrate inhibitory
    neuron_helpers.reconfigure_synaptic_input(
        connection, excitatory_biases=0, inhibitory_biases=target.i_synin_gm)

    calibration = neuron_synin.InhSynBiasCalib(
        target_leak_read=target_cadc_reads,
        parameter_range=base.ParameterRange(0, min(
            target.i_synin_gm + 250, hal.CapMemCell.Value.max)),
        target=exc_synin_calibration.target)
    calibration.n_events = exc_synin_calibration.n_events
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.i_syn_inh_gm = result.calibrated_parameters
    calib_result.success = np.all(
        [calib_result.success, result.success], axis=0)

    calibration = neuron_synin.InhSynReferenceCalib(
        target=target_cadc_reads)
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.i_syn_inh_shift = result.calibrated_parameters
    calib_result.success = np.all([
        calib_result.success, result.success], axis=0)

    # Re-enable excitatory synaptic input after calibrating inhibitory
    # Recalibrating the excitatory synaptic input reference potentials
    # only here mitigates CapMem crosstalk.
    neuron_helpers.reconfigure_synaptic_input(
        connection, excitatory_biases=calib_result.i_syn_exc_gm)
    calibration = neuron_synin.ExcSynReferenceCalib(
        target=target_cadc_reads)
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.i_syn_exc_shift = result.calibrated_parameters
    calib_result.success = np.all([
        calib_result.success, result.success], axis=0)

    # Calibrate leak potential
    calibration = neuron_potentials.LeakPotentialCalib(
        target_cadc_reads)
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.v_leak = result.calibrated_parameters
    calib_result.success = np.all([
        calib_result.success, result.success], axis=0)

    # Calibrate reset potential to be equal to leak
    calibration = neuron_potentials.ResetPotentialCalib(
        highnoise=True)
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.v_reset = result.calibrated_parameters
    calib_result.success = np.all([
        calib_result.success, result.success], axis=0)

    if target.target_noise:
        # Set Leak bias as low as possible with target readout noise
        calibration = neuron_leak_bias.LeakBiasCalib(
            target=target.target_noise, target_leak_read=target_cadc_reads)
        result = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch())
        calib_result.i_bias_leak = result.calibrated_parameters
        calib_result.success = np.all([
            calib_result.success, result.success], axis=0)

        # Re-calibrate excitatory synaptic input
        # This mitigates CapMem crosstalk and allows compensating
        # systematic drift currents (similarly below).
        calibration = neuron_synin.ExcSynReferenceCalib(
            target=target_cadc_reads)
        result = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch())
        calib_result.i_syn_exc_shift = result.calibrated_parameters
        calib_result.success = np.all([
            calib_result.success, result.success], axis=0)

        # Re-calibrate inhibitory synaptic input
        calibration = neuron_synin.InhSynReferenceCalib(
            target=target_cadc_reads)
        result = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch())
        calib_result.i_syn_inh_shift = result.calibrated_parameters
        calib_result.success = np.all([
            calib_result.success, result.success], axis=0)

        # Calibrate leak potential.
        # This re-calibration is necessary since the leak bias
        # currents were optimized and the synaptic inputs are now
        # enabled, resulting in small offset currents that can move
        # the leak potential.
        calibration = neuron_potentials.LeakPotentialCalib(
            target_cadc_reads)
        result = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch())
        calib_result.v_leak = result.calibrated_parameters
        calib_result.success = np.all([
            calib_result.success, result.success], axis=0)

        # Calibrate reset potential to be equal to resting potential
        calibration = neuron_potentials.ResetPotentialCalib(
            highnoise=True)
        result = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch())
        calib_result.v_reset = result.calibrated_parameters
        calib_result.success = np.all([
            calib_result.success, result.success], axis=0)

    # Log statistics about membrane potentials after claibration
    neuron_evaluation.measure_quadrant_results(
        connection, calib_result.success)

    return calib_result.to_neuron_calib_result(target, options)


def calibrate_baseline(connection: hxcomm.ConnectionHandle,
                       target_read: int = 128
                       ) -> base.ParameterCalibResult:
    """
    Calibrate the CADC channel offsets such that the neuron baseline,
    that is reading shortly after a reset without getting synaptic input,
    reads the given target value.

    This calibration can only be called after the neuron calibration has
    finished, as it permanently overwrites the CADC offsets. Thus, the
    CADCs can no longer read constant voltages precisely.

    This eases the calculation of amplitudes of inputs, as no extra
    baseline read is required.

    :param connection: Connection to the chip to calibrate.
    :param target_read: Target CADC read of all neurons at baseline.

    :return: CalibResult, containing:
        * Calibrated CADC channel offsets.
        * Success mask of calibration - False for CADC channels that could
          not be matched to the target read.
    """

    calibration = neuron_potentials.BaselineCalib()
    return calibration.run(
        connection, algorithm=algorithms.LinearPrediction(probe_parameters=0),
        target=target_read)
