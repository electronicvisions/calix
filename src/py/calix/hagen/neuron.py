"""
Calibrate neurons for integrating synaptic inputs, as it is desired
using the hagen mode. Call the function calibrate(connection)
to run the calibration.
"""

from typing import Dict, Optional, Union, Callable
from dataclasses import dataclass
import numpy as np
import quantities as pq
from dlens_vx_v2 import sta, halco, hal, hxcomm, lola

from calix.common import algorithms, base, helpers
from calix.hagen import neuron_helpers, neuron_evaluation, \
    neuron_leak_bias, neuron_synin, neuron_potentials
from calix import constants


@dataclass
class NeuronCalibResult:
    """
    Result object of a neuron calibration.
    Holds calibrated parameters for all neurons and their calibration success.
    """

    neurons: Dict[halco.AtomicNeuronOnDLS, lola.AtomicNeuron]
    cocos: dict()  # some coordinate, some container
    success: Dict[halco.AtomicNeuronOnDLS, bool]

    def apply(self, builder: sta.PlaybackProgramBuilder) \
            -> sta.PlaybackProgramBuilder:
        """
        Apply the calibration in the given builder.

        Configures neurons in a "default-working" state with
        calibration applied, just like after the calibration.

        :param builder: Builder to append configuration instructions to.

        :return: Builder with configuration instructions appended.
        """

        for neuron_coord, neuron in self.neurons.items():
            builder.write(neuron_coord, neuron)

        for coord, container in self.cocos.items():
            builder.write(coord, container)

        builder = helpers.wait(builder, constants.capmem_level_off_time)

        return builder

    @property
    def success_mask(self) -> np.ndarray:
        """
        Convert the success dict to a boolean numpy mask.

        :return: Numpy array containing neuron calibration success,
            ordered matching the AtomicNeuronOnDLS enum.
        """

        success_mask = np.empty(halco.AtomicNeuronOnDLS.size, dtype=bool)
        for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
            success_mask[coord.toEnum()] = self.success[coord]

        return success_mask


@dataclass
class CalibrationResultInternal:
    """
    Class providing numpy-array access to calibrated parameters.
    Used internally during calibration.
    """

    v_leak: np.ndarray = np.empty(halco.NeuronConfigOnDLS.size, dtype=int)
    v_reset: np.ndarray = np.empty(halco.NeuronConfigOnDLS.size, dtype=int)
    i_syn_exc_shift: np.ndarray = np.empty(
        halco.NeuronConfigOnDLS.size, dtype=int)
    i_syn_inh_shift: np.ndarray = np.empty(
        halco.NeuronConfigOnDLS.size, dtype=int)
    i_syn_exc_drop: np.ndarray = np.empty(
        halco.NeuronConfigOnDLS.size, dtype=int)
    i_syn_inh_drop: np.ndarray = np.empty(
        halco.NeuronConfigOnDLS.size, dtype=int)
    i_bias_leak: np.ndarray = np.empty(
        halco.NeuronConfigOnDLS.size, dtype=int)
    i_bias_reset: np.ndarray = np.empty(
        halco.NeuronConfigOnDLS.size, dtype=int)
    i_syn_exc_gm: np.ndarray = np.empty(
        halco.NeuronConfigOnDLS.size, dtype=int)
    i_syn_inh_gm: np.ndarray = np.empty(
        halco.NeuronConfigOnDLS.size, dtype=int)
    i_syn_exc_tau: np.ndarray = np.empty(
        halco.NeuronConfigOnDLS.size, dtype=int)
    i_syn_inh_tau: np.ndarray = np.empty(
        halco.NeuronConfigOnDLS.size, dtype=int)
    success: np.ndarray = np.ones(halco.NeuronConfigOnDLS.size, dtype=bool)

    def to_atomic_neuron(self,
                         neuron_coord: halco.AtomicNeuronOnDLS
                         ) -> lola.AtomicNeuron:
        """
        Returns an AtomicNeuron with calibration applied.

        :param neuron_coord: Coordinate of requested neuron.

        :return: Complete AtomicNeuron configuration.
        """

        neuron_id = neuron_coord.toEnum().value()
        atomic_neuron = lola.AtomicNeuron()
        atomic_neuron.set_from(neuron_helpers.neuron_config_default())
        atomic_neuron.set_from(neuron_helpers.neuron_backend_config_default())

        anl = atomic_neuron.leak
        anl.v_leak = hal.CapMemCell.Value(self.v_leak[neuron_id])
        anl.i_bias = hal.CapMemCell.Value(self.i_bias_leak[neuron_id])

        anr = atomic_neuron.reset
        anr.v_reset = hal.CapMemCell.Value(self.v_reset[neuron_id])
        anr.i_bias = hal.CapMemCell.Value(self.i_bias_reset[neuron_id])

        anexc = atomic_neuron.excitatory_input
        anexc.i_shift_reference = hal.CapMemCell.Value(
            self.i_syn_exc_shift[neuron_id])
        anexc.i_bias_gm = hal.CapMemCell.Value(
            self.i_syn_exc_gm[neuron_id])
        anexc.i_bias_tau = hal.CapMemCell.Value(
            self.i_syn_exc_tau[neuron_id])
        anexc.i_drop_input = hal.CapMemCell.Value(
            self.i_syn_exc_drop[neuron_id])

        aninh = atomic_neuron.inhibitory_input
        aninh.i_shift_reference = hal.CapMemCell.Value(
            self.i_syn_inh_shift[neuron_id])
        aninh.i_bias_gm = hal.CapMemCell.Value(
            self.i_syn_inh_gm[neuron_id])
        aninh.i_bias_tau = hal.CapMemCell.Value(
            self.i_syn_inh_tau[neuron_id])
        aninh.i_drop_input = hal.CapMemCell.Value(
            self.i_syn_inh_drop[neuron_id])

        return atomic_neuron

    def to_neuron_calib_result(self) -> NeuronCalibResult:
        """
        Conversion to NeuronCalibResult.
        The numpy arrays get merged into lola AtomicNeurons.

        :return: Equivalent NeuronCalibResult.
        """

        result = NeuronCalibResult(dict(), dict(), dict())

        # set neuron configuration, including CapMem
        for neuron_coord in halco.iter_all(halco.AtomicNeuronOnDLS):
            result.neurons[neuron_coord] = self.to_atomic_neuron(neuron_coord)

            neuron_id = neuron_coord.toEnum().value()
            result.success[neuron_coord] = self.success[neuron_id]

        # set global CapMem parameters
        dumper = sta.PlaybackProgramBuilderDumper()
        dumper = neuron_helpers.configure_integration(dumper)
        dumper = neuron_helpers.set_global_capmem_config(dumper)
        cocolist = dumper.done().tolist()

        for coord, config in cocolist:
            # remove Timer-commands like waits, we only want to collect
            # coord/container pairs (and would have many entries for the
            # timer coordinate otherwise)
            if coord == halco.TimerOnDLS():
                continue
            result.cocos[coord] = config

        return result


# pylint: disable=too-many-statements
def calibrate(
        connection: hxcomm.ConnectionHandle, *,
        target_leak_read: Union[int, np.ndarray] = 120,
        tau_mem: pq.quantity.Quantity = 60 * pq.us,
        tau_syn: pq.quantity.Quantity = 0.32 * pq.us,
        i_synin_gm: int = 300, target_noise: Optional[float] = None,
        readout_neuron: Optional[halco.AtomicNeuronOnDLS] = None,
        initial_configuration: Optional[
            Callable[[hxcomm.ConnectionHandle], None]] = None
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
    :param target_leak_read: Target CADC read at resting potential
        of the membrane. Due to the low leak bias currents, the spread
        of resting potentials may be high even after calibration.
    :param tau_mem: Targeted membrane time constant while calibrating the
        synaptic inputs.
        Too short values can not be achieved with this calibration routine.
        The default value of 60 us should work.
        If a target_noise is given (default), this setting does not affect
        the final leak bias currents, as those are determined by
        reaching the target noise.
    :param tau_syn: Controls the synaptic input time constant.
        If set to 0 us, the minimum synaptic input time constant will be
        used, which means different synaptic input time constants per
        neuron. If a single different Quantity is given, it is used for
        all synaptic inputs of all neurons, excitatory and inhibitory.
        If an array of Quantities is given, it can be shaped
        (2, 512) for the excitatory and inhibitory synaptic input
        of each neuron. It can also be shaped (2,) for the excitatory
        and inhibitory synaptic input of all neurons, or shaped (512,)
        for both inputs per neuron.
    :param i_synin_gm: Target synaptic input OTA bias current.
        The amplitudes of excitatory inputs using this target current are
        measured, and the median of all neurons' amplitudes is taken as target
        for calibration of the synaptic input strengths.
        The inhibitory synaptic input gets calibrated to match the excitatory.
        Some 300 LSB are proposed here. Choosing high values yields
        higher noise and lower time constants on the neurons, choosing
        low values yields less gain in a multiplication.
    :param target_noise: Noise amplitude in an integration process to
        aim for when searching the optimum leak OTA bias current,
        given as the standard deviation of successive reads in CADC LSB.
        Higher noise settings mean longer membrane time constants but
        impact reproducibility.
        Set target_noise to None to skip optimization of noise amplitudes
        entirely. In this case, the original membrane time constant
        calibration is used for leak bias currents.
    :param readout_neuron: Coordinate of the neuron to be connected to
        a readout pad, i.e. can be observed using an oscilloscope.
        The selected neuron is connected to the upper pad (channel 0),
        the lower pad (channel 1) always shows the CADC ramp of quadrant 0.
        The pads are connected via
        halco.SourceMultiplexerOnReadoutSourceSelection(0) for the neuron
        and mux 1 for the CADC ramps. When using the internal MADC
        for recording, these multiplexers can be selected directly.
        If None is given, the readout is not configured.
    :param initial_configuration: Additional function which is called before
        starting the calibration. Called with `connection`, such that the
        hardware is available within the function.
        If None (default), no additional configuration gets applied.

    :return: NeuronCalibResult, containing all calibrated parameters.

    :raises ValueError: If target parameters are not in a feasible range.
    :raises ValueError: If target parameters are not shaped as specified.
    :raises TypeError: If time constants are not given with a unit
        from the `quantities` package.
    """

    if not isinstance(tau_mem, pq.quantity.Quantity):
        raise TypeError(
            "Membrane time constant is not given as a "
            "`quantities.quantity.Quantity`.")
    if not isinstance(tau_syn, pq.quantity.Quantity):
        raise TypeError(
            "Synaptic time constant is not given as a "
            "`quantities.quantity.Quantity`.")

    if np.any([tau_mem < 0.1 * pq.us, tau_mem > 200. * pq.us]):
        raise ValueError(
            "Target membrane time constant is out of feasible range.")
    if np.any([tau_syn < 0 * pq.us, tau_syn > 50. * pq.us]):
        raise ValueError(
            "Target synaptic time constant is out of feasible range.")

    # Configure chip for calibration
    builder = sta.PlaybackProgramBuilder()
    builder, initial_config = neuron_helpers.configure_chip(
        builder, readout_neuron=readout_neuron)
    sta.run(connection, builder.done())

    # disable synaptic inputs initially
    neuron_helpers.reconfigure_synaptic_input(
        connection, excitatory_biases=0, inhibitory_biases=0)

    # call optional program for further initial configuration
    if initial_configuration is not None:
        initial_configuration(connection)

    # Initialize return object
    calib_result = CalibrationResultInternal()
    calib_result.i_syn_exc_drop = initial_config[
        halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_drop]
    calib_result.i_syn_inh_drop = initial_config[
        halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_drop]
    calib_result.i_bias_reset = initial_config[
        halco.CapMemRowOnCapMemBlock.i_bias_reset]
    calib_result.i_syn_exc_tau = initial_config[
        halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_tau]
    calib_result.i_syn_inh_tau = initial_config[
        halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_tau]

    # Calibrate synaptic input time constant using MADC
    if np.all(tau_syn == 0 * pq.us):
        pass
    elif np.ndim(tau_syn) > 0 \
            and tau_syn.shape[0] == halco.SynapticInputOnNeuron.size:
        calibration = neuron_synin.ExcSynTimeConstantCalibration(
            target=tau_syn[0])
        calib_result.i_syn_exc_tau = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch()
        ).calibrated_parameters
        calibration = neuron_synin.InhSynTimeConstantCalibration(
            target=tau_syn[1])
        calib_result.i_syn_inh_tau = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch()
        ).calibrated_parameters
    else:
        calibration = neuron_synin.ExcSynTimeConstantCalibration(
            target=tau_syn)
        calib_result.i_syn_exc_tau = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch()
        ).calibrated_parameters
        calibration = neuron_synin.InhSynTimeConstantCalibration(
            target=tau_syn)
        calib_result.i_syn_inh_tau = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch(),
        ).calibrated_parameters

    # Calibrate leak potential at target leak read
    calibration = neuron_potentials.LeakPotentialCalibration(
        target_leak_read)
    calibration.run(connection, algorithm=algorithms.NoisyBinarySearch())

    # Calibrate membrane time constant using reset
    calibration = neuron_leak_bias.MembraneTimeConstCalibCADC(
        target_time_const=tau_mem, target_amplitude=50)
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
        connection, excitatory_biases=i_synin_gm)

    exc_synin_calibration = neuron_synin.ExcSynBiasCalibration(
        target_leak_read=target_cadc_reads,
        parameter_range=base.ParameterRange(0, min(
            i_synin_gm + 250, hal.CapMemCell.Value.max)))
    result = exc_synin_calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.i_syn_exc_gm = result.calibrated_parameters
    calib_result.success = np.all(
        [calib_result.success, result.success], axis=0)

    # Disable exc. synaptic input, enable and calibrate inhibitory
    neuron_helpers.reconfigure_synaptic_input(
        connection, excitatory_biases=0, inhibitory_biases=i_synin_gm)

    calibration = neuron_synin.InhSynBiasCalibration(
        target_leak_read=target_cadc_reads,
        parameter_range=base.ParameterRange(0, min(
            i_synin_gm + 250, hal.CapMemCell.Value.max)),
        target=exc_synin_calibration.target)
    calibration.n_events = exc_synin_calibration.n_events
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.i_syn_inh_gm = result.calibrated_parameters
    calib_result.success = np.all(
        [calib_result.success, result.success], axis=0)

    calibration = neuron_synin.InhSynReferenceCalibration(
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
    calibration = neuron_synin.ExcSynReferenceCalibration(
        target=target_cadc_reads)
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.i_syn_exc_shift = result.calibrated_parameters
    calib_result.success = np.all([
        calib_result.success, result.success], axis=0)

    # Calibrate leak potential
    calibration = neuron_potentials.LeakPotentialCalibration(
        target_cadc_reads)
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.v_leak = result.calibrated_parameters
    calib_result.success = np.all([
        calib_result.success, result.success], axis=0)

    # Calibrate reset potential to be equal to leak
    calibration = neuron_potentials.ResetPotentialCalibration(
        highnoise=True)
    result = calibration.run(
        connection, algorithm=algorithms.NoisyBinarySearch())
    calib_result.v_reset = result.calibrated_parameters
    calib_result.success = np.all([
        calib_result.success, result.success], axis=0)

    if target_noise:
        # Set Leak bias as low as possible with target readout noise
        calibration = neuron_leak_bias.LeakBiasCalibration(
            target=target_noise, target_leak_read=target_cadc_reads)
        result = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch())
        calib_result.i_bias_leak = result.calibrated_parameters
        calib_result.success = np.all([
            calib_result.success, result.success], axis=0)

        # Re-calibrate excitatory synaptic input
        # This mitigates CapMem crosstalk and allows compensating
        # systematic drift currents (similarly below).
        calibration = neuron_synin.ExcSynReferenceCalibration(
            target=target_cadc_reads)
        result = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch())
        calib_result.i_syn_exc_shift = result.calibrated_parameters
        calib_result.success = np.all([
            calib_result.success, result.success], axis=0)

        # Re-calibrate inhibitory synaptic input
        calibration = neuron_synin.InhSynReferenceCalibration(
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
        calibration = neuron_potentials.LeakPotentialCalibration(
            target_cadc_reads)
        result = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch())
        calib_result.v_leak = result.calibrated_parameters
        calib_result.success = np.all([
            calib_result.success, result.success], axis=0)

        # Calibrate reset potential to be equal to resting potential
        calibration = neuron_potentials.ResetPotentialCalibration(
            highnoise=True)
        result = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch())
        calib_result.v_reset = result.calibrated_parameters
        calib_result.success = np.all([
            calib_result.success, result.success], axis=0)

    # Log statistics about membrane potentials after claibration
    neuron_evaluation.measure_quadrant_results(
        connection, calib_result.success)

    return calib_result.to_neuron_calib_result()


def calibrate_baseline(connection: hxcomm.ConnectionHandle,
                       target_read: int = 128) -> base.CalibrationResult:
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

    :return: CalibrationResult, containing:
        * Calibrated CADC channel offsets.
        * Success mask of calibration - False for CADC channels that could
          not be matched to the target read.
    """

    calibration = neuron_potentials.BaselineCalibration()
    return calibration.run(
        connection, algorithm=algorithms.LinearPrediction(probe_parameters=0),
        target=target_read)
