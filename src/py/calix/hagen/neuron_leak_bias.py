# pylint: disable=too-many-lines

"""
Provides functions to set the neurons' leak OTA bias currents
for different purposes.

The class MembraneTimeConstCalibCADC measures the membrane
time constant and calibrates to set these equal.
This is done before the calibration of synaptic input amplitudes.

The class LeakBiasCalib is targeted at setting the leak OTA bias
current as low as possible while keeping the noise under control.
This is called after the synaptic inputs are calibrated.
"""

from typing import Optional, Tuple, Union, List
import numbers
import os
import numpy as np
import quantities as pq
from scipy.optimize import curve_fit
from dlens_vx_v3 import hal, halco, logger, hxcomm

from calix.common import algorithms, base, exceptions, madc_base, helpers
from calix.hagen import neuron_potentials, neuron_helpers
from calix import constants


class MembraneTimeConstCalibCADC(base.Calib):
    """
    Find the leak conductivity by resetting the neuron's membrane to a
    potential significantly below the leak potential.

    The reset potential v_reset is set below the leak potential v_leak.
    After a reset the membrane voltage rises towards the leak potential v_leak.
    As the membrane voltage rises, the CADCs are used to read the voltage at a
    specified time after the reset.
    The leak bias current gets calibrated such that the voltage after the
    desired time has risen to `v_leak - (v_leak - v_reset)/e`. Therefore, the
    specified time should be roughly the membrane time constant.

    The leak potential gets recalibrated every time the bias current
    is changed, as changing the OTA bias current also changes the
    required offset between its inputs. The target CADC reads at leak
    potential are acquired at the beginning of the calibration with the
    existing neuron configuration. The median of all reads is taken as target.

    This class provides functions for configuring leak bias currents
    and reading the potentials of neurons a specific time after reset.

    Requirements:
    * Neuron membrane readout is connected to the CADCs (causal and acausal).
    * The currently selected leak potential is not too low, such that the
      target_amplitude can still be subtracted from the CADC reads at leak
      potential without leaving the desired operating range of the leak
      OTA or the dynamic range of the CADCs.
    * The target membrane time constant is long, above some 30 us.

    :ivar target_amplitude: Target amplitude for (v_leak - v_reset), given
        in CADC reads.
    :ivar target_time_const: Target membrane time constant.
    :ivar target_leak_read: Target CADC read at leak potential.
    :ivar leak_calibration: LeakPotentialCalib class instance used for
        recalibration of leak potential after changing the bias current.

    :raises ValueError: if target_time_constant is not a single value.
    """

    def __init__(self, target_time_const: pq.quantity.Quantity = 60 * pq.us,
                 target_amplitude: Union[int, np.ndarray] = 50):
        super().__init__(
            parameter_range=base.ParameterRange(0, 1022),
            n_instances=halco.NeuronConfigOnDLS.size, inverted=False)
        self.target_amplitude = target_amplitude
        self.leak_calibration: Optional[
            neuron_potentials.LeakPotentialCalib] = None
        self.target_leak_read: Optional[int] = None

        if target_time_const.size != 1:
            raise ValueError("This calibration routine only supports a single "
                             "'target_time_const' for all enurons . Please "
                             "choose a value of size one.")

        self.target_time_const = target_time_const

    def prelude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Read leak potentials, calibrate reset potential to leak - amplitude.

        :param connection: Connection to the chip to run on.

        :raises CalibNotSuccessful: If the target CADC read at
            reset potential is below the reliable range of the CADC
            for more than 5% of the neurons.
        """

        log = logger.get(
            "calix.hagen.neuron_leak_bias.MembraneTimeConstCalibCADC")

        # Measure leak potential
        self.target_leak_read = \
            neuron_helpers.cadc_read_neuron_potentials(connection)

        # Create instance of leak potential calibration
        self.leak_calibration = neuron_potentials.LeakPotentialCalib(
            target=self.target_leak_read)

        # Calibrate reset potential to leak - amplitude
        target_reset_read = self.target_leak_read - self.target_amplitude
        too_low_mask = target_reset_read < constants.cadc_reliable_range.lower
        if np.any(too_low_mask):
            log.WARN(
                "Target reset CADC read was obtained low for neurons "
                + str(np.nonzero(too_low_mask)[0]) + ": "
                + str(target_reset_read[too_low_mask]))
        if np.sum(too_low_mask) > int(halco.NeuronConfigOnDLS.size * 0.05):
            raise exceptions.CalibNotSuccessful(
                f"Reset CADC read was obtained at {target_reset_read}, which "
                + "is lower than the reliable range of the "
                + "CADCs for many neurons: "
                + str(np.nonzero(target_reset_read
                                 < constants.cadc_reliable_range.lower)[0])
                + ". Increase the leak voltage or reduce the target "
                + "amplitude.")
        target_reset_read[too_low_mask] = constants.cadc_reliable_range.lower

        calibration = neuron_potentials.ResetPotentialCalib(
            target=target_reset_read)
        calibration.run(connection, algorithm=algorithms.NoisyBinarySearch())

        # Calculate target CADC read after one time constant
        self.target = self.target_leak_read - (
            self.target_leak_read - target_reset_read) / np.e

    def configure_parameters(
            self, builder: base.WriteRecordingPlaybackProgramBuilder,
            parameters: np.ndarray) \
            -> base.WriteRecordingPlaybackProgramBuilder:
        """
        Configure the leak OTA bias currents of all neurons to the given
        values.

        :param builder: Builder to append configuration instructions to.
        :param parameters: Array of leak OTA bias current settings to
            configure the CapMem cells to.

        :return: Builder with configuration appended.
        """

        builder = helpers.capmem_set_neuron_cells(
            builder,
            {halco.CapMemRowOnCapMemBlock.i_bias_leak: parameters})

        builder = helpers.wait(builder, constants.capmem_level_off_time)
        return builder

    def _reset_wait_read(self, connection: hxcomm.ConnectionHandle,
                         runs_per_builder: int = 100
                         ) -> Tuple[List, List]:
        """
        Reset neuron, wait for the target time constant, read the
        neuron's membrane potentials using the CADCs.
        This is done separately for all neurons on the chip.
        An array of the individual neurons' reads is returned.

        :param connection: Connection to chip to run on.
        :param runs_per_builder: How many runs to execute in one builder.
            Choosing too many runs can result in host/FPGA timeouts, as
            the FPGA program buffer will get full and waiting on chip then
            prevents the FPGA from accepting new instructions.

        :return: Array of all neurons' CADC reads.
        """

        read_tickets = []
        read_positions = []

        runs_in_builder = 0
        builder = base.WriteRecordingPlaybackProgramBuilder()
        for target_neuron in halco.iter_all(halco.NeuronConfigOnDLS):
            # Reset neuron
            builder.write(target_neuron.toNeuronResetOnDLS(),
                          hal.NeuronReset())

            # Wait for time constant
            builder = helpers.wait(builder, self.target_time_const)

            # Read CADC results, save ticket
            # Trigger measurement, causal read
            coord = halco.CADCSampleQuadOnDLS(
                block=halco.CADCSampleQuadOnSynram(
                    halco.SynapseQuadOnSynram(
                        x=target_neuron.toSynapseQuadColumnOnDLS(),
                        y=halco.SynapseRowOnSynram(0)),
                    halco.CADCChannelType.causal,
                    halco.CADCReadoutType.trigger_read),
                synram=target_neuron.toSynramOnDLS())
            read_tickets.append(
                builder.read(coord, backend=hal.Backend.Omnibus))
            read_positions.append(target_neuron.toEntryOnQuad())

            # Read buffered result, acausal read
            coord = halco.CADCSampleQuadOnDLS(
                block=halco.CADCSampleQuadOnSynram(
                    halco.SynapseQuadOnSynram(
                        x=target_neuron.toSynapseQuadColumnOnDLS(),
                        y=halco.SynapseRowOnSynram(0)),
                    halco.CADCChannelType.acausal,
                    halco.CADCReadoutType.buffered),
                synram=target_neuron.toSynramOnDLS())
            read_tickets.append(
                builder.read(coord, backend=hal.Backend.Omnibus))
            read_positions.append(target_neuron.toEntryOnQuad())

            # Run program if builder is filled
            if runs_in_builder == runs_per_builder:
                base.run(connection, builder)

                builder = base.WriteRecordingPlaybackProgramBuilder()
                runs_in_builder = 0
            else:
                runs_in_builder += 1

        if not builder.empty():
            base.run(connection, builder)

        # Process tickets, write into array
        results = np.empty(
            halco.CADCChannelType.size * halco.NeuronConfigOnDLS.size,
            dtype=int)
        for ticket_id, (ticket, coord) in enumerate(
                zip(read_tickets, read_positions)):
            quad_result = ticket.get()
            results[ticket_id] = int(quad_result.get_sample(coord))

        # Take mean of causal and acausal read
        results = np.mean(
            results.reshape(
                (halco.NeuronConfigOnDLS.size, halco.CADCChannelType.size)),
            axis=1)

        return results

    def measure_results(
            self, connection: hxcomm.ConnectionHandle,
            builder: base.WriteRecordingPlaybackProgramBuilder) \
            -> np.ndarray:
        """
        Calibrate the leak potentials such that the CADC reads match
        the read targets.
        Reset the neurons and wait for the target time constant.
        Measure the membrane potentials using the CADCs.

        :param connection: Connection to the chip to run on.
        :param builder: Builder to run.

        :return: Array of CADC reads after reset and wait.
        """

        # Run configuration builder
        base.run(connection, builder)

        # Calibrate leak potentials to target
        leak_calib_success_mask = self.leak_calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch()).success

        # Iterate all neurons: reset, wait, measure.
        results = self._reset_wait_read(connection)

        # Set result low if it is higher than leak read (i.e. membrane floats)
        results[results > self.target_leak_read + 5] = 0

        # Set result low if leak calibration failed (i.e. leak too weak):
        # Note that this could also be caused by other reasons, e.g.,
        # the leak target potential being too high and out of the
        # feasible range of the CapMem. However, the leak conductivity
        # being too weak is the main reason for a failed calib at a
        # typical operating point, and the calibration benefits from
        # increasing bias currents for failed leak potential calibs.
        results[~leak_calib_success_mask] = 0

        return results

    def postlude(self, connection: hxcomm.ConnectionHandle) -> None:
        log = logger.get("calix.hagen.neuron_leak_bias."
                         + "MembraneTimeConstCalibCADC.postlude")
        log.INFO("Calibrated membrane time constant.")
        log.DEBUG("Leak bias currents:" + os.linesep
                  + f"{self.result.calibrated_parameters}")


class MembraneTimeConstCalibReset(madc_base.Calib):
    """
    Measure neuron reset with the MADC to calibrate the membrane time constant.

    Due to the MADC's high sample rate, an exponential fit on the decaying
    potential after a reset is used to determine the membrane time
    constant. Leak bias currents are tweaked to reach the desired value.

    The leak and reset potentials are altered to ensure a good amplitude
    for the fit between the two potentials. The membrane time constant
    at a (different) desired leak potential may be slightly
    different. For target membrane time constants above some 3 us,
    a step current stimulus can be used instead of a reset, therefore
    not changing the potentials. See MembraneTimeConstCalibOffset for
    this method.

    This calibration decides whether leak division or multiplication
    is required during prelude. The given neuron configs are therefore
    altered.

    Requirements:
    - None -

    :ivar neuron_configs: List of desired neuron configurations.
        Necessary to enable leak division/multiplication.
    """

    def __init__(self,
                 target: pq.quantity.Quantity = 60 * pq.us,
                 neuron_configs: Optional[List[hal.NeuronConfig]] = None):
        """
        :param neuron_configs: List of neuron configurations. If None, the
            hagen-mode default neuron config is used for all neurons.
        """

        super().__init__(
            parameter_range=base.ParameterRange(
                hal.CapMemCell.Value.min, hal.CapMemCell.Value.max),
            inverted=True)

        self.target = target
        if neuron_configs is None:
            self.neuron_configs = [
                neuron_helpers.neuron_config_default() for _ in
                range(halco.NeuronConfigOnDLS.size)]
        else:
            self.neuron_configs = neuron_configs

        self.sampling_time = max(70 * pq.us, 8 * np.max(self.target))
        self.wait_between_neurons = 10 * self.sampling_time
        self._wait_before_stimulation = 0 * pq.us

    def prelude(self, connection: hxcomm.ConnectionHandle):
        """
        Prepares chip for calibration.

        Sets reset potential low and leak high in order to
        observe a large decay back to the leak potential.

        Also measures the membrane time constant at low and high
        bias currents to decide whether leak multiplication or
        division is required to reach the given targets.

        :param connection: Connection to the chip to calibrate.
        """

        # prepare MADC
        super().prelude(connection)

        # set reset potential low and leak potential high
        builder = base.WriteRecordingPlaybackProgramBuilder()
        builder = helpers.capmem_set_neuron_cells(
            builder, {
                halco.CapMemRowOnCapMemBlock.v_reset: 520,
                halco.CapMemRowOnCapMemBlock.v_leak: 880})
        builder = helpers.wait(builder, constants.capmem_level_off_time)
        base.run(connection, builder)

        # decide whether leak division or multiplication is required:
        # inspect the feasible range without division or multiplication.
        for neuron_id in range(self.n_instances):
            self.neuron_configs[neuron_id].enable_leak_division = False
            self.neuron_configs[neuron_id].enable_leak_multiplication = False

        # measure at low leak bias current: If time constant is still
        # smaller than the target, then enable division.
        builder = base.WriteRecordingPlaybackProgramBuilder()
        builder = self.configure_parameters(
            builder, parameters=np.ones(self.n_instances, dtype=int) * 63
            + helpers.capmem_noise(size=self.n_instances))
        maximum_timeconstant = self.measure_results(connection, builder)
        enable_division = maximum_timeconstant < self.target

        # measure at high leak bias current: If time constant is still
        # larger than the target, then enable multiplication.
        builder = base.WriteRecordingPlaybackProgramBuilder()
        builder = self.configure_parameters(
            builder, parameters=np.ones(self.n_instances, dtype=int)
            * (hal.CapMemCell.Value.max - 63)
            + helpers.capmem_noise(size=self.n_instances))
        minimum_timeconstant = self.measure_results(connection, builder)
        enable_multiplication = minimum_timeconstant > self.target

        # check sanity of decisions
        # The fit may fail for very short time constants, < 0.5 us.
        # Thus, if both division and multiplication is selected,
        # only division is applied.
        enable_multiplication[enable_division] = False

        # set up in neuron configs
        for neuron_id in range(self.n_instances):
            self.neuron_configs[neuron_id].enable_leak_division = \
                enable_division[neuron_id]
            self.neuron_configs[neuron_id].enable_leak_multiplication = \
                enable_multiplication[neuron_id]

    def postlude(self, connection: hxcomm.ConnectionHandle):
        """
        Restore original readout configuration.

        The base class postlude is overwritten to _not_ restore
        the original neuron configuration, as leak division
        and multiplication may be altered by this routine.

        :param connection: Connection to the chip to calibrate.
        """

        # restore original readout config
        builder = base.WriteRecordingPlaybackProgramBuilder()
        builder.write(halco.ReadoutSourceSelectionOnDLS(),
                      self.original_readout_config)
        base.run(connection, builder)

    def configure_parameters(
            self, builder: base.WriteRecordingPlaybackProgramBuilder,
            parameters: np.ndarray) \
            -> base.WriteRecordingPlaybackProgramBuilder:
        """
        Configure the given array of leak bias currents.

        :param builder: Builder to append configuration instructions to.
        :param parameters: Array of bias currents to set up.

        :return: Builder with configuration appended.
        """

        builder = helpers.capmem_set_neuron_cells(
            builder, {halco.CapMemRowOnCapMemBlock.i_bias_leak: parameters})

        builder = helpers.wait(builder, constants.capmem_level_off_time)
        return builder

    def stimulate(self, builder: base.WriteRecordingPlaybackProgramBuilder,
                  neuron_coord: halco.NeuronConfigOnDLS,
                  stimulation_time: hal.Timer.Value
                  ) -> base.WriteRecordingPlaybackProgramBuilder:
        """
        Reset the neuron membrane potential to a low voltage.

        :param builder: Builder to append reset instructions to.
        :param neuron_coord: Coordinate of neuron which is currently recorded.
        :param stimulation_time: Timer value at beginning of stimulation.

        :return: Builder with neuron resets appended.
        """

        builder.write(neuron_coord.toNeuronResetOnDLS(), hal.NeuronReset())
        return builder

    def neuron_config_disabled(self, neuron_coord: halco.NeuronConfigOnDLS
                               ) -> hal.NeuronConfig:
        """
        Return a neuron config with readout disabled.

        :param neuron_coord: Coordinate of neuron to get config for.

        :return: Neuron config with readout disabled.
        """

        config = hal.NeuronConfig(
            self.neuron_configs[int(neuron_coord.toEnum())])
        config.enable_threshold_comparator = False
        config.enable_readout = False
        return config

    def neuron_config_readout(self, neuron_coord: halco.NeuronConfigOnDLS
                              ) -> hal.NeuronConfig:
        """
        Return a neuron config with readout enabled.

        :param neuron_coord: Coordinate of neuron to get config for.

        :return: Neuron config with readout enabled.
        """

        config = self.neuron_config_disabled(neuron_coord)
        config.readout_source = hal.NeuronConfig.ReadoutSource.membrane
        config.enable_readout = True
        return config

    # pylint: disable=too-many-locals
    def evaluate(self, samples: List[np.ndarray]) -> np.ndarray:
        """
        Evaluates the obtained MADC samples.

        To each neuron's MADC samples, an exponential decay is fitted,
        and the resulting time constant is returned.
        This should make for a very precise measurement.

        :param samples: MADC samples obtained for each neuron.

        :return: Numpy array of fitted synaptic input time constants.
        """
        def fitfunc(time_t, scale, tau, offset):
            return scale * np.exp(-time_t / tau) + offset

        def find_exponential_start(samples: np.ndarray) -> Optional[int]:
            """
            Find start of exponential rise.

            Find index where the exponential rise begins.
            :param samples: MADC trace to investigate.

            :return: Index of detected start of the exponential rise.
            """

            leak_potential = np.mean(samples['value'][:-10])
            reset_potential = np.min(samples['value'])
            diff = leak_potential - reset_potential

            # detect end of refractory period
            refractory = leak_potential - samples['value'] > diff / 2
            return np.arange(len(refractory))[refractory][-1]

        neuron_fits = []
        for neuron_id, neuron_data in enumerate(samples):
            # remove unreliable samples
            start = int(self._wait_before_stimulation.rescale(pq.s)
                        * self.madc_config.calculate_sample_rate(
                            self.madc_input_frequency))
            stop = int(int(self.madc_config.number_of_samples) * 0.95)
            neuron_samples = neuron_data[start:stop]

            # only fit to exponential rise
            start = find_exponential_start(neuron_samples)
            neuron_samples = neuron_samples[start:]

            # estimate start values for fit
            p_0 = {}
            p_0['offset'] = np.mean(neuron_samples["value"][-10:])
            p_0['scale'] = np.min(neuron_samples["value"]) - p_0['offset']
            index_tau = np.argmax(neuron_samples["value"]
                                  > p_0['offset'] + p_0['scale'] / np.e)
            p_0['tau'] = neuron_samples["chip_time"][index_tau] - \
                neuron_samples["chip_time"][0]

            # for small time constants the estimation of tau might fail ->
            # cut at bounds
            p_0['tau'] = min(
                max(constants.tau_mem_range.lower.rescale(pq.us).magnitude,
                    p_0['tau']),
                constants.tau_mem_range.upper.rescale(pq.us).magnitude)

            boundaries = (
                [-p_0['offset'],
                 constants.tau_mem_range.lower.rescale(pq.us).magnitude,
                 p_0['offset'] - 10],
                [0,
                 constants.tau_mem_range.upper.rescale(pq.us).magnitude,
                 p_0['offset'] + 10])

            try:
                popt, _ = curve_fit(
                    fitfunc,
                    neuron_samples["chip_time"]
                    - neuron_samples["chip_time"][0],
                    neuron_samples["value"],
                    p0=[p_0['scale'], p_0['tau'], p_0['offset']],
                    bounds=boundaries)
            except RuntimeError as error:
                raise exceptions.CalibNotSuccessful(
                    f"Fitting to MADC samples failed for neuron {neuron_id}. "
                    + str(error))
            neuron_fits.append(popt[1])  # store time constant of exponential

        return np.array(neuron_fits) * pq.us


class MembraneTimeConstCalibOffset(madc_base.Calib):
    """
    Measure response to step current with MADC to calibrate the membrane time
    constant.

    Due to the MADC's high sample rate, an exponential fit on the decaying
    potential after a step current stimulus is used to determine the
    membrane time constant. Leak bias currents are tweaked to reach
    the desired membrane time constant.

    This calibration decides whether leak division is required during
    prelude. The given neuron configs are therefore altered.
    The original neuron config (configured on chip before running the
    calibration) is not restored, instead we stay with the config
    used (and altered) during calibration.

    Requirements:
    * Leak potential is calibrated to the desired value. This is useful
      since the time constant is then calibrated at and above this
      selected potential.
    * The desired membrane time constant is larger than some 3 us, since
      leak multiplication is not supported by this calibration.
    * The synaptic input should be on; it should be calibrated or the
      bias currents should be set to 0. The reason is an effect on the
      membrane dynamics for some neurons, which looks similar to leakage.
      We recommend the synaptic input to be enabled and calibrated.

    :ivar neuron_configs: List of desired neuron configurations.
        Necessary to enable leak division.
    :ivar adjust_bias_range: Enable/disable adjustment of leak division
        to extend the dynamic range of calibration. By default, this
        setting is enabled and we select appropriate settings during
        the prelude.
    """

    def __init__(self,
                 target: pq.quantity.Quantity,
                 neuron_configs: Optional[List[hal.NeuronConfig]] = None):
        """
        :param neuron_configs: List of neuron configurations. If None, the
            hagen-mode default neuron config is used for all neurons.
        """

        super().__init__(
            parameter_range=base.ParameterRange(
                hal.CapMemCell.Value.min, hal.CapMemCell.Value.max),
            inverted=True)

        self.target = target
        if neuron_configs is None:
            self.neuron_configs = [
                neuron_helpers.neuron_config_default() for _ in
                range(halco.NeuronConfigOnDLS.size)]
        else:
            self.neuron_configs = neuron_configs
        for config in self.neuron_configs:
            # The threshold is disabled to avoid possible spiking
            # if the threshold potential is already set lower than the
            # leak that will be set in the prelude.
            config.enable_threshold_comparator = False

        self._wait_before_stimulation = 100 * pq.us  # step current duration
        self.sampling_time = max(70 * pq.us, 8 * np.max(self.target)) \
            + self._wait_before_stimulation
        self.wait_between_neurons = 10 * self.sampling_time
        self.adjust_bias_range: bool = True

    def prelude(self, connection: hxcomm.ConnectionHandle):
        """
        Prepares chip for calibration.

        Sets a high offset current in all neurons.

        Also measures the membrane time constant at low leak
        bias currents to decide whether leak division is required
        to reach the given targets.

        Leak multiplication is not supported by this calibration:
        The offset current is too weak to stimulate the membrane
        significantly if multiplication gets enabled.
        The calibration can therefore only be used if the target
        membrane time constant is larger than some 3 us.

        :param connection: Connection to the chip to calibrate.
        """

        # prepare MADC
        super().prelude(connection)

        # set offset current onto membrane (enabled later)
        builder = base.WriteRecordingPlaybackProgramBuilder()
        builder = helpers.capmem_set_neuron_cells(
            builder, {halco.CapMemRowOnCapMemBlock.i_mem_offset: 1000})
        builder = helpers.wait(builder, constants.capmem_level_off_time)

        # run program
        base.run(connection, builder)

        if self.adjust_bias_range:
            # decide whether leak division is required:
            # inspect the feasible range with division.
            for neuron_id in range(self.n_instances):
                self.neuron_configs[neuron_id].enable_leak_division = True
                self.neuron_configs[neuron_id].enable_leak_multiplication = \
                    False

            # measure at maximum leak bias current: If time constant is
            # smaller than the target, then keep division enabled.
            builder = base.WriteRecordingPlaybackProgramBuilder()
            builder = self.configure_parameters(
                # subtract CapMem noise amplitude
                builder, parameters=hal.CapMemCell.Value.max - 5)
            min_tau_with_division = self.measure_results(connection, builder)
            enable_division = min_tau_with_division < self.target

            # enabling leak multiplication is not supported for this method:
            # the offset current is too weak to stimulate the membrane
            # significantly if multiplication gets enabled.
            # The calibration can therefore only be used if the target
            # membrane time constant is larger than some 3 us.

            # set up in neuron configs
            for neuron_id in range(self.n_instances):
                self.neuron_configs[neuron_id].enable_leak_division = \
                    enable_division[neuron_id]

    def postlude(self, connection: hxcomm.ConnectionHandle):
        """
        Restore original readout configuration.
        The base class postlude is overwritten to _not_ restore
        the original neuron configuration, as leak division
        may be altered by this routine.

        :param connection: Connection to the chip to calibrate.
        """

        # The original neuron config is not restored due to possible
        # issues with the spike threshold: If the previously configured
        # threshold is lower than the leak potential used here, the
        # neuron would now be spiking regularly.

        # restore original readout config
        builder = base.WriteRecordingPlaybackProgramBuilder()
        builder.write(halco.ReadoutSourceSelectionOnDLS(),
                      self.original_readout_config)

        # disable offset current
        builder = helpers.capmem_set_neuron_cells(
            builder, {halco.CapMemRowOnCapMemBlock.i_mem_offset: 0})
        builder = helpers.wait(builder, constants.capmem_level_off_time)
        base.run(connection, builder)

    def configure_parameters(
            self, builder: base.WriteRecordingPlaybackProgramBuilder,
            parameters: np.ndarray) \
            -> base.WriteRecordingPlaybackProgramBuilder:
        """
        Configure the given array of leak bias currents.

        :param builder: Builder to append configuration instructions to.
        :param parameters: Array of bias currents to set up.

        :return: Builder with configuration appended.
        """

        builder = helpers.capmem_set_neuron_cells(
            builder, {halco.CapMemRowOnCapMemBlock.i_bias_leak: parameters})

        builder = helpers.wait(builder, constants.capmem_level_off_time)
        return builder

    def neuron_config_disabled(self, neuron_coord: halco.NeuronConfigOnDLS
                               ) -> hal.NeuronConfig:
        """
        Return a neuron config with readout disabled.

        :param neuron_coord: Coordinate of neuron to get config for.

        :return: Neuron config with readout disabled.
        """

        config = hal.NeuronConfig(
            self.neuron_configs[int(neuron_coord.toEnum())])
        config.enable_readout = False
        return config

    def neuron_config_readout(self, neuron_coord: halco.NeuronConfigOnDLS
                              ) -> hal.NeuronConfig:
        """
        Return a neuron config with readout enabled.
        The step current is alredy enabled here and will be disabled
        as the stimulus.

        :param neuron_coord: Coordinate of neuron to get config for.

        :return: Neuron config with readout enabled.
        """

        config = self.neuron_config_disabled(neuron_coord)
        config.readout_source = hal.NeuronConfig.ReadoutSource.membrane
        config.enable_readout = True
        config.enable_membrane_offset = True
        return config

    def stimulate(self, builder: base.WriteRecordingPlaybackProgramBuilder,
                  neuron_coord: halco.NeuronConfigOnDLS,
                  stimulation_time: hal.Timer.Value
                  ) -> base.WriteRecordingPlaybackProgramBuilder:
        """
        Disable the membrane offset current.

        This results in a decaying potential back to the leak.

        :param builder: Builder to append instructions to.
        :param neuron_coord: Coordinate of neuron which is currently recorded.
        :param stimulation_time: Timer value at beginning of stimulation.

        :return: Builder with neuron resets appended.
        """

        config = self.neuron_config_readout(neuron_coord)
        config.enable_membrane_offset = False
        builder.write(neuron_coord, config)
        return builder

    # pylint: disable=too-many-locals
    def evaluate(self, samples: List[np.ndarray]) -> np.ndarray:
        """
        Evaluates the obtained MADC samples.

        To each neuron's MADC samples, an exponential decay is fitted,
        and the resulting time constant is returned.

        :param samples: MADC samples obtained for each neuron.

        :return: Numpy array of fitted synaptic input time constants.
        """
        def fitfunc(time, scale, tau, offset):
            return scale * np.exp(-time / tau) + offset

        def find_exponential_start(samples: np.ndarray) -> int:
            """
            Find half amplitude of exponential decay.

            Find index where the exponential decay has reached half
            its amplitude.

            In case this function fails to find the expected edge,
            it returns 0.

            :param samples: MADC trace to investigate.

            :return: Index of detected midpoint of the exponential decay.
            """

            leak_potential = np.mean(samples['value'][:-10])
            current_potential = np.max(samples['value'])
            diff = current_potential - leak_potential

            # detect midpoint of exponential decay after current is disabled
            current = samples['value'] - leak_potential > diff / 2
            fit_start_index = np.arange(len(current))[current][-1]

            # If fit start is close to the end of the trace, return
            # first sample as start (which is already the guessed
            # end time for the offset current).
            # This may happen if the leak potential is very close
            # to the potential with the current enabled.
            if (len(current) - fit_start_index) < 100:
                return 0
            return fit_start_index

        neuron_fits = []
        for neuron_id, neuron_data in enumerate(samples):
            # remove unreliable samples
            start = int(self._wait_before_stimulation.rescale(pq.s)
                        * self.madc_config.calculate_sample_rate(
                            self.madc_input_frequency))
            stop = int(int(self.madc_config.number_of_samples) * 0.95)
            neuron_samples = neuron_data[start:stop]

            # only fit to lower half of exponential decay
            start = find_exponential_start(neuron_samples)
            neuron_samples = neuron_samples[start:]

            if len(neuron_samples) < 100:
                raise AssertionError(
                    "Number of MADC samples to fit membrane time constant is "
                    + f"too small: {len(neuron_samples)}.")

            # estimate start values for fit
            p_0 = {}
            p_0['offset'] = np.mean(neuron_samples["value"][-10:])
            p_0['scale'] = np.max(neuron_samples["value"]) - p_0['offset']
            index_tau = np.argmin(neuron_samples["value"]
                                  > p_0['offset'] + p_0['scale'] / np.e)
            p_0['tau'] = neuron_samples["chip_time"][index_tau] - \
                neuron_samples["chip_time"][0]

            # for small time constants the estimation of tau might fail ->
            # cut at bounds
            p_0['tau'] = min(
                max(constants.tau_mem_range.lower.rescale(pq.us).magnitude,
                    p_0['tau']),
                constants.tau_mem_range.upper.rescale(pq.us).magnitude)

            boundaries = (
                [0,
                 constants.tau_mem_range.lower.rescale(pq.us).magnitude,
                 p_0['offset'] - 10],
                [p_0['scale'] + 10,
                 constants.tau_mem_range.upper.rescale(pq.us).magnitude,
                 p_0['offset'] + 10])

            try:
                popt, _ = curve_fit(
                    fitfunc,
                    neuron_samples["chip_time"]
                    - neuron_samples["chip_time"][0],
                    neuron_samples["value"],
                    p0=[p_0['scale'], p_0['tau'], p_0['offset']],
                    bounds=boundaries)
            except RuntimeError as error:
                raise exceptions.CalibNotSuccessful(
                    f"Fitting to MADC samples failed for neuron {neuron_id}. "
                    + str(error))
            neuron_fits.append(popt[1])  # store time constant of exponential

        return np.array(neuron_fits) * pq.us


class LeakBiasCalib(base.Calib):
    """
    Set leak bias currents as low as possible while preventing the
    membrane voltage from floating and preventing too high membrane noise.

    Search all neurons' leak OTA bias current settings such that the potential
    does not float away, i.e. the noise in successive CADC reads is close to
    the given target noise and the resting potential is not far from
    the expected leak potential CADC value. The leak current is set as
    low as possible fulfilling this requirement.

    When integrating inputs on the neuron, a low leak conductivity is desired.
    However, some neurons show strong variations on the membrane potential
    when the leak conductivity is set very low, as the voltage now floats.
    Some neurons may require a higher leak setting in order to keep
    the membrane potential from floating.

    The noise on the membrane is quantified by reading the
    CADCs multiple times. The leak current is set such that the target
    noise is achieved. This yields a minimum leak conductivity
    while keeping the membrane from floating.

    Calling the run() function returns this tuple of parameters:
        * Array of calibrated leak OTA bias current settings
        * Mask of too noisy neurons. False for neurons exceeding the
        masking_threshold.

    Requirements:
    * Neuron membrane readout is connected to the CADCs (causal and acausal).
    * Reset potential is equal to the leak potential.
    * Synaptic inputs are calibrated such that the membrane potential is
      not affected by constant currents.

    :ivar masking_threshold: Threshold for noise to set calibration
        success to False. Should be higher than the target noise.
        By default, target + 0.8 is used, as the default
        noise target is only 1.2 LSB.
        Note that besides the pure statistical noise, a penalty is
        added to the "noise" result in case the mean resting potential
        is far from the expected value, i.e., the potential drifts
        too far. This pentalty is also considered when finding
        calibration success. See the function optimization_objective()
        in measure_results() for the specific penalty conditions.
    :ivar target_leak_read: Expected CADC reads at leak potential.
        Used to decide whether the membrane potential is floating.
    """

    def __init__(self, target: Union[numbers.Number, np.ndarray] = 1.2,
                 target_leak_read: Optional[Union[
                     numbers.Number, np.ndarray]] = None,
                 masking_threshold: Optional[Union[
                     numbers.Number, np.ndarray]] = None):
        """
        :param target: Noise on each neuron's CADC reads (standard
            deviation of successive CADC reads).
        """

        super().__init__(
            parameter_range=base.ParameterRange(
                hal.CapMemCell.Value.min, hal.CapMemCell.Value.max),
            n_instances=halco.NeuronConfigOnDLS.size,
            inverted=True)
        self.target = target
        self.target_leak_read = target_leak_read
        self.masking_threshold = masking_threshold or target + 0.8

    def prelude(self, connection: hxcomm.ConnectionHandle):
        """
        Measure the current resting potentials and use them as
        `target_leak_read`, if the parameter was not provided.

        :param connection: Connection to the chip to run on.
        """

        if self.target_leak_read is None:
            self.target_leak_read = \
                neuron_helpers.cadc_read_neuron_potentials(connection)

    def configure_parameters(
            self, builder: base.WriteRecordingPlaybackProgramBuilder,
            parameters: np.ndarray) \
            -> base.WriteRecordingPlaybackProgramBuilder:
        """
        Configure leak OTA bias currents to the given parameters.

        :param builder: Builder to append configuration to.
        :param parameters: Leak OTA bias current settings for each neuron.

        :return: Builder with configuration commands appended.
        """

        builder = helpers.capmem_set_neuron_cells(
            builder,
            {halco.CapMemRowOnCapMemBlock.i_bias_leak: parameters})

        builder = helpers.wait(builder, constants.capmem_level_off_time)
        return builder

    def measure_results(self, connection: hxcomm.ConnectionHandle,
                        builder: base.WriteRecordingPlaybackProgramBuilder
                        ) -> np.ndarray:
        """
        Measure the noise on each neuron's membrane.

        To do that, successive CADC reads are done. These happen a small
        amount of time after a reset. The reset - wait - read scheme
        is what we will use in hagen mode, therefore we measure the
        noise in the same fashion. The standard deviation of these
        reads is treated as noise.

        If the resting potential is far from the potential after reset,
        or the potential after reset is far from the expected leak
        potential, the membrane potential is assumed to be floating;
        therefore, the result is artificially increased in order to have the
        leak bias current increased for these neurons.

        :param connection: Connection to the chip to run on.
        :param builder: Builder to append read instructions to.

        :return: Array with standard deviations of CADC reads of each neuron.
        """

        neuron_results_noreset = neuron_helpers.cadc_read_neuron_potentials(
            connection, builder)

        n_reads = 60
        synram_results = []
        for synram in halco.iter_all(halco.SynramOnDLS):
            synram_results.append(
                neuron_helpers.cadc_read_neurons_repetitive(
                    connection,
                    builder=base.WriteRecordingPlaybackProgramBuilder(),
                    synram=synram, n_reads=n_reads, wait_time=5000 * pq.us,
                    reset=True))

        neuron_results = np.hstack(synram_results)

        def optimization_objective(neuron_results: np.ndarray,
                                   neuron_results_noreset: np.ndarray
                                   ) -> np.ndarray:
            """
            Calculate a result value for the calibration algorithm
            to treat correctly. The basis is the statistical noise
            of the neurons, but the result is increased artificially
            for high deviations of the neurons' membrane potentials.

            :param neuron_results: Array of neuron membrane potentials
                measured shortly after reset. Shaped (n_reads, n_neurons).
            :param neuron_results_noreset: Array of neuron resting
                potentials, i.e. reads without a reset. Shaped (n_neurons).

            :return: Array containing results for the calibration algorithm.
            """

            # add punishment noise for high deviations from expected read
            means = np.mean(neuron_results, axis=0)
            level_1_mask = np.abs(means - self.target_leak_read) > 15
            level_2_mask = np.abs(means - self.target_leak_read) > 30

            # add punishment for high deviations from leak
            level_1_mask[np.abs(means - neuron_results_noreset) > 15] = True
            level_2_mask[np.abs(means - neuron_results_noreset) > 30] = True

            # calculate noise
            noise = np.std(neuron_results, axis=0)
            noise[level_1_mask] += 1
            noise[level_2_mask] += 10

            return noise

        return optimization_objective(neuron_results, neuron_results_noreset)

    def postlude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Logs warnings for neurons with higher noise and statistics.
        Change mask of calibration success to False for neurons with
        noise significantly higher than the target.

        :param connection: Connection to the chip to run on.
        """

        log = logger.get("calix.hagen.neuron_leak_bias.LeakBiasCalib")

        # Measure noise again
        builder = base.WriteRecordingPlaybackProgramBuilder()
        results = self.measure_results(connection, builder)

        # Mask too noisy neurons
        high_mask = results > self.masking_threshold
        if np.any(high_mask):
            log.WARN(("Deviations in membrane potential for neurons {0} "
                      + "have exceeded {1}.").format(
                          high_mask.nonzero()[0],
                          self.masking_threshold))
            log.DEBUG("Deviations in read measurements of neurons which "
                      + "exceed the masking threshold: ",
                      results[high_mask])

        self.result.success = ~high_mask

        # Print results
        log.INFO(
            "Calibrated i_bias_leak, obtained noise: "
            + f"{np.mean(results[~high_mask]):5.2f} +- "
            + f"{np.std(results[~high_mask]):4.2f}")

        log.DEBUG("Leak bias currents:" + os.linesep
                  + f"{self.result.calibrated_parameters}")
