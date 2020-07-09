"""
Provides functions to set the neurons' leak OTA bias currents
for different purposes.

The class MembraneTimeConstCalibCADC measures the membrane
time constant and calibrates to set these equal.
This is done before the calibration of synaptic input amplitudes.

The class LeakBiasCalibration is targeted at setting the leak OTA bias
current as low as possible while keeping the noise under control.
This is called after the synaptic inputs are calibrated.
"""

from typing import Optional, Tuple, Union, List
import numbers
import os
import numpy as np
import quantities as pq
from scipy.optimize import curve_fit
from dlens_vx_v2 import hal, sta, halco, logger, hxcomm

from calix.common import algorithms, base, exceptions, madc_base, helpers
from calix.hagen import neuron_potentials, neuron_helpers
from calix import constants


class MembraneTimeConstCalibCADC(base.Calibration):
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
    :ivar leak_calibration: LeakPotentialCalibration class instance used for
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
            neuron_potentials.LeakPotentialCalibration] = None
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

        :raises CalibrationNotSuccessful: If the target CADC read at
            reset potential is below the reliable range of the CADC
            for more than 5% of the neurons.
        """

        log = logger.get(
            "calix.hagen.neuron_leak_bias.MembraneTimeConstCalibCADC")

        # Measure leak potential
        self.target_leak_read = \
            neuron_helpers.cadc_read_neuron_potentials(connection)

        # Create instance of leak potential calibration
        self.leak_calibration = neuron_potentials.LeakPotentialCalibration(
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
            raise exceptions.CalibrationNotSuccessful(
                f"Reset CADC read was obtained at {target_reset_read}, which "
                + "is lower than the reliable range of the "
                + "CADCs for many neurons: "
                + str(np.nonzero(target_reset_read
                                 < constants.cadc_reliable_range.lower)[0])
                + ". Increase the leak voltage or reduce the target "
                + "amplitude.")
        target_reset_read[too_low_mask] = constants.cadc_reliable_range.lower

        calibration = neuron_potentials.ResetPotentialCalibration(
            target=target_reset_read)
        calibration.run(connection, algorithm=algorithms.NoisyBinarySearch())

        # Calculate target CADC read after one time constant
        self.target = self.target_leak_read - (
            self.target_leak_read - target_reset_read) / np.e

    def configure_parameters(self, builder: sta.PlaybackProgramBuilder,
                             parameters: np.ndarray
                             ) -> sta.PlaybackProgramBuilder:
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

        read_tickets = list()
        read_positions = list()

        runs_in_builder = 0
        builder = sta.PlaybackProgramBuilder()
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
                builder = helpers.wait(builder, 100 * pq.us)
                sta.run(connection, builder.done())
                builder = sta.PlaybackProgramBuilder()
                runs_in_builder = 0
            else:
                runs_in_builder += 1

        if not builder.empty():
            builder = helpers.wait(builder, 100 * pq.us)
            sta.run(connection, builder.done())

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

    def measure_results(self, connection: hxcomm.ConnectionHandle,
                        builder: sta.PlaybackProgramBuilder) -> np.ndarray:
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
        sta.run(connection, builder.done())

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


class MembraneTimeConstCalibMADC(madc_base.Calibration):
    """
    Use the fast MADC to calibrate the membrane time constant.

    Due to the MADC's high sample rate, an exponential fit on the decaying
    potential after a reset is used to determine the membrane time
    constant. Leak bias currents are tweaked to reach the desired value.

    This calibration does not alter the neuron configuration, i.e. the
    membrane capacitance and the leak division/multiplication must be
    set suitably beforehand.

    Requirements:
    - None -

    :ivar neuron_config_default: List of desired neuron configurations.
        Necessary to enable leak division/multiplication.
    """

    def __init__(self,
                 target: pq.quantity.Quantity = 60 * pq.us,
                 neuron_configs: Optional[List[hal.NeuronConfig]] = None):
        """
        :param neuron_configs: List of neuron configurations. If None, the
            hagen-mode default neuron config is used.
        """

        super().__init__(
            parameter_range=base.ParameterRange(
                hal.CapMemCell.Value.min, hal.CapMemCell.Value.max),
            inverted=True)

        self.target = target
        if neuron_configs is None:
            self.neuron_config_default = [
                neuron_helpers.neuron_config_default() for _ in
                range(halco.NeuronConfigOnDLS.size)]
        else:
            self.neuron_config_default = neuron_configs

        self.sampling_time = max(70 * pq.us, 8 * np.max(self.target))
        self.wait_between_neurons = 10 * self.sampling_time
        self._wait_before_stimulation = 0 * pq.us

    def prelude(self, connection: hxcomm.ConnectionHandle):
        """
        Prepares chip for calibration.

        Sets reset potential low in order to observe a large decay
        back to the leak potential.

        :param connection: Connection to the chip to calibrate.
        """

        # prepare MADC
        super().prelude(connection)

        builder = sta.PlaybackProgramBuilder()

        # set reset potential low and leak potential high
        builder = helpers.capmem_set_neuron_cells(
            builder, {
                halco.CapMemRowOnCapMemBlock.v_reset: 520,
                halco.CapMemRowOnCapMemBlock.v_leak: 880})
        builder = helpers.wait(builder, constants.capmem_level_off_time)

        # run program
        sta.run(connection, builder.done())

    def configure_parameters(self, builder: sta.PlaybackProgramBuilder,
                             parameters: np.ndarray
                             ) -> sta.PlaybackProgramBuilder:
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

    def stimulate(self, builder: sta.PlaybackProgramBuilder,
                  neuron_coord: halco.NeuronConfigOnDLS,
                  stimulation_time: hal.Timer.Value
                  ) -> sta.PlaybackProgramBuilder:
        """
        Reset the neuron membrane potential to a low voltage.

        :param builder: Builder to append reset instructions to.
        :param neuron_coord: Coordinate of neuron which is currently recorded.
        :param stimulation_time: Timer value at beginning of stimulation.

        :return: Builder with PADI events appended.
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
            self.neuron_config_default[int(neuron_coord.toEnum())])
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

        fit_slice = slice(125, int(self.madc_config.number_of_samples) - 100)
        neuron_fits = list()
        for neuron_id, neuron_data in enumerate(samples):
            neuron_samples = neuron_data[fit_slice]
            try:
                popt, _ = curve_fit(
                    fitfunc,
                    neuron_samples["chip_time"]
                    - neuron_samples["chip_time"][0],
                    neuron_samples["value"], p0=[-120, 10, 420])
            except RuntimeError as error:
                raise exceptions.CalibrationNotSuccessful(
                    f"Fitting to MADC samples failed for neuron {neuron_id}. "
                    + str(error))
            neuron_fits.append(popt[1])  # store time constant of exponential

        return np.array(neuron_fits) * pq.us


class LeakBiasCalibration(base.Calibration):
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

    def configure_parameters(self, builder: sta.PlaybackProgramBuilder,
                             parameters: np.ndarray
                             ) -> sta.PlaybackProgramBuilder:
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
                        builder: sta.PlaybackProgramBuilder
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
        synram_results = list()
        for synram in halco.iter_all(halco.SynramOnDLS):
            synram_results.append(
                neuron_helpers.cadc_read_neurons_repetitive(
                    connection, builder=sta.PlaybackProgramBuilder(),
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

        log = logger.get("calix.hagen.neuron_leak_bias.LeakBiasCalibration")

        # Measure noise again
        builder = sta.PlaybackProgramBuilder()
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
            + "{0:5.2f} +- {1:4.2f}".format(
                np.mean(results[~high_mask]),
                np.std(results[~high_mask])))

        log.DEBUG("Leak bias currents:" + os.linesep
                  + f"{self.result.calibrated_parameters}")
