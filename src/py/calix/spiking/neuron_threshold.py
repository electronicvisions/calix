from copy import deepcopy
from typing import List, Union, Optional
import numpy as np
import quantities as pq
from dlens_vx_v3 import hal, halco, logger, hxcomm

from calix.common import algorithms, base, cadc_helpers, helpers, madc_base
from calix.hagen import neuron_helpers
from calix import constants
from calix.common.boundary_check import check_range_boundaries


class _SpikeCounterCalib(base.Calib):
    """
    Base class for threshold calibrations that will use the spike
    counters in order to determine the spike rate of all neurons.

    :ivar accumulation_time: Time to record spikes for during calibration.
    :ivar target: Target number of spikes during accumulation time.
        Has to be non-zero as the spike threshold would otherwise
        drift upwards.
    """

    def __init__(self):
        super().__init__(
            parameter_range=base.ParameterRange(
                hal.CapMemCell.Value.min, hal.CapMemCell.Value.max),
            n_instances=halco.NeuronConfigOnDLS.size,
            inverted=True,
            errors=["Spike threshold for neurons {0} has reached {1}."] * 2)

        self.accumulation_time: pq.quantity.Quantity = 2000 * pq.us
        self.target: int = 30  # number of spikes in accumulation time

    def configure_parameters(
            self, builder: base.WriteRecordingPlaybackProgramBuilder,
            parameters: np.ndarray) \
            -> base.WriteRecordingPlaybackProgramBuilder:
        """
        Configure the given spike threshold potentials in the
        given builder.

        :param builder: Builder to append configuration instructions to.
        :param parameters: Threshold potential for each neuron.

        :return: Builder with configuration appended.
        """

        builder = helpers.capmem_set_neuron_cells(
            builder, {halco.CapMemRowOnCapMemBlock.v_threshold: parameters})
        builder = helpers.wait(builder, constants.capmem_level_off_time)

        return builder

    def measure_results(self, connection: hxcomm.ConnectionHandle,
                        builder: base.WriteRecordingPlaybackProgramBuilder
                        ) -> np.ndarray:
        """
        Resets the spike counters, waits for a suitable amount of time,
        and reads the spike counters.

        :param connection: Connection to a chip.
        :param builder: Builder to append read instructions to.

        :return: Array with the number of spikes during the accumulation time.
            If the counter has overflown, the value is set to a high value.
        """

        # run previous program (may be large)
        base.run(connection, builder)
        builder = base.WriteRecordingPlaybackProgramBuilder()

        # Start timing-critical program: reset timer, wait a bit
        initial_wait = 100 * pq.us
        builder = helpers.wait(builder, initial_wait)
        builder.block_until(halco.BarrierOnFPGA(), hal.Barrier.omnibus)

        # Reset spike counters
        for neuron_id, coord in enumerate(
                halco.iter_all(halco.NeuronConfigOnDLS)):
            builder.write(coord.toSpikeCounterResetOnDLS(),
                          hal.SpikeCounterReset())
            builder.block_until(halco.BarrierOnFPGA(), hal.Barrier.omnibus)
            # second write to mach timing with reads below, reads take
            # twice as long as writes
            builder.write(coord.toSpikeCounterResetOnDLS(),
                          hal.SpikeCounterReset())
            builder.block_until(halco.BarrierOnFPGA(), hal.Barrier.omnibus)

        # Wait for accumulation time
        builder.block_until(halco.TimerOnDLS(), hal.Timer.Value(
            int(int(hal.Timer.Value.fpga_clock_cycles_per_us)
                * (self.accumulation_time + initial_wait).rescale(pq.us))))

        # Read spike counters
        tickets = []
        for neuron_id, coord in enumerate(
                halco.iter_all(halco.NeuronConfigOnDLS)):
            tickets.append(builder.read(coord.toSpikeCounterReadOnDLS()))
            builder.block_until(halco.BarrierOnFPGA(), hal.Barrier.omnibus)

        base.run(connection, builder)

        # Analyze results
        results = np.zeros(self.n_instances, dtype=int)
        for neuron_id, ticket in enumerate(tickets):
            result = ticket.get()
            results[neuron_id] = result.count if not result.overflow \
                else hal.SpikeCounterRead.Count.max ** 2

        return results


class NeuronThresholdCalib(_SpikeCounterCalib):
    """
    Calibrate the spike threshold of all neurons to be slightly above the
    resting potential, such that even small inputs can trigger spikes.

    During calibration, a low spike rate is desired, indicating the spike
    threshold roughly matching the resting potential. After a spike,
    the membrane potential is set as low as possible, and the next spike
    should only happen after multiple membrane time constants.

    To prevent self-induced spiking (without excitatory input),
    the spike threshold is increased by a safe_margin after calibration.
    In this state, a single input of medium weight should trigger spikes
    in most neurons, while most neurons should not spike on their own.
    Decreasing the safe_margin allows smaller inputs to trigger spikes,
    at the cost of having more neurons spike without input.

    Requirements:
    * The leak potential is currently set at the desired threshold
      potential.

    Caution:
    * The reset potential is set to the minimum after calibration.

    :ivar safe_margin: Amount to increase spike threshold after
        the calibration. Given in CapMem LSB. The default value of 40
        seems to work for inputs of weight 32, for a hagen-mode-like
        calibration on HICANN-X v2.
    """

    def __init__(self, safe_margin: int = 40):
        super().__init__()
        self.safe_margin = safe_margin

    def prelude(self, connection: hxcomm.ConnectionHandle):
        """
        Configure neurons for easy spiking.

        The membrane capacitance is set low to have a short membrane
        time constant and easily reach the leak potential after a reset.
        The reset potential is set to the minimum as it must be below the
        spike threshold, which is not yet known.

        :param connection: Connection to the chip to configure.
        """

        # Read current neuron configs
        tickets = []
        builder = base.WriteRecordingPlaybackProgramBuilder()
        for coord in halco.iter_all(halco.NeuronConfigOnDLS):
            tickets.append(builder.read(coord))
        base.run(connection, builder)

        # Enable spiking in neurons
        builder = base.WriteRecordingPlaybackProgramBuilder()
        for coord, ticket in zip(
                halco.iter_all(halco.NeuronConfigOnDLS), tickets):
            neuron_config = ticket.get()
            neuron_config.membrane_capacitor_size = 4
            neuron_config.enable_threshold_comparator = True
            neuron_config.enable_fire = True
            builder.write(coord, neuron_config)

        # Set reset potential low, such that threshold is crossed
        builder = helpers.capmem_set_neuron_cells(
            builder, config={
                halco.CapMemRowOnCapMemBlock.v_reset:
                hal.CapMemCell.Value.min})

        builder = helpers.wait(builder, constants.capmem_level_off_time)
        base.run(connection, builder)

    def postlude(self, connection: hxcomm.ConnectionHandle):
        """
        Increase parameters found during calibration by safe margin.
        Measure spikes for one second and log data for neurons which
        still spike frequently if safe_margin is at least 10 (otherwise
        we assume frequent spiking is expected).

        :param connection: Connection to the chip to calibrate.
        """

        # increase parameters, check for CapMem range
        self.result.calibrated_parameters += self.safe_margin
        result = check_range_boundaries(
            self.result.calibrated_parameters, self.parameter_range,
            ["Safe margin could not be added to neuron threshold for "
             + "neurons {0} after reaching value "
             + str(self.parameter_range.lower),
             "Safe margin could not be added to neuron threshold for "
             + "neurons {0} after reaching value "
             + str(self.parameter_range.upper)])
        self.result.calibrated_parameters = result.parameters
        log = logger.get(
            "calix.spiking.neuron_threshold.NeuronThresholdCalib"
            + ".postlude")
        for error in result.messages:
            log.WARN(error)

        builder = base.WriteRecordingPlaybackProgramBuilder()
        builder = self.configure_parameters(
            builder, self.result.calibrated_parameters)
        log.INFO("Calibrated neuron threshold potentials.")

        if self.safe_margin >= 10:
            self.accumulation_time = 1 * pq.s
            results = self.measure_results(connection, builder)

            warn_threshold = 50  # spikes during 1 second
            spiked_neurons = np.arange(halco.NeuronConfigOnDLS.size)[
                results > warn_threshold]
            if np.any(results > warn_threshold):
                log.INFO(
                    f"Neurons {spiked_neurons} have spiked without input. "
                    + "Number of spikes within 1 s: "
                    + f"{results[results > warn_threshold]}. ".replace(
                        # replace overflow value:
                        str(hal.SpikeCounterRead.Count.max ** 2), " >255")
                    + "If too many neurons spike frequently, the safe margin "
                    + "can be increased, sacrificing ease of causal spiking.")


class LeakOverThresholdCalib(_SpikeCounterCalib):
    """
    Calibrate the neurons' spike threshold such that the given
    spike rate is achieved in a leak over threshold setup,
    without any synaptic input.

    Requirements:
    * The leak potential is set sufficiently high and the reset potential
      is set sufficiently low, such that there is some range for the
      threshold potential in between the two.
    * While the target spike rates can be different between neurons, their
      dynamic range is limited to roughly a factor of 10 (i.e. the maximum
      target should not be more than 10 times the minimum target).
      This is due to the precision of the spike counter: Slowly spiking
      neurons would spike only a few times during the accumulation time,
      while quickly spiking neurons would almost saturate the counter.
      You can work around this by running the calibration multiple times
      and cherry-picking from the results.

    :ivar threshold_at_reset: Threshold parameters calibrated at the
        reset potential. Will be used as a lower boundary during the
        calibration to spike rates.
    :ivar parameters: Latest threshold parameters configured on the
        chip, saved during configure_parameters and reused during
        measure_results. There, we compare them against
        threshold_at_reset in order to keep the threshold potential
        above the reset potential.
    """

    def __init__(self, target: pq.Quantity = 100 * pq.kHz):
        """
        :param target: Target spike rate in leak over threshold setup.
        Neurons with a target of 0 are calibrated to an arbitrary firing
        rate between the maximum and minimum requested value.
        :raises ValueError: if the target is negative or all zero.
        """

        super().__init__()

        target = deepcopy(target)  # do not modify given parameter

        # handle possibly different targets
        if not np.any(target):
            raise ValueError(
                "Target spike rates are all zero, indicating none of "
                + "the neurons' thresholds shall be adjusted.")
        if np.any(target < 0):
            raise ValueError("Target spike rate cannot be negative.")
        max_spikerate = np.max(target[target != 0])
        target[target == 0] = max_spikerate

        # maximum expected spike counter value:
        # set to 80% of available counter range to leave headroom
        # for calibration
        max_counter_value = int(hal.SpikeCounterRead.Count.max * 0.8)

        self.accumulation_time = (max_counter_value / max_spikerate).simplified
        self.target = (target * self.accumulation_time).simplified.magnitude

        if np.any(self.target < 0.1 * max_counter_value):
            log = logger.get(
                "calix.spiking.neuron_threshold.LeakOverThresholdCalib")
            log.WARN(
                "Dynamic range of target spike rates spans more than "
                + "a factor of 10. Slow-spiking neurons may not be "
                + "calibrated accurately. You may run the calibration "
                + "multiple times and cherry-pick from the results in "
                + "order to achieve vastly different target spike rates.")

        # due to a bug in the digital neuron backend, we need to ensure
        # the parameters stay above the reset potential. We will use
        # these two ivars for that.
        self.parameters: Optional[np.ndarray] = None
        self.threshold_at_reset: Optional[np.ndarray] = None

    def prelude(self, connection: hxcomm.ConnectionHandle):
        """
        Measure reset potential and store it as a lower boundary for
        the parameter space.

        Since the spike rate will be zero both if the threshold is too
        high (above the leak) or too low (below the reset), we calibrate
        the threshold at reset potential and will later use these value
        as a lower boundary for feasible parameters.

        Note: This is likely caused by a bug in the digital neuron backend.
        Starting with HICANN-X v3, it may be possible to drop this entire
        prelude, as spike rates should then be highest if the threshold
        is lower than the reset.

        :param connection: Connection to the chip to run on.
        """

        # read original neuron config and CapMem values
        builder = base.WriteRecordingPlaybackProgramBuilder()
        neuron_tickets = []
        for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
            neuron_tickets.append(builder.read(coord))

        # read original common neuron backend configs
        common_backend_tickets = []
        for coord in halco.iter_all(halco.CommonNeuronBackendConfigOnDLS):
            common_backend_tickets.append(builder.read(coord))

        # select long reset duration
        config = hal.CommonNeuronBackendConfig()
        config.clock_scale_fast = 9
        config.clock_scale_slow = 9
        for coord in halco.iter_all(halco.CommonNeuronBackendConfigOnDLS):
            builder.write(coord, config)

        config = hal.NeuronBackendConfig()
        config.refractory_time = hal.NeuronBackendConfig.RefractoryTime.max
        for coord in halco.iter_all(halco.NeuronBackendConfigOnDLS):
            builder.write(coord, config)

        # obtain CADC read at reset potential
        reset_tickets = []
        for synram in halco.iter_all(halco.SynramOnDLS):
            builder = neuron_helpers.reset_neurons(builder, synram)
            builder, ticket = cadc_helpers.cadc_read_row(builder, synram)
            reset_tickets.append(ticket)
        base.run(connection, builder)
        reset_cadcread = neuron_helpers.inspect_read_tickets(
            reset_tickets).flatten()

        # calibrate threshold at reset potential
        calibration = ThresholdCalibCADC(target=reset_cadcread)
        self.threshold_at_reset = calibration.run(
            connection, algorithm=algorithms.NoisyBinarySearch()
        ).calibrated_parameters

        log = logger.get(
            "calix.spiking.neuron_threshold.LeakOverThresholdCalib")
        log.DEBUG("Threshold at reset:", self.threshold_at_reset)

        # restore original neuron config and CapMem values
        builder = base.WriteRecordingPlaybackProgramBuilder()
        for coord, ticket in zip(
                halco.iter_all(halco.AtomicNeuronOnDLS), neuron_tickets):
            builder.write(coord, ticket.get())

        # restore original common neuron backend configs
        for coord, ticket in zip(
                halco.iter_all(halco.CommonNeuronBackendConfigOnDLS),
                common_backend_tickets):
            builder.write(coord, ticket.get())

        builder = helpers.wait(builder, constants.capmem_level_off_time)
        base.run(connection, builder)

    def configure_parameters(
            self, builder: base.WriteRecordingPlaybackProgramBuilder,
            parameters: np.ndarray) \
            -> base.WriteRecordingPlaybackProgramBuilder:
        """
        Configure the given spike threshold potentials in the
        given builder.

        Also saves the parameters in an ivar, so we can check them against
        the minimum feasible range (at the reset potential) later.

        Note: Saving the parameters is part of working around a bug in
        the digital neuron backend. On HICANN-X v3, it may be possible
        to use the base function if the bug is fixed.

        :param builder: Builder to append configuration instructions to.
        :param parameters: Threshold potential for each neuron.

        :return: Builder with configuration appended.
        """

        self.parameters = parameters
        return super().configure_parameters(builder, parameters)

    def measure_results(self, connection: hxcomm.ConnectionHandle,
                        builder: base.WriteRecordingPlaybackProgramBuilder
                        ) -> np.ndarray:
        """
        Measures the spike rate without synaptic input.

        Measure number of spikes with spike counters. Also, the
        previously configured parameters are inspected:
        If they indicate the threshold is set below the reset
        potential, a high spike count is returned, in order to have the
        threshold potential be increased by the algorithm.

        Note: The replacing low spike counts with high numbers works
        around a bug in the digital neuron backend. On HICANN-X v3, it
        may be possible to just use the base function, if the bug is fixed.

        :param connection: Connection to a chip.
        :param builder: Builder to append read instructions to.

        :return: Array with the number of spikes during the accumulation time.
            If the counter has overflown, the value is set to a high value.
        """

        results = super().measure_results(connection, builder)

        # The threshold should not be below the reset potential.
        # If it is, we return a high spike count such that the
        # threshold is increased.
        results[self.parameters < self.threshold_at_reset] = \
            hal.SpikeCounterRead.Count.max ** 2
        return results


class ThresholdCalibMADC(madc_base.Calib):
    """
    Calibrate the neurons' spike threshold potential using the MADC.

    A constant current is injected onto the membrane, with the leak
    disabled and the reset potential set low. This means we observe
    regular spikes. The maximum recorded voltage at the MADC is just
    below the threshold potential, we use it as threshold measurement.

    Requirements:
    - None -

    :ivar neuron_configs: List of neuron configs that will be used as
        basis for the `neuron_config_disabled` and `neuron_config_readout`
        functions.
    """

    def __init__(self, target: Union[int, np.ndarray] = 125):
        super().__init__(
            parameter_range=base.ParameterRange(
                hal.CapMemCell.Value.min, hal.CapMemCell.Value.max),
            inverted=False,
            errors=["Spike threshold for neurons {0} has reached {1}."] * 2)

        self.target = target

        self.sampling_time = 200 * pq.us
        self.wait_between_neurons = 10 * self.sampling_time
        self._wait_before_stimulation = 0 * pq.us

        config = neuron_helpers.neuron_config_default()
        config.enable_synaptic_input_excitatory = False
        config.enable_synaptic_input_inhibitory = False
        self.neuron_configs = [
            hal.NeuronConfig(config) for _ in
            halco.iter_all(halco.NeuronConfigOnDLS)]

    def prelude(self, connection: hxcomm.ConnectionHandle):
        """
        Prepares chip for calibration.

        Disable leak, set a low reset voltage and the offset current. Note
        that only the amplitude of the current is set but it is not enabled.

        :param connection: Connection to the chip to calibrate.
        """

        # prepare MADC
        super().prelude(connection)

        builder = base.WriteRecordingPlaybackProgramBuilder()

        # set reset potential low, set constant current, disable leak
        builder = helpers.capmem_set_neuron_cells(
            builder, {
                halco.CapMemRowOnCapMemBlock.i_mem_offset: 250,
                halco.CapMemRowOnCapMemBlock.i_bias_leak: 0,
                halco.CapMemRowOnCapMemBlock.v_reset: 0})
        builder = helpers.wait(builder, constants.capmem_level_off_time)

        # run program
        base.run(connection, builder)

    def configure_parameters(
            self, builder: base.WriteRecordingPlaybackProgramBuilder,
            parameters: np.ndarray) \
            -> base.WriteRecordingPlaybackProgramBuilder:
        """
        Configure the given array of threshold voltages.

        :param builder: Builder to append configuration instructions to.
        :param parameters: Array of threshold voltages to set up.

        :return: Builder with configuration appended.
        """

        builder = helpers.capmem_set_neuron_cells(
            builder, {halco.CapMemRowOnCapMemBlock.v_threshold: parameters})

        builder = helpers.wait(builder, constants.capmem_level_off_time)
        return builder

    def stimulate(self, builder: base.WriteRecordingPlaybackProgramBuilder,
                  neuron_coord: halco.NeuronConfigOnDLS,
                  stimulation_time: hal.Timer.Value
                  ) -> base.WriteRecordingPlaybackProgramBuilder:
        """
        Empty function. The offset current is enabled already in the
        `neuron_config_readout`, therefore no stimuli are neccesary.

        :param builder: Builder to append neuron resets to.
        :param neuron_coord: Coordinate of neuron which is currently recorded.
        :param stimulation_time: Timer value at beginning of stimulation.

        :return: Builder with neuron resets appended.
        """

        config = self.neuron_config_readout(neuron_coord)
        config.enable_threshold_comparator = True
        config.enable_membrane_offset = True
        builder.write(neuron_coord, config)

        # reset neuron - some may be stuck in reset or otherwise broken
        # cf. issue 3996
        builder.write(neuron_coord.toNeuronResetOnDLS(), hal.NeuronReset())

        return builder

    def neuron_config_disabled(self, neuron_coord: halco.NeuronConfigOnDLS
                               ) -> hal.NeuronConfig:
        """
        Return a neuron config with readout disabled.

        The synaptic input and the offset current are disabled as well.

        :param neuron_coord: Coordinate of neuron to get config for.

        :return: Neuron config with readout disabled.
        """

        config = self.neuron_configs[int(neuron_coord)]
        config.enable_readout = False
        config.enable_membrane_offset = False
        return config

    def neuron_config_readout(self, neuron_coord: halco.NeuronConfigOnDLS
                              ) -> hal.NeuronConfig:
        """
        Return a neuron config with readout enabled.
        Also, the offset current and threshold comparator are enabled
        for regular spiking.

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

        For each neuron's MADC samples the maximum is determined which is
        assumed to be near the threshold voltage.

        :param samples: MADC samples obtained for each neuron.

        :return: Numpy array of measured threshold voltages.
        """

        max_reads = np.empty(halco.NeuronConfigOnDLS.size, dtype=int)
        for neuron_id, neuron_samples in enumerate(samples):
            max_reads[neuron_id] = np.max(neuron_samples["value"][50:-50])
        # the first and last 50 samples are cut off since they may contain
        # bad data (the first few MADC samples may not be plausible,
        # the last few samples may already be acquired at the next neuron)

        return max_reads

    def postlude(self, connection: hxcomm.ConnectionHandle):
        """
        Disable offset current CapMem cell again.

        :param connection: Connection to the chip to run on.
        """

        super().postlude(connection)

        builder = base.WriteRecordingPlaybackProgramBuilder()
        builder = helpers.capmem_set_neuron_cells(
            builder, {
                halco.CapMemRowOnCapMemBlock.i_mem_offset: 0})
        builder = helpers.wait(builder, constants.capmem_level_off_time)
        base.run(connection, builder)


class ThresholdCalibCADC(base.Calib):
    """
    Calibrate the neurons' spike threshold potential using the CADC.

    An offset current is injected onto the membranes, which results in
    regular spiking. The CADC samples all neurons at its maximum
    sample rate. The maximum read for each neuron is treated as the
    threshold potential, as it should be just below the threshold.

    We use a high number of CADC samples to ensure we catch the neurons
    at a potential close to the threshold, even with the slow sampling
    rate of the CADC. As long as the sample rate and inter-spike-interval
    is not perfectly aligned, the low sample rate should not be an issue,
    and ultimately electrical noise should ensure the two will never stay
    perfectly synchronous.

    Requirements:
    * Neuron membrane readout is connected to the CADCs (causal and acausal).

    :ivar original_neuron_configs: List of neuron configs from before
        the calibration, used to restore them in the postlude.
    """

    def __init__(self, target: Union[int, np.ndarray] = 125):
        super().__init__(
            parameter_range=base.ParameterRange(
                hal.CapMemCell.Value.min, hal.CapMemCell.Value.max),
            n_instances=halco.NeuronConfigOnDLS.size,
            inverted=False,
            errors=["Spike threshold for neurons {0} has reached {1}."] * 2)

        self.target = target
        self.original_neuron_configs: List[hal.NeuronConfig] = []

    def prelude(self, connection: hxcomm.ConnectionHandle):
        """
        Prepares chip for calibration, reads original neuron config in
        order to restore it after calibration.

        Disable leak, set a low reset voltage and the offset current. Note
        that only the amplitude of the current is set but it is not enabled.

        :param connection: Connection to the chip to calibrate.
        """

        builder = base.WriteRecordingPlaybackProgramBuilder()

        # read original neuron config
        tickets = []
        for coord in halco.iter_all(halco.NeuronConfigOnDLS):
            tickets.append(builder.read(coord))

        # set reset potential low, set constant current, disable leak
        builder = helpers.capmem_set_neuron_cells(
            builder, {
                halco.CapMemRowOnCapMemBlock.i_mem_offset: 70,
                halco.CapMemRowOnCapMemBlock.i_bias_leak: 0,
                halco.CapMemRowOnCapMemBlock.v_reset: 0})
        builder = helpers.wait(builder, constants.capmem_level_off_time)

        # set neuron config suitably
        config = neuron_helpers.neuron_config_default()
        config.enable_synaptic_input_excitatory = False
        config.enable_synaptic_input_inhibitory = False
        config.enable_threshold_comparator = True
        config.enable_membrane_offset = True

        for coord in halco.iter_all(halco.NeuronConfigOnDLS):
            builder.write(coord, config)

        # run program
        base.run(connection, builder)

        # save original neuron configs to ivar
        self.original_neuron_configs = []
        for coord, ticket in zip(
                halco.iter_all(halco.NeuronConfigOnDLS), tickets):
            self.original_neuron_configs.append(ticket.get())

    def configure_parameters(
            self, builder: base.WriteRecordingPlaybackProgramBuilder,
            parameters: np.ndarray) \
            -> base.WriteRecordingPlaybackProgramBuilder:
        """
        Configure the given array of threshold voltages.

        :param builder: Builder to append configuration instructions to.
        :param parameters: Array of threshold voltages to set up.

        :return: Builder with configuration appended.
        """

        builder = helpers.capmem_set_neuron_cells(
            builder, {halco.CapMemRowOnCapMemBlock.v_threshold: parameters})

        builder = helpers.wait(builder, constants.capmem_level_off_time)
        return builder

    def measure_results(
            self, connection: hxcomm.ConnectionHandle,
            builder: base.WriteRecordingPlaybackProgramBuilder) \
            -> np.ndarray:
        """
        Samples the membrane repeatedly and returns the maximum obtained
        value as the voltage measurement closest to the threshold.

        :param connection: Connection to the chip.
        :param builder: Builder to append measurement to.

        :return: Array of near-threshold CADC reads.
        """

        n_samples_per_batch = 1000
        n_batches = 3
        results = np.empty((n_batches, n_samples_per_batch,
                            halco.NeuronConfigOnDLS.size))
        for batch_id in range(n_batches):
            # reset neurons - some may be stuck in reset or otherwise broken
            # cf. issue 3996
            builder = neuron_helpers.reset_neurons(builder)

            results[batch_id, :, :halco.SynapseOnSynapseRow.size] = \
                neuron_helpers.cadc_read_neurons_repetitive(
                    connection, builder, synram=halco.SynramOnDLS.top,
                    n_reads=n_samples_per_batch, wait_time=0 * pq.us)
            results[batch_id, :, halco.SynapseOnSynapseRow.size:] = \
                neuron_helpers.cadc_read_neurons_repetitive(
                    connection, builder, synram=halco.SynramOnDLS.bottom,
                    n_reads=n_samples_per_batch, wait_time=0 * pq.us)

        return np.max(results, axis=(0, 1))

    def postlude(self, connection: hxcomm.ConnectionHandle):
        """
        Disable offset current CapMem cell again and restore original
        neuron configs.

        :param connection: Connection to the chip to run on.
        """

        super().postlude(connection)

        builder = base.WriteRecordingPlaybackProgramBuilder()
        builder = helpers.capmem_set_neuron_cells(
            builder, {
                halco.CapMemRowOnCapMemBlock.i_mem_offset: 0})
        builder = helpers.wait(builder, constants.capmem_level_off_time)

        for coord, config in zip(
                halco.iter_all(halco.NeuronConfigOnDLS),
                self.original_neuron_configs):
            builder.write(coord, config)
        base.run(connection, builder)
