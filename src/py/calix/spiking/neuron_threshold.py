from typing import List, Union
import numpy as np
import quantities as pq
from dlens_vx_v2 import hal, halco, sta, logger, hxcomm

from calix.common import base, helpers, madc_base
from calix.hagen import neuron_helpers
from calix import constants
from calix.common.boundary_check import check_range_boundaries


class NeuronThresholdCalibration(base.Calibration):
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
    :ivar accumulation_time: Time to record spikes for during calibration.
    :ivar target: Target number of spikes during accumulation time.
        Has to be non-zero as the spike threshold would otherwise
        drift upwards, not allowing small inputs to trigger a spike.
    """

    def __init__(self, safe_margin: int = 40):
        super().__init__(
            parameter_range=base.ParameterRange(
                hal.CapMemCell.Value.min, hal.CapMemCell.Value.max),
            n_instances=halco.NeuronConfigOnDLS.size,
            inverted=True,
            errors=["Spike threshold for neurons {0} has reached {1}."] * 2)
        self.safe_margin = safe_margin
        self.accumulation_time: pq.quantity.Quantity = 2000 * pq.us
        self.target: int = 30  # number of spikes in accumulation time

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
        tickets = list()
        builder = sta.PlaybackProgramBuilder()
        for coord in halco.iter_all(halco.NeuronConfigOnDLS):
            tickets.append(builder.read(coord))
        base.run(connection, builder)

        # Enable spiking in neurons
        builder = sta.PlaybackProgramBuilder()
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

    def configure_parameters(self, builder: sta.PlaybackProgramBuilder,
                             parameters: np.ndarray
                             ) -> sta.PlaybackProgramBuilder:
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
                        builder: sta.PlaybackProgramBuilder
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
        builder = sta.PlaybackProgramBuilder()

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
        builder.wait_until(halco.TimerOnDLS(), hal.Timer.Value(
            int(int(hal.Timer.Value.fpga_clock_cycles_per_us)
                * (self.accumulation_time + initial_wait).rescale(pq.us))))

        # Read spike counters
        tickets = list()
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
            "calix.spiking.neuron_threshold.NeuronThresholdCalibration"
            + ".postlude")
        for error in result.messages:
            log.WARN(error)

        builder = sta.PlaybackProgramBuilder()
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


class ThresholdCalibMADC(madc_base.Calibration):
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

        builder = sta.PlaybackProgramBuilder()

        # set reset potential low, set constant current, disable leak
        builder = helpers.capmem_set_neuron_cells(
            builder, {
                halco.CapMemRowOnCapMemBlock.i_mem_offset: 250,
                halco.CapMemRowOnCapMemBlock.i_bias_leak: 0,
                halco.CapMemRowOnCapMemBlock.v_reset: 450})
        builder = helpers.wait(builder, constants.capmem_level_off_time)

        # run program
        base.run(connection, builder)

    def configure_parameters(self, builder: sta.PlaybackProgramBuilder,
                             parameters: np.ndarray
                             ) -> sta.PlaybackProgramBuilder:
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

    def stimulate(self, builder: sta.PlaybackProgramBuilder,
                  neuron_coord: halco.NeuronConfigOnDLS,
                  stimulation_time: hal.Timer.Value
                  ) -> sta.PlaybackProgramBuilder:
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


class ThresholdCalibCADC(base.Calibration):
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
    """

    def __init__(self, target: Union[int, np.ndarray] = 125):
        super().__init__(
            parameter_range=base.ParameterRange(
                hal.CapMemCell.Value.min, hal.CapMemCell.Value.max),
            n_instances=halco.NeuronConfigOnDLS.size,
            inverted=False,
            errors=["Spike threshold for neurons {0} has reached {1}."] * 2)

        self.target = target

    def prelude(self, connection: hxcomm.ConnectionHandle):
        """
        Prepares chip for calibration.

        Disable leak, set a low reset voltage and the offset current. Note
        that only the amplitude of the current is set but it is not enabled.

        :param connection: Connection to the chip to calibrate.
        """

        builder = sta.PlaybackProgramBuilder()

        # set reset potential low, set constant current, disable leak
        builder = helpers.capmem_set_neuron_cells(
            builder, {
                halco.CapMemRowOnCapMemBlock.i_mem_offset: 70,
                halco.CapMemRowOnCapMemBlock.i_bias_leak: 0,
                halco.CapMemRowOnCapMemBlock.v_reset: 450})
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

    def configure_parameters(self, builder: sta.PlaybackProgramBuilder,
                             parameters: np.ndarray
                             ) -> sta.PlaybackProgramBuilder:
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

    def measure_results(self, connection: hxcomm.ConnectionHandle,
                        builder: sta.PlaybackProgramBuilder) -> np.ndarray:
        """
        Samples the membrane repeatedly and returns the maximum obtained
        value as the voltage measurement closest to the threshold.

        :param connection: Connection to the chip.
        :param builder: Builder to append measurement to.

        :return: Array of near-threshold CADC reads.
        """

        n_samples = 2000
        results = np.empty((n_samples, halco.NeuronConfigOnDLS.size))
        results[:, :halco.SynapseOnSynapseRow.size] = \
            neuron_helpers.cadc_read_neurons_repetitive(
                connection, builder, synram=halco.SynramOnDLS.top,
                n_reads=n_samples, wait_time=0 * pq.us)
        results[:, halco.SynapseOnSynapseRow.size:] = \
            neuron_helpers.cadc_read_neurons_repetitive(
                connection, builder, synram=halco.SynramOnDLS.bottom,
                n_reads=n_samples, wait_time=0 * pq.us)

        return np.max(results, axis=0)
