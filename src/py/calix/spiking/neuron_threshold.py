import numpy as np
import quantities as pq
from dlens_vx_v2 import hal, halco, sta, logger, hxcomm

from calix.common import base, helpers
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
        builder = helpers.wait(builder, 100 * pq.us)
        sta.run(connection, builder.done())

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
        sta.run(connection, builder.done())

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
        sta.run(connection, builder.done())
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

        # End of timing-critical program: Wait for transfers, run program
        builder = helpers.wait(builder, 100 * pq.us)
        sta.run(connection, builder.done())

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
