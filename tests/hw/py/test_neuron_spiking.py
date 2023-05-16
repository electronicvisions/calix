import unittest
from typing import Optional

import numpy as np
import quantities as pq

from dlens_vx_v3 import hal, halco, hxcomm, logger

from connection_setup import ConnectionSetup

from calix.common import base, helpers
from calix.hagen import neuron_helpers
import calix.spiking


log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class TestNeuronCalib(ConnectionSetup):
    """
    Runs a neuron calibration and ensures the results are ok.

    Caution: This test does not perform a quantitative analysis of
    the calibration results. We should add a more extensive test
    in the future, see issue 3965.

    :cvar log: Logger used for output.
    :cvar calib_result: Result of calibration, stored for re-applying.
    :cvar target_equal_rates: Target spike rate in first test with equal
        spikerates for all neurons.
    """

    log = logger.get("calix.tests.hw.test_neuron_spiking")
    calib_result: Optional[calix.spiking.SpikingCalibResult] = None
    target_equal_rates = 100 * pq.kHz

    @classmethod
    def measure_spikes(cls, connection: hxcomm.ConnectionHandle, *,
                       excitatory: bool = True, n_synapse_rows: int = 3,
                       n_events: int = 200,
                       wait_between_events: pq.quantity.Quantity = 2 * pq.us
                       ) -> np.ndarray:
        """
        Sends test inputs to the synapse drivers.
        Records the number of spikes during stimulation.

        Note that the timing of spike counter resets and reads is
        not precise. The last neurons will record spikes for a
        longer duration than the first neurons. If the neurons
        spike by themselves (without stimulation), this may lead
        to an increase of the number of spikes with the neuron id.

        :param connection: Connection to the chip to run on.
        :param excitatory: Switch between excitatory and inhibitory events.
        :param n_synapse_rows: Number of synapse rows to enable.
        :param n_events: Number of events to send in one run.
        :param wait_between_events: Time between events.

        :return: Number of spikes for each neuron.
        """

        tickets = []

        builder = base.WriteRecordingPlaybackProgramBuilder()
        builder = neuron_helpers.enable_all_synapse_drivers(
            builder, row_mode=hal.SynapseDriverConfig.RowMode.excitatory
            if excitatory else hal.SynapseDriverConfig.RowMode.inhibitory)
        builder = neuron_helpers.configure_synapses(
            builder, n_synapse_rows, weight=hal.SynapseQuad.Weight.max)

        # Send events on PADI bus
        padi_event = hal.PADIEvent()
        for coord in halco.iter_all(halco.PADIBusOnPADIBusBlock):
            padi_event.fire_bus[coord] = True  # pylint: disable=unsupported-assignment-operation

        # Reset spike counters
        for coord in halco.iter_all(halco.NeuronConfigOnDLS):
            builder.write(coord.toSpikeCounterResetOnDLS(),
                          hal.SpikeCounterReset())

        builder.write(halco.TimerOnDLS(), hal.Timer())

        for event in range(n_events):  # send some spikes
            builder.block_until(
                halco.TimerOnDLS(),
                hal.Timer.Value(
                    int((event * wait_between_events.rescale(pq.us)) * int(
                        hal.Timer.Value.fpga_clock_cycles_per_us))))
            for coord in halco.iter_all(halco.PADIEventOnDLS):
                builder.write(coord, padi_event)

        # Read spike counters
        for coord in halco.iter_all(halco.NeuronConfigOnDLS):
            tickets.append(builder.read(coord.toSpikeCounterReadOnDLS()))

        base.run(connection, builder)

        results = np.empty(halco.NeuronConfigOnDLS.size, dtype=int)
        for neuron_id, ticket in enumerate(tickets):
            result = ticket.get()
            results[neuron_id] = result.count if not result.overflow \
                else hal.SpikeCounterRead.Count.max ** 2

        return results

    def helper_test_spikes(self):
        """
        Stimulate neurons in excitatory and inhibitory fashion.
        Assert excitatory stimuli increase spike frequency.
        """

        exc_spikes = self.measure_spikes(
            self.connection, excitatory=True)
        self.log.DEBUG("Excitatory:", exc_spikes)

        inh_spikes = self.measure_spikes(
            self.connection, excitatory=False)
        self.log.DEBUG("Inhibitory:", inh_spikes)

        self.assertGreater(
            np.sum(exc_spikes > 10), np.sum(inh_spikes > 10),
            "Stimulation seems not to affect neurons.")

    @staticmethod
    def measure_spikerate(
            connection: hxcomm.ConnectionHandle, *,
            accumulation_time: pq.Quantity = 1.5 * pq.ms) -> pq.Quantity:
        """
        Measure spike rate for a specified accumulation time.

        Since we wait precisely for the accumulation_time for each neuron,
        the timing is much more precise than if we were using
        measure_spikes() without any input.

        :param connection: Connection to the chip to run on.
        :param accumulation_time: Wait time for each neuron, between spike
            counter reset and read.

        :return: Array of spike rates, i.e. the number of spikes divided
            by the accumulation time. If the counter overflowed, we
            return np.nan for those neurons.
        """

        tickets = []
        builder = base.WriteRecordingPlaybackProgramBuilder()

        for coord in halco.iter_all(halco.NeuronConfigOnDLS):
            # Reset spike counters
            builder.write(coord.toSpikeCounterResetOnDLS(),
                          hal.SpikeCounterReset())

            # wait for accumulation time
            helpers.wait(builder, accumulation_time)

            # Read spike counters
            tickets.append(builder.read(coord.toSpikeCounterReadOnDLS()))

        base.run(connection, builder)

        # evaluate results: calculate spike rate
        results = np.empty(halco.NeuronConfigOnDLS.size, dtype=float)
        for neuron_id, ticket in enumerate(tickets):
            result = ticket.get()
            results[neuron_id] = result.count \
                if not result.overflow else np.nan

        return results / accumulation_time

    def test_00_calibration(self):
        """
        Load calibration and test spike response.
        """

        self.__class__.calib_result = self.apply_calibration("spiking")
        self.helper_test_spikes()

    def test_01_leak_over_threshold(self):
        """
        Calibrate threshold for given firing rates. Use an equal rate
        for all neurons.
        """

        # Increase range between reset and leak, to place threshold
        calix.spiking.neuron.refine_potentials(
            self.connection, self.__class__.calib_result.neuron_result,
            calix.spiking.neuron.NeuronCalibTarget(
                leak=110, reset=60, threshold=120))

        # Calibrate threshold for equal firing rates
        calix.spiking.neuron.calibrate_leak_over_threshold(
            self.connection, self.__class__.calib_result.neuron_result,
            leak_over_threshold_rate=self.target_equal_rates)

        # Measure spikes based on spike counters in neuron backend
        rate_without_stimulation = self.measure_spikerate(self.connection)
        self.log.DEBUG("Rate without stimulation, measured with spike "
                       + "counters:\n", rate_without_stimulation)
        self.assertGreater(
            np.sum(rate_without_stimulation > 0.9 * self.target_equal_rates),
            halco.NeuronConfigOnDLS.size * 0.9,
            "Too few spikes in leak over threshold setup without syn. input.")
        self.assertGreater(
            np.sum(rate_without_stimulation < 1.15 * self.target_equal_rates),
            halco.NeuronConfigOnDLS.size * 0.9,
            "Too many spikes in leak over threshold setup without syn. input.")

        # Measure spikes with stimulation
        self.helper_test_spikes()

    def test_02_lot_variable_targets(self):
        """
        Calibrate threshold for different firing rates, and only part of
        the neurons. The other neurons are left untouched.
        """

        # Initial config: leak below threshold
        calix.spiking.neuron.refine_potentials(
            self.connection, self.__class__.calib_result.neuron_result,
            calix.spiking.neuron.NeuronCalibTarget(
                leak=110, reset=60, threshold=120))
        original_thresholds = np.empty(halco.NeuronConfigOnDLS.size, dtype=int)
        for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
            original_thresholds[int(coord.toEnum())] = \
                self.__class__.calib_result.neuron_result.neurons[
                    coord].threshold.v_threshold

        # Calibrate threshold for different firing rates
        targets = np.linspace(20, 120, 16)  # 32 neurons per target
        targets = np.repeat(targets, halco.NeuronConfigOnDLS.size / 16)
        targets[::4] = 0  # leave every 4th neuron untouched
        targets = targets * pq.kHz

        calix.spiking.neuron.calibrate_leak_over_threshold(
            self.connection, self.__class__.calib_result.neuron_result,
            leak_over_threshold_rate=targets)

        # Measure spikes without stimulation
        rate_without_stimulation = self.measure_spikerate(self.connection)
        self.log.DEBUG("Rate without stimulation:", rate_without_stimulation)
        success = np.isclose(
            rate_without_stimulation.rescale(pq.kHz).magnitude,
            targets.rescale(pq.kHz).magnitude, atol=8, rtol=0.08)
        self.assertGreater(
            np.sum(success), halco.NeuronConfigOnDLS.size * 0.85,
            "Too large deviations from target spike rate in leak over "
            + "threshold setup with variable targets.")

        # Assert every 4th neuron is untouched
        new_thresholds = np.empty(halco.NeuronConfigOnDLS.size, dtype=int)
        for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
            new_thresholds[int(coord.toEnum())] = \
                self.__class__.calib_result.neuron_result.neurons[
                    coord].threshold.v_threshold
        np.testing.assert_array_equal(
            new_thresholds[::4], original_thresholds[::4],
            err_msg="Threshold parameter was changed for neurons that "
            + "were not to be calibrated.")

    def test_03_without_synin_calib(self):
        """
        Calibrate with synaptic input strength uncalibrated.
        Test spike response afterwards.
        """

        target = calix.spiking.SpikingCalibTarget()
        target.neuron_target.i_synin_gm = np.ones(
            halco.NeuronConfigOnDLS.size, dtype=int) * 600

        calix.calibrate(target,
                        cache_paths=[],  # don't cache in tests
                        connection=self.connection)

        self.helper_test_spikes()

    def test_04_dense_default(self):
        """
        Calibrate with dense default neuron targets.

        Also, we check whether the dense default target is complete,
        in the sense that every parameter that is configurable per neuron
        is an array and shaped accordingly.
        """

        target = calix.spiking.SpikingCalibTarget()
        neuron_target = calix.spiking.neuron.NeuronCalibTarget().DenseDefault

        # test shape of all contained targets: They must be an array and
        # one dimension must match the number of neurons.
        for name, value in vars(neuron_target).items():
            # Skip check for variables where configuration per neuron
            # is not feasible:
            # - synapse_dac_bias is configurable only per quadrant
            # - i_synin_gm is a single target for all neurons as it is
            #   already a CapMem parameter.
            if name in ["synapse_dac_bias", "i_synin_gm"]:
                continue

            if not isinstance(value, np.ndarray):
                raise TypeError(
                    "The spiking neuron calib target dense default "
                    f"contains a parameter {name} that is not a numpy "
                    f"array: {value}.")
            if halco.NeuronConfigOnDLS.size not in value.shape:
                raise ValueError(
                    "The spiking neuron calib target dense default "
                    f"contains a parameter {name} that does not represent "
                    f"the number of neurons in its shape: {value.shape}.")

        target.neuron_target = neuron_target

        calix.calibrate(target,
                        cache_paths=[],  # don't cache in tests
                        connection=self.connection)

        self.helper_test_spikes()


if __name__ == "__main__":
    unittest.main()
