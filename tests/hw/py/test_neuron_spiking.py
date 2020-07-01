import unittest
from typing import Optional
import numpy as np
import quantities as pq
from dlens_vx_v1 import hal, halco, sta, hxcomm, logger, lola

from calix.common import helpers
from calix.hagen import neuron_helpers
from calix.spiking import neuron
import calix.spiking

from connection_setup import ConnectionSetup


class TestNeuronCalib(ConnectionSetup):
    """
    Runs a neuron calibration and ensures the results are ok.

    Caution: This test does not perform a quantitative analysis of
    the calibration results. We should add a more extensive test
    in the future, see issue 3965.

    :cvar log: Logger used for output.
    :cvar calib_result: Result of calibration, stored for re-applying.
    """

    log = logger.get("calix.tests.hw.test_neuron_spiking")
    calib_result: Optional[calix.spiking.SpikingCalibrationResult] = None

    @classmethod
    def measure_spikes(cls, connection: hxcomm.ConnectionHandle, *,
                       excitatory: bool = True, n_synapse_rows: int = 3,
                       n_events: int = 200,
                       wait_between_events: pq.quantity.Quantity = 2 * pq.us
                       ) -> np.ndarray:
        """
        Sends test inputs to the synapse drivers.
        Records the change in membrane potential of all neurons.

        :param connection: Connection to the chip to run on.
        :param excitatory: Switch between excitatory and inhibitory events.
        :param n_synapse_rows: Number of synapse rows to enable.
        :param n_events: Number of events to send in one run.
        :param wait_between_events: Time between events.

        :return: Number of spikes for each neuron.
        """

        tickets = list()

        builder = sta.PlaybackProgramBuilder()
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
            builder.wait_until(
                halco.TimerOnDLS(),
                int((event * wait_between_events.rescale(pq.us)) * int(
                    hal.Timer.Value.fpga_clock_cycles_per_us)))
            for coord in halco.iter_all(halco.PADIEventOnDLS):
                builder.write(coord, padi_event)

        # Read spike counters
        for coord in halco.iter_all(halco.NeuronConfigOnDLS):
            tickets.append(builder.read(coord.toSpikeCounterReadOnDLS()))

        # Wait for transfers, execute
        builder = helpers.wait(builder, 100 * pq.us)
        sta.run(connection, builder.done())

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

    def test_00_calibration(self):
        """
        Calibrates neurons to the given parameters.
        Test spike response afterwards.
        """

        self.__class__.calib_result = calix.spiking.calibrate(self.connection)
        self.helper_test_spikes()

    def test_01_reapply(self):
        """
        Test spike response after overwrite and re-application of stored calib.
        """

        builder = sta.PlaybackProgramBuilder()
        for neuron_coord in halco.iter_all(halco.AtomicNeuronOnDLS):
            builder.write(neuron_coord, lola.AtomicNeuron())
        sta.run(self.connection, builder.done())

        builder = sta.PlaybackProgramBuilder()
        self.__class__.calib_result.apply(builder)
        sta.run(self.connection, builder.done())

        self.helper_test_spikes()

    def test_02_without_synin(self):
        """
        Calibrate with synaptic input strength uncalibrated.
        Test spike response afterwards.
        """

        neuron.calibrate(
            self.connection,
            i_synin_gm=np.ones(
                halco.NeuronConfigOnDLS.size, dtype=int) * 400)

        self.helper_test_spikes()


if __name__ == "__main__":
    unittest.main()
