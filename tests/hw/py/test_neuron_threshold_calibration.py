import unittest
import numpy as np
from dlens_vx_v1 import hal, halco, sta, logger, hxcomm

from calix.common import algorithms, cadc, helpers
from calix.hagen import neuron, neuron_potentials, neuron_helpers
from calix.spiking import neuron_threshold

from connection_setup import ConnectionSetup


class NeuronThresholdTest(ConnectionSetup):
    """
    Calibrates chip with neuron spike threshold slightly above
    resting potential. Sends some inputs, asserts the neurons
    respond to them.

    :cvar log: Logger used for outputs.
    :cvar n_events: Number of events to send during final testing.
    :cvar wait_time: Wait time in us between two events during
        final testing.
    """

    log = logger.get("calix.tests.hw.test_neuron_threshold_calibration")
    n_events = 10
    wait_time = 10  # us between two events

    def test_00_neuron_calibration(self):
        """
        Calibrates neurons for equal membrane time constant and
        synaptic input amplitudes.
        """

        cadc.calibrate(self.connection)

        # Increase i_synin_gm a bit to facilitate spiking.
        # Set target_noise to None in order to keep the membrane time
        # constant calibrated, and not optimize for noise.
        # Set tau_syn in order to have it calibrated and not just configured
        # to the minimum possible setting.
        neuron.calibrate(
            self.connection, i_synin_gm=140, target_noise=None, tau_syn=2.)

    def test_01_threshold_calibration(self):
        """
        Calibrates spike threshold.
        """

        # calibrate spike threshold to be shortly above leak
        calibration = neuron_threshold.NeuronThresholdCalibration()
        calibration.run(
            self.connection, algorithm=algorithms.NoisyBinarySearch())

        # Calibrate reset potential to be equal to leak
        calibration = neuron_potentials.ResetPotentialCalibration(
            highnoise=True)
        calibration.run(
            self.connection, algorithm=algorithms.NoisyBinarySearch())

    @staticmethod
    def preconfigure(connection: hxcomm.ConnectionHandle,
                     weight: hal.SynapseQuad.Weight):
        """
        Configure neurons and inputs for spike test.

        Enables spike output in neuron backend config and sets a suitable
        output address. Enables one row of synapses at given weight.

        :param connection: Connection to the chip to configure.
        :param weight: Weight of synaptic input.
        """

        # Read neuron backend configs
        builder = sta.PlaybackProgramBuilder()
        tickets = list()
        for coord in halco.iter_all(halco.NeuronBackendConfigOnDLS):
            tickets.append(builder.read(coord))
        builder = helpers.wait_for_us(builder, 100)
        sta.run(connection, builder.done())

        # Write modified neuron backend configs
        builder = sta.PlaybackProgramBuilder()
        for coord, ticket in zip(
                halco.iter_all(halco.NeuronBackendConfigOnDLS), tickets):
            config = ticket.get()
            config.enable_spike_out = True
            builder.write(coord, config)

        # Enable all synapse drivers
        builder = neuron_helpers.enable_all_synapse_drivers(
            builder, row_mode=hal.SynapseDriverConfig.RowMode.excitatory)

        # Set synapses
        builder = neuron_helpers.configure_synapses(
            builder, n_synapse_rows=1, weight=weight)
        builder = neuron_helpers.configure_stp_and_padi(builder)

        sta.run(connection, builder.done())

    def analyze_spikes(self, program_spikes: sta.PlaybackProgram.spikes_type):
        """
        Inspect the given spikes, as returned by a program,
        and assert they are as expected.

        :param program_spikes: Spikes returned by playback program.
        """

        spikes = program_spikes.to_numpy()["chip_time"] / int(
            hal.Timer.Value.fpga_clock_cycles_per_us)

        # have bins to cover twice the event-timeframe (to assert no spikes
        # are recorded afterwards), and two bins per event (to assert one
        # bin with spikes, one without). The unit of time is us.
        time_bins = np.arange(
            0, self.n_events * self.wait_time * 2 + 0.1, self.wait_time / 2)

        # calculate the bins with expected spikes
        spike_times = np.arange(
            0, self.n_events * self.wait_time, self.wait_time)
        spikes_expected = np.histogram(spike_times, bins=time_bins)[0].astype(
            np.bool)

        # sort recorded spikes into bins
        spikes_recorded = np.histogram(spikes, bins=time_bins)[0]
        self.__class__.log.DEBUG("Spikes per time-bin:", spikes_recorded)

        # The number of spikes observed after input can be reduced by
        # dropped spike packets since all neurons fire in a short timeframe.
        expected_spikes_required = 200  # per bin when spike is expected
        other_spikes_allowed = 100  # per bin when no spike shoud happen

        # Assert expected spikes are present
        self.assertTrue(
            np.all(spikes_recorded[spikes_expected]
                   > expected_spikes_required),
            "Too few spikes observed in response to inputs: "
            + f"{spikes_recorded[spikes_expected]}")

        # Assert bins without input have only few spikes
        self.assertGreater(
            np.sum(spikes_recorded[~spikes_expected] < other_spikes_allowed),
            len(spikes_recorded) - np.sum(spikes_expected) - 3,
            "Too many timeframes without spikes expected have spikes: "
            + f"{spikes_recorded[~spikes_expected]}")

    def test_02_spikes(self):
        """
        Send some inputs at relatively low weight, assert neurons
        spike in response to inputs but not otherwise.
        """

        self.preconfigure(self.connection, weight=32)

        builder = sta.PlaybackProgramBuilder()

        # buffer program for a bit
        initial_wait = 1000  # us
        builder = helpers.wait_for_us(builder, initial_wait)

        # reset all timers
        builder.write(halco.TimerOnDLS(), hal.Timer())
        builder.write(halco.SystimeSyncOnFPGA(), hal.SystimeSync())

        # enable recording of spikes
        config = hal.EventRecordingConfig()
        config.enable_event_recording = True
        builder.write(halco.EventRecordingConfigOnFPGA(), config)

        # Send test events to neurons
        padi_event = hal.PADIEvent()
        for bus in halco.iter_all(halco.PADIBusOnPADIBusBlock):
            padi_event.fire_bus[bus] = True  # pylint: disable=unsupported-assignment-operation

        for i in range(self.n_events):
            builder.wait_until(
                halco.TimerOnDLS(),
                int((i * self.wait_time) * int(
                    hal.Timer.Value.fpga_clock_cycles_per_us)))
            for synram in halco.iter_all(halco.SynramOnDLS):
                builder.write(synram.toPADIEventOnDLS(), padi_event)

        # disable recording of spikes after some time
        builder = helpers.wait_for_us(
            builder, self.n_events * self.wait_time * 2)
        config = hal.EventRecordingConfig()
        config.enable_event_recording = False
        builder.write(halco.EventRecordingConfigOnFPGA(), config)

        # run program
        program = builder.done()
        sta.run(self.connection, program)
        self.__class__.log.DEBUG("Got {0} spikes".format(len(program.spikes)))

        self.analyze_spikes(program.spikes)


if __name__ == "__main__":
    unittest.main()
