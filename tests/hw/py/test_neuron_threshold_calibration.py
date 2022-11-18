import unittest

import quantities as pq
import numpy as np

from dlens_vx_v3 import hal, halco, sta, logger, hxcomm

from connection_setup import ConnectionSetup

from calix.common import algorithms, base, cadc, helpers
from calix.hagen import neuron, neuron_potentials, neuron_helpers
from calix.spiking import neuron_threshold


log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class NeuronThresholdTest(ConnectionSetup):
    """
    Calibrates chip with neuron spike threshold slightly above
    resting potential. Sends some inputs, asserts the neurons
    respond to them.

    :cvar log: Logger used for outputs.
    :cvar n_events: Number of events to send during final testing.
    :cvar wait_time: Wait time between two events during
        final testing.
    """

    log = logger.get("calix.tests.hw.test_neuron_threshold_calibration")
    n_events = 10
    wait_time = 10 * pq.us

    def test_00_neuron_calibration(self):
        """
        Calibrates neurons for equal membrane time constant and
        synaptic input amplitudes.
        """

        cadc.calibrate(self.connection)

        # using hagen mode calib: set syn. input bias and tau_syn
        # more suitable for spiking operation
        neuron.calibrate(
            self.connection, i_synin_gm=300, tau_syn=2 * pq.us)

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
        tickets = []
        for coord in halco.iter_all(halco.NeuronBackendConfigOnDLS):
            tickets.append(builder.read(coord))
        base.run(connection, builder)

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

        base.run(connection, builder)

    def analyze_spikes(self, program_spikes: sta.PlaybackProgram.spikes_type):
        """
        Inspect the given spikes, as returned by a program,
        and assert they are as expected.

        This function is used to test the NeuronThresholdCalibration,
        which yields a threshold shortly above the leak potential.

        :param program_spikes: Spikes returned by playback program.
        """

        spikes = program_spikes.to_numpy()["chip_time"] / int(
            hal.Timer.Value.fpga_clock_cycles_per_us)

        # have bins to cover twice the event-timeframe (to assert no spikes
        # are recorded afterwards), and two bins per event (to assert one
        # bin with spikes, one without). The unit of time is us.
        bin_stop = float((self.n_events * self.wait_time * 2
                          + 0.1 * pq.us).rescale(pq.us))
        bin_size = float((self.wait_time / 2).rescale(pq.us))
        time_bins = np.arange(0, bin_stop, bin_size)

        # calculate the bins with expected spikes
        spike_times = np.arange(
            0,
            float(self.n_events * self.wait_time.rescale(pq.us)),
            float(self.wait_time.rescale(pq.us)))
        spikes_expected = np.histogram(spike_times, bins=time_bins)[0].astype(
            bool)

        # sort recorded spikes into bins
        spikes_recorded = np.histogram(spikes, bins=time_bins)[0]
        self.__class__.log.DEBUG("Spikes per time-bin:", spikes_recorded)

        # The number of spikes observed after input can be reduced by
        # dropped spike packets since all neurons fire in a short timeframe.
        expected_spikes_required = 250  # per bin when spike is expected
        other_spikes_allowed = 20  # per bin when no spike shoud happen

        # Assert expected spikes are present
        self.assertTrue(
            np.all(spikes_recorded[spikes_expected]
                   > expected_spikes_required),
            "Too few spikes observed in response to inputs: "
            + f"{spikes_recorded[spikes_expected]}")

        # Assert bins without input have only few spikes
        self.assertEqual(
            np.sum(spikes_recorded[~spikes_expected] < other_spikes_allowed),
            len(spikes_recorded) - np.sum(spikes_expected),
            "Too many timeframes without spikes expected have spikes: "
            + f"{spikes_recorded[~spikes_expected]}")

    def test_01_threshold_calibration(self):
        """
        Calibrates spike threshold using the NeuronThresholdCalibration,
        which sets the threshold shortly above the leak potential.

        Send some inputs at relatively low weight, assert neurons
        spike in response to inputs but not otherwise.
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

        # test for spike response
        self.preconfigure(self.connection, weight=32)

        builder = sta.PlaybackProgramBuilder()

        # buffer program for a bit
        builder = helpers.wait(builder, 1000 * pq.us)

        # reset all timers, systime sync
        builder.write(halco.SystimeSyncOnFPGA(), hal.SystimeSync())
        builder.block_until(halco.BarrierOnFPGA(), hal.Barrier.systime)
        builder.write(halco.TimerOnDLS(), hal.Timer())

        # enable recording of spikes
        config = hal.EventRecordingConfig()
        config.enable_event_recording = True
        builder.write(halco.EventRecordingConfigOnFPGA(), config)

        # Send test events to neurons
        padi_event = hal.PADIEvent()
        for bus in halco.iter_all(halco.PADIBusOnPADIBusBlock):
            padi_event.fire_bus[bus] = True  # pylint: disable=unsupported-assignment-operation

        for i in range(self.n_events):
            builder.block_until(
                halco.TimerOnDLS(),
                int(i * self.wait_time.rescale(pq.us) * int(
                    hal.Timer.Value.fpga_clock_cycles_per_us)))
            for synram in halco.iter_all(halco.SynramOnDLS):
                builder.write(synram.toPADIEventOnDLS(), padi_event)

        # disable recording of spikes after some time
        builder = helpers.wait(builder, self.n_events * self.wait_time * 2)
        config = hal.EventRecordingConfig()
        config.enable_event_recording = False
        builder.write(halco.EventRecordingConfigOnFPGA(), config)

        # run program
        program = base.run(self.connection, builder)
        self.__class__.log.DEBUG(f"Got {len(program.spikes)} spikes")

        self.analyze_spikes(program.spikes)

    def test_02_madc_based(self):
        """
        Calibrate the spike threshold via the MADC-based method,
        ThresholdCalibMADC. Use the CADC-based method to measure the
        results.
        """

        for target in range(400, 561, 80):  # threshold in MADC LSB
            # calibrate using MADC
            calibration = neuron_threshold.ThresholdCalibMADC(target=target)
            calibration.run(
                self.connection, algorithm=algorithms.NoisyBinarySearch())

            # verify using CADC
            check = neuron_threshold.ThresholdCalibCADC()
            check.prelude(self.connection)
            results = check.measure_results(
                self.connection, builder=sta.PlaybackProgramBuilder())

            # assert at most 5% of neurons deviate by more than 8 CADC LSB
            # from median threshold of all neurons
            median = np.median(results)
            allowed_deviation = 8  # CADC LSB
            outliers = np.sum([
                results > median + allowed_deviation,
                results < median - allowed_deviation])
            self.assertLess(
                outliers, halco.NeuronConfigOnDLS.size * 0.05,
                "MADC-based threshold calib shows too many outliers "
                + f"at target {target} when checked via the CADC.")

    def test_03_cadc_based(self):
        """
        Calibrate the spike threshold via the CADC-based method,
        ThresholdCalibCADC. Use the MADC-based method to measure the
        results.
        """

        for target in range(100, 161, 30):  # threshold in CADC LSB
            # calibrate using CADC
            calibration = neuron_threshold.ThresholdCalibCADC(target=target)
            calibration.run(
                self.connection, algorithm=algorithms.NoisyBinarySearch())

            # verify using MADC
            check = neuron_threshold.ThresholdCalibMADC()
            check.prelude(self.connection)
            results = check.measure_results(
                self.connection, builder=sta.PlaybackProgramBuilder())
            self.__class__.log.TRACE(
                f"MADC results at target {target}:", results)

            # assert at most 5% of neurons deviate by more than 20 MADC LSB
            # from median threshold of all neurons
            median = np.median(results)
            allowed_deviation = 20  # MADC LSB
            outliers = np.sum([
                results > median + allowed_deviation,
                results < median - allowed_deviation])

            self.__class__.log.INFO(
                f"{outliers} neurons deviated significantly from median.")
            self.assertLess(
                outliers, halco.NeuronConfigOnDLS.size * 0.05,
                "CADC-based threshold calib shows too many outliers when "
                + f"at target {target} when checked via the MADC.")


if __name__ == "__main__":
    unittest.main()
