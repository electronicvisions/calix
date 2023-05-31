import unittest
from typing import Tuple

import numpy as np
import quantities as pq
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from dlens_vx_v3 import hal, halco, logger

from connection_setup import ConnectionSetup

import calix.spiking
from calix.common import algorithms, base, madc_base
from calix.hagen import neuron_helpers, neuron_potentials


log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class TestNeuronCalib(ConnectionSetup):
    """
    Calibrate neurons for COBA modulation. Assert they behave as
    expected.

    :cvar n_neurons_in_coba_mode: Number of neurons to calibrate
        in COBA mode. The remaining neurons will use CUBA mode and
        assert that there is no modulation.
    :cvar v_leaks: Array of v_leak targets to calibrate before
        measuring.
    :cvar exc_baselines: Baselines (resting potentials) of excitatory
        amplitude measurements.
    :cvar exc_amplitudes: Excitatory amplitude measurements.
    :cvar inh_baselines: Baselines (resting potentials) of inhibitory
        amplitude measurements.
    :cvar inh_amplitudes: Inhibitory amplitude measurements.
    """

    # Use COBA mode for most neurons, but disable it for a few
    n_neurons_in_coba_mode = 490

    v_leaks = np.arange(40, 161, 20)
    exc_baselines: np.ndarray = \
        np.empty((len(v_leaks), halco.NeuronConfigOnDLS.size))
    exc_amplitudes: np.ndarray = \
        np.empty((len(v_leaks), halco.NeuronConfigOnDLS.size))
    inh_baselines: np.ndarray = \
        np.empty((len(v_leaks), halco.NeuronConfigOnDLS.size))
    inh_amplitudes: np.ndarray = \
        np.empty((len(v_leaks), halco.NeuronConfigOnDLS.size))

    def test_00_calibrate_and_measure(self):
        """
        Run calibration and measure the amplitudes.
        """

        e_coba_reversal = np.array([
            np.ones(halco.NeuronConfigOnDLS.size) * 300,
            np.ones(halco.NeuronConfigOnDLS.size) * 30])
        e_coba_reversal[0, self.__class__.n_neurons_in_coba_mode:] = np.inf
        e_coba_reversal[1, self.__class__.n_neurons_in_coba_mode:] = -np.inf

        self.__class__.calib_result = calix.calibrate(
            calix.spiking.SpikingCalibTarget(
                neuron_target=calix.spiking.neuron.NeuronCalibTarget(
                    i_synin_gm=np.array([180, 250]),
                    e_coba_reference=np.array([150, np.nan]),
                    e_coba_reversal=e_coba_reversal)),
            cache_paths=[],  # don't cache in tests
            connection=self.connection)

        self.__class__.exc_baselines, self.__class__.exc_amplitudes = \
            self.measure_amplitudes(excitatory=True)
        self.__class__.inh_baselines, self.__class__.inh_amplitudes = \
            self.measure_amplitudes(excitatory=False)

    class Recorder(madc_base.MembraneRecorder):
        """
        MADC Recorder for measuring amplitudes.
        """

        def __init__(self):
            super().__init__()
            self._wait_before_stimulation = 10 * pq.us
            self.sampling_time = 100 * pq.us

        def stimulate(self,
                      builder: base.WriteRecordingPlaybackProgramBuilder,
                      neuron_coord: halco.NeuronConfigOnDLS,
                      stimulation_time: hal.Timer.Value
                      ) -> base.WriteRecordingPlaybackProgramBuilder:
            padi_event = hal.PADIEvent()
            for bus in halco.iter_all(halco.PADIBusOnPADIBusBlock):
                padi_event.fire_bus[bus] = True
            builder.write(
                neuron_coord.toSynramOnDLS().toPADIEventOnDLS(),
                padi_event)

            return builder

    # pylint: disable=too-many-locals
    def measure_amplitudes(
            self, excitatory: bool = True) -> Tuple[
                np.ndarray, np.ndarray]:
        """
        Measure amplitudes depending on leak potentials.

        :param excitatory: Switch between excitatory and inhibitory mode.

        :return: Array of baselines and amplitudes, per v_leak and per neuron.
        """

        # disable threshold
        builder = base.WriteRecordingPlaybackProgramBuilder()
        tickets = []
        for coord in halco.iter_all(halco.NeuronConfigOnDLS):
            tickets.append(builder.read(coord))
        base.run(self.connection, builder)

        builder = base.WriteRecordingPlaybackProgramBuilder()
        for coord, ticket in zip(
                halco.iter_all(halco.NeuronConfigOnDLS), tickets):
            config = ticket.get()
            config.enable_threshold_comparator = False
            builder.write(coord, config)

        # set synapses for stimulus
        neuron_helpers.configure_stp_and_padi(builder)
        neuron_helpers.enable_all_synapse_drivers(
            builder,
            row_mode=hal.SynapseDriverConfig.RowMode.excitatory if excitatory
            else hal.SynapseDriverConfig.RowMode.inhibitory)
        builder = neuron_helpers.configure_synapses(
            builder,
            n_synapse_rows=4,
            weight=hal.SynapseQuad.Weight(32))
        base.run(self.connection, builder)

        baselines = np.empty((len(self.__class__.v_leaks),
                              halco.NeuronConfigOnDLS.size))
        amplitudes = np.empty((len(self.__class__.v_leaks),
                               halco.NeuronConfigOnDLS.size))
        for v_leak_id, v_leak in enumerate(self.__class__.v_leaks):
            # Calibrate leak:
            # Generally, we cannot expect the leak calibration to work
            # well with COBA mode enabled, since the large swings in
            # resting potential during the leak potential calib may lead
            # to constant currents onto the membrane by the CUBA OTA,
            # which gets its bias current changed (by the COBA OTA) but
            # its reference potential is not adjusted.
            # Here in the test, though, we use a COBA modulation that is
            # not too strong, so the CUBA OTA's bias current is not changed
            # that much, and we don't expect ideal results from the
            # leak calibration.
            leak_calibration = neuron_potentials.LeakPotentialCalib(
                v_leak)
            leak_calibration.run(
                self.connection, algorithm=algorithms.NoisyBinarySearch())

            recorder = self.Recorder()
            recorder.prepare_recording(self.connection)
            samples = recorder.record_traces(
                self.connection,
                builder=base.WriteRecordingPlaybackProgramBuilder())

            for neuron_id, neuron_samples in enumerate(samples):
                neuron_samples = neuron_samples["value"]
                baselines[v_leak_id, neuron_id] = \
                    np.mean(neuron_samples[-100:])
                amplitudes[v_leak_id, neuron_id] = \
                    (np.max(neuron_samples) if excitatory
                     else np.min(neuron_samples))

        return baselines, amplitudes - baselines

    def plot_results(self):
        """
        Save a plot of the obtained amplitudes.
        """

        plt.figure(figsize=(9, 6))
        plt.plot(self.__class__.exc_baselines, self.__class__.exc_amplitudes,
                 'x', markersize=0.3, alpha=0.5, color="C0")
        plt.plot(self.__class__.inh_baselines, self.__class__.inh_amplitudes,
                 'x', markersize=0.3, alpha=0.5, color="C1")
        plt.axhline(0, linewidth=0.5)
        plt.xlabel("Resting potential [MADC LSB]")
        plt.ylabel("Observed amplitude [MADC LSB]")
        plt.savefig("coba_modulation.png", dpi=300)
        plt.close()

    def test_01_modulation(self):
        """
        Estimate amplitudes from an MADC trace, assert the results show
        COBA modulation for the neurons calibrated accordingly.
        Also, assert there is no modulation for neurons in CUBA mode.
        """

        self.plot_results()

        # fit linear functions to amplitudes
        def fitfunc(resting_potential, slope, x_offset):
            return slope * (resting_potential - x_offset)

        fits = []
        for neuron_id in range(halco.NeuronConfigOnDLS.size):
            fits.append(curve_fit(
                fitfunc, self.__class__.exc_baselines[:, neuron_id],
                self.__class__.exc_amplitudes[:, neuron_id],
                p0=(-0.3, 350), bounds=([-10, -10000], [10, 10000]))[0])
        fits = np.array(fits)

        # check slope: lose at least 5 MADC LSB of amplitude
        # over 100 MADC LSB of resting potential.
        # The value will depend on the desired modulation strength, which
        # is determined by target reference and reversal potentials.
        success = fits[:self.__class__.n_neurons_in_coba_mode, 0] < -5 / 100
        self.assertGreater(
            np.sum(success), self.__class__.n_neurons_in_coba_mode * 0.9,
            "Excitatory amplitudes decrease too slowly.")

        success = np.all(
            [fits[self.__class__.n_neurons_in_coba_mode:, 0] > -8 / 100,
             fits[self.__class__.n_neurons_in_coba_mode:, 0] < 8 / 100],
            axis=0)
        self.assertGreater(
            np.sum(success),
            (halco.NeuronConfigOnDLS.size
             - self.__class__.n_neurons_in_coba_mode) * 0.9,
            "CUBA neurons show modulation of excitatory amplitudes.")

        # check x_offset: Expect it near e_coba_reversal, but in MADC units
        success = np.all(
            [fits[:self.__class__.n_neurons_in_coba_mode, 1] > 650,
             fits[:self.__class__.n_neurons_in_coba_mode, 1] < 1500], axis=0)
        self.assertGreater(
            np.sum(success), self.__class__.n_neurons_in_coba_mode * 0.9,
            "Extrapolated excitatory COBA reversal potential deviates "
            + "strongly from target.")

        # repeat for inhibitory case
        fits = []
        for neuron_id in range(halco.NeuronConfigOnDLS.size):
            # exclude measurements at low potentials for inhibitory
            # fits, since inh. amplitudes get low there anyway
            mask = self.__class__.inh_baselines[:, neuron_id] > 330
            fits.append(curve_fit(
                fitfunc, self.__class__.inh_baselines[mask, neuron_id],
                -self.__class__.inh_amplitudes[mask, neuron_id],
                p0=(0.4, 30), bounds=([-10, -10000], [10, 10000]))[0])
        fits = np.array(fits)

        # check slope: gain at least 10 MADC LSB of amplitude
        # over 100 MADC LSB of resting potential
        success = fits[:self.__class__.n_neurons_in_coba_mode, 0] > 10 / 100
        self.assertGreater(
            np.sum(success), self.__class__.n_neurons_in_coba_mode * 0.9,
            "Inhibitory amplitudes increase too slowly.")

        success = np.all(
            [fits[self.__class__.n_neurons_in_coba_mode:, 0] > -8 / 100,
             fits[self.__class__.n_neurons_in_coba_mode:, 0] < 8 / 100],
            axis=0)
        self.assertGreater(
            np.sum(success),
            (halco.NeuronConfigOnDLS.size
             - self.__class__.n_neurons_in_coba_mode) * 0.9,
            "CUBA neurons show modulation of inhibitory amplitudes.")

        # check x_offset: Expect it near e_coba_reversal, but in MADC units
        success = np.all(
            [fits[:self.__class__.n_neurons_in_coba_mode, 1] > 0,
             fits[:self.__class__.n_neurons_in_coba_mode, 1] < 330], axis=0)
        self.assertGreater(
            np.sum(success), self.__class__.n_neurons_in_coba_mode * 0.9,
            "Extrapolated inhibitory COBA reversal potential deviates "
            + "strongly from target.")


if __name__ == "__main__":
    unittest.main()
