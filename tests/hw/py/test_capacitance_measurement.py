#!/usr/bin/env python3

import unittest
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import quantities as pq

from dlens_vx_v3 import hal, halco, hxcomm, logger

from connection_setup import ConnectionSetup

from calix.hagen.neuron_leak_bias import MembraneTimeConstCalibOffset
import calix.spiking
from calix.common import base, helpers
from calix import constants
from calix.measurement.intrinsic_capacitance import measure, fit, fit_function


log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class TauMemMeasurement(MembraneTimeConstCalibOffset):
    """
    Measure the membrane time constant with the MADC.

    This class is used to measure the increases in the membrane
    time constant due to the intrinsic capacitance.

    In additon to the functionality of the base class, the class supports
    to connect several neuron circuits and measure the time constant.
    In this case only the first neuron circuit is ``activated'', i.e.,
    for all other neuron circuits the leak and capacitance is disabled.

    Additional instance variables compared to the base class:
    :ivar neurons: Neurons to measure.
    :ivar n_connected: Number of neuron circuits to connect.
    """

    def __init__(self,
                 neurons: np.ndarray,
                 neuron_configs: List[hal.NeuronConfig],
                 n_connected: int):
        # check that neurons do not collide with each other
        used_circuits = np.concatenate(
            [np.arange(n_connected - 1) + neuron + 1
             for neuron in neurons])
        if set(neurons) & set(used_circuits):
            raise RuntimeError("Neuron circuits of different 'neurons' "
                               "overlap. Chose a larger space between "
                               "neurons or reduce the number of connected "
                               "circuits.")

        super().__init__(target=50 * pq.us, neuron_configs=neuron_configs)
        self.neurons = neurons
        self.n_connected = n_connected
        self.adjust_bias_range = False

    def prelude(self, connection: hxcomm.ConnectionHandle):
        """
        Prepares chip for calibration.

        Executes the prelude of the base class and in addition
        already configures the neuron circuits which will later
        be connected:
        - disable leak
        - disable capacitance
        - connect neuron circuits (all but the first)

        :param connection: Connection to the chip to calibrate.
        """
        super().prelude(connection)

        # Disable synaptic input for all neurons
        for config in self.neuron_configs:
            config.enable_synaptic_input_excitatory = False
            config.enable_synaptic_input_inhibitory = False
        # Prepare neuron circuits for connection
        for neuron in self.neurons:
            for offset in range(1, self.n_connected):
                config = self.neuron_configs[neuron + offset]
                config.enable_synaptic_input_excitatory = False
                config.enable_synaptic_input_inhibitory = False
                config.enable_leak_division = True
                config.membrane_capacitor_size = 0

                if offset < self.n_connected - 1:
                    config.connect_membrane_right = True

        # Set leak biases to zero for non-readout neurons
        original_leak_biases = self.read_leak_bias(connection)
        leak_biases = np.zeros_like(original_leak_biases)
        leak_biases[self.neurons] = original_leak_biases[self.neurons]

        builder = base.WriteRecordingPlaybackProgramBuilder()
        builder = helpers.capmem_set_neuron_cells(
            builder, {halco.CapMemRowOnCapMemBlock.i_bias_leak: leak_biases})
        builder = helpers.wait(builder, constants.capmem_level_off_time)

        for coord, config in zip(halco.iter_all(halco.NeuronConfigOnDLS),
                                 self.neuron_configs):
            builder.write(coord, config)
        base.run(connection, builder)

    @staticmethod
    def read_leak_bias(connection: hxcomm.ConnectionHandle):
        builder = base.WriteRecordingPlaybackProgramBuilder()
        capmem_row = halco.CapMemRowOnCapMemBlock.i_bias_leak
        tickets = []
        for neuron in halco.iter_all(halco.NeuronConfigOnDLS):
            capmem_col = \
                neuron.toNeuronConfigOnNeuronConfigBlock()\
                .toCapMemColumnOnCapMemBlock()
            coord = halco.CapMemCellOnDLS(
                halco.CapMemCellOnCapMemBlock(capmem_col, capmem_row),
                neuron.toNeuronConfigBlockOnDLS().toCapMemBlockOnDLS())
            tickets.append(builder.read(coord))
        base.run(connection, builder)
        return np.array([ticket.get().value.value() for ticket in tickets])

    def select_connection(self, connect: bool = True):
        """
        Change whether the readout neurons are connected to several
        other circuits.

        :param connect: Whether the neurons should be connected to
            other circuits or not.
        """
        for neuron in self.neurons:
            config = self.neuron_configs[neuron]
            config.connect_membrane_right = connect


class TestCapacitanceCharacterization(ConnectionSetup):
    """
    Test the characterization of the intrinsic capacitance by connecting
    several neuron circuits and measuring the membrane time constant.

    The membrane time constant is given by the capacitance divided by the
    leak conductance. When the membrane capacitance increases, the time
    constant increases as well.
    In this test, we connect several neuron circuits, disable the leak
    term as well as the capacitance for all but one circuit.
    We then compare the time constant of a single circuit
    with the time constant of several connected circuits. When we do not
    account for the intrinsic capacitance of the circuits, the membrane
    time constant increases (assuming the leak is fully disabled). If we
    measured the correct intrinsic capacitances and account for them,
    the time constant should stay the same.

    :cvar calib_result: Result of calibration, stored for re-applying.
    :cvar n_connected: Number of connected neuron circuits which are
        used to test the characterization. The number should not be
        too high such that the capacitance of a single neuron circuit
        is still sufficient to counter balance the intrinsic capacitances
        of the other circuits.
    :cvar max_dev: Maximum relative deviation between connected and
        unconnected circuits. If the measured deviation is higher than this
        value, the calibration fails.
    """

    calib_result: Optional[calix.spiking.SpikingCalibResult] = None
    n_connected = 6
    max_dev = 0.12

    def get_neurons(self):
        """
        Get coordinates of neurons to measure.
        """
        # handle each quadrant separately (there are no connections from
        # the left to the right hemisphere.
        neurons = []
        for quad in halco.iter_all(halco.NeuronConfigBlockOnDLS):
            tmp_neurons = np.arange(
                0,
                halco.NeuronConfigOnNeuronConfigBlock.size - self.n_connected,
                self.n_connected)
            neurons.append(
                tmp_neurons
                + halco.NeuronConfigOnNeuronConfigBlock.size * int(quad))
        return np.concatenate(neurons)

    def _get_capacitance_offset(self, capacitance: np.ndarray) -> np.ndarray:
        neurons_per_quad = halco.NeuronConfigOnNeuronConfigBlock.size
        highes_neuron_coord = \
            neurons_per_quad // self.n_connected * self.n_connected
        # reshape to easier calculate the sum
        cap = capacitance.reshape(-1, neurons_per_quad)
        cap = cap[:, :highes_neuron_coord].reshape(-1, self.n_connected)

        return np.sum(cap[:, 1:], axis=1)

    def measure(self, intrinsic_capacitances: Optional[np.ndarray] = None):
        calib_result = self.apply_calibration("spiking")
        neuron_configs = [neuron.asNeuronConfig() for neuron
                          in calib_result.neuron_result.neurons.values()]

        neurons = self.get_neurons()
        measurement = TauMemMeasurement(neurons, neuron_configs,
                                        n_connected=self.n_connected)
        measurement.prelude(self.connection)

        # measure unconnected
        for neuron_coord in self.get_neurons():
            config = measurement.neuron_configs[neuron_coord]
            config.membrane_capacitor_size = \
                hal.NeuronConfig.MembraneCapacitorSize.max
        measurement.select_connection(False)
        builder = base.WriteRecordingPlaybackProgramBuilder()
        res_single = measurement.measure_results(self.connection, builder)

        # measure connected
        if intrinsic_capacitances is not None:
            offsets = self._get_capacitance_offset(intrinsic_capacitances)
            assert len(offsets) == len(self.get_neurons())
            for neuron_coord, cap in zip(self.get_neurons(), offsets):
                config = measurement.neuron_configs[neuron_coord]
                config.membrane_capacitor_size = \
                    max(0, hal.NeuronConfig.MembraneCapacitorSize.max
                        - round(cap))

        measurement.select_connection(True)
        builder = base.WriteRecordingPlaybackProgramBuilder()
        res_con = measurement.measure_results(self.connection, builder)

        return res_single, res_con

    @staticmethod
    def plot_fit(capacitances: np.ndarray,
                 taus: np.ndarray,
                 fitresult: List) -> None:
        """
        Plot the measured membrane time constants and the fitted line.

        Plot two neurons per quadrant.

        :param capacitances: Membrane capacitance at which the membrane
            time constants have been measured.
        :param taus: Measured membrane time constants.
        :param fitresult: Result of the fits.
        """

        # Determine neurons which should be plotted
        n_neurons = 2
        n_blocks = halco.NeuronConfigBlockOnDLS.size
        neurons = np.tile(np.arange(n_neurons), n_blocks) \
            + np.repeat(np.arange(n_blocks), n_neurons) \
            * halco.NeuronConfigOnNeuronConfigBlock.size

        fig, axs = plt.subplots(n_blocks, n_neurons,
                                sharex=True, sharey=True,
                                figsize=(3 * n_blocks, 4 * n_neurons),
                                tight_layout=True)

        # plot neurons
        for ax, neuron in zip(axs.flatten(), neurons):
            ax.set_title(f"Neuron {neuron}")
            neuron_data = taus[neuron]
            fit_data = fitresult[neuron]

            ax.errorbar(capacitances,
                        np.mean(neuron_data, axis=1),
                        yerr=np.std(neuron_data, axis=1),
                        marker=".", ls='none', label="data")

            values = np.linspace(-5, 70, 100)
            ax.plot(values,
                    fit_function(values, *fit_data["parameters"]),
                    label="fit")
            ax.text(1, 0,
                    f"params: {fit_data['parameters']}\n"
                    f"cov: {np.sqrt(np.diag(fit_data['covariance']))}",
                    transform=ax.transAxes, ha='right', va='bottom')
            ax.set_xlabel("Capacitance / LSB")
            ax.set_ylabel(r"Membrane Time Constant / $\mu$s")
        fig.savefig("test_capacitance_measurement")

    def test_00_uncorrected(self):
        """
        Test that the membrane time constant increases when we do not
        account for the intrinsic capacitances.
        """
        single, connected = self.measure()
        rel_dev = np.mean(np.abs(connected - single) / single)
        self.assertGreater(
            rel_dev, 2 * self.max_dev,
            "Relative deviation low for calibrated case. Maximum relative "
            f"deviation seems to be too low. Current value: {self.max_dev}.")

    def test_01_characterization(self):
        """
        Characterize the internal capacitances and check that the membrane
        time constant stays constant.
        """
        # measure and fit
        test_values = np.linspace(
            10, hal.NeuronConfig.MembraneCapacitorSize.max, 5, dtype=int)
        taus = measure(self.connection, test_values=test_values, n_rep=2)
        fit_res = fit(test_values, taus)
        offsets = np.array([res["parameters"][0] for res in fit_res])

        self.plot_fit(test_values, taus, fit_res)

        # test calibration
        single, connected = self.measure(np.array(offsets))
        rel_dev = np.mean(np.abs(connected - single) / single)
        self.assertLess(
            rel_dev, self.max_dev,
            "Relative deviation high after calibration")


if __name__ == "__main__":
    unittest.main()
