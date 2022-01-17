import unittest
from typing import Optional, List, Dict

import numpy as np
import scipy.stats
import quantities as pq

from dlens_vx_v2 import hal, halco, sta, hxcomm, logger

from calix.common import base
from calix.hagen import neuron_potentials, neuron_leak_bias, neuron_synin
from calix.spiking import neuron_threshold
import calix.spiking

from connection_setup import ConnectionSetup


log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class TestNeuronDistribution(ConnectionSetup):
    """
    Runs a neuron calibration and ensures the properties
    of LIF neurons show less deviations in calibrated state
    than with all parameters set to their median.

    :cvar log: Logger used for output.
    :cvar calib_result: Result of calibration, stored for re-applying.
    :cvar neuron_configs: List of neuron configs from calibration.
    :cvar results: Results from evaluation.
    """

    log = logger.get("calix.tests.hw.test_neuron_distribution")
    calib_result: Optional[calix.spiking.SpikingCalibrationResult] = None
    neuron_configs: List[hal.NeuronConfig] = list()
    results: Dict[str, np.ndarray] = dict()

    @classmethod
    def tearDownClass(cls) -> None:
        super().tearDownClass()
        np.savez("calib_eval.npz", **cls.results)

    def find_median_config(self, key: str) -> bool:
        """
        Find majority vote on a certain setting of the NeuronConfig.

        :param key: Configuration entry to scan. Has to be a boolean
            getter of a hal.NeuronConfig property.

        :return: Majority vote of the config.
        """

        settings = np.empty(halco.NeuronConfigOnDLS.size, dtype=bool)
        for neuron_id in range(halco.NeuronConfigOnDLS.size):
            settings[neuron_id] = getattr(self.neuron_configs[neuron_id], key)

        return bool(np.median(settings))

    class ParameterTest(unittest.TestCase):
        """
        Template for testing a parameter from the LIF neuron model.

        Uses a calibration instance to measure results in the
        original (calibrated) state and with parameters equalized
        to the median (uncalibrated state).

        From both distributions, the standard deviations are calculated
        after cutting off a few percent from both tails - mitigating the
        effect of outliers on the standard deviation. The parameter
        `proportion_to_cut` controls the amount to cut off.
        Finally, we assert the calibration has decreased the standard
        deviation down to a factor `assertion_limit` of the original
        distribution.

        :cvar calib_result: Original calibration result.
        :cvar log: Logger used for output.

        :ivar name: Name of parameter to test (for debug logging).
        :ivar calibration: Instance of a calibration class for the
            tested parameter.
        :ivar uncalibrated_value: CapMem value to be configured for the
            uncalibrated state. Typically the median of the calibrated
            parameters.
        :ivar proportion_to_cut: Proportion to cut from both tails of
            the calibrated and uncalibrated distribution before asserting
            the standard deviation was decreased. This mitigates the
            effect of outliers.
        :ivar assertion_limit: Fraction of the uncalibrated standard
            deviation that is allowed after calibration as a maximum.
        """

        calib_result: Optional[calix.spiking.SpikingCalibrationResult] = None
        log = logger.get("calix.tests.hw.test_neuron_distribution")

        def __init__(self, name: str, calibration: base.Calibration,
                     uncalibrated_value: int, *,
                     proportion_to_cut: float = 0.05,
                     assertion_limit: float = 0.3):
            super().__init__()

            self.name = name
            self.calibration = calibration
            self.uncalibrated_value = uncalibrated_value
            self.proportion_to_cut = proportion_to_cut
            self.assertion_limit = assertion_limit

        def equalize_parameters(self, connection: hxcomm.ConnectionHandle):
            """
            Set the respective parameter into an uncalibrated state.

            :param connection: Connection to the chip to run on.
            """

            builder = sta.PlaybackProgramBuilder()
            self.calibration.configure_parameters(
                builder, self.uncalibrated_value)
            base.run(connection, builder)

        def run_test(self, connection: hxcomm.ConnectionHandle,
                     results: Optional[Dict[str, np.ndarray]] = None):
            """
            Run the test template as described in the class docstring.

            Calls the `equalize_parameter` function internally once
            the calibrated measurement is done and before initiating
            the uncalibrated measurement.

            :param connection: Connection to the chip to run on.
            :param results: Dictionary of results. If given, an entry
                with the name of the test (ivar `name`) is created,
                containing the calibrated measurements, as well as
                one with containing the suffix "_uncalibrated".
            """

            # apply original calib
            builder = sta.PlaybackProgramBuilder()
            self.calib_result.apply(builder)
            base.run(connection, builder)

            # measure in calibrated state
            self.calibration.prelude(connection)
            calibrated_result = self.calibration.measure_results(
                connection, builder=sta.PlaybackProgramBuilder())
            self.log.DEBUG(self.name, ": ", calibrated_result)
            if results is not None:
                results.update({self.name: calibrated_result})

            # equalize parameters, measure in uncalibrated state
            self.equalize_parameters(connection)
            uncalibrated_result = self.calibration.measure_results(
                connection, builder=sta.PlaybackProgramBuilder())
            self.log.DEBUG(self.name + "_uncalibrated: ", uncalibrated_result)
            if results is not None:
                results.update(
                    {self.name + "_uncalibrated": uncalibrated_result})

            # calculate ratio of std. deviations, assert calib affects it
            calibrated_std = np.std(scipy.stats.trimboth(
                calibrated_result,
                proportiontocut=self.proportion_to_cut))
            uncalibrated_std = np.std(scipy.stats.trimboth(
                uncalibrated_result,
                proportiontocut=self.proportion_to_cut))
            ratio = calibrated_std / uncalibrated_std
            self.log.INFO(f"Calib reduced mismatch in {self.name} "
                          + f"by a factor {ratio}.")
            self.assertLess(ratio, self.assertion_limit,
                            f"Calibration of {self.name} yields bad results.")

    def test_00_calibration(self):
        """
        Calibrate neurons to default parameters.
        """

        self.__class__.calib_result = calix.spiking.calibrate(self.connection)
        self.ParameterTest.calib_result = self.__class__.calib_result

        self.__class__.neuron_configs = list()
        for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
            self.__class__.neuron_configs.append(
                self.__class__.calib_result.neuron_result.neurons[
                    coord].asNeuronConfig())

    def test_01_leak(self):
        """
        Measure and evaluate distribution of leak potentials.
        """

        # initialize with random target
        calibration = neuron_potentials.LeakPotentialCalibration(target=100)

        # extract calibrated parameters from calibration result
        parameters = np.empty(halco.AtomicNeuronOnDLS.size, dtype=int)
        for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
            parameters[int(coord.toEnum())] = \
                self.__class__.calib_result.neuron_result.neurons[
                    coord].leak.v_leak
        median = int(np.median(parameters))

        test = self.ParameterTest(
            "v_leak", calibration, median, assertion_limit=0.3)
        test.run_test(self.connection, self.__class__.results)

    def test_02_reset(self):
        """
        Measure and evaluate distribution of reset potentials.
        """

        # initialize with random target
        calibration = neuron_potentials.ResetPotentialCalibration(target=90)

        # extract calibrated parameters from calibration result
        parameters = np.empty(halco.AtomicNeuronOnDLS.size, dtype=int)
        for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
            parameters[int(coord.toEnum())] = \
                self.__class__.calib_result.neuron_result.neurons[
                    coord].reset.v_reset
        median = int(np.median(parameters))

        test = self.ParameterTest(
            "v_reset", calibration, median, assertion_limit=0.2)
        test.run_test(self.connection, self.__class__.results)

    def test_03_threshold(self):
        """
        Measure and evaluate distribution of threshold potentials.
        """

        # initialize with random target
        calibration = neuron_threshold.ThresholdCalibCADC(target=140)

        # extract calibrated parameters from calibration result
        parameters = np.empty(halco.AtomicNeuronOnDLS.size, dtype=int)
        for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
            parameters[int(coord.toEnum())] = \
                self.__class__.calib_result.neuron_result.neurons[
                    coord].threshold.v_threshold
        median = int(np.median(parameters))

        test = self.ParameterTest(
            "v_threshold", calibration, median, assertion_limit=0.35)
        test.run_test(self.connection, self.__class__.results)

    def test_04_tau_mem(self):
        """
        Measure and evaluate distribution of membrane time constants.
        """

        # initialize with random target
        calibration = neuron_leak_bias.MembraneTimeConstCalibOffset(
            target=3 * pq.us, neuron_configs=self.__class__.neuron_configs)

        # disable leak division/multiplication adjustment
        calibration.adjust_bias_range = False

        # extract calibrated parameters from calibration result
        parameters = np.empty(halco.AtomicNeuronOnDLS.size, dtype=int)
        for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
            parameters[int(coord.toEnum())] = \
                self.__class__.calib_result.neuron_result.neurons[
                    coord].leak.i_bias
        median = int(np.median(parameters))

        # extract leak division/multiplication from calibration result
        config = hal.NeuronConfig(self.__class__.neuron_configs[0])
        config.enable_leak_division = self.find_median_config(
            "enable_leak_division")
        config.enable_leak_multiplication = self.find_median_config(
            "enable_leak_multiplication")
        self.assertFalse(
            config.enable_leak_division and config.enable_leak_multiplication,
            "Both division and multiplication is enabled for most neurons.")

        # use a second instance of the calibration class with different
        # (i.e. equalized) neuron configs. Switch to this class when
        # parameters are to be equalized.
        second_calibration = neuron_leak_bias.MembraneTimeConstCalibOffset(
            target=3 * pq.us,
            neuron_configs=[hal.NeuronConfig(config) for _ in
                            range(halco.NeuronConfigOnDLS.size)])

        # disable leak division/multiplication adjustment
        second_calibration.adjust_bias_range = False

        class TauMemTest(self.ParameterTest):
            def equalize_parameters(self, connection: hxcomm.ConnectionHandle):
                self.calibration = second_calibration
                self.calibration.prelude(connection)
                builder = sta.PlaybackProgramBuilder()
                self.calibration.configure_parameters(
                    builder, self.uncalibrated_value)
                base.run(connection, builder)

        test = TauMemTest(
            "tau_mem", calibration, median, assertion_limit=0.45)
        test.run_test(self.connection, self.__class__.results)

    def test_05_tau_syn_exc(self):
        """
        Measure and evaluate distribution of excitatory synaptic input
        time constants.
        """

        calibration = neuron_synin.ExcSynTimeConstantCalibration(
            neuron_configs=self.__class__.neuron_configs)

        # extract calibrated parameters from calibration result
        parameters = np.empty(halco.AtomicNeuronOnDLS.size, dtype=int)
        for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
            parameters[int(coord.toEnum())] = \
                self.__class__.calib_result.neuron_result.neurons[
                    coord].excitatory_input.i_bias_tau
        median = int(np.median(parameters))

        test = self.ParameterTest(
            "tau_syn_exc", calibration, median, assertion_limit=0.2)
        test.run_test(self.connection, self.__class__.results)

    def test_06_tau_syn_inh(self):
        """
        Measure and evaluate distribution of excitatory synaptic input
        time constants.
        """

        calibration = neuron_synin.InhSynTimeConstantCalibration(
            neuron_configs=self.__class__.neuron_configs)

        # extract calibrated parameters from calibration result
        parameters = np.empty(halco.AtomicNeuronOnDLS.size, dtype=int)
        for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
            parameters[int(coord.toEnum())] = \
                self.__class__.calib_result.neuron_result.neurons[
                    coord].inhibitory_input.i_bias_tau
        median = int(np.median(parameters))

        test = self.ParameterTest(
            "tau_syn_inh", calibration, median, assertion_limit=0.2)
        test.run_test(self.connection, self.__class__.results)


if __name__ == "__main__":
    unittest.main()
