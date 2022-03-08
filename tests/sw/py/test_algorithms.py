import numbers
from typing import Optional, Union, ClassVar
import unittest
import numpy as np
from dlens_vx_v3 import sta, logger

from calix.common import algorithms, base

from mock_connection_setup import ConnectionSetup


log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class _AlgorithmsGeneral(ConnectionSetup):
    """
    Tests all the available algorithms by providing artificial
    measurements that are just the parameters that were set up, possibly
    inverted. Creates the given number of instances and calibrates
    them to targets 0 ... (n_instances - 1), while the allowed
    range of parameters is also that.

    :cvar n_instances: Number of instances to calibrate.
    :cvar max_noise_amplitude: Maximum amplitude of noise applied to
        the NoisyBinarySearch.
    """

    n_instances = ClassVar[int]
    max_noise_amplitude: ClassVar[int] = 30

    @classmethod
    def parameter_range(cls) -> base.ParameterRange:
        """
        Returns the parameter range used for calibrations,
        depending on the configured number of instances.

        :return: ParameterRange of the calibration parameters.
        """

        return base.ParameterRange(0, cls.n_instances - 1)

    class TestCalib(base.Calibration, unittest.TestCase):
        """
        Calibration to test non-inverted algorithms.
        Returns the configured parameters as measured results.

        :ivar stored_parameters: Configured parameters, stored to
            return them when results are measured.
        """

        stored_parameters: Optional[np.ndarray] = None

        def configure_parameters(self, builder, parameters):
            self.assertEqual(parameters.dtype, int)
            self.stored_parameters = parameters.copy()
            return builder

        def measure_results(self, connection, builder):
            return self.stored_parameters

    class TestCalibInverted(TestCalib):
        """
        Calibration to test inverted algorithms.
        Returns the inverse of the configured parameters as result.
        """

        def invert(self, variable: Union[numbers.Number, np.ndarray]
                   ) -> Union[numbers.Number, np.ndarray]:
            """
            Inverts a variable in the parameter range of the class:
            Calculates the difference between the upper end of the
            parameter range and the variable.

            :param variable: Variable to be inverted.

            :return: Inverted result.
            """

            return self.parameter_range.upper - variable

        def measure_results(self, connection, builder):
            return self.invert(super().measure_results(connection, builder))

    def test_binary_search_not_inverted(self):
        calibration = self.TestCalib(
            parameter_range=self.parameter_range(),
            n_instances=self.n_instances)
        calibrated_parameters = calibration.run(
            self.connection, algorithm=algorithms.BinarySearch(),
            target=np.arange(self.n_instances)
        ).calibrated_parameters

        np.testing.assert_allclose(
            calibrated_parameters, np.arange(self.n_instances),
            err_msg="Calibration using Binary Search (non-inverted) failed.")

    def test_binary_search_inverted(self):
        calibration = self.TestCalibInverted(
            parameter_range=self.parameter_range(),
            n_instances=self.n_instances, inverted=True)
        calibrated_parameters = calibration.run(
            self.connection, algorithm=algorithms.BinarySearch(),
            target=np.arange(self.n_instances)
        ).calibrated_parameters

        np.testing.assert_allclose(
            calibrated_parameters, np.arange(self.n_instances)[::-1],
            err_msg="Calibration using Binary Search (inverted) failed.")

    def test_noisy_search_not_inv(self):
        for noise_amplitude in range(1, self.max_noise_amplitude):
            calibration = self.TestCalib(
                parameter_range=self.parameter_range(),
                n_instances=self.n_instances)
            calibrated_parameters = calibration.run(
                self.connection, algorithm=algorithms.NoisyBinarySearch(
                    noise_amplitude=noise_amplitude),
                target=np.arange(self.n_instances)
            ).calibrated_parameters

            np.testing.assert_allclose(
                calibrated_parameters, np.arange(self.n_instances),
                err_msg="Calibration using Extended Binary Search "
                + "(non-inverted) failed.")

    def test_noisy_search_inverted(self):
        for noise_amplitude in range(1, self.max_noise_amplitude):
            calibration = self.TestCalibInverted(
                parameter_range=self.parameter_range(),
                n_instances=self.n_instances, inverted=True)
            calibrated_parameters = calibration.run(
                self.connection, algorithm=algorithms.NoisyBinarySearch(
                    noise_amplitude=noise_amplitude),
                target=np.arange(self.n_instances)
            ).calibrated_parameters

            np.testing.assert_allclose(
                calibrated_parameters, np.arange(self.n_instances)[::-1],
                err_msg="Calibration using Extended Binary Search "
                + "(inverted) failed.")

    def test_linear_search_not_inv(self):
        calibration = self.TestCalib(
            parameter_range=self.parameter_range(),
            n_instances=self.n_instances)
        parameters = np.random.randint(
            200, 800, size=self.n_instances, dtype=int)

        algorithm = algorithms.LinearSearch(
            initial_parameters=parameters, step_size=20)
        calibrated_parameters = calibration.run(
            self.connection, algorithm=algorithm,
            target=np.arange(self.n_instances)).calibrated_parameters

        algorithm.initial_parameters = calibrated_parameters
        algorithm.step_size = 1
        calibrated_parameters = calibration.run(
            self.connection, algorithm=algorithm,
            target=np.arange(self.n_instances)).calibrated_parameters

        np.testing.assert_allclose(
            calibrated_parameters, np.arange(self.n_instances),
            err_msg="Calibration using linear search (non-inverted) failed.")

    def test_linear_search_inverted(self):
        calibration = self.TestCalibInverted(
            parameter_range=self.parameter_range(),
            n_instances=self.n_instances, inverted=True)
        parameters = np.random.randint(
            200, 800, size=self.n_instances, dtype=int)

        algorithm = algorithms.LinearSearch(
            initial_parameters=parameters, step_size=20)
        calibrated_parameters = calibration.run(
            self.connection, algorithm=algorithm,
            target=np.arange(self.n_instances)).calibrated_parameters

        algorithm.initial_parameters = calibrated_parameters
        algorithm.step_size = 1
        calibrated_parameters = calibration.run(
            self.connection, algorithm=algorithm,
            target=np.arange(self.n_instances)).calibrated_parameters

        np.testing.assert_allclose(
            calibrated_parameters, np.arange(self.n_instances)[::-1],
            err_msg="Calibration using linear search (inverted) failed.")

    def test_prediction_poly_fit(self):
        calibration = self.TestCalib(
            parameter_range=self.parameter_range(),
            n_instances=self.n_instances, inverted=False)

        # Generate data from non-inverted calib
        parameters = np.repeat(
            np.linspace(0, 1000, 20, dtype=int)[:, np.newaxis],
            self.n_instances, axis=1)
        results = np.empty_like(parameters, dtype=float)
        for parameter_id, parameter in enumerate(parameters):
            builder = sta.PlaybackProgramBuilder()
            builder = calibration.configure_parameters(builder, parameter)
            results[parameter_id] = calibration.measure_results(
                self.connection, builder)

        # Fit second degree polynomial prediction to results
        prediction = algorithms.PolynomialPrediction.from_data(
            parameters.flatten(), results.flatten(), degree=2)
        fitted_polynomial = prediction.polynomial.convert()

        # Assert x^0 term is 0, as set up in TestCalib
        self.assertAlmostEqual(
            fitted_polynomial.coef[0], 0,
            msg="Model offset does not represent identity function.")

        # Assert x^1 term is 1, as set up in TestCalib
        self.assertAlmostEqual(
            fitted_polynomial.coef[1], 1,
            msg="Model slope does not represent identity function.")

        # Assert x^2 term is zero as model is linear only
        self.assertAlmostEqual(
            fitted_polynomial.coef[2], 0,
            msg="Second order term is non-zero for linear model.")

    def test_linear_prediction_not_inv(self):
        calibration = self.TestCalib(
            parameter_range=self.parameter_range(),
            n_instances=self.n_instances, inverted=False)
        probe_points = np.arange(self.n_instances)[::-1]
        prediction = algorithms.LinearPrediction(
            probe_parameters=probe_points, offset=10, slope=1)

        # Algorithm should notice changed offset and still
        # predict correct parameters
        calibrated_parameters = calibration.run(
            self.connection, algorithm=prediction,
            target=np.arange(self.n_instances)
        ).calibrated_parameters

        np.testing.assert_allclose(
            calibrated_parameters, np.arange(self.n_instances),
            err_msg="Calibration using linear prediction failed.")

    def test_linear_prediction_inverted(self):
        calibration = self.TestCalibInverted(
            parameter_range=self.parameter_range(),
            n_instances=self.n_instances, inverted=True)
        probe_points = np.arange(self.n_instances)
        prediction = algorithms.LinearPrediction(
            probe_parameters=probe_points, offset=10, slope=-1)

        # Algorithm should notice incorrect offset and still
        # predict correct parameters
        calibrated_parameters = calibration.run(
            self.connection, algorithm=prediction,
            target=np.arange(self.n_instances)
        ).calibrated_parameters

        np.testing.assert_allclose(
            calibrated_parameters, np.arange(self.n_instances)[::-1],
            err_msg="Calibration using linear prediction failed.")


# Instantiations of the test-class with arbitrary, but even and un-even
# numbers of instances
class TestAlgorithms16Instances(_AlgorithmsGeneral):
    """
    Tests all the algorithms using 16 instances and parameters
    ranged 0...15.
    """

    n_instances = 16
    max_noise_amplitude = 4


class TestAlgorithms511Instances(_AlgorithmsGeneral):
    """
    Tests all the algorithms using 511 instances and parameters
    ranged 0...510.
    """

    n_instances = 511


class TestAlgorithms1024Instances(_AlgorithmsGeneral):
    """
    Tests all the algorithms using 1024 instances and parameters
    ranged 0...1023.
    """

    n_instances = 1024


if __name__ == "__main__":
    unittest.main()
