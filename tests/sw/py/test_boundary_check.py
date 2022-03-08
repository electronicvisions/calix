import unittest
import numpy as np

from dlens_vx_v3 import logger
from calix.common import base, boundary_check


log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class BoundaryCheckTest(unittest.TestCase):
    """
    Tests the boundary checker.
    """

    def test_no_violations(self):
        """
        Calls boundary checker with parameters which do not exceed the
        boundaries.

        Expect two true values in the error mask but no error messages since
        no template for an error message was provided. As no value exceeds
        the boundaries, no value will be changed.
        """

        parameters = np.arange(900)
        boundaries = base.ParameterRange(0, 899)

        expected_parameters = np.copy(parameters)

        result = boundary_check.check_range_boundaries(parameters, boundaries)
        np.testing.assert_allclose(
            result.parameters, expected_parameters,
            err_msg="Boundary checker modifies parameters that did not "
            + "exceed limits.")

        expected_errors = np.zeros_like(parameters, dtype=np.bool)
        expected_errors[[0, -1]] = True
        np.testing.assert_allclose(
            result.error_mask, expected_errors,
            err_msg="Boundary checker marks errors where limits "
            + "are not reached.")

        self.assertEqual(
            len(result.messages), 0,
            "No error messages should be returned.")

    def test_violations(self):
        """
        Calls boundary checker with parameters exceeding the boundaries.
        Expects parameters to be clipped and errors to be returned.
        """

        parameters = np.arange(900)
        boundaries = base.ParameterRange(200, 700)

        expected_parameters = np.copy(parameters)
        expected_parameters[parameters < 200] = 200
        expected_parameters[parameters > 700] = 700

        result = boundary_check.check_range_boundaries(
            parameters, boundaries, errors=["Entries {0} reached limits."] * 2)
        np.testing.assert_allclose(
            result.parameters, expected_parameters,
            err_msg="Boundary checker modifies parameters incorrectly.")

        expected_errors = np.zeros_like(parameters, dtype=np.bool)
        expected_errors[parameters <= 200] = True
        expected_errors[parameters >= 700] = True
        np.testing.assert_allclose(
            result.error_mask, expected_errors,
            err_msg="Boundary checker marks errors incorrectly.")

        self.assertEqual(
            len(result.messages), 2,
            "Expected two error messages for reaching "
            + "lower and upper boundary.")


if __name__ == "__main__":
    unittest.main()
