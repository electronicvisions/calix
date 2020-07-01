"""
Provides algorithms for running calibrations.
Those algorithms need a Calibration instance which, among other settings,
yields instructions for configuring parameters and measuring results.
"""

from __future__ import annotations
from typing import Union, Optional, ClassVar
import numbers
from abc import abstractmethod
import numpy as np
from dlens_vx_v1 import sta, hxcomm

from calix.common import base, boundary_check, exceptions, helpers


class BinarySearch(base.Algorithm):
    """
    Perform a binary search within the parameter range of a calibration.

    Using the class requires basic settings, which come from an
    instance of a Calibration class. Therefore it is required to hook
    the algorithm to a calibration before calling run().
    Calling run(connection, target_result) then performs the calibration.
    The algorithm starts with the usual binary search algorithms and adds
    two additional steps in the end. This aims to explore the value above
    as well as below the last tested value of the normal binary search
    algorithm. The best-suited parameter tested during the last few steps
    is returned.

    :ivar calibration: Class derived from base class Calibration implementing
        functions for configuring parameters and measuring results as well as
        yielding important parameters for the algorithm to run.
    :ivar n_steps: Number of steps to use during calibration.
    :ivar step_increments: Amount to change parameter with each step.
    :ivar initial_parameters: Parameters to start calibration with.

    :cvar n_steps_best: Number of steps (counted from last)
        to take into account when finding the best parameters in the end.
    """

    n_steps_best: ClassVar[int] = 3

    def __init__(self):
        super().__init__()

        self.n_steps: Optional[int] = None
        self.step_increments: Optional[np.ndarray] = None
        self.initial_parameters: Optional[np.ndarray] = None

    def hook_to_calibration(self, calibration: base.Calibration):
        """
        Make an instance of a calibration known to the algorithm, allowing
        to use parameters set there.

        :param calibration: Instance of calibration class that will
            be run with this algorithm.
        """

        super().hook_to_calibration(calibration)

        possibilities = self.calibration.parameter_range.upper \
            - self.calibration.parameter_range.lower + 1

        # Set number of bisection steps such that the given range is covered
        bisection_steps = int(np.ceil(np.log2(possibilities)))
        # 2 extra steps varying parameter by 1
        self.n_steps = bisection_steps + 2
        if self.n_steps < 3:
            raise ValueError("Parameter range is too small for a calibration "
                             + "with the binary search algorithm.")

        # Calculate increments in every step
        self.step_increments = np.empty(self.n_steps, dtype=int)
        for step in range(self.n_steps):
            self.step_increments[step] = int(np.ceil(possibilities
                                                     / (2 ** (step + 2))))

        # Set start parameters as middle of parameter range
        self.initial_parameters = \
            np.ones(self.calibration.n_instances, dtype=int) \
            * int(possibilities / 2) + self.calibration.parameter_range.lower

    def run(self, connection: hxcomm.ConnectionHandle,
            target_result: Union[int, float, np.ndarray]
            ) -> np.ndarray:
        """
        Perform a binary search within given parameter range.
        In the end, the parameter is moved by 1 additionally.
        This way, the result of the binary search and the two neighboring
        settings are tested and the best-suitable parameter is returned.

        :param connection: Connection that will be used when setting up
            parameters and measuring results.
        :param target_result: Target for the calibration. The parameters will
            be set up such that the results match the target as close as
            possible. The target can be either a single value that will be used
            for all instances of parameters or an array containing a specific
            target for each instance. When using an array, its length has to
            match n_instances.

        :return: Best suited parameters to reach given targets.

        :raises ValueError: if the measure_parameters() function of the
            provided calibration class returns an array with bad length.
        :raises ValueError: if the number of steps to take into account
            when finding the best parameters in the end is smaller than 3.
            A value of 3 is required for the additional step to make sense.
        """

        parameters = self.initial_parameters

        if self.n_steps_best < 3:
            raise ValueError("Number of steps to take best parameters "
                             + "from needs to be > 3.")

        last_parameters = np.empty((self.n_steps_best,
                                    self.calibration.n_instances),
                                   dtype=int)
        last_deviations = np.empty((self.n_steps_best,
                                    self.calibration.n_instances))

        for step, increment in enumerate(self.step_increments):
            # Configure parameters, measure results
            builder = sta.PlaybackProgramBuilder()
            builder = self.calibration.configure_parameters(
                builder, parameters)
            results = self.calibration.measure_results(connection, builder)
            if not len(results) == self.calibration.n_instances:
                raise ValueError(
                    "Array of results does not match number of instances.")

            # Store parameters/results of last steps for optimization
            steps_left = self.n_steps - step - 1
            if steps_left < self.n_steps_best:
                last_parameters[steps_left] = parameters.copy()
                last_deviations[steps_left] = np.abs(results - target_result)

            # Update parameters
            if steps_left > 1:  # normal binary search
                high_mask = results > target_result
                if not self.calibration.inverted:
                    parameters[high_mask] -= increment
                    parameters[~high_mask] += increment
                else:
                    parameters[high_mask] += increment
                    parameters[~high_mask] -= increment

            elif steps_left == 1:  # additional test: neighboring setting
                parameters = 2 * last_parameters[2] - last_parameters[1]
                # Add/subtract one if last two tested parameters are at the
                # lower/upper end of the parameter range
                parameters[np.all(
                    [last_parameters[2] == last_parameters[1],
                     last_parameters[2]
                     == self.calibration.parameter_range.lower],
                    axis=0)] += 1
                parameters[np.all(
                    [last_parameters[2] == last_parameters[1],
                     last_parameters[2]
                     == self.calibration.parameter_range.upper],
                    axis=0)] -= 1

            # check range boundaries
            parameters = boundary_check.check_range_boundaries(
                parameters, self.calibration.parameter_range).parameters

        # Find best parameter in array of last steps
        best_parameters = last_parameters[
            np.argmin(last_deviations, axis=0),
            np.arange(self.calibration.n_instances)]

        # Set up best parameters again
        builder = sta.PlaybackProgramBuilder()
        builder = self.calibration.configure_parameters(
            builder, best_parameters)
        sta.run(connection, builder.done())

        return best_parameters


class NoisyBinarySearch(BinarySearch):
    """
    Perform binary search with noisy start parameters.

    The start of the search is not the middle of the range, but some
    integer noise given by `noise_amplitude` is added beforehand.
    This is useful to avoid issues that arise from setting many CapMem
    cells to the same value (CapMem crosstalk, issue 2654).
    The binary search takes an extra step to compensate for the
    initial offsets.

    The NoisyBinarySearch algorithm is typically used to prevent
    that many CapMem cells are set to the same value. Since a binary
    search could theoretically drive many cells to the same CapMem
    value in the final steps, more steps are taken into account
    in the end to find an optimal value.

    :ivar noise_amplitude: Amount of noise to add at the start
        of the calibration. Must be a positive integer; use the
        algorithm `BinarySearch` to work without noise.
    """

    def __init__(self, noise_amplitude: int = 5):
        """
        :raises ValueError: if `noise_amplitude` is not greater than zero.
        """

        super().__init__()

        if noise_amplitude < 1:
            raise ValueError("Noise must be a positive integer.")
        self.noise_amplitude = noise_amplitude

    def hook_to_calibration(self, calibration: base.Calibration):
        """
        Make an instance of a calibration known to the algorithm, allowing
        to use parameters set there.

        :param calibration: Instance of calibration class that will
            be run with this algorithm.

        :raises ExcessiveNoiseError: If the applied noise amplitude is
            more than a quarter of the calibration range, since that
            noise would lead to hitting the range boundaries instantly.
        """

        super().hook_to_calibration(calibration)

        # update calib with an additional step to adjust for noise
        if self.noise_amplitude > self.step_increments[0]:
            raise exceptions.ExcessiveNoiseError(
                "Excessive amounts of noise applied to the binary search "
                + "algorithm. The noise would partly be canceled within "
                + "the first step due to reaching the range boundaries.")
        self.n_steps += 1
        self.step_increments = np.sort(np.append(
            self.step_increments, self.noise_amplitude))[::-1]

        # take more steps into account when finding best parameters:
        # The last steps, which potentially reduce the spread below
        # the initially added noise, are always considered. Refer to
        # the class docstring for motivation.
        self.n_steps_best += int(np.ceil(np.log2(self.noise_amplitude)))

        # add noise to initial values of normal binary search
        self.initial_parameters += helpers.capmem_noise(
            start=-self.noise_amplitude, end=self.noise_amplitude + 1,
            size=self.calibration.n_instances)
        self.initial_parameters = boundary_check.check_range_boundaries(
            self.initial_parameters, self.calibration.parameter_range
        ).parameters


class LinearSearch(base.Algorithm):
    """
    Perform a linear search within the given parameter range to calibrate.
    From an initial guess, the parameter is moved in constant steps until
    the calibration target is crossed.

    Initializing the class requires basic settings, which come from an
    instance of a Calibration class.
    Calling run(connection, target_result) then performs the calibration.
    The parameters are sweeped until the results crosses the target, or the
    maximum number of steps is exceeded.

    :ivar calibration: Instance of calibration class used to provide
        settings and functions.
    :ivar max_steps: Maximum number of steps to use.
        If set too low, it can limit the available range if the step_size
        during the calibration is set low. When having reached the target
        value with all instances, the calibration stops early.
    :ivar initial_parameters: Initial (guessed) parameters to start
        algorithm with.
    :ivar step_size: Amount by which the parameters are moved in every step.

    :raises ValueError: If step_size is not a positive number.
    """

    def __init__(self,
                 initial_parameters: Optional[Union[int, np.ndarray]] = None,
                 step_size: int = 1, max_steps: int = 50):
        if step_size < 1:
            raise ValueError("`step_size` has to be a positve number.")

        super().__init__()
        self.initial_parameters = initial_parameters
        self.step_size = step_size
        self.max_steps = max_steps

    def run(self, connection: hxcomm.ConnectionHandle,
            target_result: Union[float, int, np.ndarray]
            ) -> np.ndarray:
        """
        Performs the calibration sweeping the parameter by step_size.

        The algorithm starts with the parameters supplied in the argument as a
        guess. It then moves in steps of `step_size` towards the target until
        the result crosses the target. The calibration then stops and returns
        the parameter that yields the results closest to the target in the last
        2 steps, i.e. while testing the values around crossing the threshold.

        Note that this function can be called multiple times with different
        step_sizes in order to approach the target step-wise. Doing so it is
        important to supply the results of the previous run as parameters.

        :param connection: Connection to use during configuring and measuring.
        :param target_result: Target result for calibration.

        :return: Best suited parameters to reach given targets.

        :raises TypeError: if `step_size` does not allow addition.
        :raises ValueError: if the measure_parameters() function of the
            provided calibration class returns an array with bad length.
        """

        # Handle inputs
        if self.initial_parameters is None:
            parameters = int(np.mean(self.calibration.parameter_range))
        else:
            parameters = self.initial_parameters
        if isinstance(parameters, numbers.Integral):
            parameters = np.ones(self.calibration.n_instances,
                                 dtype=int) * parameters
        if not isinstance(self.step_size, numbers.Integral):
            raise TypeError(
                "The step size for ParameterSweep needs to be a single "
                + "number. Call run() multiple times to vary step_size.")

        running = np.ones(self.calibration.n_instances, dtype=np.bool)
        last_parameters = np.empty((2, self.calibration.n_instances),
                                   dtype=int)
        last_deviations = np.ones((2, self.calibration.n_instances)) * np.inf

        for run in range(self.max_steps):
            # break if all instances are calibrated
            if not np.any(running):
                break

            # check range boundaries
            parameters = boundary_check.check_range_boundaries(
                parameters, self.calibration.parameter_range).parameters

            # Configure parameters, measure results
            builder = sta.PlaybackProgramBuilder()
            builder = self.calibration.configure_parameters(
                builder, parameters)
            results = self.calibration.measure_results(connection, builder)
            if not len(results) == self.calibration.n_instances:
                raise ValueError(
                    "Array of results does not match number of instances.")

            # Store parameters/results of last steps for optimization
            last_parameters[run % 2][running] = parameters[running]
            last_deviations[run % 2][running] = \
                np.abs(results - target_result)[running]

            # Update parameters
            high_mask = results > target_result
            if not self.calibration.inverted:
                parameters[high_mask & running] -= self.step_size
                parameters[~high_mask & running] += self.step_size
            else:
                parameters[high_mask & running] += self.step_size
                parameters[~high_mask & running] -= self.step_size

            # stop once threshold has been crossed,
            # i.e. setting moves opposite to the initial direction
            if run == 0:
                initial_deviations = high_mask
            else:
                running[initial_deviations != high_mask] = False

        # Find best parameter in array of last 2 steps
        parameters = last_parameters[
            np.argmin(last_deviations, axis=0),
            np.arange(self.calibration.n_instances)]

        # Set up best parameters again
        builder = sta.PlaybackProgramBuilder()
        builder = self.calibration.configure_parameters(builder, parameters)
        sta.run(connection, builder.done())

        return parameters


class PredictiveModel(base.Algorithm):
    """
    Base class for predictive models, which calculate the ideal parameters
    to achieve a target result based on a model.
    The models predict() function translates desired results into parameters.
    As there are differences between instances of the same thing on the chip,
    which is usually target of calibration, a shift of the model per instance
    is determined before calculating the optimal parameters.
    This means probe_parameters are configured and measured, used as shift,
    then the model prediction is calculated and returned.
    Running the algorithm, the shifts described above are measured at the
    probe parameters, then the parameters returned from the prediction
    are ensured to be in the allowed parameter range and set up
    on the chip before returning.

    :ivar probe_parameters: Parameters to use when determining the shift
        each instance has with respect to the expected model.
    """

    def __init__(self, probe_parameters: Union[int, np.ndarray]):
        super().__init__()
        self.probe_parameters = probe_parameters

    @abstractmethod
    def _predict(self, probe_results: np.ndarray,
                 target_result: Union[float, int, np.ndarray]) -> np.ndarray:
        """
        Predict parameters which need to be set to achieve the given
        target_result. Use the probe_results at self.probe_parameters as
        baseline to determine offsets of individual instances.

        :param probe_results: Array of results achieved using
            self.probe_parameters.
        :param target_result: Array or single value to aim results for.

        :return: Array containing the parameters which are expected to yield
            results closest to the desired target results.
        """

        raise NotImplementedError

    def run(self, connection: hxcomm.ConnectionHandle,
            target_result: Union[float, int, np.ndarray]) -> np.ndarray:
        """
        Run the algorithm, i.e. configure the probe parameters,
        measure results there, calculate optimal parameters based on
        this measurement and the model, and set these up.

        :param connection: Connection connected to the chip to run on.
        :param target_result: Array or single value to be aimed for.

        :return: Array containing the parameters which are expected to yield
            results closest to the desired target results.
            All parameters are in the allowed ParameterRange.
        """

        if isinstance(self.probe_parameters, np.ndarray):
            probe_parameters = self.probe_parameters
        else:
            probe_parameters = np.ones(
                self.calibration.n_instances, dtype=int
            ) * self.probe_parameters

        builder = sta.PlaybackProgramBuilder()
        builder = self.calibration.configure_parameters(
            builder, probe_parameters)
        probe_results = self.calibration.measure_results(connection, builder)

        optimal_parameters = self._predict(probe_results, target_result)
        optimal_parameters = boundary_check.check_range_boundaries(
            optimal_parameters, self.calibration.parameter_range).parameters

        builder = sta.PlaybackProgramBuilder()
        builder = self.calibration.configure_parameters(
            builder, optimal_parameters)
        sta.run(connection, builder.done())

        return optimal_parameters


class PolynomialPrediction(PredictiveModel):
    """
    Class implementing a polynomial predictive model.
    Construction of this class requires an instance of numpy Polynomial
    which already contains suitable parameters, or measured data from
    characterization of the parameter/result pair to be calibrated
    (via `from_data()`). In the latter case, a fit to the data is
    performed to obtain the numpy polynomial.

    :ivar polynomial: Instance of numpy polynomial, used as a model
        to map target values to parameters.
    """

    def __init__(self, probe_parameters: Union[int, np.ndarray],
                 polynomial: np.polynomial.polynomial.Polynomial):
        """
        :param probe_parameters: Parameter at which a measurement
            was or will be taken in order to determine the offsets
            for each calibration instance. Refer to `_predict` for
            details.
        """

        super().__init__(probe_parameters)
        self.polynomial = polynomial

    @classmethod
    def from_data(cls, parameters: np.ndarray,
                  results: np.ndarray, degree: int) -> PolynomialPrediction:
        """
        Construct the polynomial prediction from data.
        A polynomial of given degree is fitted to the parameters and results.
        Refer to the documentation of numpy.polynomial.polynomial.Polynomial
        for details on how the fit is performed.

        The median of the parameters used during characterization is used
        as probe point when determining the offset of each instance
        during prediction.

        :param parameters: Array of parameters where values have been recored.
        :param results: Array of results obtained at parameters.
        :param degree: Degree of the fitted polynomial.
        """

        return cls(
            probe_parameters=int(np.median(parameters)),
            polynomial=np.polynomial.polynomial.Polynomial.fit(
                results, parameters, degree))

    def _predict(self, probe_results: np.ndarray,
                 target_result: Union[float, int, np.ndarray]) -> np.ndarray:
        """
        Predict parameters to obtain the given target results.

        Evaluate the polynomial at probe parameters and compare with
        the results obtained there to determine the required offset
        of each instance. Evaluate the shifted polynomial at the target
        results to obtain the results predicted by the model.

        :param probe_results: Array of results achieved using
            self.probe_parameters.
        :param target_result: Array or single value to aim results for.

        :return: Array containing the parameters which are expected to yield
            results closest to the desired target results.
            Results are rounded to the nearest integer parameters.
        """

        offset = self.probe_parameters - self.polynomial(probe_results)
        return np.rint(self.polynomial(target_result) + offset).astype(int)


class LinearPrediction(PolynomialPrediction):
    """
    Class implementing a first degree polynomial.
    """

    def __init__(self, probe_parameters: Union[int, np.ndarray],
                 offset: float = 0, slope: float = 1):
        """
        :param probe_parameters: Parameter at which a measurement
            was or will be taken in order to determine the offsets
            for each calibration instance. Refer to
            `PolynomialPrediction._predict` for details.
        :param offset: Offset (zeroth-degree) term of the linar function.
        :param slope: Slope (first-degree) term of the linear function.
        """

        super().__init__(
            probe_parameters,
            polynomial=np.polynomial.polynomial.Polynomial([offset, slope]))
