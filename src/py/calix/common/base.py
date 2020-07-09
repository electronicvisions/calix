"""
Provides abstract base classes for calibrations and algorithms.
"""

from __future__ import annotations
from collections import namedtuple
import numbers
from typing import List, Union, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from dlens_vx_v2 import sta, logger, hxcomm

from calix.common.boundary_check import check_range_boundaries


ParameterRange = namedtuple("ParameterRange", ["lower", "upper"])


@dataclass
class CalibrationResult:
    """
    Data structure for calibration results.
    Contains an array of calibrated parameters and a mask
    indicating calibration success.
    """
    calibrated_parameters: np.ndarray
    success: np.ndarray


class Calibration(ABC):
    """
    Base class for calibration applications.

    Accepts parameter range and other generally important settings.
    From here, Calibrations can be derived, implementing functions for
    at least measuring results and configuring parameters.
    Only one-dimensional calibrations are supported, i.e. only one variable is
    adjusted to move towards a desired target value.

    :ivar parameter_range: Allowed range for parameter.
    :ivar inverted: Invert calibration, i.e. expect a higher parameter
        returning a lower result. Defaults to False.
    :ivar n_instances: Number of objects (e.g. neurons) to calibrate.
    :ivar errors: Error messages to be logged when instances of the
        calibration have reached the parameter_range boundaries. If None,
        nothing will be logged. The first (second) error message is used
        when the lower (upper) bound is reached.
        Error messages are expected to support the .format() operation.
        As parameter {0}, the indices of failed instances are inserted, as
        parameter {1}, the respective boundary value is inserted.
    :ivar target: Target read for calibration. Can be found during
        prelude, otherwise needs to be supplied to the run() method.
    :ivar result: List: Calibrated parameters and success mask.
        Only available after run() has been called.

    :raises ValueError: if parameter_range has a bad shape or is too small.
    """

    def __init__(self, parameter_range: ParameterRange, n_instances: int,
                 inverted: bool = False, errors: Optional[List[str]] = None):
        self.parameter_range = parameter_range
        if parameter_range.upper - parameter_range.lower < 1:
            raise ValueError("Boundaries of parameter_range are too close.")
        self.inverted = inverted
        self.n_instances = n_instances
        self.errors = errors
        self.target: Union[numbers.Integral, np.ndarray, None] = None
        self.result: Optional[CalibrationResult] = None

    def prelude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Function to do necessary pre-configuration or other things that
        are required during the actual calibration, like finding the
        target values.
        If not overwritten, this function does nothing.

        :param connection: Connection to the chip to be calibrated.
        """

    def postlude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Function to do necessary postconfiguring of the chip.
        If not overwritten, this function does nothing.

        :param connection: Connection to the chip to be calibrated.
        """

    @abstractmethod
    def configure_parameters(self, builder: sta.PlaybackProgramBuilder,
                             parameters: np.ndarray
                             ) -> sta.PlaybackProgramBuilder:
        """
        Function to set up parameters for testing.

        Use the provided builder to append instructions that configure the
        provided parameters. Then return the builder again.
        If one needs to wait between configuring the parameters and measuring,
        e.g. after setting a new CapMem value, this should be done here.

        :param builder: Program Builder to which configuration instructions
            can be appended.
        :param parameters: Array of parameters that should be set up. Only
            integer parameters will be used.

        :return: The Program Builder with instructions to set up the given
            parameters appended.
        """

        raise NotImplementedError

    @abstractmethod
    def measure_results(self, connection: hxcomm.ConnectionHandle,
                        builder: sta.PlaybackProgramBuilder) -> np.ndarray:
        """
        Function to measure the observable.

        Typically this will read the CADC channels and return an array of
        the obtained reads. Use the provided builder to append read
        instructions, run it using the provided connection. If necessary,
        treat the results in any way, e.g. calculate statistics per quadrant
        if calibrating global settings. Then return an array that fits the
        number of parameters to update.

        :param connection: Connection where builder can be executed in order to
            measure results.
        :param builder: Builder where instructions to read something can be
            appended.

        :return: An array with the measurement results in a shape that
            corresponds to the calibration instances. Any type that allows
            comparison to a calibration target using an operator `<` is
            allowed, in particular int and float.
        """

        raise NotImplementedError

    def run(self, connection: hxcomm.ConnectionHandle,
            algorithm: Algorithm,
            target: Union[numbers.Integral, np.ndarray, None] = None
            ) -> CalibrationResult:
        """
        Use the provided algorithm to find optimal parameters.

        Calls prelude function, calls algorithm.run(), calls postlude function.
        If self.target is not defined after the prelude function, the
        calibration target is required as a parameter to this function.

        :param connection: Connection to the chip to calibrate.
        :param algorithm: Instance of an algorithm to be used for
            calibration. Will be hooked to the calibration during running
            and unhooked afterwards.
        :param target: Target result to be aimed for. If given, this target
            is used. If None, the attribute self.target is used.

        :raises TypeError: If neither the target parameter or the attribute
            self.target are defined.

        :return: CalibrationResult, containing the optimal parameters.
        """

        # Call prelude, check if target exists
        self.prelude(connection)

        if target is None:
            if self.target is None:
                raise TypeError("Calibration target is not defined.")
        else:
            self.target = target

        # Run calibration
        algorithm.hook_to_calibration(self)
        calibrated_parameters = algorithm.run(connection, self.target)
        algorithm.unhook_from_calibration()

        # Check range boundaries
        if self.errors is not None:
            # format {1} in error messages to boundary values
            self.errors = [
                self.errors[0].format("{0}", self.parameter_range.lower),
                self.errors[1].format("{0}", self.parameter_range.upper)]

            result = check_range_boundaries(
                calibrated_parameters, self.parameter_range, self.errors)

            for error in result.messages:
                logger.get("calix.common.base.Calibration.run").WARN(error)

        else:
            result = check_range_boundaries(
                calibrated_parameters, self.parameter_range)

        self.result = CalibrationResult(result.parameters, ~result.error_mask)

        # Call postlude
        self.postlude(connection)

        return self.result


class Algorithm(ABC):
    """
    Base class for calibration Algorithms.
    From here, algorithms are derived, which implement a run() method.

    :ivar calibration: Calibration instance that contains important parameters
        and functions for the Algorithm to run.
    """

    # The current Algorithm ABC is rather "unconstrained" and should
    # become more expressive. See issue 3954.

    def __init__(self):
        self.calibration: Optional[Calibration] = None

    def hook_to_calibration(self, calibration: Calibration):
        """
        Make an instance of a calibration known to the algorithm, allowing
        to use parameters set there.

        :param calibration: Instance of calibration class that will
            be run with this algorithm.
        """

        self.calibration = calibration

    def unhook_from_calibration(self):
        """
        Clear a calibration instance previously hooked to this algorithm.
        """

        self.calibration = None

    @abstractmethod
    def run(self, connection: hxcomm.ConnectionHandle,
            target_result: Union[float, int, np.ndarray]
            ) -> np.ndarray:
        """
        Function that iterates setting parameters up, testing the results,
        comparing them to the target result(s) and changing the parameters.
        The specifics are implemented in the derived algorithms.
        Generally, the run function also configures the optimum parameters
        on hardware, in addition to returning them.

        :param connection: Connection that will be used when setting up
            parameters and measuring results.
        :param target_result: Target for the calibration. The parameters will
            be set up such that the results match the target as close as
            possible. The target can be either a single value that will be used
            for all instances of parameters or an array containing a specific
            target for each instance. When using an array, its length has to
            match n_instances.

        :return: Best suited parameters to reach given targets.
        """

        raise NotImplementedError
