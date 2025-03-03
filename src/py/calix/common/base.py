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
from dlens_vx_v3 import halco, hal, sta, logger, hxcomm, lola

from pyccalix import CalibOptions
from calix.common.boundary_check import check_range_boundaries


class WriteRecordingPlaybackProgramBuilder:
    def __init__(self):
        self.builder = sta.PlaybackProgramBuilder()
        self.dumper = sta.PlaybackProgramBuilderDumper()
        self.cocos = {}

    def done(self):
        for coord, config in self.cocos.items():
            self.dumper.write(coord, config)
        self.cocos = {}
        return self.builder.done(), self.dumper.done()

    def merge_back(self, other):
        self.builder.merge_back(other.builder)
        self.dumper.merge_back(other.dumper)
        self.cocos.update(other.cocos)
        other.cocos = {}

    def copy_back(self, other):
        self.builder.copy_back(other.builder)
        self.dumper.copy_back(other.dumper)
        self.cocos.update(other.cocos)

    def block_until(self, coord, config):
        self.builder.block_until(coord, config)

    def write(self, coord, config, backend=None):
        if backend is not None:
            self.builder.write(coord, config, backend)
        else:
            self.builder.write(coord, config)
        self.cocos.update({coord: config})

    def read(self, coord, backend=None):
        if backend is not None:
            return self.builder.read(coord, backend)
        return self.builder.read(coord)

    def empty(self):
        return self.builder.empty() and self.dumper.empty()


class StatefulConnection:
    def __init__(
            self,
            connection: hxcomm.ConnectionHandle,
            init: sta.ExperimentInit = sta.ExperimentInit()):
        self.connection = connection
        self.reinit = sta.ReinitStackEntry(self.connection)
        self.init = init
        self.dumper = sta.PlaybackProgramBuilderDumper()

        builder, _ = sta.generate(init)
        self.reinit.set(builder.done(), enforce=True)

    def update_reinit(self, dumperdone: sta.DumperDone):
        dumperdone.remove_block_until()
        self.dumper.merge_back(sta.convert_to_dumper(dumperdone))
        dumperdone = self.dumper.done()
        dumperdone.squash()
        self.dumper = sta.convert_to_dumper(dumperdone)

        dumper_copy = sta.PlaybackProgramBuilderDumper()
        dumper_copy.copy_back(self.dumper)

        sta_builder, _ = sta.generate(self.init)
        sta_builder.merge_back(sta.convert_to_builder(dumper_copy))
        sta_builder.block_until(halco.BarrierOnFPGA(), hal.Barrier.omnibus)
        sta_builder.write(halco.TimerOnDLS(), hal.Timer())
        sta_builder.block_until(
            halco.TimerOnDLS(), hal.Timer.Value(
                hal.Timer.Value.fpga_clock_cycles_per_us * 1000 * 100))
        self.reinit.set(sta_builder.done(), enforce=False)

    def get_unique_identifier(self):
        return self.connection.get_unique_identifier()


def run(connection: StatefulConnection,
        builder: WriteRecordingPlaybackProgramBuilder) -> sta.PlaybackProgram:
    """
    Wraps the stadls run function.

    Records all write accesses and updates the connection's reinit program.

    Includes a barrier, blocking for the omnibus being idle, before
    the end of the program. The finished program is returned,
    such that, e.g., spikes are accessible.

    :param connection: Connection to the chip to execute on.
    :param builder: Builder to execute.

    :return: Program compiled by builder.done() that is executed.
    """

    builder.block_until(halco.BarrierOnFPGA(), hal.Barrier.omnibus)
    program, dumperdone = builder.done()
    sta.run(connection.connection, program)
    connection.update_reinit(dumperdone)
    return program


ParameterRange = namedtuple("ParameterRange", ["lower", "upper"])


def check_values(name, value, feasible_range: ParameterRange):
    """
    Check whether the provided target parameters are feasible
    for calibration.

    :param name: Name of parameter used in logging
    :param value: Value to check
    :param feasible_ranges: Dict of feasible ranges for each parameter.
        These values have been determined experimentally. Some setups
        might support larger ranges but in general subpar calibrations
        are expected outside this range.
        Warnings will be logged in case they are exceeded.
    """
    log = logger.get("calix.common.base.check_values")

    if value is None:
        return

    if not isinstance(value, np.ndarray):
        value = np.array(value)
    if np.any([value < feasible_range.lower,
               value > feasible_range.upper]):
        log.WARN(
            f"Parameter {name} was chosen at {value}, which is "
            + f"outside the standard range of {feasible_range}. "
            + "Please expect imperfect results.")


@dataclass
class CalibTarget(ABC):
    """
    Data structure for collecting targets for higher-level calibration
    functions into one logical unit.

    Targets are parameters that directly affect how a circuit is
    configured. They have a standard range, where the circuits will
    work well. Exceeding the standard range may work better for some
    instances (e.g., neurons) than others.
    """


class TopLevelCalibTarget(CalibTarget):
    """
    :class:`CalibTarget` that can be calibrated for as part of the public
    calix API via :func:`calix.calibrate`.
    """
    @abstractmethod
    def calibrate(self,
                  connection: hxcomm.ConnectionHandle,
                  options: Optional[CalibOptions] = None) -> CalibResult:
        """
        Execute a calibration for this target.

        :param connection: Connection to be used
        :param options: Calibration options
        :return: Calibration result
        """
        raise NotImplementedError


@dataclass
class CalibResult(ABC):
    """
    Data structure for higher-level calibration results, that combine
    multiple parameters into a logal unit.

    Used as base type for hagen and spiking calibration results.

    :ivar target: Target parameters that were used to achieve these
        results.
    :ivar options: Further options for calibration, that also may
        affect the results.
    """

    target: Optional[CalibTarget]
    options: Optional[CalibOptions]

    # The following function should take a union type exposed from
    # haldls as an argument, cf. issue 3995.
    # To be updated also in derived classes.
    @abstractmethod
    def apply(self, builder: Union[sta.PlaybackProgramBuilder,
                                   sta.PlaybackProgramBuilderDumper]):
        """
        Apply the saved calib result to the chip.

        This function should work with either a builder or a dumper.

        :param builder: Builder or dumper to append instructions to.
        """

        raise NotImplementedError

    def to_chip(
            self,
            initial_config: Optional[sta.PlaybackProgramBuilderDumper] = None,
            chip: Optional[lola.Chip] = None) -> lola.Chip:
        """
        Apply the calibration into a lola.chip object.

        :param initial_config: Optional dumper filled with configuration
            instructions that have been run before creating the
            calibration.
        :param chip: Chip object to merge the calibration into.
            Defaults to a default-constructed `lola.Chip()`.

        :return: Chip object with calibration applied.
        """

        if initial_config is None:
            initial_config = sta.PlaybackProgramBuilderDumper()

        builder = WriteRecordingPlaybackProgramBuilder()
        builder.dumper = initial_config
        dumper_copy = sta.PlaybackProgramBuilderDumper()
        dumper_copy.copy_back(builder.dumper)
        builder.builder = sta.convert_to_builder(dumper_copy)
        self.apply(builder)
        _, dumperdone = builder.done()

        if chip is None:
            chip = lola.Chip()

        return sta.convert_to_chip(dumperdone, chip)


@dataclass
class ParameterCalibResult:
    """
    Data structure for calibration results of single parameters.

    Contains an array of calibrated parameters and a mask
    indicating calibration success.
    """

    calibrated_parameters: np.ndarray
    success: np.ndarray


class Calib(ABC):
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
        self.result: Optional[ParameterCalibResult] = None

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
    def configure_parameters(
            self, builder: WriteRecordingPlaybackProgramBuilder,
            parameters: np.ndarray) \
            -> WriteRecordingPlaybackProgramBuilder:
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
    def measure_results(
            self, connection: hxcomm.ConnectionHandle,
            builder: WriteRecordingPlaybackProgramBuilder) -> np.ndarray:
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
            ) -> ParameterCalibResult:
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

        :return: ParameterCalibResult, containing the optimal
            parameters.
        """

        # Call prelude, check if target exists
        self.prelude(connection)

        if target is None:
            if self.target is None:
                raise TypeError("Calib target is not defined.")
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
                logger.get("calix.common.base.Calib.run").WARN(error)

        else:
            result = check_range_boundaries(
                calibrated_parameters, self.parameter_range)

        self.result = ParameterCalibResult(
            result.parameters, ~result.error_mask)

        # Call postlude
        self.postlude(connection)

        return self.result


class Algorithm(ABC):
    """
    Base class for calibration Algorithms.
    From here, algorithms are derived, which implement a run() method.

    :ivar calibration: Calib instance that contains important parameters
        and functions for the Algorithm to run.
    """

    # The current Algorithm ABC is rather "unconstrained" and should
    # become more expressive. See issue 3954.

    def __init__(self):
        self.calibration: Optional[Calib] = None

    def hook_to_calibration(self, calibration: Calib):
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
