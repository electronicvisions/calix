import inspect
import pkgutil
import unittest
from typing import Callable, Set, Type

import calix
from calix.common import exceptions
from calix.common.base import Algorithm, Calibration
from dlens_vx_v1.sta import ExperimentInit, generate, run
from dlens_vx_v1.hxcomm import ManagedConnection, ConnectionHandle


class GenericCalibrationTest(unittest.TestCase):
    CONNECTION_MANAGER = ManagedConnection()
    CONNECTION: ConnectionHandle

    # Calibration target for all calibrated properties. A value of '1' works
    # for all current and planned algorithms and is approved by JW. It might
    # require more complex selection logic in the future.
    CALIBRATION_TARGET = 1.

    @classmethod
    def setUpClass(cls) -> None:
        # Connect (sim or hardware)
        cls.CONNECTION = cls.CONNECTION_MANAGER.__enter__()

        # Initialize the chip
        init_builder, _ = generate(ExperimentInit())
        run(cls.CONNECTION, init_builder.done())

    @classmethod
    def tearDownClass(cls) -> None:
        # Disconnect
        cls.CONNECTION_MANAGER.__exit__()

    @classmethod
    def generate_cases(cls):
        """
        Generate test cases for all combinations of implementations of
        :class:`calix.hagen.base.Algorithm` and
        :class:`calix.hagen.base.Calibration` and run them.
        """
        for algorithm in cls.implementations(Algorithm):
            assert issubclass(algorithm, Algorithm)
            for calibration in cls.implementations(Calibration):
                assert issubclass(calibration, Calibration)
                test_method = cls.generate_single(algorithm, calibration)
                test_method.__name__ = f"test_" \
                                       f"{calibration.__name__}_" \
                                       f"{algorithm.__name__}"

                setattr(cls, test_method.__name__, test_method)

    @staticmethod
    def generate_single(algorithm_type: Type[Algorithm],
                        calibration_type: Type[Calibration]) -> Callable:
        """
        Generate a test function for running a calibration of given type with
        an algorithm of given type.

        :param algorithm_type: Algorithm type to be used for calibrating
        :param calibration_type: Calibration type to be executed
        :return: Function testing a single run of
        """

        def test_func(self: GenericCalibrationTest):
            try:
                calibration = calibration_type()
            except TypeError as error:
                self.skipTest(f"{calibration_type.__name__} cannot be "
                              f"default-constructed: {error}")
                raise

            try:
                algorithm = algorithm_type()
            except TypeError as error:
                self.skipTest(f"{algorithm_type.__name__} cannot be "
                              f"default-constructed: {error}")
                raise

            try:
                calib_result = calibration.run(self.CONNECTION,
                                               algorithm,
                                               target=self.CALIBRATION_TARGET)
            except exceptions.ExcessiveNoiseError as error:
                self.skipTest(f"{algorithm_type.__name__} cannot be "
                              + f"used with {calibration_type.__name__}: "
                              + f"{error}")
            except exceptions.CalibrationNotSuccessful as error:
                self.skipTest("Calibration was not successful, which is "
                              + "to be expected in this test.")

            self.assertIsNotNone(calib_result)

        return test_func

    @staticmethod
    def implementations(class_: Type) -> Set[type]:
        """
        Recursively get all implementations/non-abstract children of a given
        parent/interface.

        :param class_: Parent class to be crawled for non-abstract subclasses.
        :return: Set of subclasses found.
        """
        ret = set()
        todo = {class_}

        while todo:
            current = todo.pop()
            todo.update(current.__subclasses__())

            if not inspect.isabstract(current):
                ret.add(current)

        return ret


# Recursively import all submodules to ensure all children are known
for submodule in pkgutil.walk_packages(calix.__path__, f"{calix.__name__}."):
    __import__(submodule.name)

GenericCalibrationTest.generate_cases()

if __name__ == '__main__':
    unittest.main()
