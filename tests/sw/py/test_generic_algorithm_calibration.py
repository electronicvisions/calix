import inspect
import pkgutil
import unittest
from typing import Callable, Set, Type

from dlens_vx_v3 import logger
import calix
from calix.common import exceptions
from calix.common.base import Algorithm, Calibration

from mock_connection_setup import ConnectionSetup


log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class GenericCalibrationTest(ConnectionSetup):
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
                calib_result = calibration.run(self.connection, algorithm)
            except exceptions.ExcessiveNoiseError as error:
                self.skipTest(f"{algorithm_type.__name__} cannot be "
                              + f"used with {calibration_type.__name__}: "
                              + f"{error}")
            except exceptions.CalibrationNotSuccessful as error:
                self.skipTest("Calibration was not successful, which is "
                              + "to be expected in this test.")
            except exceptions.CalibrationNotSupported as error:
                self.skipTest(
                    f"Calibration is deliberately not supported: {error}")
            except exceptions.TooFewSamplesError as error:
                self.skipTest(
                    "Too few MADC samples were received, which is "
                    + "to be expected here since the ZeroMockConnection "
                    + "does not provide MADC samples.")

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
