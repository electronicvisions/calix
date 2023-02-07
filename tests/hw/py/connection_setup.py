import unittest
from pathlib import Path
import pickle
import os

from dlens_vx_v3 import sta, hxcomm, logger

from calix.common import base
import calix.scripts.calix_generate_default_calibration as calib_generator


class ConnectionSetup(unittest.TestCase):
    """
    Base class for hardware tests:
    Provides connection and ExperimentInit() initialization
    during setup and disconnects during teardown.

    :cvar conn_manager: Connection context manager.
    :cvar connection: Connection to chip to use.
    :cvar target_directory: Directory to store calibration results.
    :cvar available_calibrations: Dictionary of calibration types
        (as string) and corresponding class in calib-generation script.
        Needs to contain all calibrations that are requestable in
        `get_calibration()`.
    :cvar use_cache: Switch whether stored calibration results are
        used. If False, all calibrations are performed new. The
        results are still saved in the target_directory.
    """

    conn_manager = hxcomm.ManagedConnection()
    connection = None
    target_directory = Path("./calibs")
    available_calibrations = {
        "hagen": calib_generator.HagenCalib(),
        "hagen_synin": calib_generator.HagenSyninCalib(),
        "spiking": calib_generator.SpikingCalib(),
        "spiking2": calib_generator.SpikingCalib2()}
    use_cache = True

    @classmethod
    def setUpClass(cls) -> None:
        cls.connection = cls.conn_manager.__enter__()

        # extend target_directory by connection id
        cls.target_directory /= cls.connection.get_unique_identifier()

        # Initialize the chip
        builder, _ = sta.ExperimentInit().generate()
        base.run(cls.connection, builder)

    @classmethod
    def apply_calibration(cls, identifier: str) -> base.CalibResult:
        """
        Search for requested calibration, load and apply it.

        If it doesn't exist on disk, run it on the chip. In this case,
        the chip is re-initialized, then the calibration is applied again.
        Return the result.

        :param identifier: String denoting the type of requested
            calibration. Possible values are listed in the cvar
            available_calibrations.

        :return: Corresonding calibration result.
        """

        # check if requested calibration exists already, run it otherwise
        filename = cls.target_directory / "_".join(
            [identifier, "calix-native.pkl"])
        log = logger.get("calix.tests.hw.py.connection_setup")

        if cls.use_cache and filename.exists():
            log.INFO(f"Loading calibration from {filename}")
            with open(filename, "rb") as calibfile:
                result = pickle.load(calibfile)
        else:
            # generate and save calibration
            log.INFO("No cached calibration found or cache disabled, "
                     + f"generating to {filename}")
            cls.target_directory.mkdir(parents=True, exist_ok=True)
            cls.available_calibrations[identifier].record_and_dump(
                cls.connection, cls.target_directory)

            with open(filename, "rb") as calibfile:
                result = pickle.load(calibfile)

            # re-initialize chip after calibration has run
            builder, _ = sta.ExperimentInit().generate()
            base.run(cls.connection, builder)

        builder = sta.PlaybackProgramBuilder()
        result.apply(builder)
        base.run(cls.connection, builder)
        return result

    @classmethod
    def tearDownClass(cls) -> None:
        cls.conn_manager.__exit__()


def string_to_bool(string: str) -> bool:
    """
    Convert string to boolean.

    Note: This reimplements the distutils.util.strtobool
    function [1], which is deprecated and will be removed in Python 3.12.

    [1]: https://docs.python.org/3/distutils/apiref.html#distutils.util.strtobool  # pylint: disable=line-too-long

    :param string: Input string.
    :return: Boolean interpretation of string.
    :raises ValueError: If string doesn't contain a known
        expression representing a boolean value.
    """

    if string.lower() in ["y", "yes", "t", "true", "on", "1", "enabled"]:
        return True
    if string.lower() in ["n", "no", "f", "false", "off", "0", "disabled"]:
        return False
    raise ValueError("Failed to interpret {string} as a boolean.")


# use stored calib if not disabled in environment
ConnectionSetup.use_cache = string_to_bool(os.environ.get(
    "CALIX_USE_CACHE", default="True"))
