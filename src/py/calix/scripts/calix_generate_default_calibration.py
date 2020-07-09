#!/usr/bin/env python3
"""
Execute all available default calibration and saves them in all available data
formats. If result files already exist, they are overwritten.
"""

import argparse
import pickle
import gzip
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Union, List

from dlens_vx_v2.hxcomm import ConnectionHandle, ManagedConnection
from dlens_vx_v2.sta import PlaybackProgramBuilderDumper, ExperimentInit, \
    run, to_json, to_portablebinary

import calix.hagen
from calix.hagen import HagenCalibrationResult
import calix.spiking
from calix.spiking import SpikingCalibrationResult

# TODO @JW: There should be a common base for these (Hagen, Spiking, ...)
CalibrationResult = Union[HagenCalibrationResult, SpikingCalibrationResult]


class CalibrationRecorder(metaclass=ABCMeta):
    """
    Recorder for various calibration results, to be implemented for HAGEN,
    Spiking etc.
    """

    @property
    @abstractmethod
    def calibration_type(self) -> str:
        """
        Identifier for the acquired calibration data.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_calib(self,
                       connection: ConnectionHandle) -> CalibrationResult:
        """
        Execute calibration routines and create a result object.
        :param connection: Connection used for acquiring calibration data.
        :return: Calibration result data
        """
        raise NotImplementedError


class CalibrationDumper(metaclass=ABCMeta):
    """
    Dumper for calibration data into various formats, e.g. calix internal,
    write instructions etc.
    """

    @property
    @abstractmethod
    def format_name(self) -> str:
        """
        Identifier for the dumped data format.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def format_extension(self) -> str:
        """
        File extension for the dumped data format, including a leading '.'.
        """
        raise NotImplementedError

    @abstractmethod
    def dump_calibration(self, calibration_result: CalibrationResult,
                         target_file: Path):
        """
        Read a calibration result and serialize it to a given file.
        :param calibration_result: Calibration result to be serialized.
        :param target_file: Path to the file the result is written to.
        """
        raise NotImplementedError


class RecorderAndDumper(metaclass=ABCMeta):
    """
    Record and dump calibration data.
    """

    @property
    @abstractmethod
    def recorder(self) -> CalibrationRecorder:
        """
        Recorder used for acquiring the calibration data
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dumpers(self) -> List[CalibrationDumper]:
        """
        List of Dumpers used for serializing the calibration data. All will
        serialize the data that has been acquired in a single calibration run.
        """
        raise NotImplementedError

    def record_and_dump(self, connection: ConnectionHandle,
                        deployment_folder: Path):
        """
        Record calibration data and dump it to a file.
        :param connection: Connection used to acquire the calibration data
        :param deployment_folder: Folder the file with serialized results is
                                  created in.
        """
        result = self.recorder.generate_calib(connection)

        for dumper in self.dumpers:
            filename = f"{self.recorder.calibration_type}_" \
                       f"{dumper.format_name}{dumper.format_extension}"
            target_file = deployment_folder.joinpath(filename)
            dumper.dump_calibration(result, target_file)


class HagenCalibRecorder(CalibrationRecorder):
    """
    Recorder for a canonical Hagen-Mode calibration.
    """
    calibration_type = "hagen"

    def generate_calib(self,
                       connection: ConnectionHandle) -> CalibrationResult:
        builder, _ = ExperimentInit().generate()
        run(connection, builder.done())
        return calix.hagen.calibrate(connection)


class SpikingCalibRecorder(CalibrationRecorder):
    """
    Recorder for a default Spiking-Mode calibration.
    """
    calibration_type = "spiking"

    def generate_calib(self,
                       connection: ConnectionHandle) -> CalibrationResult:
        builder, _ = ExperimentInit().generate()
        run(connection, builder.done())
        return calix.spiking.calibrate(connection)


class CalixFormatDumper(CalibrationDumper):
    """
    Dumper for the calix-internal data format.
    """
    format_name = "calix-native"
    format_extension = ".pkl"

    def dump_calibration(self, calibration_result: CalibrationResult,
                         target_file: Path):
        with target_file.open(mode="wb") as target:
            pickle.dump(calibration_result, target)


class CocoListPortableBinaryFormatDumper(CalibrationDumper):
    """
    Dumper for the Coordinate-Container-List data in portable binary format.
    """
    format_name = "cocolist"
    format_extension = ".pbin"

    def dump_calibration(self, calibration_result: CalibrationResult,
                         target_file: Path):
        builder = PlaybackProgramBuilderDumper()
        calibration_result.apply(builder)

        with target_file.open(mode="wb") as target:
            target.write(to_portablebinary(builder.done()))


class CocoListJsonFormatDumper(CalibrationDumper):
    """
    Dumper for the Coordinate-Container-List data in json format
    with gzip compression.
    """
    format_name = "cocolist"
    format_extension = ".json.gz"

    def dump_calibration(self, calibration_result: CalibrationResult,
                         target_file: Path):
        builder = PlaybackProgramBuilderDumper()
        calibration_result.apply(builder)

        with gzip.open(target_file, mode="wt") as target:
            target.write(to_json(builder.done()))


class HagenCalibration(RecorderAndDumper):
    recorder = HagenCalibRecorder()
    dumpers = [CalixFormatDumper(),
               CocoListPortableBinaryFormatDumper(),
               CocoListJsonFormatDumper()]


class SpikingCalibration(RecorderAndDumper):
    recorder = SpikingCalibRecorder()
    dumpers = [CalixFormatDumper(),
               CocoListPortableBinaryFormatDumper(),
               CocoListJsonFormatDumper()]


def run_and_save_all(deployment_folder: Path):
    """
    Executes all available default calibrations and saves them to all
    available file formats.

    :param deployment_folder: Path calibration results are deployed to.
    """
    with ManagedConnection() as connection:
        for calib in [HagenCalibration(), SpikingCalibration()]:
            calib.record_and_dump(connection, deployment_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "deployment_folder",
        help="Path to save all calibration result to. Directories are "
        + "created if not already present.")
    args = parser.parse_args()

    depl_folder = Path(args.deployment_folder)
    depl_folder.mkdir(parents=True, exist_ok=True)
    run_and_save_all(depl_folder)
