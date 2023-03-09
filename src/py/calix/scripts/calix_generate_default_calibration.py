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
from typing import List

import quantities as pq

from dlens_vx_v3 import logger
from dlens_vx_v3.hxcomm import ConnectionHandle, ManagedConnection
from dlens_vx_v3.sta import PlaybackProgramBuilderDumper, to_json, \
    to_portablebinary

import calix.hagen
import calix.spiking
from calix.common.base import CalibResult, TopLevelCalibTarget
from calix.hagen import HagenCalibTarget, HagenSyninCalibTarget
from calix.spiking import SpikingCalibTarget

log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class CalibRecorder(metaclass=ABCMeta):
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

    @property
    @abstractmethod
    def calibration_target(self) -> TopLevelCalibTarget:
        """
        Calibration target to be calibrated for.
        """
        raise NotImplementedError


class CalibDumper(metaclass=ABCMeta):
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
    def dump_calibration(self, calibration_result: CalibResult,
                         target_file: Path):
        """
        Read a calibration result and serialize it to a given file.
        :param calibration_result: Calib result to be serialized.
        :param target_file: Path to the file the result is written to.
        """
        raise NotImplementedError


class RecorderAndDumper(metaclass=ABCMeta):
    """
    Record and dump calibration data.
    """

    @property
    @abstractmethod
    def recorder(self) -> CalibRecorder:
        """
        Recorder used for acquiring the calibration data
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def dumpers(self) -> List[CalibDumper]:
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
        result = calix.calibrate(self.recorder.calibration_target,
                                 cache_paths=[],  # don't cache
                                 connection=connection)

        for dumper in self.dumpers:
            filename = f"{self.recorder.calibration_type}_" \
                       f"{dumper.format_name}{dumper.format_extension}"
            target_file = deployment_folder.joinpath(filename)
            dumper.dump_calibration(result, target_file)


class HagenCalibRecorder(CalibRecorder):
    """
    Recorder for a canonical Hagen-Mode calibration.
    """
    calibration_type = "hagen"
    calibration_target = HagenCalibTarget()


class HagenSyninCalibRecorder(CalibRecorder):
    """
    Recorder for a Hagen-Mode calibration with integration on the
    synaptic input lines, as opposed to neuron membranes.
    """
    calibration_type = "hagen_synin"
    calibration_target = HagenSyninCalibTarget()


class SpikingCalibRecorder(CalibRecorder):
    """
    Recorder for a default Spiking-Mode calibration.
    """
    calibration_type = "spiking"
    calibration_target = SpikingCalibTarget()


class SpikingCalibRecorder2(CalibRecorder):
    """
    Recorder for a second Spiking-Mode calibration.

    In contrast to the default "spiking" calibration, we use smaller
    synaptic and membrane time constants, but stronger currents per event.
    Synaptic events should be shaprer in the sense that they are stronger,
    but decay more quickly. Also, the spike threshold is increased.

    This calibration will, among other uses, be used in the demo
    notebooks, for the yin-yang example.
    """

    calibration_type = "spiking2"
    calibration_target = SpikingCalibTarget(
        neuron_target=calix.spiking.neuron.NeuronCalibTarget(
            leak=80,
            reset=80,
            threshold=150,
            tau_mem=6 * pq.us,
            tau_syn=6 * pq.us,
            i_synin_gm=500,
            synapse_dac_bias=1000)
    )


class CalixFormatDumper(CalibDumper):
    """
    Dumper for the calix-internal data format.
    """
    format_name = "calix-native"
    format_extension = ".pkl"

    def dump_calibration(self, calibration_result: CalibResult,
                         target_file: Path):
        with target_file.open(mode="wb") as target:
            pickle.dump(calibration_result, target)


class CocoListPortableBinaryFormatDumper(CalibDumper):
    """
    Dumper for the Coordinate-Container-List data in portable binary format.
    """
    format_name = "cocolist"
    format_extension = ".pbin"

    def dump_calibration(self, calibration_result: CalibResult,
                         target_file: Path):
        builder = PlaybackProgramBuilderDumper()
        calibration_result.apply(builder)

        with target_file.open(mode="wb") as target:
            target.write(to_portablebinary(builder.done()))


class CocoListJsonFormatDumper(CalibDumper):
    """
    Dumper for the Coordinate-Container-List data in json format
    with gzip compression.
    """
    format_name = "cocolist"
    format_extension = ".json.gz"

    def dump_calibration(self, calibration_result: CalibResult,
                         target_file: Path):
        builder = PlaybackProgramBuilderDumper()
        calibration_result.apply(builder)

        with gzip.open(target_file, mode="wt") as target:
            target.write(to_json(builder.done()))


class HagenCalib(RecorderAndDumper):
    recorder = HagenCalibRecorder()
    dumpers = [CalixFormatDumper(),
               CocoListPortableBinaryFormatDumper(),
               CocoListJsonFormatDumper()]


class HagenSyninCalib(RecorderAndDumper):
    recorder = HagenSyninCalibRecorder()
    dumpers = [CalixFormatDumper(),
               CocoListPortableBinaryFormatDumper(),
               CocoListJsonFormatDumper()]


class SpikingCalib(RecorderAndDumper):
    recorder = SpikingCalibRecorder()
    dumpers = [CalixFormatDumper(),
               CocoListPortableBinaryFormatDumper(),
               CocoListJsonFormatDumper()]


class SpikingCalib2(RecorderAndDumper):
    recorder = SpikingCalibRecorder2()
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
        for calib in [HagenCalib(), HagenSyninCalib(),
                      SpikingCalib(), SpikingCalib2()]:
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
