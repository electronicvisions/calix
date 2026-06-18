#!/usr/bin/env python3
"""
Execute more specific, long-running calibrations and save them in all
available data formats. If result files already exist, they are
overwritten.
"""

import argparse
from pathlib import Path

import quantities as pq

from dlens_vx_v3 import logger
from dlens_vx_v3.hxcomm import ManagedConnection
from dlens_vx_v3.halco import iter_all, AtomicNeuronOnDLS

from pyccalix import CorrelationCalibOptions
import calix.spiking
from calix.spiking import correlation
from calix.scripts.calix_generate_default_calibration import \
    CalibRecorder, RecorderAndDumper, CalixFormatDumper, \
    CocoListPortableBinaryFormatDumper, CocoListJsonFormatDumper
from calix.common.base import StatefulConnection

log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class CorrelationCalibRecorder(CalibRecorder):
    """
    Recorder for a spiking neuron calibration with correlation sensors
    calibrated.
    """

    calibration_type = "correlation"
    calibration_target = calix.spiking.SpikingCalibTarget()
    for an in iter_all(AtomicNeuronOnDLS):
        calibration_target.neuron_target.tau_syn[an].fill(5e-6)  # pq.s
    calibration_target.neuron_target.tau_mem.fill(20e-6)  # pq.s
    calibration_target.neuron_target.reset.fill(80)
    calibration_target.neuron_target.leak.fill(100)
    calibration_target.neuron_target.threshold.fill(130)
    calibration_target.neuron_target.synapse_dac_bias = 1000
    calibration_target.neuron_target.cuba_synin.i_synin_gm = 400
    calibration_target.correlation_target.amplitude = 1.5
    calibration_target.correlation_target.time_constant = 30e-6  # pq.s
    calibration_options = calix.spiking.SpikingCalibOptions()
    calibration_options.correlation_options.calibrate_synapses = True
    calibration_options.correlation_options.branches = \
        correlation.CorrelationBranches.CAUSAL
    calibration_options.correlation_options.default_amp_calib = 1
    calibration_options.correlation_options.v_res_meas = 0.95  # pq.V


class CorrelationCalib(RecorderAndDumper):
    recorder = CorrelationCalibRecorder()
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
        for calib in [CorrelationCalib()]:
            calib.record_and_dump(
                StatefulConnection(connection), deployment_folder)


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
