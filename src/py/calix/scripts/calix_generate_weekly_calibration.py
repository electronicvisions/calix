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
    calibration_target = calix.spiking.SpikingCalibTarget(
        neuron_target=calix.spiking.neuron.NeuronCalibTarget(
            tau_syn=5 * pq.us,
            tau_mem=20 * pq.us,
            reset=80,
            leak=100,
            threshold=130,
            synapse_dac_bias=1000,
            i_synin_gm=400),
        correlation_target=correlation.CorrelationCalibTarget(
            amplitude=1.5, time_constant=30 * pq.us)
    )
    calibration_options = calix.spiking.SpikingCalibOptions(
        correlation_options=correlation.CorrelationCalibOptions(
            calibrate_synapses=True,
            branches=correlation.CorrelationBranches.CAUSAL,
            default_amp_calib=1, v_res_meas=0.95 * pq.V))


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
