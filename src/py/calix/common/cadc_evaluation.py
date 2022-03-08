"""
Provides functions which are helpful when evaluating the success of a
CADC calibration.
"""

import numpy as np
from dlens_vx_v3 import halco, sta, logger, hxcomm

from calix.common import cadc_helpers


def compare_results(uncalibrated_data: np.ndarray, calibrated_data: np.ndarray
                    ) -> None:
    """
    Compare results with calibrated offset and uncalibrated reads.

    Logs the mean and standard deviation of CADC reads before and after
    calibration.

    :param uncalibrated_data: Array of CADC samples that will be used as data
        before calibration.
    :param calibrated_data: Array of CADC samples after calibration.
    """
    log = logger.get("calix.common.cadc_evaluation.compare_results")

    uncalibrated_data = cadc_helpers.reshape_cadc_quadrants(uncalibrated_data)
    calibrated_data = cadc_helpers.reshape_cadc_quadrants(calibrated_data)

    for quadrant in range(halco.NeuronConfigBlockOnDLS.size):
        quadrant_mean_before = np.mean(
            uncalibrated_data[quadrant])
        quadrant_mean_after = np.mean(calibrated_data[quadrant])
        quadrant_std_before = np.std(
            uncalibrated_data[quadrant])
        quadrant_std_after = np.std(calibrated_data[quadrant])

        log.DEBUG((
            "Quadrant {0:1d}: before {1:5.2f} +- {2:4.2f}, "
            + "after {3:5.2f} +- {4:4.2f}").format(
                quadrant, quadrant_mean_before, quadrant_std_before,
                quadrant_mean_after, quadrant_std_after))


def check_calibration_success(connection: hxcomm.ConnectionHandle,
                              builder: sta.PlaybackProgramBuilder,
                              read_data: np.ndarray) -> None:
    """
    Compare the given uncalibrated reads to a new read that is done
    with the supplied builder/connection. Logs a warning if the
    deviation of reads after calibration is still high.

    :param connection: Connection to the chip to run on.
    :param builder: Builder to append read instructions to.
    :param read_data: Array of reads before calibration.
    """

    log = logger.get(
        "calix.common.cadc_evaluation.check_calibration_success")

    calibrated_data = cadc_helpers.read_cadcs(connection, builder)
    compare_results(uncalibrated_data=read_data,
                    calibrated_data=calibrated_data)

    if np.std(calibrated_data) > 3:
        log.WARN("CADC channels show a standard deviation of "
                 + f"{np.std(calibrated_data):4.2f} after calibration.")
