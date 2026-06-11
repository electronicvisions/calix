#!/usr/bin/env python3
"""Record MADC samples versus an applied external voltage, fit the
relation linearly, store the fit in JSON and optionally plot the result."""
import json
from argparse import ArgumentParser
from datetime import datetime
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
from matplotlib.axes import Axes
from scipy import stats

from calix.common.base import StatefulConnection
from calix.common.madc_characterization import MADCCharacterization
from dlens_vx_v3 import hxcomm


def main(connection: StatefulConnection,
         start_voltage: float,
         stop_voltage: float,
         n_steps: int = 10) -> Tuple[Dict, "stats.LinregressResult"]:
    """Run the MADC characterization and a linear fit.

    :param start_voltage: Start value (in V) for the external voltage.
    :param stop_voltage: Stop value (in V) for the external voltage.
    :param n_steps: Number of voltage steps between start and stop.
    :return: Tuple of the calibration result and the linear fit result.
    """
    calibration = MADCCharacterization(
        np.linspace(start_voltage, stop_voltage, n_steps) * pq.V)
    calib_result = calibration.run(connection)

    fit_result = stats.linregress(calib_result['madc_values'],
                                  calib_result['voltages'])
    return calib_result, fit_result


def plot_linear_fit(ax: Axes,
                    calib_result: Dict,
                    fit_result: "stats.LinregressResult") -> None:
    """Plot measured points and the linear fit.

    :param ax: Axes to draw the measurement and fit onto.
    :param calib_result: Calibration result returned by ``main``.
    :param fit_result: Linear fit result returned by ``main``.
    """
    ax.errorbar(
        x=calib_result['madc_values'],
        y=calib_result['voltages'],
        xerr=calib_result['madc_values_std'],
        ls='',
        marker='.',
        label='Measurement Point')

    x = np.linspace(np.min(calib_result['madc_values']),
                    np.max(calib_result['madc_values']),
                    1000)
    y = x * fit_result.slope + fit_result.intercept
    ax.plot(x, y, 'r', label='Linear Fit', lw=0.3)

    ax.set_xlabel('MADC value')
    ax.set_ylabel('External Voltage (V)')
    ax.legend()


def plot_residuals(ax: Axes,
                   calib_result: Dict,
                   fit_result: "stats.LinregressResult") -> None:
    """Plot the fit residuals.

    :param ax: Axes to draw the residuals onto.
    :param calib_result: Calibration result returned by ``main``.
    :param fit_result: Linear fit result returned by ``main``.
    """
    residuals_y = (
        calib_result['madc_values'] * fit_result.slope
        + fit_result.intercept) - calib_result['voltages'].magnitude
    ax.plot(
        calib_result['madc_values'], residuals_y, 'r',
        label=f"Residuals (std: {np.std(residuals_y):.5f} V)",
        ls='', marker='.')

    ax.set_xlabel('MADC value')
    ax.set_ylabel('External Voltage (V)')
    ax.legend()
    ax.figure.tight_layout()


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Record MADC samples while applying an external voltage '
                    'with DAC25. Perform linear fit to madc_value vs voltage '
                    'and save result in JSON file. Create plot with measured '
                    'values and if desired.')
    parser.add_argument('-start_value',
                        type=int,
                        default=0.1,
                        help='Start value (in V) for external voltage supply')
    parser.add_argument('-stop_value',
                        type=int,
                        default=1.0,
                        help='Stop value (in V) for external voltage supply')
    parser.add_argument('-n_steps',
                        type=int,
                        default=10,
                        help='Number of steps to take between start and stop '
                             'value.')
    parser.add_argument('--plot',
                        action='store_true',
                        help='Save plot of measured values and linear fit')
    args = parser.parse_args()

    with hxcomm.ManagedConnection() as conn:
        stateful_connection = StatefulConnection(conn)
        calib_res, fit_res = main(stateful_connection,
                                  args.start_value, args.stop_value, args.n_steps)

    result = dict(description='MADC calibration result: voltage = slope * '
                              'madc_value + intercept. The voltage is given '
                              'in V.',
                  date=str(datetime.now()),
                  slope=fit_res.slope,
                  intercept=fit_res.intercept)
    with open('madc_calibration_result.json', 'w',
              encoding='utf-8') as json_file:
        json.dump(result, json_file, indent=4)

    if args.plot:
        fig, ax = plt.subplots()
        plot_linear_fit(ax, calib_res, fit_res)
        fig.savefig('madc_calibration_residuals.pdf')

        fig, ax = plt.subplots()
        plot_residuals(ax, calib_res, fit_res)
        fig.savefig('madc_calibration_plot.pdf')
