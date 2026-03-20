#!/usr/bin/env python3
from argparse import ArgumentParser
from datetime import datetime
from scipy import stats
from typing import Dict, List
import numpy as np
import json
import quantities as pq
from matplotlib.axes import Axes
import matplotlib.pyplot as plt

from calix.common.base import StatefulConnection
from calix.common.cadc_characterization import CADCCharacterization
from calix.common import cadc
from dlens_vx_v3 import hxcomm


def main(connection: StatefulConnection,
         start_value: float,
         stop_value: float,
         n_steps: int):
    # calibrate CADCs
    cadc_result = cadc.calibrate(stateful_connection)

    calibration = CADCCharacterization(
        np.linspace(args.start_value, args.stop_value, args.n_steps) * pq.V)
    calib_result = calibration.run(stateful_connection)

    # Perform linear fit
    slope = []
    intercept = []
    for channel_type in range(calib_result['cadc_values'].shape[1]):
        slope.append([])
        intercept.append([])
        for channel in range(calib_result['cadc_values'].shape[2]):
            res = stats.linregress(
                calib_result['cadc_values'][:, channel_type, channel],
                calib_result['voltages'])
            slope[-1].append(res.slope)
            intercept[-1].append(res.intercept)

    return calib_result, slope, intercept


def plot_linear_fit(
        ax: Axes,
        calib_result: Dict,
        slope: List,
        intercept: List):
    # Measurement
    for channel_type in range(calib_result['cadc_values'].shape[1]):
        for channel in range(calib_result['cadc_values'].shape[2]):
            ax.errorbar(
                x=calib_result['cadc_values'][:, channel_type, channel],
                y=calib_result['voltages'],
                xerr=calib_result['cadc_values_std'][:, channel_type, channel],
                ls='',
                marker='.')

            # Fit
            x = np.linspace(np.min(calib_result['cadc_values'][:, channel_type, channel]),
                            np.max(calib_result['cadc_values'][:, channel_type, channel]),
                            100)
            y = x * slope[channel_type][channel] + intercept[channel_type][channel]
            ax.plot(x, y, 'r', lw=0.3)

    # Labels
    ax.set_xlabel('CADC value')
    ax.set_ylabel('External Voltage (V)')


def plot_residuals(
        ax: Axes,
        calib_result: Dict,
        slope: List,
        intercept: List):
    # Measurement
    all_residuals = []
    for channel_type in range(calib_result['cadc_values'].shape[1]):
        for channel in range(calib_result['cadc_values'].shape[2]):
            local_slope = slope[channel_type][channel]
            local_intercept = slope[channel_type][channel]
            y = calib_result['cadc_values'][:, channel_type, channel] * local_slope + local_intercept
            residuals_y = y - calib_result['voltages'].rescale(pq.V).magnitude
            ax.plot(calib_result['cadc_values'][:, channel_type, channel], residuals_y, 'r', lw=0.3, ls='', marker='.', alpha=0.1)
            all_residuals.append(residuals_y)

    # Labels
    ax.set_xlabel('CADC value')
    ax.set_ylabel('External Voltage (V)')


if __name__ == '__main__':
    parser = ArgumentParser(
        description='Record CADC samples while applying an external voltage with '
                    'DAC25. Perform linear fit to cadc_value vs voltage and save '
                    'result in JSON file. Create plot with measured values and '
                    'if desired.')
    parser.add_argument('-start_value',
                        type=float,
                        default=0.2,
                        help='Start value (in V) for external voltage supply')
    parser.add_argument('-stop_value',
                        type=float,
                        default=1.0,
                        help='Stop value (in V) for external voltage supply')
    parser.add_argument('-n_steps',
                        type=int,
                        default=15,
                        help='Number of steps to take between start and stop '
                             'value.')
    parser.add_argument('--plot',
                        action='store_true',
                        help='Save plot of measured values and linear fit')
    args = parser.parse_args()

    with hxcomm.ManagedConnection() as connection:
        stateful_connection = StatefulConnection(connection)
        calib_results, slope, intercept = main(
            stateful_connection, args.start_value,
            args.stop_value, args.n_steps)

        # Save result as JSON file
        result = dict(description='CADC calibration result: voltage = slope * '
                                  'cadc_value + intercept. The voltage is given in V.'
                                  'Voltage and slope are of shape (n_channel_types, '
                                  'n_channels). I.e. one channel per neuron for causal'
                                  '/acausal channels.',
                      date=str(datetime.now()),
                      slope=slope,
                      intercept=intercept)
        with open('cadc_calibration_result.json', 'w') as json_file:
            json.dump(result, json_file, indent=4)

    if args.plot:
        fig, ax = plt.subplots()
        plot_linear_fit(ax, calib_results, slope, intercept)
        fig.savefig('cadc_calibration_plot.pdf')

        fig, ax = plt.subplots()
        plot_residuals(ax, calib_results, slope, intercept)
        fig.savefig('cadc_calibration_residuals.pdf')
