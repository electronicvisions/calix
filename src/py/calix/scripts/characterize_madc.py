#!/usr/bin/env python3
from argparse import ArgumentParser
from datetime import datetime
from scipy import stats
import numpy as np
import json
import quantities as pq

from calix.common.base import StatefulConnection
from calix.common.madc_characterization import MADCCharacterization
from dlens_vx_v3 import hxcomm

parser = ArgumentParser(
    description='Record MADC samples while applying an external voltage with '
                'DAC25. Perform linear fit to madc_value vs voltage and save '
                'result in JSON file. Create plot with measured values and '
                'if desired.')
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


with hxcomm.ManagedConnection() as connection:
    stateful_connection = StatefulConnection(connection)
    calibration = MADCCharacterization(
        np.linspace(args.start_value, args.stop_value, args.n_steps) * pq.V)
    calib_result = calibration.run(stateful_connection)

# Perform linear fit
res = stats.linregress(calib_result['madc_values'],
                       calib_result['voltages'])

# Save result as JSON file
result = dict(description='MADC calibration result: voltage = slope * '
                          'madc_value + intercept. The voltage is given in V.',
              date=str(datetime.now()),
              slope=res.slope,
              intercept=res.intercept)
with open('madc_calibration_result.json', 'w') as json_file:
    json.dump(result, json_file, indent=4)

if args.plot:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    # Measurement
    ax.errorbar(
        x=calib_result['madc_values'],
        y=calib_result['voltages'],
        xerr=calib_result['madc_values_std'],
        ls='',
        marker='.',
        label='Measurement Point')

    # Fit
    x = np.linspace(np.min(calib_result['madc_values']),
                    np.max(calib_result['madc_values']),
                    1000)
    y = x * res.slope + res.intercept
    ax.plot(x, y, 'r', label='Linear Fit', lw=0.3)

    # Labels
    ax.set_xlabel('MADC value')
    ax.set_ylabel('External Voltage (V)')
    ax.legend()
    fig.savefig('madc_calibration_plot.pdf')

    fig, ax = plt.subplots()

    # Residuals
    residuals_y = (calib_result['madc_values'] * res.slope + res.intercept) - calib_result['voltages'].magnitude
    ax.plot(calib_result['madc_values'], residuals_y, 'r', label=f"Residuals (std: {np.std(residuals_y):.5f} V)", ls='', marker='.')

    # Labels
    ax.set_xlabel('MADC value')
    ax.set_ylabel('External Voltage (V)')
    ax.legend()
    fig.tight_layout()
    fig.savefig('madc_calibration_residuals.pdf')
