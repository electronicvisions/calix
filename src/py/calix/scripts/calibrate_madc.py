#!/usr/bin/env python3
from argparse import ArgumentParser
from datetime import datetime
from scipy import stats
import numpy as np
import json

from calix.common.base import StatefulConnection
from calix.common.madc import MADCCharacterization
from dlens_vx_v3 import hxcomm

parser = ArgumentParser(
    description='Record MADC samples while applying an external voltage with '
                'DAC25. Perform linear fit to madc_value vs voltage and save '
                'result in JSON file. Create plot with measured values and '
                'if desired.')
parser.add_argument('-start_value',
                    type=int,
                    default=200,
                    help='Start value (in DAC) for external voltage supply')
parser.add_argument('-stop_value',
                    type=int,
                    default=1800,
                    help='Stop value (in DAC) for external voltage supply')
parser.add_argument('-step_size',
                    type=int,
                    default=10,
                    help='Difference between external DAC values')
parser.add_argument('--save_plot',
                    action='store_true',
                    help='Save plot of measured values and linear fit')
args = parser.parse_args()


with hxcomm.ManagedConnection() as connection:
    stateful_connection = StatefulConnection(connection)
    calibration = MADCCharacterization(range(args.start_value,
                                             args.stop_value,
                                             args.step_size))
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

if args.save_plot:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    # Measurement
    ax.plot(calib_result['madc_values'], calib_result['voltages'], 'k+',
            label='Measurement Point')

    # Fit
    x = np.linspace(np.min(calib_result['madc_values']),
                    np.max(calib_result['madc_values']),
                    1000)
    y = x * res.slope + res.intercept
    ax.plot(x, y, 'r', label='Linear Fit')

    # Labels
    ax.set_xlabel('MADC value')
    ax.set_ylabel('External Voltage (V)')
    ax.legend()
    fig.savefig('madc_calibration_plot.svg')
