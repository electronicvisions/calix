"""
Functions to print statistics of the neuron membrane potentials
per quadrant.
"""

import numpy as np
from dlens_vx_v1 import halco, logger, hxcomm
from calix.hagen import neuron_helpers


def measure_quadrant_results(connection: hxcomm.ConnectionHandle,
                             success_mask: np.ndarray = np.ones(
                                 halco.NeuronConfigOnDLS.size,
                                 dtype=np.bool)):
    """
    Measure the current membrane potentials (without any inputs) and
    log statistics by quadrant at DEBUG level.

    :param connection: Connection to chip to run on.
    :param success_mask: Mask of successfully calibrated neurons.
        Neurons which have failed are excluded when calculating statistics.
    """

    log = logger.get("calix.hagen.neuron_evaluation.measure_quadrant_results")

    results = neuron_helpers.cadc_read_neuron_potentials(connection)
    results = neuron_helpers.reshape_neuron_quadrants(results)
    success_mask = neuron_helpers.reshape_neuron_quadrants(success_mask)

    for target_quadrant in range(halco.NeuronConfigBlockOnDLS.size):
        log.DEBUG("Quadrant {0:1d}: resting pot. {1:5.2f} +- {2:4.2f}".format(
            target_quadrant,
            np.mean(results[target_quadrant][success_mask[target_quadrant]]),
            np.std(results[target_quadrant][success_mask[target_quadrant]])))
