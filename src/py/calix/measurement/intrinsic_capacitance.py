"""
Measure the intrinsic capacitance.

Each neuron circuit has an intrinsic capacitance C_i. The total
capacitance of the neuron circuit is given by the intrinsic capacitance
and the adjustable capacitance C_a: C = C_a + C_i.

We will measure the time constant tau at different values
of the capacitance and perform a fit to determine the intrinsic
capacitance:

tau = (C_a + C_i) / g,

where g is the leak conductance.
"""

from copy import deepcopy
from typing import Sequence, List, Optional
import numpy as np
import quantities as pq
from scipy.optimize import curve_fit

from dlens_vx_v3 import hal, halco

from calix import constants
from calix.common import algorithms, base, helpers
from calix.hagen import neuron_leak_bias


def _prepare_chip(connection: base.StatefulConnection,
                  tau_mem: pq.Quantity = 80 * pq.us,
                  v_leak_cap_mem: int = 500):
    """
    Configure neurons.

    Steps:
    - disable firing
    - set membrane as a readout source
    - set leak potential
    - calibrate membrane time constant

    :param connection: Connection to chip to run on.
    :param tau_mem: Membrane time constant to calibrate. Should be
        chosen rather high since the time constant gets smaller when we
        decrease the capacitance.
    :param v_leak_cap_mem: CapMem value of leak potential. The exact value
        of the leak potential is not important for the characterization
        but we want to ensure that the neurons are in a "good" state.
    """

    # Use default configuration, i.e., firing and synaptic
    # inputs disabled
    neuron_config = hal.NeuronConfig()
    neuron_config.enable_readout_amplifier = True
    neuron_config.membrane_capacitor_size = \
        hal.NeuronConfig.MembraneCapacitorSize.max
    neuron_config.readout_source = hal.NeuronConfig.ReadoutSource.membrane

    neuron_configs = [deepcopy(neuron_config) for _
                      in halco.iter_all(halco.NeuronConfigOnDLS)]
    builder = base.WriteRecordingPlaybackProgramBuilder()
    for coord, config in zip(halco.iter_all(halco.NeuronConfigOnDLS),
                             neuron_configs):
        builder.write(coord, config)

    # Set leak potential
    builder = helpers.capmem_set_neuron_cells(
        builder, {halco.CapMemRowOnCapMemBlock.v_leak: v_leak_cap_mem})

    builder = helpers.wait(builder, constants.capmem_level_off_time)
    base.run(connection, builder)

    # Calibrate membrane time constant
    calibration = neuron_leak_bias.MembraneTimeConstCalibOffset(
        target=tau_mem, neuron_configs=neuron_configs)
    calibration.run(connection, algorithm=algorithms.NoisyBinarySearch())

    return neuron_configs


def fit(capacitances: np.ndarray,
        time_constants: np.ndarray) -> List:
    """
    Determine the intrinsic capacitance with a linear fit.

    See module doc-string for more details.

    :param capacitances: Capacitance values at which the membrane time
        constants were recorded.
    :param time_constants: Recorded time constants for each capacitance
        and each neuron. Should have shape (#capacitances, #neurons).
    :return: List with a dictionary for each neuron. The dictionary
        contains the "parameters" of the fit as well as the "covariance"
        matirx.
    """
    result = []
    for neuron_data in np.array(time_constants):
        mean_tau = np.mean(neuron_data, axis=1)

        if neuron_data.shape[-1] > 1:
            std_tau = np.std(neuron_data, axis=1)
            absolute_sigma = True
        else:
            std_tau = 1
            absolute_sigma = False
        fit_res = curve_fit(fit_function,
                            capacitances,
                            mean_tau,
                            sigma=std_tau,
                            absolute_sigma=absolute_sigma)
        result.append({"parameters": fit_res[0],
                       "covariance": fit_res[1]})
    return result


def fit_function(set_capacitance: float,
                 intrinsic_capacitance: float,
                 leak_conductance: float) -> float:
    """
    Membrane time constant as a function of the capacitance and
    leak conductance.

    We assume that the total capacitance is given by a capacitance C_a
    which we can set and by an intrinsic capacitance C_i which is always
    present:

    tau = (C_a + C_i) / g,

    where g is the leak conductance.

    :param set_capacitance: Capacitance which can be controlled C_a.
    :param intrinsic_capacitance: Intrinsic capacitance which is always
        present.
    :param leak_conductance: Leak conductance.
    :return: Membrane time constant.
    """
    return (set_capacitance + intrinsic_capacitance) / leak_conductance


def measure(connection: base.StatefulConnection,
            tau_mem: pq.Quantity = 100 * pq.us,
            v_leak_cap_mem: int = 400,
            test_values: Optional[Sequence[int]] = None,
            n_rep: int = 1) -> np.ndarray:
    """
    Measure the membrane time constant for each neuron at different
        capacitance values.

    :param connection: Connection to chip to run on.
    :param tau_mem: Membrane time constant at the maximum capacitance.
        This will be calibrated and should be chosen rather high since the
        time constant gets smaller when we decrease the capacitance.
    :param v_leak_cap_mem: CapMem value of leak potential. The exact value
        of the leak potential is not important for the characterization
        but we want to ensure that the neurons are in a "good" state.
    :param n_rep: how often to measure the membrane time constant at
        each capacitance setting. The measured values are averaged
        before the fit is performed.
    :return: Measured membrane time constant. The output is arranged as
        follows: (neuron, capacitance, repetition).
    """
    if test_values is None:
        test_values = np.linspace(
            10, hal.NeuronConfig.MembraneCapacitorSize.max, 5, dtype=int)
    # prepare measurement
    neuron_configs = _prepare_chip(
        connection, tau_mem=tau_mem, v_leak_cap_mem=v_leak_cap_mem)
    calibration = neuron_leak_bias.MembraneTimeConstCalibOffset(
        target=tau_mem, neuron_configs=neuron_configs)
    calibration.adjust_bias_range = False
    calibration.prelude(connection)

    # measure time constant for different capacitance values
    time_constants = []
    for value in test_values:
        builder = base.WriteRecordingPlaybackProgramBuilder()
        for coord, config in zip(halco.iter_all(halco.NeuronConfigOnDLS),
                                 calibration.neuron_configs):
            config.membrane_capacitor_size = \
                hal.NeuronConfig.MembraneCapacitorSize(value)
            builder.write(coord, config)
        builder = helpers.wait(builder, 10 * tau_mem)
        base.run(connection, builder)

        for _ in range(n_rep):
            builder = base.WriteRecordingPlaybackProgramBuilder()
            time_constants.append(
                calibration.measure_results(connection, builder))

    calibration.postlude(connection)

    # currently, we have the values saved in a list of list.
    # rearrange values such that we have (neuron, capacitance, repetition)
    time_constants = np.array(time_constants).reshape(
        (len(test_values), n_rep, -1))
    return np.transpose(time_constants, (2, 0, 1))
