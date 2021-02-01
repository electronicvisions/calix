#!/usr/bin/env python3
"""
Calibrate neurons to generate plateau potentials when spiking and saving the
calibration result to disk.
"""
from typing import Union
from pathlib import Path
import numpy as np
import quantities as pq
from dlens_vx_v2 import sta, hxcomm

import calix.spiking
from calix.spiking import SpikingCalibrationResult


###############################################################################
# Helper function
###############################################################################
def plateau_potential_decay(*,
                            v_leak: Union[float, np.ndarray],
                            v_reset: Union[float, np.ndarray],
                            v_thres: Union[float, np.ndarray],
                            tau_mem: pq.quantity.Quantity) -> pq.Quantity:
    """
    Calculate time needed to exponentially decay below threshold.

    In case of plateau potentials the reset potential is above the threshold
    potential. In this function we calculate how long the membrane needs to
    relax below the threshold potential if a simple leaky-integrate-and-fire
    neuron is assumed.
    If the reset potential is below threshold, i.e. no  plateau potential is
    configured, zero is returned.

    :param v_leak: Target leak potentials.
    :param v_reset: Target reset potentials.
    :param v_thres: Target threshold potentials.
    :param tau_mem: Target membrane time constants.
    """
    t_fall = np.zeros(tau_mem.size) * tau_mem.units

    # Only perform following calculations for v_reset > v_thres
    sel = v_reset > v_thres

    # Consider a neuron whichs threshold voltage v_thres is below the reset
    # potential v_reset. Assume that the membrane voltage V_m decreases
    # exponentially with time constant tau_m back to the leak potential
    # v_leak:
    #   V_m = (v_reset - v_leak) * e^(-t/tau_m) + v_leak.
    # Therefore, it needs the time t_fall to reach threshold:
    #                     V_m != v_thres
    #   <=> reset_thres_ratio := (v_reset - v_leak) / (v_thres - v_leak)
    #                          = e^(t_fall/tau_m)
    #   <=>             t_fall = tau_m * ln(reset_thres_ratio)
    reset_thres_ratio = np.asarray((v_reset - v_leak) / (v_thres - v_leak))

    t_fall[sel] = tau_mem[sel] * np.log(reset_thres_ratio[sel])

    return t_fall


###############################################################################
# Define calibration target
###############################################################################
# On BSS-2 plateau potentials are emulated by the refractory mechanism.
# Instead of choosing a reset near the leak potential it is chosen above the
# threshold:
neuron_args = {'leak': 50,
               'threshold': 80,
               'reset': 100,
               'tau_mem': 10 * pq.us}

# At the end of the plateau potential the neuron has to have enough time to
# relax back below threshold in order to not trigger an additional spike.
# On BSS-2 the refractory period is made up of a 'reset period', in which the
# membrane is clamped to the reset voltage and no new spike can be triggered,
# and a 'holdoff period', in which the neuron potential evolves freely but
# still new spikes can be triggered.
# We now chose a holdoff period such that the membrane has enough time to
# relax back below the threshold potential. We add a factor >1 to account for
# possible excitatory synaptic input:
neuron_args['holdoff_time'] = 1.5 * plateau_potential_decay(
    v_leak=neuron_args['leak'],
    v_reset=neuron_args['reset'],
    v_thres=neuron_args['threshold'],
    tau_mem=neuron_args['tau_mem'])

# We choose the refractory time such that we have a reset period of 20 us. This
# will be the length of or plateau potential.
neuron_args['refractory_time'] = 20 * pq.us + neuron_args['holdoff_time']

###############################################################################
# Perform the calibration
###############################################################################
with hxcomm.ManagedConnection() as connection:
    builder, _ = sta.ExperimentInit().generate()
    sta.run(connection, builder.done())

    calibration_result = calix.spiking.calibrate(connection,
                                                 neuron_kwargs=neuron_args)

###############################################################################
# Save calibration as portable binary
###############################################################################
target_file = Path('my_calibration.pbin')

builder = sta.PlaybackProgramBuilderDumper()
calibration_result.apply(builder)

with target_file.open(mode="wb") as target:
    target.write(sta.to_portablebinary(builder.done()))
