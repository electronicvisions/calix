'''
Calculations related to determining the refractory clock scales as well as
counter values.
'''
from dataclasses import dataclass
import math
from typing import Optional

import numpy as np
import quantities as pq

from dlens_vx_v2 import hal


# Assume refractory clock frequency yo be unchanged after
# sta.ExperimentInit:
# This should be replaced by looking it up from a chip object,
# see issue 3955.
_clock_base_frequency = hal.ADPLL().calculate_output_frequency(
    output=hal.ADPLL().Output.core_1) * pq.Hz


@dataclass
class Settings:
    """
    Class providing array-like access to settings related to the refractory
    clock.
    """
    refractory_counters: Optional[np.ndarray] = None
    reset_holdoff: Optional[np.ndarray] = None
    input_clock: Optional[np.ndarray] = None
    fast_clock: int = 0
    slow_clock: int = 0


def calculate_settings(tau_ref: pq.Quantity,
                       holdoff_time: pq.Quantity = 0 * pq.us) -> Settings:
    """
    Determine scales of fast and slow clock as well as refractory counter
    settings of individual neurons.

    On BSS-2 the refractory period is made up of a "reset period" where the
    potential is clamped to the reset potential and a "holdoff period" in
    which the membrane can evolve freely but no new spikes are triggered.

    In this function the clock scalers and counter values are chosen such that
    the given targets for both parameters can be fulfilled.

    :param tau_ref: Target refractory times. This is the total time in which
        the neuron does not emit new spikes and therefore includes the holdoff
        period.
    :param holdoff_time: Target length of the holdoff period.
        Note: Due to hardware constraint the holdoff period is always at least
        one clock cycle long. For a target of zero the minimal holdoff period
        is chosen.

    :return: Clock settings.

    :raises ValueError: If `tau_ref` and `holdoff_time` have different sizes
        and none of them is a scalar value.
    """
    if tau_ref.size != holdoff_time.size:
        # try to resize arrays to same sizes
        if tau_ref.size == 1:
            tau_ref = tau_ref * np.ones(holdoff_time.size)
        elif holdoff_time.size == 1:
            holdoff_time = holdoff_time * np.ones(tau_ref.size)
        else:
            raise ValueError("Shapes of `tau_ref` and `holdoff_time` do not "
                             "match and can not be resized properly: "
                             f"{tau_ref.size} != {holdoff_time.size}.")

    # The factor of 2 and addition of 1 accounts for the fact that the
    # lowest bit of the refractory counter is always used for holdoff
    max_cycles_holdoff = 2 * hal.NeuronBackendConfig.ResetHoldoff.max + 1

    clock_settings = Settings(
        refractory_counters=np.empty(tau_ref.size, dtype=int),
        reset_holdoff=hal.NeuronBackendConfig.ResetHoldoff.max
        * np.ones(tau_ref.size, dtype=int),
        input_clock=np.ones(tau_ref.size, dtype=int))

    # Assign neurons to fast clock
    fast_neurons = tau_ref <= np.mean(tau_ref)

    if np.sum(fast_neurons) > 0:
        clock_settings.fast_clock = _calculate_clock_scale(
            np.max(tau_ref[fast_neurons])
            / hal.NeuronBackendConfig.RefractoryTime.max,
            _clock_base_frequency)
        clock_settings.refractory_counters[fast_neurons] = \
            _calculate_clock_cycles(tau_ref[fast_neurons],
                                    clock_settings.fast_clock,
                                    _clock_base_frequency)

    slow_neurons = ~fast_neurons
    # Reassign neurons to the slow clock if the `holdoff_time` can not be
    # supported by the fast scale
    slow_neurons |= holdoff_time > \
        max_cycles_holdoff * _clock_period(clock_settings.fast_clock,
                                           _clock_base_frequency)

    if np.sum(slow_neurons) > 0:
        clock_settings.input_clock[slow_neurons] = 0

        min_time_per_cycle_holdoff = np.max(holdoff_time) / max_cycles_holdoff
        min_time_per_cycle_refractory = np.max(tau_ref[slow_neurons]) \
            / hal.NeuronBackendConfig.RefractoryTime.max

        clock_settings.slow_clock = _calculate_clock_scale(
            max(min_time_per_cycle_holdoff, min_time_per_cycle_refractory),
            _clock_base_frequency)
        clock_settings.refractory_counters[slow_neurons] = \
            _calculate_clock_cycles(tau_ref[slow_neurons],
                                    clock_settings.slow_clock,
                                    _clock_base_frequency)

        real_clock_cycles = _calculate_clock_cycles(holdoff_time[slow_neurons],
                                                    clock_settings.slow_clock,
                                                    _clock_base_frequency)
        clock_settings.reset_holdoff[slow_neurons] = np.ceil(
            hal.NeuronBackendConfig.ResetHoldoff.max
            - (real_clock_cycles - 1) / 2).astype(int)

    return clock_settings


def _calculate_clock_scale(min_time_per_cycle: pq.Quantity,
                           f_base: pq.Quantity) -> int:
    '''
    Calculate clock scale needed to have at least the given amount of time
    per cycle.

    :param min_time_per_cycle: Minimum time one cycle should take.
    :param f_base: Base frequency of the refractory clock.
    :return: Minimum clock scale for which one cycle needs at least the
        given time.
    '''
    # number of "base clock cycles" per desired cycle length.
    # Factor of 2 since clock is always down scaled by 2.
    cycles = (min_time_per_cycle * f_base / 2).simplified
    if cycles > 1:
        clock_scale = math.ceil(np.log2(cycles))
    else:
        clock_scale = 0

    if clock_scale > hal.CommonNeuronBackendConfig.ClockScale.max:
        raise ValueError("Refractory times or holdoff times are larger "
                         "than feasible.")
    return clock_scale


def _calculate_clock_cycles(refractory_times: pq.Quantity,
                            clock_scale: int,
                            f_base: pq.Quantity) -> np.ndarray:
    '''
    Convert refractory time to clock cycles.

    :param refractory_times: Refractory times in time unit.
    :param clock_scale: Clock scaler of refrectory clock
    :param f_base: Base frequency of the refractory clock.
    :return: Refractory times in clock cycles.
    '''
    # output of the refractory clock
    frequency = f_base / 2**(clock_scale + 1)

    # Last bit of refractory counter is always used for holdoff
    return (refractory_times * frequency).simplified.astype(int) + 1


def _clock_period(clock_scale: int, f_base: pq.Quantity) -> pq.Quantity:
    '''
    Calculate period of refractory clock.

    :param clock_scale: Scalinf factor of refrctory clock.
    :param f_base: Base frequency of the refractory clock.
    :return: Period length of refractory clock
    '''
    # output of the refractory clock
    return 2**(clock_scale + 1) / f_base
