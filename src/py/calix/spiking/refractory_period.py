'''
Calculations related to determining the refractory clock scales as well as
counter values.
'''
from dataclasses import dataclass
from typing import Optional

import numpy as np
import quantities as pq

from dlens_vx_v3 import hal, halco, logger


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

    :raises ValueError: If `tau_ref` and `holdoff_time` have shapes
        that are incompatible with the number of neurons on chip.
    """

    try:
        tau_ref = np.broadcast_to(tau_ref, (halco.NeuronConfigOnDLS.size,)) \
            * tau_ref.units  # quantities lose units when reshaping...
        holdoff_time = np.broadcast_to(
            holdoff_time, (halco.NeuronConfigOnDLS.size,)) \
            * holdoff_time.units  # quantities lose units when reshaping...
    except ValueError as err:
        raise ValueError(
            f"Shapes of `tau_ref` ({tau_ref.shape}) and `holdoff_time` "
            + f"({tau_ref.shape}) can not be resized to match the "
            + f"number of neurons on chip ({halco.NeuronConfigOnDLS.size})."
        ) from err

    # The factor of 2 and addition of 1 accounts for the fact that the
    # lowest bit of the refractory counter is always used for holdoff
    max_cycles_holdoff = 2 * hal.NeuronBackendConfig.ResetHoldoff.max + 1

    # Calculate sensible minimum cycles:
    # We choose the minimum number of clock cycles to be 30 in order to
    # keep the relative quantization error below approximately 3%.
    # Also, there are issues when setting the refractory counter too low,
    # cf. issue 3741.
    min_cycles_refractory = 30
    min_cycles_holdoff = 1

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
            np.min(tau_ref[fast_neurons]) / min_cycles_refractory,
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
        max_time_per_cycle_holdoff = np.min(holdoff_time) / min_cycles_holdoff
        max_time_per_cycle_refractory = np.min(tau_ref[slow_neurons]) \
            / min_cycles_refractory

        clock_settings.slow_clock = _calculate_clock_scale(
            max(min_time_per_cycle_holdoff, min_time_per_cycle_refractory),
            min(max_time_per_cycle_holdoff, max_time_per_cycle_refractory),
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
                           max_time_per_cycle: pq.Quantity,
                           f_base: pq.Quantity) -> int:
    '''
    Calculate clock scale needed to have a suitable time per cycle.

    The clock scaler is chosen such that the minimum time per cycle
    is fulfilled. Then, it tries to also fulfil the maximum time per
    cycle. If there's still headroom, a medium clock scaler will
    be preferred.

    :param min_time_per_cycle: Minimum time one cycle should take.
    :param max_time_per_cycle: Maximum time one cycle is allowed to take.
    :param f_base: Base frequency of the refractory clock.

    :return: Minimum clock scale for which one cycle needs at least the
        given time.

    :raises ValueError: If requested minimum time per clock cycle is
        too large for BSS-2.
    '''

    # Only fulfil maximum condition in case it's not too tight
    if max_time_per_cycle < 2 * min_time_per_cycle:
        max_time_per_cycle = 2 * min_time_per_cycle

    # Find matching clock scaler, starting at ideal value:
    # We start with a clock scaler of 4 since this scaler should still
    # give enough head room to alter the refractory counters manually
    # after the calibration. This facilitates the exploration of
    # neuron properties in higher level software stacks.
    clock_scale = 4

    for _ in range(hal.CommonNeuronBackendConfig.ClockScale.max):
        time_per_cycle = 1 / (f_base / 2) * (2 ** clock_scale)
        if min_time_per_cycle <= time_per_cycle <= max_time_per_cycle:
            break
        if time_per_cycle < min_time_per_cycle:
            clock_scale += 1
        elif time_per_cycle > max_time_per_cycle:
            clock_scale -= 1

    if clock_scale < hal.CommonNeuronBackendConfig.ClockScale.min:
        # Reaching the fastest clock is not a hard limit, as the counter will
        # just be smaller than ideal.
        clock_scale = hal.CommonNeuronBackendConfig.ClockScale.min
        logger.get("calix.spiking.refractory_period").warn(
            "Refractory clock scaler has reached the fastest possible "
            + "setting. Expect refractory and holdoff times to be less "
            + "precise.")
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
