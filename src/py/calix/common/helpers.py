"""
Various helper functions that are more general and thus not contained
in one of the named modules.
"""

from typing import Union, Dict, Optional, Tuple
import warnings
import numpy as np
import quantities as pq
from dlens_vx_v3 import hal, sta, halco

import pyccalix


def wait_for_us(builder: sta.PlaybackProgramBuilder, waiting_time: float
                ) -> sta.PlaybackProgramBuilder:
    """
    Waits for a given amount of time in us.

    This function appends instructions to the given builder which
    first reset the timer and then wait until the given time is reached.

    :note: This function is deprecated. Use `wait()` instead.

    :param builder: Builder to add wait instruction to.
    :param waiting_time: Time in us to wait for.

    :return: Builder with wait instruction added to.
    """
    warnings.warn("This function is deprecated. Use "
                  "`calix.common.helpers.wait()` instead.",
                  DeprecationWarning, stacklevel=2)

    # Returning the modified builder is regarded as bad style.
    # This returning should be removed, also elsewhere, cf. issue 3952
    builder.write(halco.TimerOnDLS(), hal.Timer())
    builder.block_until(halco.TimerOnDLS(), int(
        waiting_time * int(hal.Timer.Value.fpga_clock_cycles_per_us)))
    return builder


def wait(builder: sta.PlaybackProgramBuilder,
         waiting_time: pq.quantity.Quantity) -> sta.PlaybackProgramBuilder:
    """
    Waits for a given amount of time.

    This function appends instructions to the given builder which
    first reset the timer and then wait until the given time is reached.

    :param builder: Builder to add wait instruction to.
    :param waiting_time: Time to wait for.

    :return: Builder with wait instruction added to.
    """

    # Returning the modified builder is regarded as bad style.
    # This returning should be removed, also elsewhere, cf. issue 3952
    builder.write(halco.TimerOnDLS(), hal.Timer())
    builder.block_until(halco.TimerOnDLS(), int(
        waiting_time.rescale(pq.us)
        * int(hal.Timer.Value.fpga_clock_cycles_per_us)))
    return builder


def capmem_noise(start: int = -5, end: int = 6,
                 size: Optional[Union[int, Tuple[int]]] = None
                 ) -> Union[int, np.ndarray]:
    """
    Creates random integers between start and end-1.
    Used mainly to vary CapMem settings in order to avoid
    setting many cells to the same value.

    :param start: Lower end of the random range.
    :param end: One above the upper end of the random range.
    :param size: Number/shape of values to draw. If None, a single integer
        is returned.

    :return: Array with integer noise, or single integer.
    """

    return np.random.randint(start, end, size)


def capmem_set_quadrant_cells(
        builder: sta.PlaybackProgramBuilder,
        config: Dict[halco.CapMemCellOnCapMemBlock, Union[int, np.ndarray]]
) -> sta.PlaybackProgramBuilder:
    """
    Set multiple CapMem cells that are global per quadrant to the same
    provided values.

    :param builder: Builder to append configuration to.
    :param config: Dict with the desired configuration.

    :return: Builder with configuration appended.
    """

    for capmem_block_id, capmem_block in enumerate(
            halco.iter_all(halco.CapMemBlockOnDLS)):
        for cell, value in config.items():
            coord = halco.CapMemCellOnDLS(cell, capmem_block)

            if isinstance(value, np.ndarray):
                builder.write(coord, hal.CapMemCell(
                    hal.CapMemCell.Value(value[capmem_block_id])))
            else:
                builder.write(coord, hal.CapMemCell(
                    hal.CapMemCell.Value(value)))

    return builder


def capmem_set_neuron_cells(
        builder: sta.PlaybackProgramBuilder,
        config: Dict[halco.CapMemRowOnCapMemBlock, Union[int, np.ndarray]]
) -> sta.PlaybackProgramBuilder:
    """
    Set single CapMem rows on the neurons to the desired values.
    Expects parameters to be configured along with the
    desired row coordinates.
    The parameters can either be a numpy array of integer values, which
    are written to the neurons directly, or a single value, which is
    written only after adding some noise of +/- 5, if the range allows.
    Values of zero are not changed, turning something off is always possible.

    :param builder: Builder to append configuration to.
    :param config: Dict which contains pairs of CapMemRowOnCapMemBlock
        coordinates and either a single CapMemCell value or
        an array of CapMemCell values.
        In case a single non-zero value is given, it is changed to an
        array with noise and this array is written to hardware. This aims to
        reduce the crosstalk between CapMem cells.

    :return: Builder with configuration appended.
    """

    noise_amplitude = 5

    for capmem_row, parameters in config.items():
        # Add noise if single, non-zero value is given
        if not isinstance(parameters, np.ndarray):
            value = parameters
            parameters = np.ones(halco.NeuronConfigOnDLS.size, dtype=int) \
                * value
            if value != 0:
                parameters += capmem_noise(
                    max(hal.CapMemCell.Value.min - value, -noise_amplitude),
                    min(hal.CapMemCell.Value.max - value, noise_amplitude) + 1,
                    size=halco.NeuronConfigOnDLS.size)
            config[capmem_row] = parameters

        # Append write instructions to builder
        pyccalix.helpers.write_capmem_row(builder,
                                          capmem_row,
                                          config[capmem_row])

    return builder
