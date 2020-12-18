"""
Module for calibrating the HICANN-X chips for usage in hagen mode,
i.e. for multiply-accumulate operation.
"""

from typing import Tuple
from dataclasses import dataclass
import numpy as np
from dlens_vx_v2 import sta, hxcomm, halco, hal

from calix.common import base, cadc, helpers
from calix.hagen import neuron, synapse_driver
from calix import constants


@dataclass
class HagenCalibrationResult:
    """
    Data class containing results of cadc, neuron and synapse driver
    calibration, all what is necessary for operation in hagen mode.

    Refer to the documentation of :class:`calix.hagen.cadc.CADCCalibResult`,
    :class:`calix.hagen.neuron.NeuronCalibResult` and
    :class:`calix.hagen.synapse_driver.SynapseDriverCalibResult` for
    details about the contained result objects.
    """

    cadc_result: cadc.CADCCalibResult
    neuron_result: neuron.NeuronCalibResult
    synapse_driver_result: synapse_driver.SynapseDriverCalibResult

    def apply(self, builder: sta.PlaybackProgramBuilder
              ) -> sta.PlaybackProgramBuilder:
        """
        Apply the calib to the chip.

        Assumes the chip to be initialized already, which can be done using
        the stadls ExperimentInit().

        :param builder: Builder to append instructions to.

        :return: Builder with configuration appended.
        """

        self.cadc_result.apply(builder)
        self.neuron_result.apply(builder)
        self.synapse_driver_result.apply(builder)

        return builder


def calibrate(connection: hxcomm.ConnectionHandle,
              cadc_kwargs: dict = None,
              neuron_kwargs: dict = None,
              synapse_driver_kwargs: dict = None
              ) -> HagenCalibrationResult:
    """
    Execute a full calibration for hagen mode:
    Calibrate CADCs to a suitable dynamic range.
    Calibrate neurons, calibrate synapse drivers.

    The chip has to be initialized first, which can be done using the
    stadls ExperimentInit().

    The individual calibrations' default parameters can be overwritten
    by providing the appropriate arguments.

    :param connection: Connection to the chip to calibrate.
    :param cadc_kwargs: Optional parameters for CADC calibration.
    :param neuron_kwargs: Optional parameters for neuron calibration.
        The leak bias is set to zero by default, which disables leakage
        entirely. Manually specify a membrane time constant (example:
        neuron_kwargs["tau_mem"] = 60) to avoid this. Note that even
        if the leak bias is set to zero, some pseudo-leakage may occur
        through the synaptic input OTAs.
    :param synapse_driver_kwargs: Optional parameters for synapse
        driver calibration.

    :return: HagenCalibrationResult, containing cadc, neuron and
        synapse driver results.

    :raises RuntimeError: If the CADC calibration is unsuccessful,
        based on evaluation of the found parameters. This happens some times
        when communication problems are present, in this case two FPGA resets
        are needed to recover. As chip resets / inits do not solve the
        problem, the calibration is not retried here.
    """

    # calibrate CADCs
    kwargs = {"dynamic_range": base.ParameterRange(150, 500)}
    if cadc_kwargs is not None:
        kwargs.update(cadc_kwargs)
    cadc_result = cadc.calibrate(connection, **kwargs)

    # calibrate neurons
    if neuron_kwargs is None:
        neuron_kwargs = dict()
    neuron_result = neuron.calibrate(connection, **neuron_kwargs)

    # calibrate synapse drivers
    if synapse_driver_kwargs is None:
        synapse_driver_kwargs = dict()
    synapse_driver_result = synapse_driver.calibrate(
        connection, **synapse_driver_kwargs)

    # set leak biases to zero
    # We want to have only integration on the neurons, no leakage.
    if "tau_mem" not in neuron_kwargs:
        builder = sta.PlaybackProgramBuilder()
        helpers.capmem_set_neuron_cells(
            builder, {halco.CapMemRowOnCapMemBlock.i_bias_leak: 0})
        builder = helpers.wait(builder, constants.capmem_level_off_time)
        base.run(connection, builder)

        for neuron_coord in halco.iter_all(halco.AtomicNeuronOnDLS):
            neuron_result.neurons[neuron_coord].leak.i_bias = \
                hal.CapMemCell.Value(0)

    return HagenCalibrationResult(
        cadc_result, neuron_result, synapse_driver_result)
