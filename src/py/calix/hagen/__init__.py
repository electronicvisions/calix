"""
Module for calibrating the HICANN-X chips for usage in hagen mode,
i.e. for multiply-accumulate operation.
"""

from typing import Tuple, Union, Optional
from dataclasses import dataclass
import numpy as np

from dlens_vx_v2 import sta, hxcomm, halco, hal, lola

from calix.common import algorithms, base, cadc, synapse, helpers
from calix.hagen import neuron, synapse_driver, neuron_helpers, multiplication
from calix import constants


@dataclass
class HagenSyninCalibrationResult(base.CalibrationResult):
    """
    Calibration results needed for hagen mode integration on the
    synaptic inputs.

    Contains synapse driver calibration, CADC calibration and calibrated
    bias currents of the synapse DAC.

    Refer to the documentation of :class:`calix.hagen.cadc.CADCCalibResult`
    and :class:`calix.hagen.synapse_driver.SynapseDriverCalibResult` for
    details about the contained result objects.
    """

    cadc_result: cadc.CADCCalibResult
    synapse_driver_result: synapse_driver.SynapseDriverCalibResult
    syn_i_bias_dac: np.ndarray

    def apply(self, builder: Union[sta.PlaybackProgramBuilder,
                                   sta.PlaybackProgramBuilderDumper]):
        """
        Apply the calib to the chip.

        Assumes the chip to be initialized already, which can be done using
        the stadls ExperimentInit().

        :param builder: Builder or dumper to append instructions to.
        """

        # global bias currents
        neuron_helpers.configure_chip(builder)

        # apply saved calib results
        self.cadc_result.apply(builder)
        self.synapse_driver_result.apply(builder)
        helpers.capmem_set_quadrant_cells(
            builder,
            {halco.CapMemCellOnCapMemBlock.syn_i_bias_dac:
             self.syn_i_bias_dac})

        # static neuron configuration
        for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
            config = lola.AtomicNeuron()
            multiplication.Multiplication.configure_for_integration(config)
            builder.write(coord, config)

        # Connect 1.2 V to synapse debug line:
        # This is necessary to reset the synaptic input potentials back to a
        # resting potential after accumulation has happened there.
        # First, connect synapse line to mux:
        config = hal.PadMultiplexerConfig()
        config.synin_debug_excitatory_to_synapse_intermediate_mux = True
        config.synin_debug_inhibitory_to_synapse_intermediate_mux = True
        config.synapse_intermediate_mux_to_pad = True
        builder.write(halco.PadMultiplexerConfigOnDLS(), config)

        # connect DAC from correlation v_reset to upper pad
        config = hal.ShiftRegister()
        config.select_analog_readout_mux_1_input = \
            config.AnalogReadoutMux1Input.readout_chain_0
        config.select_analog_readout_mux_2_input = \
            config.AnalogReadoutMux2Input.v_reset
        builder.write(halco.ShiftRegisterOnBoard(), config)

        # select v_reset = 1.2 V
        config = lola.DACChannelBlock().default_ldo_2
        config.set_voltage(halco.DACChannelOnBoard.mux_dac_25, 1.2)
        builder.write(halco.DACChannelBlockOnBoard(), config)

        # wait for CapMem (longer wait due to large configuration changes)
        helpers.wait(builder, constants.capmem_level_off_time * 5)


@dataclass
class HagenCalibrationResult(base.CalibrationResult):
    """
    Data class containing results of cadc, neuron and synapse driver
    calibration, all what is necessary for operation in hagen mode
    when using integration on the neurons' membranes.

    Refer to the documentation of :class:`calix.hagen.cadc.CADCCalibResult`,
    :class:`calix.hagen.neuron.NeuronCalibResult` and
    :class:`calix.hagen.synapse_driver.SynapseDriverCalibResult` for
    details about the contained result objects.
    """

    cadc_result: cadc.CADCCalibResult
    neuron_result: neuron.NeuronCalibResult
    synapse_driver_result: synapse_driver.SynapseDriverCalibResult

    def apply(self, builder: Union[sta.PlaybackProgramBuilder,
                                   sta.PlaybackProgramBuilderDumper]):
        """
        Apply the calib to the chip.

        Assumes the chip to be initialized already, which can be done using
        the stadls ExperimentInit().

        :param builder: Builder or dumper to append instructions to.
        """

        self.cadc_result.apply(builder)
        self.neuron_result.apply(builder)
        self.synapse_driver_result.apply(builder)

    def to_hagen_synin_result(
            self, connection: hxcomm.ConnectionHandle,
            cadc_kwargs: dict = None, synapse_dac_bias: int = 800
    ) -> HagenSyninCalibrationResult:
        """
        Reconfigure calibration result for integration on synaptic
        input lines. The new result is applied to the chip.

        Only the missing parts are recalibrated, which should
        only take seconds to run. Note that the neuron
        calibration is dropped as it is not required for
        integration on synaptic inputs.

        :param connection: Connection to the chip to calibrate.
        :param cadc_kwargs: Arguments for CADC calibration.
        :param synapse_dac_bias: Target value for the synapse DAC
            bias calibration.

        :return: Hagen-mode calibration result for integration on
            synaptic input lines.
        """

        # calibrate CADC to smaller range
        kwargs = {"dynamic_range": base.ParameterRange(150, 340)}
        if cadc_kwargs is not None:
            kwargs.update(cadc_kwargs)
        cadc_result = cadc.calibrate(connection, **kwargs)

        # reconnect neuron readout to CADCs
        builder = sta.PlaybackProgramBuilder()
        neuron_helpers.configure_chip(builder)

        # set target synapse DAC bias current
        builder = helpers.capmem_set_quadrant_cells(
            builder,
            {halco.CapMemCellOnCapMemBlock.syn_i_bias_dac: synapse_dac_bias})
        builder = helpers.wait(builder, constants.capmem_level_off_time)
        base.run(connection, builder)

        # Calibrate synapse DAC bias current
        calibration = synapse.DACBiasCalibCADC()
        calibrated_dac_bias = calibration.run(
            connection, algorithm=algorithms.BinarySearch()
        ).calibrated_parameters

        # pack into result class, apply
        result = HagenSyninCalibrationResult(
            cadc_result, self.synapse_driver_result, calibrated_dac_bias)
        builder = sta.PlaybackProgramBuilder()
        result.apply(builder)
        base.run(connection, builder)

        return result


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
    """

    # preparations for synapse driver calib: calibrate CADC to smaller range
    cadc.calibrate(connection, dynamic_range=base.ParameterRange(150, 340))
    builder = sta.PlaybackProgramBuilder()
    neuron_helpers.configure_chip(builder)
    base.run(connection, builder)

    # set suitable synapse DAC bias current
    builder = sta.PlaybackProgramBuilder()
    builder = helpers.capmem_set_quadrant_cells(
        builder,
        {halco.CapMemCellOnCapMemBlock.syn_i_bias_dac: 800})
    builder = helpers.wait(builder, constants.capmem_level_off_time)
    base.run(connection, builder)

    # Calibrate synapse DAC bias current
    calibration = synapse.DACBiasCalibCADC()
    calibration.run(
        connection, algorithm=algorithms.BinarySearch())

    # calibrate synapse drivers
    # (uses external DAC for resetting potentials on synaptic input lines)
    if synapse_driver_kwargs is None:
        synapse_driver_kwargs = dict()
    synapse_driver_result = synapse_driver.calibrate(
        connection, **synapse_driver_kwargs)

    # disconnect DAC from pad
    builder = sta.PlaybackProgramBuilder()
    builder.write(halco.ShiftRegisterOnBoard(), hal.ShiftRegister())
    base.run(connection, builder)

    # calibrate CADCs
    kwargs = {"dynamic_range": base.ParameterRange(150, 500)}
    if cadc_kwargs is not None:
        kwargs.update(cadc_kwargs)
    cadc_result = cadc.calibrate(connection, **kwargs)

    # calibrate neurons
    if neuron_kwargs is None:
        neuron_kwargs = dict()
    neuron_result = neuron.calibrate(connection, **neuron_kwargs)

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

    # pack into result class
    to_be_returned = HagenCalibrationResult(
        cadc_result, neuron_result, synapse_driver_result)

    # apply calibration again:
    # The calibration is re-applied since synapse driver calibration
    # may be overwritten during neuron calibration.
    builder = sta.PlaybackProgramBuilder()
    to_be_returned.apply(builder)
    base.run(connection, builder)

    return to_be_returned


def calibrate_for_synin_integration(
        connection: hxcomm.ConnectionHandle,
        cadc_kwargs: dict = None,
        synapse_driver_kwargs: dict = None,
        synapse_dac_bias: int = 800,
) -> HagenSyninCalibrationResult:
    """
    Calibrate the chip for integration on synaptic input lines.

    Calibrate CADC, synapse drivers, and synapse DAC bias.

    :param connection: Connection to the chip to calibrate.
    :param cadc_kwargs: Optional parameters for CADC calibration.
    :param synapse_driver_kwargs: Optional parameters for synapse
        driver calibration.
    :param synapse_dac_bias: Synapse DAC bias current that is desired.
        Controls the charge emitted to the synaptic input line by a
        multiplication.

    :return: Calibration result for integration on the synaptic input
        lines.
    """

    # calibrate CADCs
    kwargs = {"dynamic_range": base.ParameterRange(150, 340)}
    if cadc_kwargs is not None:
        kwargs.update(cadc_kwargs)
    cadc_result = cadc.calibrate(connection, **kwargs)

    # global configuration
    builder = sta.PlaybackProgramBuilder()
    neuron_helpers.configure_chip(builder)
    base.run(connection, builder)

    # set target synapse DAC bias current
    builder = sta.PlaybackProgramBuilder()
    builder = helpers.capmem_set_quadrant_cells(
        builder,
        {halco.CapMemCellOnCapMemBlock.syn_i_bias_dac: synapse_dac_bias})
    builder = helpers.wait(builder, constants.capmem_level_off_time)
    base.run(connection, builder)

    # Calibrate synapse DAC bias current
    calibration = synapse.DACBiasCalibCADC()
    calibrated_dac_bias = calibration.run(
        connection, algorithm=algorithms.BinarySearch()).calibrated_parameters

    # calibrate synapse drivers
    if synapse_driver_kwargs is None:
        synapse_driver_kwargs = dict()
    synapse_driver_result = synapse_driver.calibrate(
        connection, **synapse_driver_kwargs)

    # pack into result class, apply it
    to_be_returned = HagenSyninCalibrationResult(
        cadc_result, synapse_driver_result, calibrated_dac_bias)
    builder = sta.PlaybackProgramBuilder()
    to_be_returned.apply(builder)
    base.run(connection, builder)

    return to_be_returned
