"""
Module for calibrating the HICANN-X chips for usage in hagen mode,
i.e. for multiply-accumulate operation.
"""
from __future__ import annotations

from typing import Tuple, Union, Optional
from dataclasses import dataclass, field
from warnings import warn

import numpy as np

from dlens_vx_v3 import hxcomm, halco, hal, lola

from calix.common import algorithms, base, cadc, synapse, helpers
from calix.hagen import neuron, synapse_driver, neuron_helpers, multiplication
from calix import constants


@dataclass
class HagenSyninCalibTarget(base.TopLevelCalibTarget):
    """
    Dataclass collecting target parameters for Hagen-mode calibrations.
    with integration on synaptic input lines.

    :ivar cadc_target: Target parameters for CADC calibration.
    :ivar synapse_dac_bias: Target synapse DAC bias current.
        Controls the charge emitted to the synaptic input line by a
        multiplication.
    """

    cadc_target: cadc.CADCCalibTarget = field(
        default_factory=lambda: cadc.CADCCalibTarget(
            dynamic_range=base.ParameterRange(150, 340)))
    synapse_dac_bias: int = 800

    feasible_ranges = {
        "synapse_dac_bias": base.ParameterRange(
            30, hal.CapMemCell.Value.max)
    }

    def calibrate(self,
                  connection: hxcomm.ConnectionHandle,
                  options: Optional[HagenSyninCalibOptions] = None
                  ) -> HagenSyninCalibResult:
        return calibrate_for_synin_integration(connection, self, options)


@dataclass
class HagenSyninCalibOptions(base.CalibOptions):
    """
    Dataclass collecting further options for Hagen-mode calibrations
    with integration on synaptic input lines.

    :ivar cadc_options: Further options for CADC calibration.
    :ivar synapse_driver_options: Further options for synapse driver
        calibration.
    """

    cadc_options: cadc.CADCCalibOptions = field(
        default_factory=cadc.CADCCalibOptions)
    synapse_driver_options: synapse_driver.SynapseDriverCalibOptions \
        = field(default_factory=synapse_driver.SynapseDriverCalibOptions)


@dataclass
class HagenCalibTarget(base.TopLevelCalibTarget):
    """
    Dataclass collecting target parameters for Hagen-mode calibrations
    with integration on membranes.

    :ivar cadc_target: Target parameters for CADC calibration.
    :ivar neuron_target: Target parameters for neuron calibration.
    """

    cadc_target: cadc.CADCCalibTarget = field(
        default_factory=lambda: cadc.CADCCalibTarget(
            dynamic_range=base.ParameterRange(150, 500)))
    neuron_target: neuron.NeuronCalibTarget = field(
        default_factory=neuron.NeuronCalibTarget)

    def calibrate(self,
                  connection: hxcomm.ConnectionHandle,
                  options: Optional[HagenCalibOptions] = None
                  ) -> HagenCalibResult:
        return calibrate(connection, self, options)


@dataclass
class HagenCalibOptions(base.CalibOptions):
    """
    Dataclass collecting further options for Hagen-mode calibrations with
    integration on membranes.

    :ivar cadc_options: Further options for CADC calibration.
    :ivar neuron_options: Further options for neuron calibration.
    :ivar neuron_disable_leakage: Decide whether the neuron leak bias
        currents are set to zero after calibration. This is done
        by default, which disables leakage entirely. Note that even
        if the leak bias is set to zero, some pseudo-leakage may occur
        through the synaptic input OTAs.
    :ivar synapse_driver_options: Further options for synapse driver
        calibration.
    """

    cadc_options: cadc.CADCCalibOptions = field(
        default_factory=cadc.CADCCalibOptions)
    neuron_options: neuron.NeuronCalibOptions = field(
        default_factory=neuron.NeuronCalibOptions)
    neuron_disable_leakage: bool = True
    synapse_driver_options: synapse_driver.SynapseDriverCalibOptions \
        = field(default_factory=synapse_driver.SynapseDriverCalibOptions)


@dataclass
class HagenSyninCalibResult(base.CalibResult):
    """
    Calib results needed for hagen mode integration on the
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

    def apply(self, builder: base.WriteRecordingPlaybackProgramBuilder):
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
        config = lola.AtomicNeuron()
        multiplication.Multiplication.configure_for_integration(config)
        for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
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
class HagenCalibResult(base.CalibResult):
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

    def apply(self, builder: base.WriteRecordingPlaybackProgramBuilder):
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
            self, connection: hxcomm.ConnectionHandle, *,
            cadc_target: Optional[cadc.CADCCalibTarget] = None,
            cadc_options: Optional[cadc.CADCCalibOptions] = None,
            synapse_dac_bias: Optional[int] = None,
    ) -> HagenSyninCalibResult:
        """
        Reconfigure calibration result for integration on synaptic
        input lines. The new result is applied to the chip.

        Only the missing parts are recalibrated, which should
        only take seconds to run. Note that the neuron
        calibration is dropped as it is not required for
        integration on synaptic inputs.

        :param connection: Connection to the chip to calibrate.
        :param cadc_target: Target parameters for CADC calibration.
        :param cadc_options: Further options for CADC calibration.
        :param synapse_dac_bias: Target value for the synapse DAC
            bias calibration.

        :return: Hagen-mode calibration result for integration on
            synaptic input lines.
        """

        # calibrate CADC to smaller range
        if cadc_target is None:
            cadc_target = HagenSyninCalibTarget().cadc_target
        if cadc_options is None:
            cadc_options = HagenSyninCalibOptions().cadc_options
        cadc_result = cadc.calibrate(connection, cadc_target, cadc_options)

        # reconnect neuron readout to CADCs
        builder = base.WriteRecordingPlaybackProgramBuilder()
        neuron_helpers.configure_chip(builder)

        # set target synapse DAC bias current
        if synapse_dac_bias is None:
            synapse_dac_bias = HagenSyninCalibTarget().synapse_dac_bias
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
        result = HagenSyninCalibResult(
            target=HagenSyninCalibTarget(
                cadc_target=cadc_target, synapse_dac_bias=synapse_dac_bias),
            options=HagenSyninCalibOptions(
                cadc_options=cadc_options,
                synapse_driver_options=self.options.synapse_driver_options),
            cadc_result=cadc_result,
            synapse_driver_result=self.synapse_driver_result,
            syn_i_bias_dac=calibrated_dac_bias)
        builder = base.WriteRecordingPlaybackProgramBuilder()
        result.apply(builder)
        base.run(connection, builder)

        return result


def calibrate(connection: hxcomm.ConnectionHandle,
              target: Optional[HagenCalibTarget] = None,
              options: Optional[HagenCalibOptions] = None
              ) -> HagenCalibResult:
    """
    Execute a full calibration for hagen mode:
    Calibrate CADCs to a suitable dynamic range.
    Calibrate neurons, calibrate synapse drivers.

    The chip has to be initialized first, which can be done using the
    stadls ExperimentInit().

    The individual calibrations' default parameters can be overwritten
    by providing the appropriate arguments.

    :param connection: Connection to the chip to calibrate.
    :param target: Target parameters for calibration, given as an
        instance of HagenCalibTarget.
    :param options: Further options for calibration, given as an
        instance of HagenCalibOptions.

    :return: HagenCalibResult, containing cadc, neuron and
        synapse driver results.
    """

    if target is None:
        target = HagenCalibTarget()
    if options is None:
        options = HagenCalibOptions()

    target.check()

    # preparations for synapse driver calib: calibrate CADC to smaller range
    cadc.calibrate(connection, cadc.CADCCalibTarget(
        dynamic_range=base.ParameterRange(150, 340)))
    builder = base.WriteRecordingPlaybackProgramBuilder()
    neuron_helpers.configure_chip(builder)
    base.run(connection, builder)

    # set suitable synapse DAC bias current
    builder = base.WriteRecordingPlaybackProgramBuilder()
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
    synapse_driver_result = synapse_driver.calibrate(
        connection, options.synapse_driver_options)

    # disconnect DAC from pad
    builder = base.WriteRecordingPlaybackProgramBuilder()
    builder.write(halco.ShiftRegisterOnBoard(), hal.ShiftRegister())
    base.run(connection, builder)

    # calibrate CADCs
    cadc_result = cadc.calibrate(
        connection, target.cadc_target, options.cadc_options)

    # calibrate neurons
    neuron_result = neuron.calibrate(
        connection, target.neuron_target, options.neuron_options)

    # set leak biases to zero
    # We want to have only integration on the neurons, no leakage.
    if options.neuron_disable_leakage:
        builder = base.WriteRecordingPlaybackProgramBuilder()
        helpers.capmem_set_neuron_cells(
            builder, {halco.CapMemRowOnCapMemBlock.i_bias_leak: 0})
        builder = helpers.wait(builder, constants.capmem_level_off_time)
        base.run(connection, builder)

        for neuron_coord in halco.iter_all(halco.AtomicNeuronOnDLS):
            neuron_result.neurons[neuron_coord].leak.i_bias = \
                hal.CapMemCell.Value(0)

    # pack into result class
    to_be_returned = HagenCalibResult(
        target=target, options=options,
        cadc_result=cadc_result, neuron_result=neuron_result,
        synapse_driver_result=synapse_driver_result)

    # apply calibration again:
    # The calibration is re-applied since synapse driver calibration
    # may be overwritten during neuron calibration.
    builder = base.WriteRecordingPlaybackProgramBuilder()
    to_be_returned.apply(builder)
    base.run(connection, builder)

    return to_be_returned


def calibrate_for_synin_integration(
        connection: hxcomm.ConnectionHandle,
        target: Optional[HagenSyninCalibTarget] = None,
        options: Optional[HagenSyninCalibOptions] = None
) -> HagenSyninCalibResult:
    """
    Calibrate the chip for integration on synaptic input lines.

    Calibrate CADC, synapse drivers, and synapse DAC bias.

    :param connection: Connection to the chip to calibrate.
    :param target: Calib target parameters.
    :param options: Further options for calibration.

    :return: Calib result for integration on the synaptic input
        lines.
    """

    if target is None:
        target = HagenSyninCalibTarget()
    if options is None:
        options = HagenSyninCalibOptions()

    target.check()

    # calibrate CADCs
    cadc_result = cadc.calibrate(
        connection, target.cadc_target, options.cadc_options)

    # global configuration
    builder = base.WriteRecordingPlaybackProgramBuilder()
    neuron_helpers.configure_chip(builder)
    base.run(connection, builder)

    # set target synapse DAC bias current
    builder = base.WriteRecordingPlaybackProgramBuilder()
    builder = helpers.capmem_set_quadrant_cells(
        builder,
        {halco.CapMemCellOnCapMemBlock.syn_i_bias_dac:
         target.synapse_dac_bias})
    builder = helpers.wait(builder, constants.capmem_level_off_time)
    base.run(connection, builder)

    # Calibrate synapse DAC bias current
    calibration = synapse.DACBiasCalibCADC()
    calibrated_dac_bias = calibration.run(
        connection, algorithm=algorithms.BinarySearch()).calibrated_parameters

    # calibrate synapse drivers
    synapse_driver_result = synapse_driver.calibrate(
        connection, options.synapse_driver_options)

    # pack into result class, apply it
    to_be_returned = HagenSyninCalibResult(
        target=target, options=options,
        cadc_result=cadc_result, synapse_driver_result=synapse_driver_result,
        syn_i_bias_dac=calibrated_dac_bias)
    builder = base.WriteRecordingPlaybackProgramBuilder()
    to_be_returned.apply(builder)
    base.run(connection, builder)

    return to_be_returned
