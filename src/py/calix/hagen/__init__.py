"""
Module for calibrating the HICANN-X chips for usage in hagen mode,
i.e. for multiply-accumulate operation.
"""

from typing import Tuple, Union, Optional
from dataclasses import dataclass
import numpy as np
from dlens_vx_v2 import sta, hxcomm, halco, hal, lola

from calix.common import algorithms, base, cadc, synapse, helpers
from calix.hagen import neuron, synapse_driver, neuron_helpers
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


@dataclass
class HagenSyninCalibrationResult(HagenCalibrationResult):
    """
    Extension of the usual (neuron-membrane-integrating) hagen calibration
    result, modified for integration on the synaptic input lines.
    The necessary changes are stored in a program dumper.

    Note that the integration on synaptic input lines does not require
    a neuron calibration at all. Since we currently need calibrated
    neurons for the synapse driver calibration, we implemented this
    result as an extension to the the usual calibration and save
    the already generated neuron calib result there. If we'll develop
    a stanalone synapse driver calibration in the future, we may
    drop the neuron calib from this result dataclass.
    """

    dumper: sta.PlaybackProgramBuilderDumper = \
        sta.PlaybackProgramBuilderDumper()

    def apply(self, builder: Union[sta.PlaybackProgramBuilder,
                                   sta.PlaybackProgramBuilderDumper]):
        """
        Apply the calib to the chip.

        Assumes the chip to be initialized already, which can be done using
        the stadls ExperimentInit().

        :param builder: Builder or dumper to append instructions to.

        :raises TypeError: If the given builder has a wrong type.
        """

        super().apply(builder)
        if isinstance(builder, sta.PlaybackProgramBuilder):
            builder.copy_back(sta.convert_to_builder(self.dumper))
        elif isinstance(builder, sta.PlaybackProgramBuilderDumper):
            builder.copy_back(self.dumper)
        else:
            raise TypeError(
                f"Type of given `builder` not supported: {type(builder)}")

        # wait for CapMem (longer wait due to large configuration changes)
        helpers.wait(builder, constants.capmem_level_off_time * 5)


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


def calibrate_for_synin_integration(
        connection: hxcomm.ConnectionHandle,
        synapse_dac_bias: int = 800,
        cadc_range: base.ParameterRange = base.ParameterRange(150, 340),
        hagen_kwargs: Optional[dict] = None
) -> HagenSyninCalibrationResult:
    """
    Calibrate the chip for integration on synaptic input lines.

    After the usual hagen-mode calibration, change the configuration
    such that activations can be integrated on the
    synaptic input lines instead of the neurons' membranes.

    Possible limitations:
    * The synapse driver calibration still happens via standard hagen
      mode, i.e. by integrating amplitudes on the neurons' membranes.
      If we develop a future synapse driver calibration which also
      works only on the synaptic input lines, we can drop the neuron
      calibration entirely from this function.

    :param connection: Connection to the chip to calibrate.
    :param synapse_dac_bias: Synapse DAC bias current that is desired.
        Controls the charge emitted to the synaptic input line by a
        multiplication.
    :param cadc_range: Dynamic range of the CADC, given in CapMem LSB.
        Should roughly contain the range of 0.3 to 0.6 V expected at
        the synaptic input readout (via source follower).
    :param hagen_kwargs: Arguments for the usual hagen-mode calibration.
        Refer to `calix.hagen.calibrate` for details.

    :return: Calibration result with configuration for integration on
        the synaptic input lines.
    """

    if hagen_kwargs is None:
        hagen_kwargs = dict()
    result = calibrate(connection, **hagen_kwargs)

    dumper = sta.PlaybackProgramBuilderDumper()

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
    dumper = helpers.capmem_set_quadrant_cells(
        dumper,
        {halco.CapMemCellOnCapMemBlock.syn_i_bias_dac: calibrated_dac_bias})

    # Calibrate CADCs to new range
    result.cadc_result = cadc.calibrate(
        connection, dynamic_range=cadc_range)

    # choose excitatory synaptic input line as readout
    # disable pullup of synaptic input lines (tau_syn -> inf)
    # enable small capacitor to increase capacitance and reduce
    # the effect of parasitic capacitance of each synapse.
    # Note: The latter setting is named incorrectly - it refers to
    # a small capacitance mode, which is set to False (= cap connected).
    for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
        config = result.neuron_result.neurons[coord]
        config.excitatory_input.enable_small_capacitor = False
        config.excitatory_input.enable_high_resistance = True
        config.excitatory_input.i_bias_tau = 0
        config.inhibitory_input.enable_small_capacitor = False
        config.inhibitory_input.enable_high_resistance = True
        config.inhibitory_input.i_bias_tau = 0
        config.readout.source = lola.AtomicNeuron.Readout.Source.exc_synin

    # Connect 1.2 V to synapse debug line:
    # This is necessary to reset the synaptic input potentials back to a
    # resting potential after accumulation has happened there.
    # First, connect synapse line to mux:
    config = hal.PadMultiplexerConfig()
    config.synin_debug_excitatory_to_synapse_intermediate_mux = True
    config.synin_debug_inhibitory_to_synapse_intermediate_mux = True
    config.synapse_intermediate_mux_to_pad = True
    dumper.write(halco.PadMultiplexerConfigOnDLS(), config)

    # connect DAC from correlation v_reset to upper pad
    config = hal.ShiftRegister()
    config.select_analog_readout_mux_1_input = \
        config.AnalogReadoutMux1Input.readout_chain_0
    config.select_analog_readout_mux_2_input = \
        config.AnalogReadoutMux2Input.v_reset
    dumper.write(halco.ShiftRegisterOnBoard(), config)

    # select v_reset = 1.2 V
    config = lola.DACChannelBlock().default_ldo_2
    config.set_voltage(halco.DACChannelOnBoard.mux_dac_25, 1.2)
    dumper.write(halco.DACChannelBlockOnBoard(), config)

    # pack into result class, apply it
    to_be_returned = HagenSyninCalibrationResult(
        result.cadc_result, result.neuron_result,
        result.synapse_driver_result, dumper)
    builder = sta.PlaybackProgramBuilder()
    to_be_returned.apply(builder)
    base.run(connection, builder)

    return to_be_returned
