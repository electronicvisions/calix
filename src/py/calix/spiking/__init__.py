from __future__ import annotations

from typing import Optional, Union
from dataclasses import dataclass, field
from warnings import warn

from dlens_vx_v3 import hxcomm, hal, halco

from calix.common import base, cadc
from calix.spiking import neuron, correlation, synapse_driver


@dataclass
class SpikingCalibTarget(base.TopLevelCalibTarget):
    """
    Data class containing targets for spiking neuron calibration.

    :ivar cadc_target: Target parameters for CADC calibration.
    :ivar neuron_target: Target parameters for neuron calibration.
    :ivar correlation_target: Target parameters for calibration of
        correlation sensors. If None, they will not be calibrated.
    :ivar stp_target: Target for STP calibration.
    """

    cadc_target: cadc.CADCCalibTarget = field(
        default_factory=cadc.CADCCalibTarget)
    neuron_target: neuron.NeuronCalibTarget = field(
        default_factory=neuron.NeuronCalibTarget)
    correlation_target: Optional[correlation.CorrelationCalibTarget] = None
    stp_target: synapse_driver.STPCalibTarget = field(
        default_factory=synapse_driver.STPCalibTarget)

    def calibrate(self,
                  connection: hxcomm.ConnectionHandle,
                  options: Optional[SpikingCalibOptions] = None
                  ) -> SpikingCalibResult:
        return calibrate(connection, self, options)


@dataclass
class SpikingCalibOptions(base.CalibOptions):
    """
    Data class containing further options for spiking calibration.

    :ivar cadc_options: Further options for CADC calibration.
    :ivar neuron_options: Further options for neuron calibration.
    :ivar correlation_options: Further options for correlation calibration.
    :ivar stp_options: Further options for STP calibration.
    :ivar refine_potentials: Switch whether after the neuron calibration,
        the CADCs and neuron potentials are calibrated again. This mitigates
        CapMem crosstalk effects. By default, refinement is only performed
        if COBA mode is disabled.
    """

    cadc_options: cadc.CADCCalibOptions = field(
        default_factory=cadc.CADCCalibOptions)
    neuron_options: neuron.NeuronCalibOptions = field(
        default_factory=neuron.NeuronCalibOptions)
    correlation_options: correlation.CorrelationCalibOptions = field(
        default_factory=correlation.CorrelationCalibOptions)
    stp_options: synapse_driver.STPCalibOptions = field(
        default_factory=synapse_driver.STPCalibOptions)
    refine_potentials: Optional[bool] = None


@dataclass
class SpikingCalibResult(base.CalibResult):
    """
    Data class containing results of cadc and neuron
    calibration, all what is necessary for operation in spiking mode.

    Refer to the documentation of :class:`calix.common.cadc.CADCCalibResult`,
    :class:`calix.spiking.neuron.NeuronCalibResult`,
    :class:`calix.spiking.correlation.CorrelationCalibResult` and
    :class:`calix.spiking.synapse_driver.STPCalibResult`
    for details about the contained result objects.

    :ivar cadc_result: Result form CADC calibration.
    :ivar neuron_result: Result form neuron calibration.
    :ivar correlation_result: Result from correlation calibration.
    :ivar stp_result: Result from STP calibration.
    """

    cadc_result: cadc.CADCCalibResult
    neuron_result: neuron.NeuronCalibResult
    stp_result: synapse_driver.STPCalibResult
    correlation_result: Optional[correlation.CorrelationCalibResult] = None

    def apply(self, builder: base.WriteRecordingPlaybackProgramBuilder):
        """
        Apply the calib to the chip.

        Assumes the chip to be initialized already, which can be done using
        the stadls ExperimentInit().

        :param builder: Builder to append instructions to.
        """

        if self.correlation_result is not None:
            self.correlation_result.apply(builder)
        self.cadc_result.apply(builder)
        self.neuron_result.apply(builder)
        self.stp_result.apply(builder)


def calibrate(connection: hxcomm.ConnectionHandle,
              target: Optional[SpikingCalibTarget] = None,
              options: Optional[SpikingCalibOptions] = None
              ) -> SpikingCalibResult:
    """
    Execute a full calibration for spiking mode:
    Calibrate CADCs to a suitable dynamic range.
    Calibrate neurons.

    The chip has to be initialized first, which can be done using the
    stadls ExperimentInit().

    The individual calibrations' default parameters can be overwritten
    by providing cadc_ or neuron_kwargs, respectively.

    After the "usual" neuron calibration has finished, the CADC calibration
    is repeated in order to mitigate CapMem crosstalk effects.
    Afterwards, the neuron potentials are calibrated once again.

    :param connection: Connection to the chip to calibrate.
    :param target: Target parameters for calibration, given as an
        instance of SpikingCalibTarget.
    :param options: Further options for calibration, given as an
        instance of SpikingCalibOptions.

    :return: SpikingCalibResult, containing cadc and neuron results.
    """

    if target is None:
        target = SpikingCalibTarget()
    if options is None:
        options = SpikingCalibOptions()

    target.check()

    # calibrate CADCs
    cadc_result = cadc.calibrate(
        connection, target.cadc_target, options.cadc_options)

    # calibrate neurons
    neuron_result = neuron.calibrate(
        connection, target.neuron_target, options.neuron_options)

    # calibrate STP
    stp_result = synapse_driver.calibrate(
        connection, target=target.stp_target, options=options.stp_options)

    # Refine potentials only if COBA mode is not selected:
    # The large voltage swings on v_mem during refinement with the
    # default binary search algorithm may affect the synaptic input
    # bias current (modulated by COBA) so much that a constant current
    # onto the membrane is generated, shifting the membrane potential.
    # Refinement may work for low synaptic input bias currents or
    # using more benign algorithms, but generally, there may be problems.
    # Also, precise potentials will not be as important in COBA mode.
    if options.refine_potentials or (
            options.refine_potentials is None
            and target.neuron_target.e_coba_reversal is None):
        # re-calibrate CADCs
        # The newly set CapMem cells during the neuron calibration introduce
        # crosstalk on the CapMem, which means the previous CADC calibration
        # is no longer precise. We repeat the calib after the neurons are
        # configured to mitigate this crosstalk.
        cadc_result = cadc.calibrate(
            connection, target.cadc_target, options.cadc_options)

        # re-calibrate neuron potentials
        neuron.refine_potentials(
            connection, neuron_result, target.neuron_target)

    # Perform correlation calibration at the end since correlation bias
    # currents affect the neuron CADC readout. Also, a high voltage on
    # the correlation capacitor can affect the readout, cf. issue #3469.
    if target.correlation_target is None:
        correlation_result = None
    else:
        correlation_result = correlation.calibrate(
            connection, target=target.correlation_target,
            options=options.correlation_options)

    result = SpikingCalibResult(
        target=target, options=options,
        cadc_result=cadc_result, neuron_result=neuron_result,
        correlation_result=correlation_result, stp_result=stp_result)

    builder = base.WriteRecordingPlaybackProgramBuilder()
    result.apply(builder)
    base.run(connection, builder)
    return result
