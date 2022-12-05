from typing import Optional, Union
from dataclasses import dataclass
from warnings import warn

from dlens_vx_v3 import sta, hxcomm

from calix.common import base, cadc, helpers
from calix.spiking import neuron
from calix import constants


@dataclass
class SpikingCalibrationTarget(base.CalibrationTarget):
    """
    Data class containing targets for spiking neuron calibration.

    :ivar cadc_target: Target parameters for CADC calibration.
    :ivar neuron_target: Target parameters for neuron calibration.
    """

    cadc_target: cadc.CADCCalibTarget = cadc.CADCCalibTarget()
    neuron_target: neuron.NeuronCalibTarget = neuron.NeuronCalibTarget()


@dataclass
class SpikingCalibrationOptions(base.CalibrationOptions):
    """
    Data class containing further options for spiking calibration.

    :ivar cadc_options: Further options for CADC calibration.
    :ivar neuron_options: Further options for neuron calibration.
    """

    cadc_options: cadc.CADCCalibOptions = cadc.CADCCalibOptions()
    neuron_options: neuron.NeuronCalibOptions = neuron.NeuronCalibOptions()


@dataclass
class SpikingCalibrationResult(base.CalibrationResult):
    """
    Data class containing results of cadc and neuron
    calibration, all what is necessary for operation in spiking mode.

    Refer to the documentation of :class:`calix.common.cadc.CADCCalibResult`,
    :class:`calix.spiking.neuron.NeuronCalibResult` for details about the
    contained result objects.
    """

    cadc_result: cadc.CADCCalibResult
    neuron_result: neuron.NeuronCalibResult

    def apply(self, builder: Union[sta.PlaybackProgramBuilder,
                                   sta.PlaybackProgramBuilderDumper]):
        """
        Apply the calib to the chip.

        Assumes the chip to be initialized already, which can be done using
        the stadls ExperimentInit().

        :param builder: Builder to append instructions to.
        """

        self.cadc_result.apply(builder)
        self.neuron_result.apply(builder)


def calibrate(connection: hxcomm.ConnectionHandle,
              target: Optional[SpikingCalibrationTarget] = None,
              options: Optional[SpikingCalibrationOptions] = None, *,
              cadc_kwargs: Optional[dict] = None,
              neuron_kwargs: Optional[dict] = None
              ) -> SpikingCalibrationResult:
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
        instance of SpikingCalibrationTarget.
    :param options: Further options for calibration, given as an
        instance of SpikingCalibrationOptions.

    :return: SpikingCalibrationResult, containing cadc and neuron results.
    """

    if target is None:
        target = SpikingCalibrationTarget()
    if options is None:
        options = SpikingCalibrationOptions()

    used_deprecated_parameters = False
    if cadc_kwargs is not None:
        target.cadc_target = cadc.CADCCalibTarget(**cadc_kwargs)
        used_deprecated_parameters = True
    if neuron_kwargs is not None:
        target.neuron_target = neuron.NeuronCalibTarget(**neuron_kwargs)
        used_deprecated_parameters = True

    # delete deprecated arguments, to ensure the correct ones are used
    # in the following code
    del cadc_kwargs
    del neuron_kwargs

    if used_deprecated_parameters:
        warn(
            "Passing arguments directly to calibrate() functions is "
            "deprecated. Please now use the target parameter class.",
            DeprecationWarning, stacklevel=2)

    target.check()

    # calibrate CADCs
    cadc_result = cadc.calibrate(
        connection, target.cadc_target, options.cadc_options)

    # calibrate neurons
    neuron_result = neuron.calibrate(
        connection, target.neuron_target, options.neuron_options)

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

    return SpikingCalibrationResult(
        target=target, options=options,
        cadc_result=cadc_result, neuron_result=neuron_result)
