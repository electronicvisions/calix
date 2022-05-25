from typing import Optional, Union
from dataclasses import dataclass

from calix.common import base, cadc, helpers
from calix.spiking import neuron
from calix import constants
from dlens_vx_v2 import sta, hxcomm


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
    :param cadc_kwargs: Optional parameters for CADC calibration.
    :param neuron_kwargs: Optional parameters for neuron calibration.

    :return: SpikingCalibrationResult, containing cadc and neuron results.
    """

    # calibrate CADCs
    if cadc_kwargs is None:
        cadc_kwargs = dict()
    cadc_result = cadc.calibrate(connection, **cadc_kwargs)

    # calibrate neurons
    if neuron_kwargs is None:
        neuron_kwargs = dict()
    neuron_result = neuron.calibrate(connection, **neuron_kwargs)

    # re-calibrate CADCs
    # The newly set CapMem cells during the neuron calibration introduce
    # crosstalk on the CapMem, which means the previous CADC calibration
    # is no longer precise. We repeat the calib after the neurons are
    # configured to mitigate this crosstalk.
    cadc_result = cadc.calibrate(connection, **cadc_kwargs)

    # re-calibrate neuron potentials
    # filter neuron_kwargs for potentials
    potentials = ["leak", "reset", "threshold"]
    neuron_potentials = {key: neuron_kwargs[key] for key in potentials
                         if key in neuron_kwargs}
    neuron.refine_potentials(
        connection, neuron_result, **neuron_potentials)

    return SpikingCalibrationResult(cadc_result, neuron_result)
