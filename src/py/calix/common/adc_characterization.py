"""
Uses the external voltage source DAC25 to characterize an ADC
"""
from abc import abstractmethod
from typing import Sequence
import quantities as pq

from dlens_vx_v3 import halco, hal, lola

from calix.common import base


class ADCCharacterization:
    '''
    Use an external voltage source to determine the relationship between
    ADC values and voltages.

    The external voltage source is set to a number of desired values and
    the ADC is read at these values.

    :cvar recording_time: Time to record at each voltage level.
    :cvar wait_before_measurement: Waiting time (in us) before the ADC
        samples are recorded at each setting of the external voltage.
    '''
    recording_time = 10 * pq.us
    wait_before_measurement = 1e3 * pq.us

    def __init__(self,
                 test_values: Sequence[int],
                 readout_pad: int = 0):
        '''
        :param test_values: Values to set for the external voltage source.
        :param readout_pad: Pad at which the external voltage is applied.
        '''
        self.pad = readout_pad
        self.test_values = test_values

    def prelude(self, connection: base.StatefulConnection) -> None:
        """
        :param connection: Connection to the chip to calibrate.
        """

        builder = base.WriteRecordingPlaybackProgramBuilder()

        # Connect external voltage source DAC25 to pad
        shift_reg = hal.ShiftRegister()
        if self.pad == 0:
            mux_1_input = shift_reg.AnalogReadoutMux1Input.readout_chain_0
        elif self.pad == 1:
            mux_1_input = shift_reg.AnalogReadoutMux1Input.readout_chain_1

        shift_reg.select_analog_readout_mux_1_input = mux_1_input
        shift_reg.select_analog_readout_mux_2_input = \
            shift_reg.AnalogReadoutMux2Input.mux_dac_25
        builder.write(halco.ShiftRegisterOnBoard(), shift_reg)

        base.run(connection, builder)

    @staticmethod
    def configure_parameters(
            builder: base.WriteRecordingPlaybackProgramBuilder,
            parameter: pq.Quantity
    ) -> base.WriteRecordingPlaybackProgramBuilder:
        """
        Configure external voltage source.

        :param builder: Builder to append configuring instructions.
        :param parameters: Voltage for external voltage source.

        :return: Builder with configuration instructions appended.
        """
        # use DACChannelBlock to convert Voltage to DAC setting
        block = lola.DACChannelBlock()
        block.set_voltage(
            halco.DACChannelOnBoard.mux_dac_25,
            parameter.rescale(pq.V))
        channel = hal.DACChannel()
        channel.value = block.value[
            halco.DACChannelOnBoard.mux_dac_25]
        builder.write(halco.DACChannelOnBoard.mux_dac_25, channel)
        return builder

    @abstractmethod
    def run(self, connection: base.StatefulConnection) -> dict:
        """
        Perform measurement.

        :param connection: Connection to the chip to calibrate.
        """
        raise NotImplementedError
