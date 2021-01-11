"""
Uses the external voltage source DAC25 to calibrate the MADC
"""
import numpy as np
import quantities as pq

from dlens_vx_v3 import halco, hal, logger

from calix.common import helpers
from calix.common import base
from calix import constants


def external_dac_to_voltage(dac_value: int) -> float:
    """
    Convert digital value of DAC25 to V.

    :param dac_value: Value of external DAC25 to convert into V
    :returns: Voltage in V
    """
    return dac_value / 2047 * 1.2


class MADCCharacterization:
    '''
    Use an external voltage source to determine relationship between MADC
    values and voltages.
    The external voltage source is set to a number of desired values. The
    translation between MADC value and voltage is determined by a linear fit
    to these values.

    :cvar n_samples: Number of MADC samples at each setting of the external
        voltage
    :cvar wait_before_measurement: Waiting time (in us hw) before the MADC
        samples are recorded at each setting of the external voltage
    '''
    n_samples = 200
    wait_before_measurement = 1e3 * pq.us
    log = logger.get("calix.common.madc.MADCCharacterization")

    def __init__(self, test_values, readout_pad=0):
        '''
        :param readout_pad: Pad at which the external voltage is applied
        :param test_values: Values to set for the external voltage source
        '''
        self.pad = readout_pad
        self.test_values = test_values

    def prelude(self, connection: base.StatefulConnection) -> None:
        """
        :param connection: Connection to the chip to calibrate.
        """

        builder = base.WriteRecordingPlaybackProgramBuilder()

        # MADC settings (copied from pyNN)
        madc_capmem_values = {
            halco.CapMemCellOnDLS.readout_out_amp_i_bias_0: 0,
            halco.CapMemCellOnDLS.readout_out_amp_i_bias_1: 0,
            halco.CapMemCellOnDLS.readout_pseudo_diff_buffer_bias: 0,
            halco.CapMemCellOnDLS.readout_ac_mux_i_bias: 500,
            halco.CapMemCellOnDLS.readout_madc_in_500na: 500,
            halco.CapMemCellOnDLS.readout_sc_amp_i_bias: 500,
            halco.CapMemCellOnDLS.readout_sc_amp_v_ref: 400,
            halco.CapMemCellOnDLS.readout_pseudo_diff_v_ref: 400,
            halco.CapMemCellOnDLS.readout_iconv_test_voltage: 400,
            halco.CapMemCellOnDLS.readout_iconv_sc_amp_v_ref: 400}

        for cell, value in madc_capmem_values.items():
            builder.write(cell, hal.CapMemCell(value))
        builder = helpers.wait(builder, constants.capmem_level_off_time)

        config = hal.MADCConfig()
        config.number_of_samples = self.n_samples
        builder.write(halco.MADCConfigOnDLS(), config)

        # Connect external voltage source DAC25 to MADC
        shift_reg = hal.ShiftRegister()
        if self.pad == 0:
            mux_1_input = shift_reg.AnalogReadoutMux1Input.readout_chain_0
        elif self.pad == 1:
            mux_1_input = shift_reg.AnalogReadoutMux1Input.readout_chain_1

        shift_reg.select_analog_readout_mux_1_input = mux_1_input
        shift_reg.select_analog_readout_mux_2_input = \
            shift_reg.AnalogReadoutMux2Input.mux_dac_25
        builder.write(halco.ShiftRegisterOnBoard(), shift_reg)

        # Mux config
        mux_config = hal.PadMultiplexerConfig()
        mux_config.debug_to_pad = True
        builder.write(halco.PadMultiplexerConfigOnDLS(self.pad), mux_config)

        # Readout selection
        readout_selection = hal.ReadoutSourceSelection()
        mux = readout_selection.SourceMultiplexer()
        if self.pad == 0:
            mux.debug_plus = True
        elif self.pad == 1:
            mux.debug_minus = True
        else:
            self.log.ERROR("Choose either pad 0 or 1")
        readout_selection.set_buffer(
            halco.SourceMultiplexerOnReadoutSourceSelection(0), mux)
        builder.write(halco.ReadoutSourceSelectionOnDLS(0), readout_selection)

        base.run(connection, builder)

    def build_measurement_program(
            self, builder: base.WriteRecordingPlaybackProgramBuilder
    ) -> base.WriteRecordingPlaybackProgramBuilder:
        '''
        Build a program which sets the desired values of the external voltage
        source and records MADC samples.

        :param builder: Builder to append configuring instructions.
        '''

        madc_control = hal.MADCControl()
        madc_control.enable_power_down_after_sampling = False
        madc_control.start_recording = False
        madc_control.wake_up = True
        madc_control.enable_pre_amplifier = True
        builder.write(halco.MADCControlOnDLS(), madc_control)

        # initial wait and  systime sync TODO: Synchronization needed
        initial_wait = 1000 * pq.us
        builder.write(halco.SystimeSyncOnFPGA(), hal.SystimeSync())
        builder = helpers.wait(builder, initial_wait)

        # enable recording of samples TODO: what does this do?
        config = hal.EventRecordingConfig()
        config.enable_event_recording = True
        builder.write(halco.EventRecordingConfigOnFPGA(), config)

        # Measure MADC voltage for one setting of the external DAC after
        # another
        for n_value, test_value in enumerate(self.test_values):
            # set parameter
            self.configure_parameters(builder, test_value)

            # wait before measurement
            samples_per_us = 30 / pq.us  # assume about 30 samples per us
            # Wait twice as long to record all samples
            madc_recording_time = self.n_samples / samples_per_us * 2

            # Wait before triggering new measurement
            waiting_time = (initial_wait
                            + self.wait_before_measurement * (n_value + 1)
                            + madc_recording_time * n_value)
            builder = helpers.wait(builder, waiting_time)

            # trigger MADC sampling
            madc_control.wake_up = False
            madc_control.start_recording = True
            builder.write(halco.MADCControlOnDLS(), madc_control)

            # let MADC return to READY once given number of samples is acquired
            madc_control.start_recording = False
            if n_value == len(self.test_values) - 1:
                madc_control.enable_power_down_after_sampling = True
            builder.write(halco.MADCControlOnDLS(), madc_control)

            # wait for sampling to finish
            waiting_time = waiting_time + madc_recording_time
            builder = helpers.wait(builder, waiting_time)

        # disable recording of samples
        config = hal.EventRecordingConfig()
        config.enable_event_recording = False
        builder.write(halco.EventRecordingConfigOnFPGA(), config)

        return builder

    @staticmethod
    def configure_parameters(
            builder: base.WriteRecordingPlaybackProgramBuilder, parameter: int
    ) -> base.WriteRecordingPlaybackProgramBuilder:
        """
        Configure external voltage source.

        :param builder: Builder to append configuring instructions.
        :param parameters: DAC value for external voltage source

        :return: Builder with configuration instructions appended.
        """

        channel = hal.DACChannel()
        channel.value = parameter
        builder.write(halco.DACChannelOnBoard.mux_dac_25, channel)
        return builder

    def measure_results(self, connection: base.StatefulConnection,
                        builder: base.WriteRecordingPlaybackProgramBuilder
                        ) -> np.ndarray:
        """
        Executes measurement on chip and resturn recorded MADC samples

        :param connection: Connection to the chip to calibrate.
        :param builder: Builder to append measurement program to.

        :return: Numpy array of results.
        """

        builder = self.build_measurement_program(builder)
        program = base.run(connection, builder)

        return program.madc_samples.to_numpy()

    def extract_mean_madc_values(self, samples: np.ndarray) -> dict:
        """
        Extract mean madc value for each test value and return dictioanry with
        mean madc values and applied external voltages.

        :param samples: MADC samples

        :return: Dictionary of mean MADC values and applied external voltages
        """

        # Divide recorded samples in parts for different test values.
        # Use points where the time difference between recorded values is
        # larger than half of the waiting time before measurement.
        time_between_parts = \
            int(self.wait_before_measurement * 0.5
                * int(hal.Timer.Value.fpga_clock_cycles_per_us))
        madc_samples = np.sort(samples, order="chip_time")
        cycle_diff = np.diff(madc_samples['chip_time'])
        boundaries = [0] + \
            list(np.where(cycle_diff > time_between_parts)[0] + 1) \
            + [len(madc_samples)]

        assert len(boundaries) == len(self.test_values) + 1

        mean_traces = []
        for start_index, stop_index in zip(boundaries[:-1], boundaries[1:]):
            # ignore a few samples at beginning of recording
            offset = 10
            mean_traces.append(np.mean(
                madc_samples['value'][start_index + offset:stop_index]))

        assert len(self.test_values) == len(mean_traces)

        return dict(madc_values=np.array(mean_traces),
                    voltages=np.array([external_dac_to_voltage(dac) for dac
                                       in self.test_values]))

    def run(self, connection: base.StatefulConnection) -> dict:
        """
        Perform measurement.

        :param connection: Connection to the chip to calibrate.
        """

        # Call prelude, check if target exists
        self.prelude(connection)

        # Run calibration
        builder = base.WriteRecordingPlaybackProgramBuilder()
        madc_samples = self.measure_results(connection, builder)

        return self.extract_mean_madc_values(madc_samples)
