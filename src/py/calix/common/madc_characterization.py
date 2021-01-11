"""
Uses the external voltage source DAC25 to characterize the MADC
"""
import numpy as np
import quantities as pq

from dlens_vx_v3 import halco, hal, logger

from calix.common import helpers
from calix.common import base
from calix.common import adc_characterization
from calix import constants


class MADCCharacterization(adc_characterization.ADCCharacterization):
    '''
    Use an external voltage source to determine the relationship between
    MADC values and voltages.

    The external voltage source is set to a number of desired values and
    the MADC is read at these values.
    '''
    log = logger.get("calix.common.madc_characterization.MADCCharacterization")

    def prelude(self, connection: base.StatefulConnection) -> None:
        """
        :param connection: Connection to the chip to calibrate.
        """
        adc_characterization.ADCCharacterization.prelude(self, connection)

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
        builder.write(halco.MADCConfigOnDLS(), config)

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

        # disable recording of samples
        config = hal.EventRecordingConfig()
        config.enable_event_recording = False
        builder.write(halco.EventRecordingConfigOnFPGA(), config)

        madc_control = hal.MADCControl()
        madc_control.enable_power_down_after_sampling = False
        madc_control.start_recording = False
        madc_control.wake_up = True
        madc_control.enable_pre_amplifier = True
        madc_control.enable_continuous_sampling = True
        builder.write(halco.MADCControlOnDLS(), madc_control)
        madc_control.wake_up = False  # no need to wake up again later

        # wait some time for MADC to wake up
        builder = helpers.wait(builder, 1 * pq.ms)

        # trigger MADC recording
        madc_control.start_recording = True
        madc_control.stop_recording = False
        builder.write(halco.MADCControlOnDLS(), madc_control)
        builder = helpers.wait(builder, self.recording_time)

        # iterate over voltages
        for test_value in self.test_values:
            self.configure_parameters(builder, test_value)
            builder = helpers.wait(builder, self.wait_before_measurement)

            # enable recording of samples
            config = hal.EventRecordingConfig()
            config.enable_event_recording = True
            builder.write(halco.EventRecordingConfigOnFPGA(), config)

            builder = helpers.wait(builder, self.recording_time)

            # disable recording of samples
            config = hal.EventRecordingConfig()
            config.enable_event_recording = False
            builder.write(halco.EventRecordingConfigOnFPGA(), config)

        # stop MADC recording
        madc_control.start_recording = False
        madc_control.stop_recording = True
        madc_control.enable_power_down_after_sampling = True
        builder.write(halco.MADCControlOnDLS(), madc_control)

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
        std_traces = []
        for start_index, stop_index in zip(boundaries[:-1], boundaries[1:]):
            mean_traces.append(np.mean(
                madc_samples['value'][start_index + stop_index]))
            std_traces.append(np.std(
                madc_samples['value'][start_index + stop_index]))

        assert len(self.test_values) == len(mean_traces)
        assert len(std_traces) == len(mean_traces)

        # we expect only small temporal noise
        if max(std_traces) >= 2.:
            self.log.warn(
                "Trace standard deviation maximum larger than 2: "
                + str(std_traces))

        return {'madc_values': np.array(mean_traces),
                'madc_values_std': np.array(std_traces),
                'voltages': self.test_values,
                'madc_samples': madc_samples['value']}

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
