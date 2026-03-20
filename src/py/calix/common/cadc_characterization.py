"""
Uses the external voltage source DAC25 to calibrate the CADC
"""
from typing import List, Tuple
import numpy as np

from dlens_vx_v3 import halco, hal, logger, sta

from calix.common import helpers
from calix.common import base
from calix.common import adc_characterization


class CADCCharacterization(adc_characterization.ADCCharacterization):
    '''
    Use an external voltage source to determine the relationship between
    CADC values and voltages.

    The external voltage source is set to a number of desired values and
    the CADC is read at these values.

    :cvar num_samples: Number of samples to take and average.
    '''
    num_samples = 5
    log = logger.get("calix.common.cadc_characterization.CADCCharacterization")

    def prelude(self, connection: base.StatefulConnection) -> None:
        """
        :param connection: Connection to the chip to calibrate.
        """
        adc_characterization.ADCCharacterization.prelude(self, connection)

        builder = base.WriteRecordingPlaybackProgramBuilder()

        # Mux config
        mux_config = hal.PadMultiplexerConfig()
        mux_config.cadc_debug_acausal_to_synapse_intermediate_mux = True
        mux_config.cadc_debug_causal_to_synapse_intermediate_mux = True
        mux_config.synapse_intermediate_mux_to_pad = True
        builder.write(halco.PadMultiplexerConfigOnDLS(self.pad), mux_config)

        base.run(connection, builder)

    def build_measurement_program(
            self, builder: base.WriteRecordingPlaybackProgramBuilder
    ) -> Tuple[base.WriteRecordingPlaybackProgramBuilder,
               List[List[sta.ContainerTicket]]]:
        '''
        Build a program which sets the desired values of the external voltage
        source and records CADC samples.

        :param builder: Builder to append configuring instructions.
        '''

        tickets = []

        # iterate over voltages
        for _, test_value in enumerate(self.test_values):
            tickets.append([])
            self.configure_parameters(builder, test_value)
            builder = helpers.wait(builder, self.wait_before_measurement)

            for _ in range(self.num_samples):
                tickets[-1].append(builder.read(halco.CADCSamplesOnDLS()))

        return builder, tickets

    def measure_results(self, connection: base.StatefulConnection,
                        builder: base.WriteRecordingPlaybackProgramBuilder
                        ) -> np.ndarray:
        """
        Executes measurement on chip and return recorded CADC samples

        :param connection: Connection to the chip to calibrate.
        :param builder: Builder to append measurement program to.

        :return: Numpy array of results with shape (n_value, n_sample, 2, 512),
                 where the second last dimension are the causal/acausal
                 channels and the last is the channel.
        """

        builder, tickets = self.build_measurement_program(builder)
        base.run(connection, builder)

        samples = []
        for tickets_per_value in tickets:
            samples.append([])
            for ticket in tickets_per_value:
                local_samples = ticket.get()
                samples[-1].append([
                    local_samples.causal.to_numpy().flatten(),
                    local_samples.acausal.to_numpy().flatten()])

        return np.array(samples)

    def extract_mean_cadc_values(self, samples: np.ndarray) -> dict:
        """
        Extract mean CADC value for each test value.

        :param samples: CADC samples

        :return: Dictionary of mean CADC values, their std, applied
                 external voltages and the raw samples
        """
        return {'cadc_values': np.mean(samples, axis=1),
                'cadc_values_std': np.std(samples, axis=1),
                'voltages': self.test_values,
                'cadc_samples': samples}

    def run(self, connection: base.StatefulConnection) -> dict:
        """
        Perform measurement.

        :param connection: Connection to the chip to calibrate.
        """

        # Call prelude, check if target exists
        self.prelude(connection)

        # Run calibration
        builder = base.WriteRecordingPlaybackProgramBuilder()
        cadc_samples = self.measure_results(connection, builder)

        return self.extract_mean_cadc_values(cadc_samples)
