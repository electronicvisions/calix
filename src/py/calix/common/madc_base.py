"""
Provides an abstract base class for calibrations utilizing the MADC.
"""

from abc import abstractmethod
from typing import Optional, List, Tuple, Union
import numbers
import os
import numpy as np
import quantities as pq
import matplotlib.pyplot as plt
from dlens_vx_v2 import hal, sta, halco, hxcomm

from calix.common import base, exceptions, helpers


class Calibration(base.Calibration):
    """
    Abstract base class for neuron calibrations using the MADC.

    During prelude, the MADC is enabled and the current readout section
    config is saved. During calibration, the readout config is changed
    continuously to measure one neuron's properties after another.
    The stimuli during measurement have to be implemented
    as well as the evaluation of samples.
    Also, the recording timing per neuron can be set.

    :ivar original_readout_config: Source selection mux config
        before start of calibration. Used to restore state after
        calibration has finished.
    :ivar original_neuron_configs: Neuron configs before start of
        calibration. Used to restore state after calibration.
    :ivar sampling_time: Time to record MADC samples for each
        neuron. The requested sampling time must be achievable with
        the maximum number of samples in the MADCConfig - without
        using continuous sampling. With default settings, this equals
        roughly 2.18 ms of recording time per neuron.
    :ivar _wait_before_stimulation: Time to wait after triggering
        MADC sampling before stimulation. Marked private since analyzing
        the trace is typically not robust against changing this number.
    :ivar _dead_time: Time to wait before triggering the MADC
        sampling after the previous sampling period ended. Within
        this time, the neuron connections are changed. A minimum of 1 us
        is recommended.
    :ivar wait_between_neurons: Total time to wait for samples
        for each neuron. Has to be larger than the sum of sampling_time,
        wait_before_stimulation and dead_time.
    :ivar madc_input_frequency: Expected input clock frequency for the
        MADC, supplied by the PLL. Defaults to 500 MHz, which is present
        after the default stadls ExperimentInit.
    :ivar madc_config: Static configuration of the MADC. Written during
        the prelude and used to obtain the sample rate. The number of
        samples is calculated automatically based on the sampling_time.
    """

    def __init__(self, parameter_range: base.ParameterRange, inverted: bool,
                 errors: Optional[List[str]] = None,
                 n_instances: int = halco.NeuronConfigOnDLS.size):
        super().__init__(
            parameter_range=parameter_range,
            n_instances=n_instances, inverted=inverted, errors=errors)
        self.original_readout_config: Optional[
            hal.ReadoutSourceSelection] = None
        self.original_neuron_configs: Optional[List[hal.NeuronConfig]] = None
        self.sampling_time = 40 * pq.us
        self.wait_between_neurons = 10 * self.sampling_time
        self._wait_before_stimulation = 2 * pq.us
        self._dead_time = 1 * pq.us

        # Assume MADC input frequency to be unchanged after sta.ExperimentInit:
        # This should be replaced by looking it up from a chip object,
        # see issue 3955.
        self.madc_input_frequency = hal.ADPLL().calculate_output_frequency(
            output=hal.ADPLL().Output.dco)  # default MADC clock
        self.madc_config = hal.MADCConfig()

    def prelude(self, connection: hxcomm.ConnectionHandle):
        """
        Prepares chip for calibration.

        Configures the MADC and sets necessary bias currents.
        Reads and saves readout config and neuron config to
        restore original state after calibration.

        :param connection: Connection to the chip to calibrate.
        """

        builder = sta.PlaybackProgramBuilder()

        # read current readout config
        readout_ticket = builder.read(halco.ReadoutSourceSelectionOnDLS())

        # read current neuron config
        neuron_tickets = list()
        for neuron_coord in halco.iter_all(halco.NeuronConfigOnDLS):
            neuron_tickets.append(builder.read(neuron_coord))

        # enable MADC biases
        builder.write(halco.CapMemCellOnDLS.readout_ac_mux_i_bias,
                      hal.CapMemCell(500))
        builder.write(halco.CapMemCellOnDLS.readout_madc_in_500na,
                      hal.CapMemCell(500))
        builder.write(halco.CapMemCellOnDLS.readout_sc_amp_i_bias,
                      hal.CapMemCell(500))
        builder.write(halco.CapMemCellOnDLS.readout_pseudo_diff_v_ref,
                      hal.CapMemCell(400))
        builder.write(halco.CapMemCellOnDLS.readout_sc_amp_v_ref,
                      hal.CapMemCell(400))

        # static MADC config
        self.madc_config.number_of_samples = int(
            self.madc_config.calculate_sample_rate(self.madc_input_frequency)
            * self.sampling_time.rescale(pq.s))
        builder.write(halco.MADCConfigOnDLS(), self.madc_config)

        # run program
        base.run(connection, builder)

        # inspect reads
        self.original_readout_config = readout_ticket.get()
        self.original_neuron_configs = list()
        for ticket in neuron_tickets:
            self.original_neuron_configs.append(ticket.get())

        # disable readout for all neurons
        # runs in a separate program so that self.original_neuron_configs
        # is available when calling self.neuron_config_disabled()
        builder = sta.PlaybackProgramBuilder()
        for neuron_coord in halco.iter_all(halco.NeuronConfigOnDLS):
            builder.write(
                neuron_coord, self.neuron_config_disabled(neuron_coord))
        base.run(connection, builder)

    @abstractmethod
    def neuron_config_readout(self, neuron_coord: halco.NeuronConfigOnDLS
                              ) -> hal.NeuronConfig:
        """
        Return a neuron config with readout active and connected
        to the readout lines.

        :param neuron_coord: Coordinate of neuron to get config for.

        :return: Neuron config with readout enabled.
        """

        raise NotImplementedError

    @abstractmethod
    def neuron_config_disabled(self, neuron_coord: halco.NeuronConfigOnDLS
                               ) -> hal.NeuronConfig:
        """
        Return a neuron config in silent state, i.e. disconnected
        from the common readout lines.

        :param neuron_coord: Coordinate of neuron to get config for.

        :return: Neuron config with readout disabled.
        """

        raise NotImplementedError

    @abstractmethod
    def stimulate(self, builder: sta.PlaybackProgramBuilder,
                  neuron_coord: halco.NeuronConfigOnDLS,
                  stimulation_time: hal.Timer.Value
                  ) -> sta.PlaybackProgramBuilder:
        """
        Execute some commands after triggering MADC sampling.

        E.g., synaptic inputs could be sent to observe their shape.

        :param builder: Builder to append stimulation instructions to.
        :param neuron_coord: Coordinate of neuron which is currently recorded.
        :param stimulation_time: Timer value at beginning of stimulation.

        :return: Builder with stimulation instructions appended.
        """

        raise NotImplementedError

    # pylint: disable=too-many-statements
    def build_measurement_program(
            self, builder: sta.PlaybackProgramBuilder) -> Tuple[
                sta.PlaybackProgramBuilder, List[
                    sta.ContainerTicket_EventRecordingConfig]]:
        """
        Builds a program to measure an arbitrary MADC trace for each
        neuron.

        One neuron after another is connected to the MADC,
        the stimulate function is called and samples are recorded.
        The timing of each neuron is recorded, tickets containing the
        time of stimulation are returned along with the builder.

        :param builder: Builder to append instructions to.

        :return: Tuple containing:
            * Builder with instructions appended.
            * List of read tickets just before stimulating each neuron.
        """

        # wake up MADC
        madc_control = hal.MADCControl()
        madc_control.enable_power_down_after_sampling = False
        madc_control.start_recording = False
        madc_control.wake_up = True
        madc_control.enable_pre_amplifier = True
        builder.write(halco.MADCControlOnDLS(), madc_control)

        # connect first neurons to MADC already:
        # this minimizes drift in voltages once sampling begins
        builder.write(
            halco.NeuronConfigOnDLS(0),
            self.neuron_config_readout(halco.NeuronConfigOnDLS(0)))
        builder.write(
            halco.NeuronConfigOnDLS(1),
            self.neuron_config_readout(halco.NeuronConfigOnDLS(1)))

        mux_config = hal.ReadoutSourceSelection.SourceMultiplexer()
        mux_config.neuron_even[halco.HemisphereOnDLS(0)] = True
        config = hal.ReadoutSourceSelection()
        config.set_buffer(
            halco.SourceMultiplexerOnReadoutSourceSelection(0),
            mux_config)
        config.enable_buffer_to_pad[
            halco.SourceMultiplexerOnReadoutSourceSelection(0)] = True
        builder.write(halco.ReadoutSourceSelectionOnDLS(), config)

        # enable recording of samples
        config = hal.EventRecordingConfig()
        config.enable_event_recording = True
        builder.write(halco.EventRecordingConfigOnFPGA(), config)

        # initial wait, systime sync
        initial_wait = 1000 * pq.us
        builder.write(halco.SystimeSyncOnFPGA(), hal.SystimeSync())
        builder = helpers.wait(builder, initial_wait)

        switching_time_tickets = list()
        for neuron_coord in halco.iter_all(halco.NeuronConfigOnDLS):
            # connect neuron to shared line
            builder.write(
                neuron_coord, self.neuron_config_readout(neuron_coord))

            # connect shared line to MADC
            mux_config = hal.ReadoutSourceSelection.SourceMultiplexer()
            if neuron_coord.toAtomicNeuronOnDLS().toNeuronColumnOnDLS() % 2:
                mux_config.neuron_odd[
                    neuron_coord.toNeuronRowOnDLS().toHemisphereOnDLS()] = True
            else:
                mux_config.neuron_even[
                    neuron_coord.toNeuronRowOnDLS().toHemisphereOnDLS()] = True

            config = hal.ReadoutSourceSelection()
            config.set_buffer(
                halco.SourceMultiplexerOnReadoutSourceSelection(0),
                mux_config)
            config.enable_buffer_to_pad[
                halco.SourceMultiplexerOnReadoutSourceSelection(0)] = True
            builder.write(halco.ReadoutSourceSelectionOnDLS(), config)

            # trigger MADC sampling
            current_time = initial_wait + self._dead_time \
                + self.wait_between_neurons * int(neuron_coord.toEnum())
            builder.block_until(
                halco.TimerOnDLS(),
                int(current_time.rescale(pq.us)
                    * int(hal.Timer.Value.fpga_clock_cycles_per_us)))
            madc_control.wake_up = False
            madc_control.start_recording = True
            builder.write(halco.MADCControlOnDLS(), madc_control)

            # read something to get time of relevant samples
            switching_time_tickets.append(builder.read(
                halco.EventRecordingConfigOnFPGA()))

            # let MADC return to READY once given number of samples is acquired
            # We need to disable the `start_recording` setting quickly, since
            # the MADC will trigger multiple times if any of the following
            # waits would be larger than the sampling time.
            madc_control.start_recording = False
            if int(neuron_coord.toEnum()) == halco.NeuronConfigOnDLS.max:
                # turn off MADC after last neuron is measured
                madc_control.enable_power_down_after_sampling = True
            builder.write(halco.MADCControlOnDLS(), madc_control)

            # wait before stimulation
            stimulation_time = initial_wait + self._dead_time \
                + self._wait_before_stimulation + self.wait_between_neurons \
                * int(neuron_coord.toEnum())
            stimulation_time = hal.Timer.Value(int(
                stimulation_time.rescale(pq.us)
                * int(hal.Timer.Value.fpga_clock_cycles_per_us)))
            builder.block_until(halco.TimerOnDLS(), stimulation_time)

            # stimulate
            builder = self.stimulate(builder, neuron_coord, stimulation_time)

            # wait for sampling to finish
            final_time = initial_wait + self.wait_between_neurons \
                * (int(neuron_coord.toEnum()) + 1)
            builder.block_until(
                halco.TimerOnDLS(),
                int(final_time.rescale(pq.us)
                    * int(hal.Timer.Value.fpga_clock_cycles_per_us)))

            # disconnect neuron
            builder.write(
                neuron_coord, self.neuron_config_disabled(neuron_coord))

        # disable recording of samples
        config = hal.EventRecordingConfig()
        config.enable_event_recording = False
        builder.write(halco.EventRecordingConfigOnFPGA(), config)

        return builder, switching_time_tickets

    @abstractmethod
    def evaluate(self, samples: List[np.ndarray]) -> np.ndarray:
        """
        Process the given array of samples and return an array of results.

        The given samples are a list of samples for each neuron.
        Evaluation will typically involve fitting some parameters.
        The relevant obtained result is to be returned.

        :param samples: MADC samples received per neuron.

        :return: Array containing results per neuron.
        """

        raise NotImplementedError

    def record_traces(self, connection: hxcomm.ConnectionHandle,
                      builder: sta.PlaybackProgramBuilder
                      ) -> List[np.ndarray]:
        """
        Executes measurement on chip, returns samples per neuron.

        :param connection: Connection to the chip to calibrate.
        :param builder: Builder to append measurement program to.

        :return: List of numpy arrays, containing samples for each neuron.

        :raises TooFewSamplesError: If the number of received MADC samples
            is significantly smaller than expected for at least one neuron.
            The evaluate function may fail in this case, thus the error
            is caught here.
        """

        builder, switching_times = self.build_measurement_program(builder)

        # run program
        program = base.run(connection, builder)

        # convert switching times to us
        for neuron_id in range(halco.NeuronConfigOnDLS.size):
            switching_times[neuron_id] = (
                float(switching_times[neuron_id].fpga_time)
                / int(hal.Timer.Value.fpga_clock_cycles_per_us))
        switching_times.append(np.inf)  # last neuron's samples

        # convert chip_time of samples to us
        madc_samples = np.sort(
            program.madc_samples.to_numpy(), order="chip_time")
        madc_samples = madc_samples.astype(
            [("value", int), ("channel", int), ("fpga_time", int),
             ("chip_time", float)])
        madc_samples["chip_time"] /= int(
            hal.Timer.Value.fpga_clock_cycles_per_us)

        # split MADC samples by neuron
        switching_indices = np.searchsorted(
            madc_samples["chip_time"], switching_times)
        neuron_samples = list()
        for neuron_id in range(halco.NeuronConfigOnDLS.size):
            neuron_slice = slice(switching_indices[neuron_id],
                                 switching_indices[neuron_id + 1])
            neuron_samples.append(madc_samples[neuron_slice])

        # raise error if less than 95% of requested samples are received
        n_samples_required = int(int(
            self.madc_config.number_of_samples) * 0.95)
        n_samples_received = np.array([len(res) for res in neuron_samples])
        if np.any(n_samples_received < n_samples_required):
            raise exceptions.TooFewSamplesError(
                "Too few MADC samples were received. "
                + f"Expected more than {n_samples_required} samples. "
                + os.linesep
                + "Neurons with too few samples:" + os.linesep
                + f"{np.where(n_samples_received < n_samples_required)[0]}"
                + os.linesep
                + "Sample count per neuron:" + os.linesep
                + f"{n_samples_received}")

        return neuron_samples

    def measure_results(self, connection: hxcomm.ConnectionHandle,
                        builder: sta.PlaybackProgramBuilder) -> np.ndarray:
        """
        Measure and evaluate the results.

        Calls record_traces() to acquire an MADC trace per neuron,
        and calls evaluate() to get an array of results from, e.g., fits.

        :param connection: Connection to the chip to calibrate.
        :param builder: Builder to append measurement program to.

        :return: Numpy array, containing evaluated results.
        """

        samples = self.record_traces(connection, builder)
        return self.evaluate(samples)

    def postlude(self, connection: hxcomm.ConnectionHandle):
        """
        Restore original readout and neuron configuration.

        :param connection: Connection to the chip to calibrate.
        """

        builder = sta.PlaybackProgramBuilder()

        # restore original readout config
        builder.write(halco.ReadoutSourceSelectionOnDLS(),
                      self.original_readout_config)

        # restore original neuron config
        for neuron_coord, neuron_config in zip(
                halco.iter_all(halco.NeuronConfigOnDLS),
                self.original_neuron_configs):
            builder.write(neuron_coord, neuron_config)

        base.run(connection, builder)


class _Recorder(Calibration):
    """
    Record, plot and save a trace of each neuron via the MADC.

    This class is reserved for debugging purposes, therefore marked private.

    Note that here, the neuron is recorded without any stimuli. You may
    want to derive a class for your experiment setup and configure a
    suitable stimulate() function.

    Example usage:
    >>> from calix.common import madc_base
    >>> recorder = madc_base._Recorder()
    >>> recorder.prelude(connection)
    >>> samples = recorder.record_traces(
    ...     connection, builder=sta.PlaybackProgramBuilder())
    >>> recorder.plot_traces(samples)
    """

    def __init__(self):
        super().__init__(
            parameter_range=base.ParameterRange(0, 1), inverted=False)

    def neuron_config_disabled(self, neuron_coord: halco.NeuronConfigOnDLS
                               ) -> hal.NeuronConfig:
        config = hal.NeuronConfig(
            self.original_neuron_configs[int(neuron_coord.toEnum())])
        config.enable_readout = False
        return config

    def neuron_config_readout(self, neuron_coord: halco.NeuronConfigOnDLS
                              ) -> hal.NeuronConfig:
        config = self.neuron_config_disabled(neuron_coord)
        config.enable_readout = True
        return config

    def stimulate(self, builder: sta.PlaybackProgramBuilder,
                  neuron_coord: halco.NeuronConfigOnDLS,
                  stimulation_time: hal.Timer.Value
                  ) -> sta.PlaybackProgramBuilder:
        """
        Send no stimuli to the observed neuron.

        .. note:: Overwrite this function in order to send arbitrary
            stimuli during the MADC recording.

        :param builder: Builder to append stimulation instructions to.
        :param neuron_coord: Coordinate of neuron which is currently recorded.
        :param stimulation_time: Timer value at beginning of stimulation.

        :return: Builder with stimulation instructions appended.
        """

        return builder

    def configure_parameters(self, builder: sta.PlaybackProgramBuilder,
                             parameters: np.ndarray
                             ) -> sta.PlaybackProgramBuilder:
        """
        Raises an exception since this recorder does not support
        calibration, i.e. cannot be used to configure parameters.
        """

        raise exceptions.CalibrationNotSupported(
            "Recorder is used for debugging, it does not set parameters.")

    def evaluate(self, samples: List[np.ndarray]) -> np.ndarray:
        """
        Raises an exception since this recorder does not support
        calibration, i.e. does not perform any evaluation of results.
        """

        raise exceptions.CalibrationNotSupported(
            "Recorder is used for debugging and does not return results.")

    @staticmethod
    def plot_traces(samples: List[np.ndarray]):
        """
        Plot recorded traces, and save them as png figures.

        Each neurons' trace is plotted in its own figure, saved postfixed
        with the neuron id from enumerating the list. You may slice the
        original samples in order to save runtime while plotting.

        :param samples: List of recorded MADC samples, as returned by the
            record_traces() function.
        """

        # restrict plotting range:
        # The first samples may be invalid, and the last samples
        # may already contain samples of the next observed neuron,
        # since the switching times between neurons are not recorded
        # accurately within Calibration.record_traces().
        plot_slice = slice(10, -100)

        for neuron_id, neuron_samples in enumerate(samples):
            plt.figure()
            plt.plot(neuron_samples[plot_slice]["chip_time"]
                     - neuron_samples[plot_slice]["chip_time"][0],
                     neuron_samples[plot_slice]["value"])
            plt.savefig(f"trace_neuron{neuron_id}.png", dpi=300)
            plt.close()

    def run(self, connection: hxcomm.ConnectionHandle,
            algorithm: base.Algorithm,
            target: Union[numbers.Integral, np.ndarray, None] = None
            ) -> base.CalibrationResult:
        """
        Raises an exception since this recorder does not support
        calibration, i.e. being called with an algorithm.

        :raises CalibrationNotSupported: if called.
        """

        raise exceptions.CalibrationNotSupported(
            "Recorder is used for debugging and does not support calibration.")
