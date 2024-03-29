# pylint: disable=too-many-lines

"""
Provides functions to calibrate the neurons' synaptic input
reference potentials and synaptic input OTA bias currents.
Also allows calibration of synaptic input time constant.
"""

from abc import abstractmethod
from typing import Union, Optional, Type, List
import os
import numpy as np
import quantities as pq
from scipy.optimize import curve_fit
from dlens_vx_v3 import hal, halco, logger, hxcomm

from calix.common import algorithms, base, madc_base, cadc_helpers, helpers, \
    exceptions
from calix.hagen import neuron_helpers
from calix import constants


class SynReferenceCalib(base.Calib):
    """
    Calibrate synaptic reference voltage such that no currents are
    generated from the OTA without input.

    Calibrate the neuron's synaptic input reference voltage such that
    the membrane potential after enabling the OTA is the same as before.
    The membrane potential without enabled OTA has to be supplied as a target.
    This calibration routine sets reference voltages to the corresponding
    CapMem value to meet this calibration target.

    If bias_currents are supplied during initialization, the synaptic
    input to be calibrated gets turned on using these bias currents.

    Requirements:
    * Neuron membrane readout is connected to the CADCs (causal and acausal).
    * The desired OTA bias currents are already set up and the synaptic
      input is enabled. Alternatively, the desired configuration can be
      supplied as `bias_currents`.

    :ivar bias_currents: Synaptic input bias currents to configure
        for the synaptic input to be calibrated. If None, the
        configuration is unchanged.
    """

    def __init__(
            self, bias_currents: Optional[Union[int, np.ndarray]] = None,
            target: Union[int, np.ndarray] = 120):
        """
        :param bias_currents: Synaptic input bias currents to configure
            for the synaptic input to be calibrated. If None, the
            configuration is unchanged.
        :param target: Target CADC reads with synaptic input enabled.
            Optional, can also be supplied to the run() call.
            These reads need to be taken with the synaptic input
            disabled.
        """

        super().__init__(
            parameter_range=base.ParameterRange(
                hal.CapMemCell.Value.min, hal.CapMemCell.Value.max),
            n_instances=halco.NeuronConfigOnDLS.size, inverted=self._inverted)

        self.bias_currents = bias_currents
        self.target = target

    @property
    @abstractmethod
    def _inverted(self) -> bool:
        """
        Inversion property of the calibration, used to configure the
        base class.

        For the excitatory input, we must not invert it,
        for the inhibitory input, we must invert it.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def _log_abbreviation(self) -> str:
        """
        Abbreviation of exc/inh synaptic input used for log messages.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def _capmem_parameter_coord(self) -> halco.CapMemRowOnCapMemBlock:
        """
        Coordinate of CapMem parameters for the synaptic input
        reference potential.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def _bias_current_keyword(self) -> str:
        """
        String representation of the keyword required to preconfigure
        the synaptic inputs.
        """

        raise NotImplementedError

    def prelude(self, connection: hxcomm.ConnectionHandle):
        """
        If desired, configure the synaptic input bias currents
        to the given values. If no bias currents were given,
        the configuration is unchanged.

        :param connection: Connection to the chip to calibrate.
        """

        if self.bias_currents:
            neuron_helpers.reconfigure_synaptic_input(
                connection, **{self._bias_current_keyword: self.bias_currents})

    def configure_parameters(
            self, builder: base.WriteRecordingPlaybackProgramBuilder,
            parameters: np.ndarray) \
            -> base.WriteRecordingPlaybackProgramBuilder:
        """
        Configure the given parameters to the respective CapMem cell
        determined by whether the excitatory or inhibitory synaptic input
        is to be calibrated.

        :param builder: Builder to append configuration to.
        :param parameters: Array of reference potential settings to configure.

        :return: Builder with configuration instructions appended.
        """

        builder = helpers.capmem_set_neuron_cells(
            builder, {self._capmem_parameter_coord: parameters})

        builder = helpers.wait(builder, constants.capmem_level_off_time)
        return builder

    def measure_results(
            self, connection: hxcomm.ConnectionHandle,
            builder: base.WriteRecordingPlaybackProgramBuilder) \
            -> np.ndarray:
        """
        Measure the membrane potentials of each neuron.

        :param connection: Connection to chip to run measurement.
        :param builder: Builder to append read instructions to.

        :return: Array with CADC reads of neuron membrane potentials.
        """

        return neuron_helpers.cadc_read_neuron_potentials(
            connection, builder)

    def postlude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Print statistics of the resting potentials with synaptic input
        enabled and reference potential calibrated.

        :param connection: Connection to the chip to run on.
        """

        builder = base.WriteRecordingPlaybackProgramBuilder()
        results = self.measure_results(connection, builder)
        logger.get(
            "calix.hagen.neuron_synin.SynReferenceCalib.postlude"
        ).INFO(
            ("Calibrated i_bias_syn_in_{0}_shift, CADC statistics: "
             + "{1:5.2f} +- {2:4.2f}").format(
                 self._log_abbreviation,
                 np.mean(results[self.result.success]),
                 np.std(results[self.result.success])))


class ExcSynReferenceCalib(SynReferenceCalib):
    """
    Class for calibrating the excitatory synaptic input reference potential.
    """

    _log_abbreviation = "exc"
    _capmem_parameter_coord = \
        halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_shift
    _inverted = True
    _bias_current_keyword = "excitatory_biases"


class InhSynReferenceCalib(SynReferenceCalib):
    """
    Class for calibrating the inhibitory synaptic input reference potential.
    """

    _log_abbreviation = "inh"
    _capmem_parameter_coord = \
        halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_shift
    _inverted = False
    _bias_current_keyword = "inhibitory_biases"


class SynBiasCalib(base.Calib):
    """
    Calibrate the strength of synaptic inputs to match for all neurons.

    This is done using inputs of multiple rows of synapses at medium
    weight, with STP disabled, as set during the prelude.
    With these longer pulses, the mismatch of synapses should be minimal.
    The bias currents of the synaptic input OTAs are tweaked such that
    the difference in CADC reads before and after sending inputs match.

    Requirements:
    * Neuron membrane readout is connected to the CADCs (causal and acausal).
    * Synaptic currents can reach the neurons (`hal.ColumnCurrentSwitch`).
    * The membrane time constant is long, so that sucessive input events
      accumulate.
    * The selected synaptic time constant and synapse DAC bias current
      are not too large, so that the amplitudes resulting from a single
      stimulus don't already saturate the usable dynamic range at the CADC.

    :ivar syn_ref_calib: Instance of a SynReferenceCalib, configured
        with the given target_leak_read. The synaptic input reference
        potentials are recalibrated every time after changing the synaptic
        input OTA bias current.
    :ivar n_events: Number of events to send during integration of amplitudes
        in measure_results(). During prelude the number of events might
        be reduced if the measured amplitudes are too high.
    :ivar n_runs: Number of runs to average results from during measurement
        of amplitudes.
    :ivar wait_between_events: Wait time between two successive input events.
    :ivar reliable_amplitudes: Maximum amplitude that is reliably observable
        using the CADC. Used to find n_events during prelude.
    :ivar recalibrate_reference: Decide whether the OTA's reference
        potential is recalibrated during prelude and each time the bias
        current is touched.
    """

    def __init__(self, target_leak_read: Optional[np.ndarray] = None,
                 parameter_range: base.ParameterRange
                 = base.ParameterRange(hal.CapMemCell.Value.min,
                                       hal.CapMemCell.Value.max),
                 target: Optional[np.ndarray] = None,
                 recalibrate_reference: bool = True):
        """
        :param target_leak_read: Target CADC read for synaptic input
            reference potential calibration, which is called after each
            change of the bias current. This parameter must be given if
            recalibrate_reference is True (default), but can be left
            None otherwise.
        :param parameter_range: Allowed range of synaptic input OTA
            bias current.
        :param target: Target amplitudes for events. If given, the
            measurement of target amplitudes in prelude is skipped.
            This can be useful when calibrating inhibitory and excitatory
            synaptic inputs to the same amplitude targets.
        :param recalibrate_reference: Decide whether the OTA's reference
            potential is recalibrated during prelude and each time the bias
            current is touched.

        :raises ValueError: If target_leak_read is None while
            recalibrate_references is True.
        """

        super().__init__(
            parameter_range=parameter_range,
            n_instances=halco.NeuronConfigOnDLS.size, inverted=False)
        self.syn_ref_calib = self._reference_calib_type(
            target=target_leak_read)
        self.target = target

        self.n_events: int = 15  # number of events to send during measuring
        self.n_runs: int = 5     # number of runs to average during measuring
        self.wait_between_events: pq.quantity.Quantity = 1 * pq.us

        self.reliable_amplitudes = 60  # use amplitude lower than this
        self.recalibrate_reference = recalibrate_reference
        if self.recalibrate_reference and target_leak_read is None:
            raise ValueError(
                "target_leak_read must be given if OTA's reference "
                + "potentials are to be recalibrated.")

    @property
    @abstractmethod
    def _reference_calib_type(self) -> Type[SynReferenceCalib]:
        """
        Type of the sub-calibration to use for recalibrating the synaptic
        input reference potentials.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def _row_mode(self) -> hal.SynapseDriverConfig.RowMode:
        """
        Row mode (excitatory / inhibitory) of synapses to enable.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def _bias_current_coord(self) -> halco.CapMemRowOnCapMemBlock:
        """
        Coordinate of the capmem bias current cells for the synaptic
        input OTA to calibrate.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def _expected_sign(self) -> int:
        """
        Expected sign of the amplitudes. The measured amplitudes get
        multiplied with this sign.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def _log_abbreviation(self) -> str:
        """
        Abbreviation of exc/inh synaptic input used for log messages.
        """

        raise NotImplementedError

    def prelude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Preconfigure synapse drivers to send excitatory/inhibitory
        signals to all connected synapses.
        Measure current amplitudes and use the median as target
        for calibration, which can be overwritten by providing
        a target to the run() method.
        If an amplitude higher than the reliable range is obtained,
        the parameter self.n_events is reduced and it is measured again.

        :param connection: Connection to the chip to run on.

        :raises CalibNotSuccessful: If obtained target amplitudes
            are still too high after reducing self.n_events.
        """

        # Enable all synapse drivers and set them to the desired row mode.
        # The actual number of enabled synapse rows is later configured with
        # the help of `neuron_helpers.configure_synapses`.
        builder = neuron_helpers.enable_all_synapse_drivers(
            builder=base.WriteRecordingPlaybackProgramBuilder(),
            row_mode=self._row_mode)
        builder = neuron_helpers.configure_synapses(builder)
        base.run(connection, builder)

        if self.target is None:
            # Recalibrate synaptic input reference potential (once)
            if self.recalibrate_reference:
                self.syn_ref_calib.run(
                    connection, algorithm=algorithms.NoisyBinarySearch())

            # Measure current amplitudes, use median as target
            max_retries = 50  # adjust n_events at most that many times

            for _ in range(max_retries):
                self.target = int(np.median(self.measure_amplitudes(
                    connection,
                    builder=base.WriteRecordingPlaybackProgramBuilder(),
                    recalibrate_syn_ref=False)))
                if self.target < self.reliable_amplitudes:
                    break
                self.n_events = int(self.n_events * 0.8)
                if self.n_events == 0:
                    raise exceptions.CalibNotSuccessful(
                        "A single event's amplitude exceeds "
                        + "the reliable range: "
                        + f"{self.target} > {self.reliable_amplitudes}.")

            if self.target > self.reliable_amplitudes:
                raise exceptions.CalibNotSuccessful(
                    "Target neuron amplitudes are too high.")

            log = logger.get(
                "calix.hagen.neuron_synin.SynBiasCalib.prelude")
            log.DEBUG(f"Using {self.n_events} events during synin calib.")
            log.DEBUG("Target amplitude for i_bias_syn calibration: "
                      + f"{self.target}")

    def configure_parameters(
            self, builder: base.WriteRecordingPlaybackProgramBuilder,
            parameters: np.ndarray) \
            -> base.WriteRecordingPlaybackProgramBuilder:
        """
        Configure the bias current of the synaptic input OTA of all neurons
        to the given parameters. The target can be the excitatory or the
        inhibitory synaptic input circuit depending on the class properties.

        :param builder: Builder to append configuration instructions to.
        :param parameters: Synaptic input OTA bias current.

        :return: Builder with configuration instructions appended.
        """

        builder = helpers.capmem_set_neuron_cells(
            builder,
            {self._bias_current_coord: parameters})

        builder = helpers.wait(builder, constants.capmem_level_off_time)
        return builder

    def measure_amplitudes(self, connection: hxcomm.ConnectionHandle,
                           builder: base.WriteRecordingPlaybackProgramBuilder,
                           recalibrate_syn_ref: Optional[bool] = None
                           ) -> np.ndarray:
        """
        Send stimuli to all neurons using a few rows of synapses (as
        set during the prelude). The number of inputs in one run as
        well as the number of runs can be varied, the mean of all
        experiments is returned.

        Returns the difference in CADC reads after and before
        stimulation. The result uses the appropriate sign depending
        on excitatory or inhibitory mode, i.e. is normally positive.

        :param connection: Connection to the chip to run on.
        :param builder: Builder to append stimulate/read instructions, then
            gets executed.
        :param recalibrate_syn_ref: Switch if the synaptic input reference
            has to be recalibrated before measuring amplitudes.

        :return: Array of amplitudes resulting from stimulation.
        """

        # Recalibrate synaptic input reference potential
        if recalibrate_syn_ref or \
                (recalibrate_syn_ref is None and self.recalibrate_reference):
            self.syn_ref_calib.run(
                connection, algorithm=algorithms.NoisyBinarySearch())

        # Send test events to neurons
        padi_event = hal.PADIEvent()
        for bus in halco.iter_all(halco.PADIBusOnPADIBusBlock):
            padi_event.fire_bus[bus] = True  # pylint: disable=unsupported-assignment-operation

        results = []
        baselines = []

        for _ in range(self.n_runs):
            for synram in halco.iter_all(halco.SynramOnDLS):
                # Measure baseline
                builder, ticket = cadc_helpers.cadc_read_row(
                    builder, synram)
                baselines.append(ticket)

                # Send events on PADI bus
                builder.write(halco.TimerOnDLS(), hal.Timer())

                for i in range(self.n_events):
                    builder.block_until(
                        halco.TimerOnDLS(),
                        hal.Timer.Value(
                            int(i * self.wait_between_events.rescale(pq.us)
                                * int(
                                    hal.Timer.Value.fpga_clock_cycles_per_us)))
                    )
                    builder.write(synram.toPADIEventOnDLS(), padi_event)

                # Wait like for 2 events (some 2 * tau_syn) before CADC read.
                # This means most synaptic current has flown onto the
                # membrane, but it does not yet significantly decay due
                # to leakage.
                builder.block_until(
                    halco.TimerOnDLS(),
                    hal.Timer.Value(
                        int((self.n_events + 1)
                            * self.wait_between_events.rescale(pq.us)
                            * int(hal.Timer.Value.fpga_clock_cycles_per_us))))

                # final measurement in the same builder
                builder, ticket = cadc_helpers.cadc_read_row(
                    builder, synram)
                results.append(ticket)

                # Wait for many membrane time constants:
                # This wait ensures that any stimuli on neurons' membranes,
                # such as noise induced by the measurement before, can decay
                # back to a resting potential before we continue with the
                # next measurement.
                builder = helpers.wait(builder, 1000 * pq.us)

            base.run(connection, builder)

            builder = base.WriteRecordingPlaybackProgramBuilder()

        # Inspect reads
        baselines = neuron_helpers.inspect_read_tickets(
            baselines).reshape(
                (self.n_runs, halco.NeuronConfigOnDLS.size))
        differences = neuron_helpers.inspect_read_tickets(
            results).reshape(
                (self.n_runs, halco.NeuronConfigOnDLS.size)) - baselines

        return np.mean(differences, axis=0) * self._expected_sign

    def measure_results(
            self, connection: hxcomm.ConnectionHandle,
            builder: base.WriteRecordingPlaybackProgramBuilder) \
            -> np.ndarray:
        """
        Measure results of calibration.

        Calls the measure_amplitudes function using the default
        setting to recalibrate the synaptic reference potentials before
        measuring. This function only exists to keep the same arguments as in
        the base calib class.

        :param connection: Connection to the chip to run on.
        :param builder: Builder to append stimulate/read instructions, then
            gets executed.

        :return: Array of amplitudes resulting from stimulation.
        """

        return self.measure_amplitudes(connection, builder)

    def postlude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Log statistics of the event amplitudes after calibration.

        :param connection: Connection to the chip to run on.
        """

        builder = base.WriteRecordingPlaybackProgramBuilder()
        results = self.measure_results(connection, builder)
        log = logger.get("calix.hagen.neuron_synin"
                         + ".SynBiasCalib.postlude")
        log.INFO(
            ("Calibrated i_bias_synin_{0}_gm, amplitudes: "
             + "{1:5.2f} +- {2:4.2f}").format(
                 self._log_abbreviation,
                 np.mean(results[self.result.success]),
                 np.std(results[self.result.success])))
        log.DEBUG(
            "Neuron synaptic input OTA bias currents:" + os.linesep
            + f"{self.result.calibrated_parameters}")


class ExcSynBiasCalib(SynBiasCalib):
    """
    Class for calibrating the excitatory synaptic input bias current.
    """

    _reference_calib_type = ExcSynReferenceCalib
    _row_mode = hal.SynapseDriverConfig.RowMode.excitatory
    _bias_current_coord = halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_gm
    _expected_sign = 1
    _log_abbreviation = "exc"


class InhSynBiasCalib(SynBiasCalib):
    """
    Class for calibrating the inhibitory synaptic input bias current.
    """

    _reference_calib_type = InhSynReferenceCalib
    _row_mode = hal.SynapseDriverConfig.RowMode.inhibitory
    _bias_current_coord = halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_gm
    _expected_sign = -1
    _log_abbreviation = "inh"


class SynTimeConstantCalib(madc_base.Calib):
    """
    Calibrate synaptic input time constant using the MADC.

    The sypantic input time constant is fitted to an MADC trace:
    after we send some stimuli to decrease the voltage on the synaptic
    input line, we fit to the exponential decay back to the baseline
    voltage.

    During prelude, the MADC is enabled and the current readout section
    config is saved. During calibration, the readout config is changed
    continuously to calibrate one neuron's synaptic input time constant
    after another.

    Synaptic inputs need to be disabled during calibration, thus the
    synaptic input bias currents are written to zero in the prelude.
    Their state is not saved and restored as this calibration will
    normally run before they are calibrated, anyway.

    Requirements:
    * Synaptic events can reach the neurons, i.e. the synapse DAC bias
      is set and the `hal.ColumnCurrentSwitch`es allow currents from
      the synapses through.

    :ivar neuron_config_default: List of desired neuron configurations.
        Necessary to enable high resistance mode.
    """

    def __init__(self, target: pq.quantity.Quantity = 1.2 * pq.us,
                 neuron_configs: Optional[List[hal.NeuronConfig]] = None):
        """
        :param neuron_configs: List of neuron configurations. If None, the
            hagen-mode default neuron config is used.
        :param target: Target synaptic input time constant.
        """

        super().__init__(
            parameter_range=base.ParameterRange(
                hal.CapMemCell.Value.min, hal.CapMemCell.Value.max),
            inverted=True)
        self._wait_before_stimulation = 5 * pq.us
        self.target = target

        if neuron_configs is None:
            self.neuron_config_default = [
                neuron_helpers.neuron_config_default()
            ] * halco.NeuronConfigOnDLS.size
        else:
            self.neuron_config_default = neuron_configs

        self.sampling_time = max(50 * pq.us, 8 * np.max(self.target))
        self.wait_between_neurons = 10 * self.sampling_time
        self._wait_before_stimulation = 5 * pq.us

    @property
    @abstractmethod
    def _bias_current_coord(self) -> halco.CapMemRowOnCapMemBlock:
        """
        Coordinate of the CapMem bias current for the synaptic
        input resistor controlling the time constant.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def _row_mode(self) -> hal.SynapseDriverConfig.RowMode:
        """
        Row mode (excitatory / inhibitory) of synapses to enable.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def _readout_source(self) -> hal.NeuronConfig.ReadoutSource:
        """
        Readout source to select in neuron readout multiplexer.
        """

        raise NotImplementedError

    def prelude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Prepares chip for calibration.

        Disables synaptic inputs.
        Configures synapse drivers to stimulate the necessary input.

        :param connection: Connection to the chip to calibrate.
        """

        # prepare MADC
        super().prelude(connection)

        builder = base.WriteRecordingPlaybackProgramBuilder()

        # Ensure synaptic inputs are disabled (the readout mux is
        # not capable of disconnecting more than 1.2 V):
        # Otherwise, high membrane potentials can leak through the
        # readout selection mux and affect synaptic input readout.
        # The digital enable for the syn. inputs is disabled in
        # `neuron_config_disabled()`, but this transmission gate is
        # not capable of disconnecting more than 1.2 V on HX-v1.
        builder = helpers.capmem_set_neuron_cells(
            builder, {halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_gm: 0,
                      halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_gm: 0})
        builder = helpers.wait(builder, constants.capmem_level_off_time)

        # Enable all synapse drivers and set them to the desired row mode.
        # The actual number of enabled synapse rows is later configured with
        # the help of `neuron_helpers.configure_synapses`.
        builder = neuron_helpers.enable_all_synapse_drivers(
            builder, row_mode=self._row_mode)
        builder = neuron_helpers.configure_synapses(builder)

        # run program
        base.run(connection, builder)

    def configure_parameters(
            self, builder: base.WriteRecordingPlaybackProgramBuilder,
            parameters: np.ndarray) \
            -> base.WriteRecordingPlaybackProgramBuilder:
        """
        Configures the given array of synaptic input resistor bias currents.

        :param builder: Builder to append configuration instructions to.
        :param parameters: Array of bias currents to set up.

        :return: Builder with configuration appended.
        """

        builder = helpers.capmem_set_neuron_cells(
            builder,
            {self._bias_current_coord: parameters})

        builder = helpers.wait(builder, constants.capmem_level_off_time)
        return builder

    def stimulate(self, builder: base.WriteRecordingPlaybackProgramBuilder,
                  neuron_coord: halco.NeuronConfigOnDLS,
                  stimulation_time: hal.Timer.Value
                  ) -> base.WriteRecordingPlaybackProgramBuilder:
        """
        Send some PADI events to the synaptic input in order to
        drop the potential.

        :param builder: Builder to append PADI events to.
        :param neuron_coord: Coordinate of neuron which is currently recorded.
        :param stimulation_time: Timer value at beginning of stimulation.

        :return: Builder with PADI events appended.
        """

        padi_event = hal.PADIEvent()
        for bus in halco.iter_all(halco.PADIBusOnPADIBusBlock):
            padi_event.fire_bus[bus] = True  # pylint: disable=unsupported-assignment-operation
        for _ in range(5):
            builder.write(neuron_coord.toSynramOnDLS().toPADIEventOnDLS(),
                          padi_event)
        return builder

    def neuron_config_disabled(self, neuron_coord) -> hal.NeuronConfig:
        """
        Return a neuron config with readout disabled.

        The synaptic input is also disabled, since the neurons' readout
        multiplexer may leak if the membrane voltage rises above 1.2 V,
        impacting the syn. input measurement. Hence, the transmission
        gates between the OTAs and the membrane are disabled. Note that on
        Hicann-X v1, this does not suffice to disable the synaptic
        input, and also the OTA bias currents need to be set to zero,
        which is done during the prelude.

        :return: Neuron config with readout disabled.
        """

        config = hal.NeuronConfig(
            self.neuron_config_default[int(neuron_coord.toEnum())])
        config.enable_synaptic_input_excitatory = False
        config.enable_synaptic_input_inhibitory = False
        config.enable_readout = False
        return config

    def neuron_config_readout(self, neuron_coord) -> hal.NeuronConfig:
        """
        Return a neuron config with readout enabled.

        :return: Neuron config with readout enabled.
        """

        config = self.neuron_config_disabled(neuron_coord)
        config.readout_source = self._readout_source
        config.enable_readout = True
        return config

    def evaluate(self, samples: List[np.ndarray]) -> np.ndarray:
        """
        Evaluates the obtained MADC samples.

        To each neuron's MADC samples, an exponential decay is fitted,
        and the resulting time constant is returned.

        :param samples: MADC samples obtained for each neuron.

        :return: Numpy array of fitted synaptic input time constants
            in microseconds.
        """

        def fitfunc(time_t, scale, tau, offset):
            return scale * np.exp(-time_t / tau) + offset

        neuron_fits = []
        for neuron_id, neuron_data in enumerate(samples):
            # remove unreliable samples at beginning/end of trace
            start_index = np.argmax(neuron_data["chip_time"] * pq.us
                                    > self._wait_before_stimulation / 2
                                    + neuron_data["chip_time"][0] * pq.us)
            stop_index = int(self.madc_config.number_of_samples - 100)
            neuron_data = neuron_data[start_index:stop_index]

            # only fit to exponential decay
            neuron_data = neuron_data[np.argmin(neuron_data["value"]):]

            # estimate start values for fit
            offset = np.mean(neuron_data["value"][-10:])
            scale = np.min(neuron_data["value"]) - offset
            index_tau = np.argmax(neuron_data["value"] > offset + scale / np.e)
            tau = neuron_data["chip_time"][index_tau] - \
                neuron_data["chip_time"][0]
            # for small time constants the estimation of tau might fail ->
            # cut at bounds
            tau = min(
                max(constants.tau_syn_range.lower.rescale(pq.us).magnitude,
                    tau),
                constants.tau_syn_range.upper.rescale(pq.us).magnitude)

            # if the synaptic input line is floating (due to a too low bias
            # current), the recorded trace is constant -> register high time
            # constant such that bias is increased
            if abs(scale) < 10:
                neuron_fits.append(np.inf)
                continue

            boundaries = (
                [-offset,
                 constants.tau_syn_range.lower.rescale(pq.us).magnitude,
                 offset - 10],
                [0,
                 constants.tau_syn_range.upper.rescale(pq.us).magnitude,
                 offset + 10])

            try:
                fit_result = curve_fit(
                    fitfunc,
                    neuron_data["chip_time"] - neuron_data["chip_time"][0],
                    neuron_data["value"], p0=[scale, tau, offset],
                    bounds=boundaries)
                neuron_fits.append(fit_result[0][1])
            except RuntimeError as error:
                raise exceptions.CalibNotSuccessful(
                    f"Fitting to MADC samples failed for neuron {neuron_id}. "
                    + str(error))

        return np.array(neuron_fits) * pq.us


class ExcSynTimeConstantCalib(SynTimeConstantCalib):
    """
    Calibrate excitatory synaptic input time constant.
    """

    _bias_current_coord = halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_tau
    _row_mode = hal.SynapseDriverConfig.RowMode.excitatory
    _readout_source = hal.NeuronConfig.ReadoutSource.exc_synin


class InhSynTimeConstantCalib(SynTimeConstantCalib):
    """
    Calibrate inhibitory synaptic input time constant.
    """

    _bias_current_coord = halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_tau
    _row_mode = hal.SynapseDriverConfig.RowMode.inhibitory
    _readout_source = hal.NeuronConfig.ReadoutSource.inh_synin


class SynReferenceCalibMADC(madc_base.Calib):
    """
    Calibrate synaptic input reference potentials using the MADC.

    This calibration requires the leak bias currents to be zero, which
    is set in the prelude. The leak bias currents are NOT restored,
    since the target use case is PPU-based hagen mode integration with
    leakage disabled.

    As for all calibrations based on the MADC calibration, neuron
    configurations and readout configuration are changed during
    calibration and restored afterwards.

    To measure the current output of the synaptic input OTA, the membrane
    potential is observed while floating (with the leak term disabled).
    Initially, the membrane is reset. Once the refractory period ends and
    the reset is released, we fit to the membrane potential: If a constant
    current flows onto the membrane, it increases or decreases linearly.
    We search the reference potential settings for minimum constant currents,
    i.e. such that the membrane potential stays constant after the reset is
    released.

    Requirements:
    * The synaptic input is enabled and configured with the desired bias
      currents (i_bias_synin_{exc,inh}_gm).
    * The refractory time has to be set as `expected_refractory_time`.
      We use this parameter to select the start time of the fit.

    :ivar n_runs: Take the average of multiple measurements in each
        step of the calibration. Defaults to 1 (no averaging).
    :ivar expected_refractory_time: Configured refractory time. Used
        to determine the start point of the linear fit, which should
        begin once the reset is released and the membrane potential
        starts drifting.
    """

    def __init__(self):
        super().__init__(
            parameter_range=base.ParameterRange(
                hal.CapMemCell.Value.min, hal.CapMemCell.Value.max),
            inverted=self._inverted)
        self.target = np.zeros(halco.NeuronConfigOnDLS.size)
        self.sampling_time = 55 * pq.us
        self.wait_between_neurons = 70 * pq.us
        self.n_runs = 1  # averaging of results
        self.expected_refractory_time = 2 * pq.us

    @property
    @abstractmethod
    def _reference_potential_coord(self) -> halco.CapMemRowOnCapMemBlock:
        """
        Coordinate of the CapMem reference potential for the synaptic
        input OTA.
        """

        raise NotImplementedError

    @property
    @abstractmethod
    def _inverted(self) -> halco.CapMemRowOnCapMemBlock:
        """
        Select whether the parameter change needs to be inverted.
        """

        raise NotImplementedError

    def prelude(self, connection: hxcomm.ConnectionHandle):
        """
        Prepares chip for calibration.

        Enables the MADC and sets the leak bias currents to zero.

        :param connection: Connection to the chip to calibrate.
        """

        # prepare MADC
        super().prelude(connection)

        builder = base.WriteRecordingPlaybackProgramBuilder()

        # disable leak OTA
        builder = helpers.capmem_set_neuron_cells(
            builder, {halco.CapMemRowOnCapMemBlock.i_bias_leak: 0})
        builder = helpers.wait(builder, constants.capmem_level_off_time)

        # run program
        base.run(connection, builder)

    def configure_parameters(
            self, builder: base.WriteRecordingPlaybackProgramBuilder,
            parameters: np.ndarray) \
            -> base.WriteRecordingPlaybackProgramBuilder:
        """
        Configures the given array of synaptic input reference potentials.

        :param builder: Builder to append configuration instructions to.
        :param parameters: Array of reference potentials to set up.

        :return: Builder with configuration appended.
        """

        builder = helpers.capmem_set_neuron_cells(
            builder,
            {self._reference_potential_coord: parameters})

        builder = helpers.wait(builder, constants.capmem_level_off_time)
        return builder

    def stimulate(self, builder: base.WriteRecordingPlaybackProgramBuilder,
                  neuron_coord: halco.NeuronConfigOnDLS,
                  stimulation_time: hal.Timer.Value
                  ) -> base.WriteRecordingPlaybackProgramBuilder:
        """
        Reset the neuron's membrane potential.

        :param builder: Builder to append resets to.
        :param neuron_coord: Coordinate of neuron which is currently recorded.
        :param stimulation_time: Timer value at beginning of stimulation.

        :return: Builder with reset appended.
        """

        builder.write(neuron_coord.toNeuronResetOnDLS(), hal.NeuronReset())
        return builder

    def neuron_config_disabled(self, neuron_coord) -> hal.NeuronConfig:
        """
        Return a neuron config with readout disabled.

        :return: Neuron config with readout disabled.
        """

        config = neuron_helpers.neuron_config_default()
        return config

    def neuron_config_readout(self, neuron_coord) -> hal.NeuronConfig:
        """
        Return a neuron config with readout enabled.

        :return: Neuron config with readout enabled.
        """

        config = self.neuron_config_disabled(neuron_coord)
        config.readout_source = hal.NeuronConfig.ReadoutSource.membrane
        config.enable_readout = True
        return config

    def evaluate(self, samples: List[np.ndarray]) -> np.ndarray:
        """
        Evaluates the obtained MADC samples.

        To each neuron's MADC samples, a linear function is fitted,
        and the slope is returned.

        :param samples: MADC samples obtained for each neuron.

        :return: Numpy array of fitted slopes.
        """

        def fitfunc(time, slope, offset):
            return slope * time + offset

        # fit synaptic input time constant
        fit_slice = slice(
            int((self._wait_before_stimulation + self.expected_refractory_time
                 ).rescale(pq.s) * self.madc_config.calculate_sample_rate(
                     self.madc_input_frequency)),
            int(int(self.madc_config.number_of_samples) * 0.9))
        neuron_fits = []
        for neuron_data in samples:
            neuron_data = neuron_data[fit_slice]
            guessed_slope = \
                (neuron_data["value"][-1] - neuron_data["value"][0]) \
                / (neuron_data["chip_time"][-1] - neuron_data["chip_time"][0])
            guessed_offset = neuron_data["value"][0]
            neuron_fits.append(curve_fit(
                fitfunc,
                neuron_data["chip_time"] - neuron_data["chip_time"][0],
                neuron_data["value"], p0=[guessed_slope, guessed_offset])[0])

        return np.array(neuron_fits)[:, 0]

    def measure_results(
            self, connection: hxcomm.ConnectionHandle,
            builder: base.WriteRecordingPlaybackProgramBuilder) \
            -> np.ndarray:
        """
        Executes multiple measurements on chip, returns the mean result.

        :param connection: Connection to the chip to calibrate.
        :param builder: Builder to append measurement program to.

        :return: Numpy array of mean results.
        """

        base.run(connection, builder)

        results = np.empty((self.n_runs, halco.NeuronConfigOnDLS.size))
        for run_id in range(self.n_runs):
            results[run_id] = super().measure_results(
                connection,
                builder=base.WriteRecordingPlaybackProgramBuilder())

        return np.mean(results, axis=0)


class ExcSynReferenceCalibMADC(SynReferenceCalibMADC):
    """
    Calibrate excitatory synaptic input reference potential.
    """

    _reference_potential_coord = \
        halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_shift
    _inverted = True


class InhSynReferenceCalibMADC(SynReferenceCalibMADC):
    """
    Calibrate inhibitory synaptic input reference potential.
    """

    _reference_potential_coord = \
        halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_shift
    _inverted = False
