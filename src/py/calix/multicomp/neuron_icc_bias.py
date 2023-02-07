import numpy as np
import quantities as pq

from typing import Optional, List
from dlens_vx_v3 import halco, sta, hxcomm, hal

from calix import constants
from scipy.optimize import curve_fit
from calix.common import algorithms, base, exceptions, madc_base, helpers
from calix.hagen import neuron_helpers, neuron_potentials, neuron_leak_bias


class ICCMADCCalib(madc_base.Calib):
    """
    Calibrate the inter-compartment conductance time constant of all neurons to
    the provided target value.

    Two neighboring, unconnected neuron circuits are set to different leak
    potentials. Then the two neurons are connected using the inter-compartment
    conductance. During that connection process the membrane of the neuron of
    interest is recorded with the MADC. Throughout this process the leak
    potential of the neighboring neurons approach one another exponentially.
    An exponential is fitted to the recorded trace and the bias current
    responsible for the inter-compartment conductance is tweaked such that the
    desired value for the time constant is reached.

    The membrane potentials decay with a time constant tau_total, which is
    given by:
        tau_total = 1 / (2 / tau_icc + 1 / tau_leak)
                    <=>
        tau_icc = 2 * tau_leak * tau_total / (tau_leak - tau_total)

    If the membrane time constant (tau_leak) is large compared to tau_icc,
    tau_total is approximately half of tau_icc (the factor 2 in the equation
    above originates from the two membrane time constants of the two involved
    neurons). For that reason the leak conductance is set minimal during
    calibration.

    This calibration decides whether inter-compartment conductance bias current
    division or multiplication is required during prelude. The given neuron
    configs are therefore altered.
    To be able to measure the total time constant some requirements have to be
    fulfilled which are set by calling prepare_for_measurement().

    Note:
    * Leak potentials should be set alternatingly with a difference of a few
      10 LSB (in CADC reads) between even and odd neurons. As the conductance
      between the compartments depends on the base voltages, it is recommended
      to choose a range, which will later also be used for experiments.

    :ivar neuron_configs: List of desired neuron configurations.
        Necessary to enable inter-compartment conductance bias
        division/multiplication.
    :ivar v_leak: List of target leak potentials. When calibrating the icc the
        leak potential will be calibrated to the provided leak potentials in
        the prelude.
    """
    def __init__(self,
                 target: pq.quantity.Quantity = 2 * pq.us,
                 v_leak: np.ndarray = np.tile(
                     [80, 110], int(halco.NeuronConfigOnDLS.size / 2)),
                 neuron_configs: Optional[List[hal.NeuronConfig]] = None):
        """
        :param target: Calib target for the inter-compartment
            conductance. Note that we measure the total time constant
            tau_total. Please refer to the class doc-string for further
            reading.
        :param v_leak: Target for the leak potentials.
        :param neuron_configs: List of neuron configurations. If None, the
            hagen-mode default neuron config is used for all neurons.
        """

        super().__init__(base.ParameterRange(
            hal.CapMemCell.Value.min, hal.CapMemCell.Value.max), inverted=True)

        self.target = target
        if neuron_configs is None:
            self.neuron_configs = [
                neuron_helpers.neuron_config_default() for _ in
                range(halco.NeuronConfigOnDLS.size)]
        else:
            self.neuron_configs = neuron_configs

        self._wait_before_stimulation = 10 * pq.us
        self.sampling_time = 100 * pq.us + self._wait_before_stimulation
        self.wait_between_neurons = 5 * self.sampling_time
        self._dead_time = 1 * pq.us
        self.v_leak = v_leak

    def prepare_for_measurement(self, connection: hxcomm.ConnectionHandle,
                                v_leak: Optional[np.ndarray] = None):
        """
        Set all necessary chip parameters such that a calibration of the inter-
        compartment conductance can be executed.

        In order to calibrate the inter compartment conductance the
        leak conductance is set to a small value, while preventing the membrane
        to float.
        Furthermore, the threshold comparator is disabled as well as the
        synaptic input.
        Note: The leak potential between two neighboring neurons on the
        same hemisphere should be calibrated some 10 LSB apart to ensure a
        gap large enough for a recordable decay.

        :param connection: Connection to the chip to calibrate.
        :param v_leak: List of targets for the leak potential calibration. If
            None is given the ivar v_leak will be used.
        """

        if v_leak is None:
            v_leak = self.v_leak
        # set i_bias_leak initially high to ensure a sufficient leak potential
        # calibration
        builder = sta.PlaybackProgramBuilder()
        builder = helpers.capmem_set_neuron_cells(
            builder, {halco.CapMemRowOnCapMemBlock.i_bias_leak:
                      hal.CapMemCell.Value.max})
        for neuron_coord, neuron_config in zip(
                halco.iter_all(halco.NeuronConfigOnDLS), self.neuron_configs):
            neuron_config.enable_leak_division = False
            neuron_config.enable_leak_multiplication = False
            # disable synaptic input and disable the threshold to set neurons
            # in non-spiking configuration
            neuron_config.enable_synaptic_input_excitatory = False
            neuron_config.enable_synaptic_input_inhibitory = False
            neuron_config.enable_threshold_comparator = False
            # disable all switches, which are required to be turned off
            # initially during the calibration.
            neuron_config.connect_soma = False
            neuron_config.connect_soma_right = False
            neuron_config.enable_multicomp_conductance = False

            builder.write(neuron_coord, neuron_config)
        base.run(connection, builder)

        # disable synaptic inputs
        neuron_helpers.reconfigure_synaptic_input(
            connection, excitatory_biases=0, inhibitory_biases=0)

        # calibrate leak potential alternating
        builder = sta.PlaybackProgramBuilder()
        leak_calib = neuron_potentials.LeakPotentialCalib(v_leak)
        leak_calib.run(connection, algorithm=algorithms.NoisyBinarySearch())

        # enable leak bias division
        for neuron_coord, neuron_config in zip(
                halco.iter_all(halco.NeuronConfigOnDLS), self.neuron_configs):
            neuron_config.enable_leak_division = True
            builder.write(neuron_coord, neuron_config)
        base.run(connection, builder)

        # calibrate leak bias current to a low value without making the
        # membrane floating. The CADC based calibration routine is used since
        # the MADC based routines lead to floating membrane voltages for some
        # neurons on some setups.
        leak_bias_calib = neuron_leak_bias.MembraneTimeConstCalibCADC(
            target_time_const=60. * pq.us, target_amplitude=30)
        leak_bias_calib.run(
            connection, algorithm=algorithms.NoisyBinarySearch())

        # rerun leak potential calibration as now the leak bias current was
        # changed
        leak_calib.run(connection, algorithm=algorithms.NoisyBinarySearch())

    def prelude(self, connection: hxcomm.ConnectionHandle):
        """
        Prepares chip for calibration.

        Disable all multicompartment related switches, which are required to be
        turned off initially during the calibration.

        Also measures the inter-compartment conductance time constant at low
        and high bias currents to decide whether multiplication or
        division of the bias is required to reach the given targets.

        :param connection: Connection to the chip to calibrate.
        """

        # prepare MADC
        super().prelude(connection)

        # set membrane potential to desired values, disable threshold
        # comparator, disabled synapses and the leak conductance is set low
        self.prepare_for_measurement(connection)

        # decide whether icc/nmda division or multiplication is required:
        # inspect the feasible range without division or multiplication and
        for neuron_config in self.neuron_configs:
            neuron_config.enable_divide_multicomp_conductance_bias = False
            neuron_config.enable_multiply_multicomp_conductance_bias = False

        # measure at low icc/nmda bias current: If time constant is still
        # smaller than the target, then enable division.
        builder = sta.PlaybackProgramBuilder()
        builder = self.configure_parameters(
            builder, parameters=200 + helpers.capmem_noise(
                size=self.n_instances))
        maximum_timeconstant = self.measure_results(connection, builder)
        enable_division = maximum_timeconstant < self.target

        # measure at high icc/nmda bias current: If time constant is still
        # larger than the target, then enable multiplication.
        builder = sta.PlaybackProgramBuilder()
        builder = self.configure_parameters(
            builder, parameters=hal.CapMemCell.Value.max - 63
            + helpers.capmem_noise(size=self.n_instances))
        minimum_timeconstant = self.measure_results(connection, builder)
        enable_multiplication = minimum_timeconstant > self.target

        # check sanity of decisions
        multiplication_and_division = np.logical_and(enable_multiplication,
                                                     enable_division)
        if multiplication_and_division.any():
            raise exceptions.CalibNotSuccessful(
                "Prelude decided to enable both i_bias_nmda multiplication and"
                + " division at the same time for neurons "
                + f"{np.where(multiplication_and_division)}")

        # set up in neuron configs
        for neuron_id in range(self.n_instances):
            self.neuron_configs[neuron_id].\
                enable_divide_multicomp_conductance_bias = \
                enable_division[neuron_id]
            self.neuron_configs[neuron_id].\
                enable_multiply_multicomp_conductance_bias = \
                enable_multiplication[neuron_id]

    def configure_parameters(self, builder: sta.PlaybackProgramBuilder,
                             parameters: np.ndarray
                             ) -> sta.PlaybackProgramBuilder:
        """
        Configure the given array of inter-compartment conductance bias
        currents.

        :param builder: Builder to append configuration instructions to.
        :param parameters: Array of nmda bias currents to set up.

        :return: Builder with configuration appended.
        """

        builder = helpers.capmem_set_neuron_cells(
            builder, {halco.CapMemRowOnCapMemBlock.i_bias_nmda: parameters})
        builder = helpers.wait(builder, constants.capmem_level_off_time)
        return builder

    def stimulate(self, builder: sta.PlaybackProgramBuilder,
                  neuron_coord: halco.NeuronConfigOnDLS,
                  stimulation_wait: hal.Timer.Value
                  ) -> sta.PlaybackProgramBuilder:
        """
        Enables the inter-compartment conductance (ICC) between the
        compartments resulting in an approach of their membrane potentials.

        A multicompartment neuron will be created by connecting two neurons via
        the somatic line with one ICC enabled. If neuron_coord is even the
        neuron will be connected with its right neighbor otherwise it will be
        connected with its left neighbor.

        :param builder: Builder to append instructions to.
        :param neuron_coord: Coordinate of neuron which is currently recorded.
        :param stimulation_wait: Timer value at beginning of stimulation.

        :return: Builder with instructions appended.
        """

        neuron_id = int(neuron_coord.toEnum())

        is_even = neuron_id % 2 == 0
        neighboring_id = neuron_id + (1 if is_even else -1)

        # neuron configs
        neuron_config = self.neuron_config_readout(neuron_coord)
        neighboring_config = self.neuron_config_disabled(
            halco.NeuronConfigOnDLS(halco.common.Enum(neighboring_id)))

        # connect compartments
        if is_even:
            neuron_config.connect_soma_right = True
        else:
            neighboring_config.connect_soma_right = True

        neuron_config.enable_multicomp_conductance = True
        neuron_config.connect_soma = False

        neighboring_config.connect_soma = True
        neighboring_config.enable_multicomp_conductance = False

        builder.write(neuron_coord, neuron_config)
        builder.write(halco.NeuronConfigOnDLS(halco.common.Enum(
            neighboring_id)), neighboring_config)

        # Disconnect compartments after wait
        builder.wait_until(
            halco.TimerOnDLS(),
            int(stimulation_wait) + int(self.sampling_time.rescale(pq.us))
            * int(hal.Timer.Value.fpga_clock_cycles_per_us))

        builder.write(
            halco.NeuronConfigOnDLS(halco.common.Enum(neighboring_id)),
            self.neuron_config_disabled(halco.NeuronConfigOnDLS(
                halco.common.Enum(neighboring_id))))
        builder.write(neuron_coord, self.neuron_config_readout(neuron_coord))

        return builder

    def neuron_config_disabled(self, neuron_coord: halco.NeuronConfigOnDLS
                               ) -> hal.NeuronConfig:
        """
        Return a neuron config with readout disabled.

        :param neuron_coord: Coordinate of neuron to get config for.

        :return: Neuron config with readout disabled.
        """

        config = hal.NeuronConfig(
            self.neuron_configs[int(neuron_coord.toEnum())])
        config.enable_readout = False
        return config

    def neuron_config_readout(self, neuron_coord: halco.NeuronConfigOnDLS
                              ) -> hal.NeuronConfig:
        """
        Return a neuron config with readout active and connected
        to the readout lines.

        :param neuron_coord: Coordinate of neuron to get config for.

        :return: Neuron config with readout enabled.
        """

        config = self.neuron_config_disabled(neuron_coord)
        config.readout_source = hal.NeuronConfig.ReadoutSource.membrane
        config.enable_readout = True
        return config

    def evaluate(self, samples: List[np.ndarray]) -> pq.Quantity:
        """
        Evaluates the obtained MADC samples.

        To each neuron's MADC samples, an exponential decay is fitted and the
        resulting time constant is returned.

        :param samples: MADC samples obtained for each neuron.

        :return: Array of the fitted total time constants as quantity.
        """
        def fitfunc(variable_x, scale, tau, offset, x_offset):
            """
            Exponential function to fit the measured data to.

            A heaviside function is used inside of the exponential to ensure
            a correct detection of the beginning of the decay.
            """
            return scale * np.exp(
                -np.heaviside(variable_x - x_offset, 1)
                * (variable_x - x_offset) / tau) + offset

        neuron_fits = []
        for neuron_id, neuron_data in enumerate(samples):
            # The first samples may be the last samples of the previous neuron
            # dicard them as well as the last few samples
            madc_sample_rate = self.madc_config.calculate_sample_rate(
                self.madc_input_frequency)
            start = int((self._wait_before_stimulation.rescale(pq.s) / 4)
                        * madc_sample_rate)
            stop = int(int(self.madc_config.number_of_samples) * 0.95)
            neuron_samples = neuron_data[start:stop]

            # set time with reference to the first
            neuron_samples['chip_time'] = neuron_samples['chip_time'] \
                - neuron_data['chip_time'][0]

            # estimate start values for fit
            p_0 = {}
            p_0['offset'] = np.mean(neuron_samples["value"][-100:])
            start_decay = np.argmax(
                neuron_samples['chip_time']
                > self._wait_before_stimulation.rescale(pq.us))
            p_0['scale'] = np.mean(neuron_samples["value"][:start_decay]) - \
                p_0['offset']
            p_0['x_offset'] = self._wait_before_stimulation.rescale(pq.us).\
                magnitude

            index_tau = np.argmax(np.abs(
                neuron_samples["value"] - p_0['offset'])
                < np.abs(p_0['scale']) / np.e)
            # absolute value is used to make initial guess of tau robust
            # against small amplitudes which may occur at large time constants
            # when the resistance is large
            p_0['tau'] = np.abs(
                neuron_samples["chip_time"][index_tau] - p_0['x_offset'])

            try:
                popt, _ = curve_fit(
                    fitfunc,
                    neuron_samples["chip_time"],
                    neuron_samples["value"],
                    p0=[p_0['scale'], p_0['tau'], p_0['offset'],
                        p_0['x_offset']],
                    bounds=([-p_0['offset'], 0.1, p_0['offset'] - 10,
                             p_0['x_offset'] - 3],
                            [p_0['offset'], 100, p_0['offset'] + 10,
                             p_0['x_offset'] + 3]))
            except RuntimeError as error:
                raise exceptions.CalibNotSuccessful(
                    f"Fitting to MADC samples failed for neuron {neuron_id}. "
                    + str(error))
            neuron_fits.append(popt[1])  # store time constant of exponential

        return np.array(neuron_fits) * pq.us
