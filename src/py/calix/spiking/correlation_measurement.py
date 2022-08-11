"""
Provides a measurement class that measures correlation amplitudes
at different timings of pre- and postspikes, and determines amplitude
and time constants.
"""

from typing import Union, Tuple

import numpy as np
from scipy.optimize import curve_fit
import quantities as pq

from dlens_vx_v3 import hal, halco, sta, lola, hxcomm, logger

import pyccalix

from calix.common import base, exceptions
from calix.hagen import helpers
from calix import constants


class CorrelationMeasurement:
    """
    Measurement and evaluation of correlation traces.

    Provides functions to measure correlation traces on some
    quads or the whole chip, and to analyze these traces in
    order to obtain amplitude and time constant parameters, using
    fits or calculations.

    Requirements:
    * CADCs are enabled and calibrated.
    * External correlation voltages (v_res_meas, v_reset) are set.

    :ivar delays: Delay between pre- and postspike for correlated spike
        pairs. Use negative numbers for anticausal spike pairs.
    :ivar n_events: Number of correlated events in one measurement.
    :ivar amp_calib: Amplitude calibration bits for synapses.
    :ivar time_calib: Time constant calibration bits for synapses.
    :ivar i_ramp: Correlation time constant calibration CapMem current.
    :ivar i_store: Correlation amplitude calibration CapMem current.
    :ivar address: Address to use for presynaptic events.
    :ivar wait_between_pairs: Wait time between the individual correlated
        spike pairs. Should be an order of magnitude greater than the
        largest delay.
    :ivar log: Logger used to log outputs.
    """

    def __init__(
            self, delays: pq.Quantity, *,
            n_events: int = 100,
            amp_calib: Union[int, np.ndarray] = 0,
            time_calib: Union[int, np.ndarray] = 0,
            i_ramp: Union[int, np.ndarray] = 80,
            i_store: Union[int, np.ndarray] = 70):
        if 0 in delays:
            raise ValueError(
                "Zero delay is not allowed: pre- and postspike cannot "
                + "arrive simultaneously.")
        self.delays = delays
        self.n_events = n_events

        if not isinstance(amp_calib, np.ndarray):
            self.amp_calib = np.ones(
                (halco.NeuronConfigOnDLS.size, halco.SynapseRowOnSynram.size),
                dtype=int) * amp_calib
        else:
            self.amp_calib = amp_calib

        if not isinstance(time_calib, np.ndarray):
            self.time_calib = np.ones(
                (halco.NeuronConfigOnDLS.size, halco.SynapseRowOnSynram.size),
                dtype=int) * time_calib
        else:
            self.time_calib = time_calib

        self.i_ramp = i_ramp
        self.i_store = i_store

        self.address = hal.SynapseQuad.Label(15)
        self.wait_between_pairs = max(500 * pq.us, 5 * np.max(np.abs(delays)))
        self.log = logger.get(
            "calix.spiking.correlation.CorrelationMeasurement")

    @staticmethod
    def configure_base(builder: sta.PlaybackProgramBuilder) -> None:
        """
        Basic config for reading correlation.

        Connect correlation readout to the CADC and configure
        forwarding of post-pulses to the synapses.

        :param builder: Builder to append configuration instructions to.
        """

        # Column Switches: Connect correlation readout to CADC
        quad_config = hal.ColumnCorrelationQuad()
        switch_config = hal.ColumnCorrelationQuad.ColumnCorrelationSwitch()
        switch_config.enable_internal_causal = True
        switch_config.enable_internal_acausal = True

        for switch_coord in halco.iter_all(halco.EntryOnQuad):
            quad_config.set_switch(switch_coord, switch_config)

        for quad_coord in halco.iter_all(halco.ColumnCorrelationQuadOnDLS):
            builder.write(quad_coord, quad_config)

        # Neuron Config
        neuron_config = hal.NeuronConfig()
        neuron_config.enable_fire = True  # necessary for artificial resets
        for neuron_coord in halco.iter_all(halco.NeuronConfigOnDLS):
            builder.write(neuron_coord, neuron_config)

        # Common Neuron Backend Config
        config = hal.CommonNeuronBackendConfig()
        for coord in halco.iter_all(halco.CommonNeuronBackendConfigOnDLS):
            builder.write(coord, config)

        # Neuron Backend Config
        config = hal.NeuronBackendConfig()
        # Set roughly 2 us refractory time, the precise value does not
        # matter. But too low values will not work (issue 3741).
        config.refractory_time = \
            hal.NeuronBackendConfig.RefractoryTime.max
        for coord in halco.iter_all(halco.NeuronBackendConfigOnDLS):
            builder.write(coord, config)

        # Common Correlation Config
        config = hal.CommonCorrelationConfig()
        for coord in halco.iter_all(halco.CommonCorrelationConfigOnDLS):
            builder.write(coord, config)

    @staticmethod
    def configure_input(builder: sta.PlaybackProgramBuilder) -> None:
        """
        Configures PADI bus and synapse drivers such that
        pre-pulses can be sent to all synapse rows.

        :param builder: Builder to append configuration instructions to.
        """

        # PADI global config
        global_padi_config = hal.CommonPADIBusConfig()

        for padi_config_coord in halco.iter_all(
                halco.CommonPADIBusConfigOnDLS):
            builder.write(padi_config_coord, global_padi_config)

        # Synapse driver config
        synapse_driver_config = hal.SynapseDriverConfig()
        synapse_driver_config.enable_address_out = True
        synapse_driver_config.enable_receiver = True
        row_mode = hal.SynapseDriverConfig.RowMode.excitatory
        synapse_driver_config.row_mode_top = row_mode
        synapse_driver_config.row_mode_bottom = row_mode
        synapse_driver_config.row_address_compare_mask = 0b00000  # reach all

        for synapse_driver_coord in halco.iter_all(
                halco.SynapseDriverOnDLS):
            builder.write(synapse_driver_coord, synapse_driver_config)

    def configure_capmem(self, builder: sta.PlaybackProgramBuilder) -> None:
        """
        Configure synapse bias currents for the correlation sensors.

        :param builder: Builder to append the configuration to.
        """

        parameters = {
            halco.CapMemCellOnCapMemBlock.syn_i_bias_corout: 600,
            halco.CapMemCellOnCapMemBlock.syn_i_bias_ramp: self.i_ramp,
            halco.CapMemCellOnCapMemBlock.syn_i_bias_store: self.i_store}

        helpers.capmem_set_quadrant_cells(builder, parameters)
        helpers.wait(builder, constants.capmem_level_off_time)

    def configure_synapses(
            self, builder: sta.PlaybackProgramBuilder,
            quad: halco.SynapseQuadColumnOnDLS) -> None:
        """
        Set the columns of synapses in the given column quad to the
        given address.

        The other synapses are set to address 0. All weights are set to 0.
        The individual correlation calib bits are set according to the
        respective ivars.

        :param builder: Builder to append configuration instructions to.
        :param quad: Coordinate of synapse quad column to be set to address.
        """

        synapses = lola.SynapseMatrix()
        weights = np.zeros((halco.SynapseRowOnSynram.size,
                            halco.SynapseOnSynapseRow.size), dtype=int)
        addresses = np.zeros(halco.SynapseOnSynapseRow.size, dtype=int)

        target_slice = [int(col) for col in quad.toNeuronColumnOnDLS()]

        addresses[target_slice] = self.address
        addresses = np.repeat(
            addresses[np.newaxis], halco.SynapseRowOnSynram.size, axis=0)

        synapses.weights.from_numpy(weights)
        synapses.labels.from_numpy(addresses)

        for synram in halco.iter_all(halco.SynramOnDLS):
            synram_slice = slice(
                int(synram.toEnum()) * (halco.NeuronConfigOnDLS.size
                                        // halco.SynramOnDLS.size),
                (int(synram.toEnum()) + 1) * (halco.NeuronConfigOnDLS.size
                                              // halco.SynramOnDLS.size))
            synapses.amp_calibs.from_numpy(self.amp_calib[synram_slice].T)
            synapses.time_calibs.from_numpy(self.time_calib[synram_slice].T)

            builder.write(synram, synapses)

    def configure_all(self, builder: sta.PlaybackProgramBuilder) -> None:
        """
        Preconfigure the chip for correlation measurements.

        :param builder: Builder to append configuration instructions to.
        """

        self.configure_capmem(builder)
        self.configure_base(builder)
        self.configure_input(builder)
        self.configure_synapses(builder, quad=halco.SynapseQuadColumnOnDLS())

    def prelude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Preconfigure the chip for correlation measurements.

        :param connection: Connection to the chip to run on.
        """

        builder = sta.PlaybackProgramBuilder()
        self.configure_all(builder)
        base.run(connection, builder)

    def measure_quad(
            self, connection: hxcomm.ConnectionHandle,
            quad: halco.SynapseQuadColumnOnDLS,
            synram: halco.SynramOnDLS) -> np.ndarray:
        """
        Measure correlation data for the given quad.

        :param connection: Connection to the chip to run on.
        :param quad: Synapse quad column coordinate to measure correlation on.
        :param synram: Synapse array coordinate to measure correlation on.

        :return: Array of correlation measurements, i.e. difference between
            baseline and result reads. Shaped (len(delays), 4, 256, 2)
            for the given delays, entries in a quad, the synapse rows,
            and causal/acausal correlation. Note that the enumeration
            of entries in a quad is reversed with respect to the
            enumeration of neurons.

        :raises HardwareError: If the observed baseline reads are lower
            than expected. This can be the case if the hardware setup is
            not equipped with the v_reset fix and therefore does not
            receive a high enough voltage to reset the correaltion
            accumulation capacitors. The fix is explained in [1].
            [1]: https://brainscales-r.kip.uni-heidelberg.de/projects/symap2ic/wiki/xboard#putting-a-new-board-into-service  # pylint: disable=line-too-long
        """

        builder = sta.PlaybackProgramBuilder()
        self.configure_synapses(builder, quad)
        base.run(connection, builder)

        to_be_returned = np.empty(
            (len(self.delays), halco.EntryOnQuad.size,
             halco.SynapseRowOnSynram.size, halco.CADCChannelType.size),
            dtype=int)

        for delay_id, delay in enumerate(self.delays):
            builder = sta.PlaybackProgramBuilder()

            # reach "stable pattern":
            # Send the second pulse of correlated pairs once before starting.
            # This is necessary as the hardware seems to have some form of
            # state that remembers the last seen pulse, even if that is
            # a long time ago. This is documented in issue 4021, which is
            # worked around here.
            if delay > 0:  # pre before post
                pyccalix.spiking.send_postpulse(builder, quad, synram)
            else:  # post before pre
                pyccalix.spiking.send_prepulse(builder, synram, self.address)
            helpers.wait(builder, 1 * pq.ms)

            # Reset correlations
            pyccalix.spiking.reset_correlation(builder, quad, synram)

            # Read baseline correlations
            baselines = pyccalix.spiking.read_correlation(
                builder, quad, synram)

            # Send correlated pulses
            builder.block_until(halco.BarrierOnFPGA(), hal.Barrier.omnibus)
            builder.write(halco.TimerOnDLS(), hal.Timer())  # reset timer
            for event_id in range(self.n_events):
                # initial wait
                time_offset = self.wait_between_pairs * (event_id + 1)
                builder.block_until(
                    halco.TimerOnDLS(),
                    int(time_offset.rescale(pq.us).magnitude * int(
                        hal.Timer.Value.fpga_clock_cycles_per_us)))

                if delay > 0:  # pre before post
                    pyccalix.spiking.send_prepulse(
                        builder, synram, self.address)
                else:  # post before pre
                    pyccalix.spiking.send_postpulse(builder, quad, synram)

                # wait for delay
                builder.block_until(
                    halco.TimerOnDLS(),
                    int((time_offset + np.abs(delay)
                         ).rescale(pq.us).magnitude
                        * int(hal.Timer.Value.fpga_clock_cycles_per_us)))

                if delay > 0:  # pre before post
                    pyccalix.spiking.send_postpulse(builder, quad, synram)
                else:  # post before pre
                    pyccalix.spiking.send_prepulse(
                        builder, synram, self.address)

            # Read correlation
            results = pyccalix.spiking.read_correlation(builder, quad, synram)

            # Run program, evaluate results
            base.run(connection, builder)
            baselines = pyccalix.spiking.evaluate_correlation(baselines)
            results = pyccalix.spiking.evaluate_correlation(results)
            to_be_returned[delay_id] = baselines - results

            self.log.TRACE(
                f"Results of synram {int(synram)} quad {int(quad)}:\n"
                + f"Baselines: {baselines.mean(axis=1)} "
                + f"+- {baselines.std(axis=1)},\n"
                + f"Amplitudes: {baselines.mean(1) - results.mean(1)}\n")

            # Assert the baseline reads are as expected. For some seupts,
            # the mux_dac_25 channel may not be routed to the V_reset
            # voltage. The fix is explained in [1].
            # [1]: https://brainscales-r.kip.uni-heidelberg.de/projects/symap2ic/wiki/xboard#putting-a-new-board-into-service  # pylint: disable=line-too-long
            if np.min(baselines) < 120:
                raise exceptions.HardwareError(
                    f"Baseline read was obtained at {np.min(baselines)}, "
                    + "which is lower than expected. The setup may not "
                    + "be equipped with the V_reset fix on the x-board. "
                    + "This can also be a result of setting the store "
                    + "bias current too low, when aiming for high "
                    + "correlation amplitudes.")

        return to_be_returned

    def measure_chip(self, connection: hxcomm.ConnectionHandle) -> np.ndarray:
        """
        Measure correlation data for all synapses on chip.

        :param connection: Connection to the chip to run on.

        :return: Array of correlation measurements, i.e. difference between
            baseline and result reads. Shaped (len(delays), 512, 256, 2)
            for the given delays, all columns on chip (following the
            enumeration of neurons), the synapse rows, and causal/acausal
            correlation.
        """

        results = np.empty(
            (len(self.delays), halco.NeuronConfigOnDLS.size,
             halco.SynapseRowOnSynram.size, halco.CADCChannelType.size),
            dtype=int)

        for synram in halco.iter_all(halco.SynramOnDLS):
            for quad in halco.iter_all(halco.SynapseQuadColumnOnDLS):
                offset = halco.SynapseOnSynapseRow.size * int(synram.toEnum())
                columns = [int(col) + offset
                           for col in quad.toNeuronColumnOnDLS()]

                run_results = self.measure_quad(
                    connection, quad=quad, synram=synram)
                results[:, columns, :, :] = run_results

                with np.printoptions(
                        formatter={"float_kind": lambda x: f"{x:>5.1f}"}):
                    self.log.DEBUG(
                        f"Mean results per delay for synram {synram} "
                        + f"quad {quad}:\n"
                        + f"Causal:  {run_results.mean(axis=(1, 2))[:, 0]}\n"
                        + f"Acausal: {run_results.mean(axis=(1, 2))[:, 1]}")

        return results

    @staticmethod
    def fitfunc(delay: float, amplitude: float, time_constant: float,
                offset: float) -> float:
        """
        Exponential fit function for correlation traces.
        """

        return amplitude * np.exp(-delay / time_constant) + offset

    def fit(self, results: np.ndarray) -> Tuple[np.ndarray, pq.Quantity]:
        """
        Fit exponential traces to the given correlation data.

        The obtained amplitude per correlated event and the time
        constant are returned.

        :param results: Measured correlation amplitudes. The data is
            assumed to be shaped (delays, n_cols, n_rows, causal/acausal),
            as returned by the measure_quad or measure_chip functions.

        :return: Tuple containing two arrays, i.e. the fitted amplitudes
            per event and time constants. The returned arrays have the
            same shape as the input, i.e. (n_cols, n_rows, causal/acausal).
            Values of numpy.nan indicate that the fitting routine failed
            and parameters could not be estimated.
        """

        # check input shape
        if results.shape[0] != len(self.delays):
            raise ValueError(
                f"Given results have a bad shape: {results.shape}. "
                + f"Expected {len(self.delays)} in outer dimension.")

        amplitudes = np.empty_like(results[0], dtype=float)
        time_constants = np.empty_like(results[0], dtype=float)

        # guess and constrain fit parameters
        # (amplitude, time constant, offset)
        fit_parameters = {
            "p0": [20, 5, 0],
            "bounds": ([0, 0.1, 0], [255, 100, 255])}

        for column_id in range(results.shape[1]):
            for row_id in range(results.shape[2]):
                # causal fit
                try:
                    popt = curve_fit(
                        self.fitfunc,
                        self.delays[self.delays > 0].rescale(pq.us).magnitude,
                        results[self.delays > 0, column_id, row_id, 0],
                        **fit_parameters)[0]
                except RuntimeError:
                    popt = [np.nan, np.nan]

                amplitudes[column_id, row_id, 0] = popt[0]
                time_constants[column_id, row_id, 0] = popt[1]

                # acausal fit
                try:
                    popt = curve_fit(
                        self.fitfunc,
                        -self.delays[self.delays < 0].rescale(pq.us).magnitude,
                        results[self.delays < 0, column_id, row_id, 1],
                        **fit_parameters)[0]
                except RuntimeError:
                    popt = [np.nan, np.nan]

                amplitudes[column_id, row_id, 1] = popt[0]
                time_constants[column_id, row_id, 1] = popt[1]

        return amplitudes / self.n_events, time_constants * pq.us

    def estimate_fit(self, results: np.ndarray) -> Tuple[
            np.ndarray, pq.Quantity]:
        """
        Guess fit parameters for exponential traces.

        The given results are analyzed to estimate amplitude and time
        constants, without actually running an optimization function.
        Time constants are calculated by the ratio of decay between
        successive results, amplitudes are extrapolated as results at
        zero delay.

        This function saves lots of runtime compared to performing the
        actual exponential fits.

        :param results: Measured correlation amplitudes. The data is
            assumed to be shaped (delays, n_cols, n_rows, causal/acausal),
            as returned by the measure_quad or measure_chip functions.

        :return: Tuple containing two arrays, i.e. the fitted amplitudes
            per event and time constants. The returned arrays have the
            same shape as the input, i.e. (n_cols, n_rows, causal/acausal).
            Values of numpy.nan indicate that the parameters could not
            be estimated, either as a result of too small amplitudes
            in the given result array, or in case the amplitudes were
            equal at different delays.
        """

        # check input shape
        if results.shape[0] != len(self.delays):
            raise ValueError(
                f"Given results have a bad shape: {results.shape}. "
                + f"Expected {len(self.delays)} in outer dimension.")

        def expand_to_result_shape(arr: np.ndarray) -> np.ndarray:
            """
            Repeat given array to match the shape of results.

            :param array: One-dimensional input array.
            :return: Repeated array, containing dimensions for synapse
                columns and rows: shaped
                (arr.shape[0], results.shape[1], results.shape[2]).
            """

            out = np.repeat(arr[..., np.newaxis], results.shape[1], axis=-1)
            out = np.repeat(out[..., np.newaxis], results.shape[2], axis=-1)
            return out

        def estimate_tau(results: np.ndarray, delays: np.ndarray
                         ) -> np.ndarray:
            """
            Estimate time constant by ratio of two results for
            two adjacent delays.

            Ratios where the amplitude stayed the same or increased are
            ignored. This could happen if delay steps are too small with
            respect to the time constant as a result of noise.

            :param results: Results for one correlation branch, ordered
                by delays (with increasing absolute value).
            :param delays: Delays corresponding to the first dimension
                of the given results.

            :return: Array of estimated time constants.
            """

            # calculate ratio of two results for two adjacent delays
            ratios = results[1:] / results[:-1]

            # ignore ratios where amplitude stayed the same or increased
            ratios[ratios >= 1] = np.nan

            # estimate time constant by ratio and time difference
            taus = expand_to_result_shape(-np.diff(delays)) \
                / np.log(ratios)

            return taus

        def estimate_amplitude(
                results: np.ndarray,
                delays: np.ndarray,
                estimated_taus: np.ndarray) -> np.ndarray:
            """
            Infer amplitude at delay zero by assuming an exponential
            decay with the given time constant.

            :param results: Results from correlation measurements, matching
                the given delays along axis 0.
            :param delays: Delays for when the results were obtained.
            :param estimated_taus: Time constant to use for inferring the
                amplitude.

            :return: Array of estimated amplitudes at delay zero.
            """

            # Calculate ratio to increase result if it were at delay zero
            ratios = np.exp(expand_to_result_shape(delays) / estimated_taus)
            amplitudes = results * ratios

            return np.nanmean(amplitudes, axis=0)

        # set results with extreme values to nan:
        # Quantisation noise would be excessive if we used low results as
        # accurate amplitudes. High results may be in saturation.
        results = results.astype(float)
        results[results < 8] = np.nan
        results[results > 160] = np.nan

        # get ordered causal and anticausal delays
        delay_order = np.argsort(self.delays)
        delay_order = [delay_order[self.delays[delay_order] > 0],
                       delay_order[self.delays[delay_order] < 0][::-1]]
        if len(delay_order[0]) < 2:
            raise ValueError("Too few causal data points provided.")
        if len(delay_order[1]) < 2:
            raise ValueError("Too few acausal data points provided.")

        taus = np.stack([
            estimate_tau(  # causal taus
                results[delay_order[0], :, :, 0],
                self.delays[delay_order[0]]).rescale(pq.us).magnitude,
            estimate_tau(  # acausal taus
                results[delay_order[1], :, :, 1],
                -self.delays[delay_order[1]]).rescale(pq.us).magnitude],
            axis=3) * pq.us  # quantities loses unit when stacking...

        # calculate mean time constant of all measurements
        taus = np.nanmean(taus, axis=0)

        # calculate median time constant, use this if estimated tau
        # is unrealistic to ensure amplitudes are not affected by this
        median_tau = np.nanmedian(taus)
        taus_for_amps = np.ma.MaskedArray(
            taus.rescale(pq.us).magnitude,  # quantities loses unit here
            mask=np.any([taus < 0.2 * median_tau, taus > 5 * median_tau],
                        axis=0),
            fill_value=median_tau.rescale(pq.us).magnitude)

        amplitudes = np.stack([
            estimate_amplitude(  # causal amplitudes
                results[delay_order[0], :, :, 0],
                self.delays[delay_order[0]],
                taus_for_amps.filled()[:, :, 0] * pq.us),
            estimate_amplitude(  # acausal amplitudes
                results[delay_order[1], :, :, 1],
                -self.delays[delay_order[1]],
                taus_for_amps.filled()[:, :, 1] * pq.us)],
            axis=2)

        # Normalize amplitude by number of events
        amplitudes = amplitudes / self.n_events

        return amplitudes, taus
