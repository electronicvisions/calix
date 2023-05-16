"""
Measures correlation of pre- and postsynaptic spikes with default
parameters (bias currents). We test functionality of correlation
sensors with default settings and assert the calibration of
CapMem biases works accurately.
"""


import unittest
import numpy as np
import quantities as pq

from dlens_vx_v3 import halco, lola, logger

from connection_setup import ConnectionSetup

from calix.hagen import base, cadc
from calix.spiking import correlation, correlation_measurement


log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class TestCorrelation(ConnectionSetup):
    """
    Configures the chip such that correlation can be measured and read
    out via the CADC.

    First, we assert the results change as expected with the
    delay between pre- and postsynaptic spikes, using default parameters.

    Then, we calibrate the CapMem biases (not individual synapses,
    as this takes a lot of runtime). We assert the calibration works.
    Also, different methods of obtaining characteristics (amplitude,
    time constant) from the correlation data are checked, as we have
    a fast estimation and an accurate fit method.

    :cvar log: Logger used to log results.
    :cvar MEASURED_QUADS: Synapse quad columns to use for measuring.
        All located on quadrant 0, will be shifted as needed.
    """

    log = logger.get("calix.tests.hw.test_correlation")
    MEASURED_QUADS = [halco.SynapseQuadColumnOnDLS(quad)
                      for quad in [1, 9, 33, 41]]

    def test_00_prepare(self):
        """
        Calibrate CADCs.
        """

        cadc.calibrate(self.connection)

    def test_01_correlation(self):
        """
        Measure correlation and assert results are as expected.

        We do not calibrate parameters yet.
        """

        # set correlation voltages
        builder = base.WriteRecordingPlaybackProgramBuilder()
        dac_config = lola.DACChannelBlock().default_ldo_2
        dac_config.set_voltage(halco.DACChannelOnBoard.v_res_meas, 0.95)
        dac_config.set_voltage(halco.DACChannelOnBoard.mux_dac_25, 1.85)
        builder.write(halco.DACChannelBlockOnBoard(), dac_config)
        base.run(self.connection, builder)

        delays = np.array([-5, -1, 1, 5]) * pq.us  # need to be sorted!
        measurement = correlation_measurement.CorrelationMeasurement(
            delays=delays, n_events=30)
        measurement.prelude(self.connection)
        results = measurement.measure_chip(self.connection)

        # assert causal > acausal for positive delay and vice versa
        is_causal = delays > 0

        is_ok = results[is_causal, :, :, 0] > results[is_causal, :, :, 1]
        self.assertGreater(
            np.sum(is_ok),
            halco.NeuronConfigOnDLS.size * halco.SynapseRowOnSynram.size
            * 0.8 * np.sum(is_causal),
            "Too few synapses respond to causal correlation.")

        is_ok = results[~is_causal, :, :, 0] < results[~is_causal, :, :, 1]
        self.assertGreater(
            np.sum(is_ok),
            halco.NeuronConfigOnDLS.size * halco.SynapseRowOnSynram.size
            * 0.8 * np.sum(~is_causal),
            "Too few synapses respond to acausal correlation.")

        # assert amplitudes decrease with time
        is_ok = results[is_causal, :, :, 0][0] > \
            results[is_causal, :, :, 0][-1]
        self.assertGreater(
            np.sum(is_ok),
            halco.NeuronConfigOnDLS.size * halco.SynapseRowOnSynram.size
            * 0.8,
            "Too few synapses show time dependency of causal correlation.")

        is_ok = results[~is_causal, :, :, 1][0] < \
            results[~is_causal, :, :, 1][-1]
        self.assertGreater(
            np.sum(is_ok),
            halco.NeuronConfigOnDLS.size * halco.SynapseRowOnSynram.size
            * 0.8,
            "Too few synapses show time dependency of acausal correlation.")

    def test_02_capmem_calibration(self):
        """
        Calibrate correlation CapMem parameters and assert the
        measured results are sensible.
        """

        # unset correlation voltages, calibration must set them itself
        builder = base.WriteRecordingPlaybackProgramBuilder()
        builder.write(halco.DACChannelBlockOnBoard(),
                      lola.DACChannelBlock().default_ldo_2)
        base.run(self.connection, builder)

        # calibrate amplitudes and time constants
        calib_result = correlation.calibrate(
            self.connection, target=correlation.CorrelationCalibTarget(
                amplitude=0.5, time_constant=5 * pq.us),
            options=correlation.CorrelationCalibOptions(
                calibrate_synapses=False))

        # check correlation parameters in a few quad columns
        delays = [-50, -30, -15, -10, -5, -2.5, -1,
                  1, 2.5, 5, 10, 15, 30, 50] * pq.us
        measurement = correlation_measurement.CorrelationMeasurement(
            delays, amp_calib=calib_result.amp_calib,
            time_calib=calib_result.time_calib)

        results = []
        for quadrant in halco.iter_all(halco.CapMemBlockOnDLS):
            for quad in self.MEASURED_QUADS:  # all on west quadrant
                if int(quadrant.toCapMemBlockOnHemisphere().toEnum()) == 1:
                    quad = quad + 16  # move quad coord to east quadrant
                results.append(measurement.measure_quad(
                    self.connection, quad=quad,
                    synram=quadrant.toHemisphereOnDLS().toSynramOnDLS()))

        # result shape: delays, columns, rows, causal/acausal
        results = np.concatenate(results, axis=1)
        self.assertEqual(
            results.shape,
            (len(delays), len(self.MEASURED_QUADS) * halco.EntryOnQuad.size
             * halco.CapMemBlockOnDLS.size, halco.SynapseRowOnSynram.size,
             halco.CADCChannelType.size),
            "Correlation result shape does not match expectation!")

        amps, taus = measurement.estimate_fit(results)
        self.assertEqual(
            amps.shape,
            (len(self.MEASURED_QUADS) * halco.EntryOnQuad.size
             * halco.CapMemBlockOnDLS.size, halco.SynapseRowOnSynram.size,
             halco.CADCChannelType.size),
            "Amplitude result shape does not match expectation!")
        self.assertEqual(
            taus.shape,
            (len(self.MEASURED_QUADS) * halco.EntryOnQuad.size
             * halco.CapMemBlockOnDLS.size, halco.SynapseRowOnSynram.size,
             halco.CADCChannelType.size),
            "Time constant result shape does not match expectation!")

        amps = amps.reshape((halco.CapMemBlockOnDLS.size, -1))
        taus = taus.reshape((halco.CapMemBlockOnDLS.size, -1))
        amp_percentiles = []
        tau_percentiles = []

        self.log.INFO("Correlation characteristics (30/50/70th percentile):")
        for quadrant in range(halco.CapMemBlockOnDLS.size):
            amp_percentiles.append(np.percentile(
                amps[quadrant][~np.isnan(amps[quadrant])], [30, 50, 70]))
            tau_percentiles.append(np.percentile(
                taus[quadrant][~np.isnan(taus[quadrant])], [30, 50, 70]))
            self.log.INFO(
                f"Quadrant {quadrant}: estimated results "
                + "(30/50/70th percentile): "
                + f"Amplitudes {amp_percentiles[quadrant]}, "
                + f"Time constants {tau_percentiles[quadrant]}")

        amp_percentiles = np.array(amp_percentiles)
        tau_percentiles = np.array(tau_percentiles)
        allowed_relative_deviation = 2.
        self.assertFalse(
            np.any(amp_percentiles < (calib_result.target.amplitude
                                      / allowed_relative_deviation)),
            "Correlation amplitudes are lower than expected.")
        self.assertFalse(
            np.any(amp_percentiles > (calib_result.target.amplitude
                                      * allowed_relative_deviation)),
            "Correlation amplitudes are higher than expected.")
        self.assertFalse(
            np.any(tau_percentiles < (calib_result.target.time_constant
                                      / allowed_relative_deviation)),
            "Correlation time constants are smaller than expected.")
        self.assertFalse(
            np.any(tau_percentiles > (calib_result.target.time_constant
                                      * allowed_relative_deviation)),
            "Correlation time constants are larger than expected.")

        # Check whether performing actual fits yields similar results:
        # To save runtime, check only the results from qudrant 0
        quadrant_slice = slice(  # select first quarter of array
            0, results.shape[1] // halco.CapMemBlockOnDLS.size)
        results = results[:, quadrant_slice, ...]
        amps, taus = measurement.fit(results)
        self.log.INFO(
            f"Fit results contained {np.sum(np.isnan(amps))} nan values.")
        self.log.INFO(
            "Quadrant 0: Results from fit: "
            + f"Median amplitude {np.median(amps)}, "
            + f"Median time constant {np.median(taus)}")
        np.testing.assert_allclose(
            np.median(amps), amp_percentiles[0, 1], rtol=0.2, atol=0.2,
            err_msg="Median amplitude obtained from fits does not match "
            + "the value estimated by direct calculations.")
        np.testing.assert_allclose(
            np.median(taus).rescale(pq.us).magnitude,
            tau_percentiles[0, 1], rtol=0.3, atol=1,
            err_msg="Median time constant obtained from fits does not match "
            + "the value estimated by direct calculations.")


if __name__ == "__main__":
    unittest.main()
