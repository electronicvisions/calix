"""
Tests functionality of hagen-mode neuron calibration. Runs it and sends
some test events afterwards. Asserts the observed amplitudes are right.
"""

from typing import Tuple, List, Optional
import unittest
import os

import numpy as np
import quantities as pq

from dlens_vx_v2 import hal, sta, halco, logger, hxcomm

from calix.common import base, cadc, cadc_helpers, helpers
from calix.hagen import neuron, neuron_helpers
from calix import constants

from connection_setup import ConnectionSetup


log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class TestNeuronCalib(ConnectionSetup):
    """
    Runs a neuron calibration and ensures the results are ok.

    :cvar log: Logger used for output.
    :cvar calibration_result: Result of hagen-mode neuron calibration,
        stored for re-applying.
    :cvar cadc_result: Result of CADC calibration, stored for re-applying.
    """

    log = logger.get("calix.tests.hw.test_neuron_integration")
    calibration_result: Optional[neuron.NeuronCalibResult] = None
    cadc_result: Optional[cadc.CADCCalibResult] = None

    # pylint: disable=too-many-locals
    @classmethod
    def measure_amplitudes(
            cls,
            connection: hxcomm.ConnectionHandle, *,
            excitatory: bool = True, n_events: int = 10,
            wait_between_events: pq.quantity.Quantity = 1.2 * pq.us,
            n_runs: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inject synaptic events and measure membrane potential of all neurons.

        Measures the baseline CADCs before any input is injected. After that
        30 runs are performed. In every run `n_events` events are injected in
        the synapse drivers and the response on all neurons' membranes are
        observed with the CADC.

        :param connection: Connection to the chip to run on.
        :param excitatory: Switch between excitatory and inhibitory events.
        :param n_events: Number of events to send in one integration.
        :param wait_between_events: Waiting time between input events.
        :param n_runs: Number of runs to execute and take statistics from.

        :return: Array of baseline CADC reads without sending events.
        :return: Two-dimensional array of CADC reads after sending inputs.
            The first dimension contains the number of runs, the second
            dimension contains all neurons' results.
        """

        baselines = list()
        results = list()

        builder = sta.PlaybackProgramBuilder()
        row_mode = hal.SynapseDriverConfig.RowMode.excitatory if excitatory \
            else hal.SynapseDriverConfig.RowMode.inhibitory
        builder = neuron_helpers.enable_all_synapse_drivers(
            builder, row_mode=row_mode)
        builder = neuron_helpers.configure_synapses(builder)

        # Read CADCs before synaptic input:
        for synram in halco.iter_all(halco.SynramOnDLS):
            builder = neuron_helpers.reset_neurons(builder)

            # wait for a typical integration time, without any inputs arriving
            builder = helpers.wait(builder, 10 * pq.us)

            builder, ticket = cadc_helpers.cadc_read_row(builder, synram)
            baselines.append(ticket)

        # Send events on PADI bus
        padi_event = hal.PADIEvent()
        for padi_bus_coord in halco.iter_all(halco.PADIBusOnPADIBusBlock):
            padi_event.fire_bus[padi_bus_coord] = True  # pylint: disable=unsupported-assignment-operation

        for _ in range(n_runs):
            for synram in halco.iter_all(halco.SynramOnDLS):
                builder = neuron_helpers.reset_neurons(builder, synram)
                builder.write(halco.TimerOnDLS(), hal.Timer())

                for event_id in range(n_events):  # send some spikes
                    builder.block_until(
                        halco.TimerOnDLS(),
                        int((event_id * wait_between_events.rescale(pq.us))
                            * int(hal.Timer.Value.fpga_clock_cycles_per_us)))
                    builder.write(synram.toPADIEventOnDLS(), padi_event)

                # wait for synaptic input time constant
                builder.block_until(
                    halco.TimerOnDLS(),
                    int((n_events + 1) * wait_between_events.rescale(pq.us)
                        * int(hal.Timer.Value.fpga_clock_cycles_per_us)))

                # Read CADCs after integration in the same builder
                builder, ticket = cadc_helpers.cadc_read_row(builder, synram)
                results.append(ticket)

        base.run(connection, builder)

        baselines = neuron_helpers.inspect_read_tickets(
            baselines).flatten()
        results = neuron_helpers.inspect_read_tickets(
            results).reshape((n_runs, halco.NeuronConfigOnDLS.size))

        return baselines, results

    @staticmethod
    def inspect_results(baseline_data: np.ndarray,
                        result_data: np.ndarray) -> List[str]:
        """
        Calculates statistics of the results and returns them as strings.
        Expects shapes of arrays as returned from measure_amplitudes().

        :param baseline_data: Baseline reads (without sending inputs).
        :param result_data: Reads after sending inputs.

        :return: List of strings (lines) that contain statistics.
        """

        out = list()
        baselines = neuron_helpers.reshape_neuron_quadrants(baseline_data)
        results = np.array([neuron_helpers.reshape_neuron_quadrants(result)
                            for result in result_data])

        for quadrant in range(halco.NeuronConfigBlockOnDLS.size):
            means = [np.mean(baselines[quadrant]),   # baselines
                     np.mean(results[:, quadrant]),  # after integration
                     np.mean(results[:, quadrant]    # amplitudes
                             - baselines[quadrant])]
            stds = [np.std(baselines[quadrant]),     # baseline spread
                    np.std(np.mean(results[:, quadrant],
                                   axis=0)),         # result spread
                    np.std(np.mean(results[:, quadrant], axis=0)
                           - baselines[quadrant])]   # systematic dev. of amps
            noise = np.std(results[:, quadrant]      # statistical noise
                           - baselines[quadrant], axis=0)

            out.append(f"Qdr {quadrant}: before {means[0]:4.2f} +- "
                       + f"{stds[0]:4.2f}, after {means[1]:5.2f} +- "
                       + f"{stds[1]:4.2f}, amplitude {means[2]:4.2f} +- "
                       + f"{stds[2]:4.2f}")

            out.append("       noise: min/10%/median/mean/90%/max "
                       + f"{np.min(noise):4.2f} / "
                       + f"{np.percentile(noise, 10):4.2f} / "
                       + f"{np.median(noise):4.2f} / "
                       + f"{np.mean(noise):4.2f} / "
                       + f"{np.percentile(noise, 90):4.2f} / "
                       + f"{np.max(noise):4.2f}")
        return out

    def evaluate_results(self, baselines: List[np.ndarray],
                         results: List[np.ndarray],
                         expected_baseline: Optional[int] = None):
        """
        Assert the amplitudes and baselines are as expected.

        :param baselines: Baseline reads of neurons. List containing an
            array for excitatory and inhibitory measurement.
        :param results: Results after sending inputs. List containing an
            array for excitatory and inhibitory measurement.
        :param expected_baseline: Expected CADC read at the baseline,
            i.e. without any inputs. If None, the baseline is assumed
            to be uncalibrated; if given, it is assumed to be calibrated
            to the given integer. The latter can be achieved by calling
            neuron.calibrate_baseline().
        """

        # log results with statistics by quadrant
        self.log.DEBUG("Sent inputs with baseline {0}calibrated.".format(
            "un" if expected_baseline is None else ""))

        self.log.DEBUG("Results of excitatory inputs:" + os.linesep
                       + os.linesep.join(self.inspect_results(
                           baselines[0], results[0])))
        self.log.DEBUG("Results of inhibitory inputs:" + os.linesep
                       + os.linesep.join(self.inspect_results(
                           baselines[1], results[1])))

        # Check baseline read
        noise = np.abs(np.diff([baselines[0], baselines[1]], axis=0))
        self.assertLess(
            np.percentile(noise, 80), 6,
            "Difference in baseline reads of excitatory and inhibitory "
            + "stimulation is high.")

        if expected_baseline is not None:
            for baseline in baselines:
                noise = np.abs(baseline - expected_baseline)
                self.assertLess(
                    np.percentile(noise, 80), 16,
                    "More than 20% of the baseline reads deviate "
                    + f"significantly (>16) from {expected_baseline}.")

        # Assert statistical noise in neurons is less than 7 CADC LSB
        # for at least 75% of the neurons of each quadrant
        for amplitudes in results:
            noise = np.std(amplitudes, axis=0)
            noise = neuron_helpers.reshape_neuron_quadrants(noise)
            for quadrant_data in noise:
                self.assertLess(
                    np.percentile(quadrant_data, 75), 7,
                    "Statistical noise in observed amplitudes is high.")

        if expected_baseline is None:
            exc_amplitudes = np.mean(results[0], axis=0) - baselines[0]
            inh_amplitudes = baselines[1] - np.mean(results[1], axis=0)
        else:
            exc_amplitudes = np.mean(results[0], axis=0) - expected_baseline
            inh_amplitudes = expected_baseline - np.mean(results[1], axis=0)

        # Assert amplitudes are above 20 for 90% of neurons in each quadrant
        for amplitudes in [exc_amplitudes, inh_amplitudes]:
            for quadrant_data in neuron_helpers.reshape_neuron_quadrants(
                    amplitudes):
                self.assertGreater(np.percentile(quadrant_data, 10), 20,
                                   "Observed amplitudes are low.")

        # Assert systematic noise between neurons' amplitudes is less
        # than 10 LSB for 10...90th percentiles of quadrants
        for amplitudes in [exc_amplitudes, inh_amplitudes]:
            sorted_amplitudes = np.sort(
                neuron_helpers.reshape_neuron_quadrants(amplitudes), axis=1
            )[:, int(0.1 * halco.NeuronConfigOnNeuronConfigBlock.size):
              int(0.9 * halco.NeuronConfigOnNeuronConfigBlock.size)]

            noise = np.std(sorted_amplitudes, axis=1)
            for quadrant_data in noise:
                self.assertLess(
                    quadrant_data, 10,
                    "Systematic deviations between neurons are high.")

        # Assert excitatory and inhibitory amplitudes are roughly equal,
        # i.e. mean exc and inh amplitudes per quadrant deviate by
        # at most 10 LSB (20 LSB if baseline calibrated, which can
        # mean additional constant offsets.)
        deviations = np.abs(
            neuron_helpers.reshape_neuron_quadrants(exc_amplitudes).mean(1)
            - neuron_helpers.reshape_neuron_quadrants(inh_amplitudes).mean(1))
        for quadrant_data in deviations:
            self.assertLess(
                quadrant_data, 20 if expected_baseline is not None else 10,
                "Strong deviations between excitatory and inhibitory "
                + "amplitudes")

    def evaluate_calibration(self, connection: hxcomm.ConnectionHandle):
        """
        Send events to the neurons and assert the calibration works
        as expected.

        :param connection: Connection to the chip to run on.
        """

        # Measure amplitudes of events
        exc_baselines, exc_results = self.measure_amplitudes(
            connection, excitatory=True)

        inh_baselines, inh_results = self.measure_amplitudes(
            connection, excitatory=False)

        # Assert amplitudes are as expected
        self.evaluate_results([exc_baselines, inh_baselines],
                              [exc_results, inh_results])

    def evaluate_baseline_calib(self, connection: hxcomm.ConnectionHandle):
        """
        Calibrate the baseline to 128, send events and assert the
        calibration works as expected.

        :param connection: Connection to the chip to run on.
        """

        # Set baseline read to 128
        expected_baseline = 128
        neuron.calibrate_baseline(connection, target_read=expected_baseline)

        # Measure amplitudes of events
        exc_baselines, exc_results = self.measure_amplitudes(
            connection, excitatory=True)

        inh_baselines, inh_results = self.measure_amplitudes(
            connection, excitatory=False)

        # Assert baseline is calibrated and amplitudes are good
        self.evaluate_results([exc_baselines, inh_baselines],
                              [exc_results, inh_results],
                              expected_baseline=expected_baseline)

    def test_00_neuron_calibration(self):
        """
        Loads calibration.
        """

        result = self.apply_calibration("hagen")

        self.__class__.cadc_result = result.cadc_result
        self.__class__.calibration_result = result.neuron_result

    def test_01_calibration_results(self):
        # Require 60% success rate of calibration
        self.assertGreater(
            sum(self.__class__.calibration_result.success.values()),
            int(halco.NeuronConfigOnDLS.size * 0.60),
            "Calibration failed for more than 40% of the neurons.")

        # Check calibration success
        self.evaluate_calibration(self.connection)

    def test_02_results_w_baseline(self):
        self.evaluate_baseline_calib(self.connection)

    def test_03_overwrite(self):
        """
        Overwrite the exiting calibration, assert the tests fail.
        """

        # Overwrite calibration
        builder = helpers.capmem_set_neuron_cells(
            sta.PlaybackProgramBuilder(),
            {halco.CapMemRowOnCapMemBlock.v_leak: 600,
             halco.CapMemRowOnCapMemBlock.v_reset: 610,
             halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_shift: 310,
             halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_shift: 310,
             halco.CapMemRowOnCapMemBlock.i_bias_leak: 250,
             halco.CapMemRowOnCapMemBlock.i_bias_synin_exc_gm: 0,
             halco.CapMemRowOnCapMemBlock.i_bias_synin_inh_gm: 0})
        builder = helpers.wait(builder, constants.capmem_level_off_time)
        base.run(self.connection, builder)

        # Measure results, assert calibration is gone
        self.assertRaises(AssertionError,
                          self.evaluate_calibration, self.connection)

    def test_04_overwrite_w_baseline(self):
        self.assertRaises(AssertionError,
                          self.evaluate_baseline_calib, self.connection)


if __name__ == "__main__":
    unittest.main()
