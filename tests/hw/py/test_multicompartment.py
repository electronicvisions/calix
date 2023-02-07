#!/usr/bin/env python

import unittest
from typing import List
import numpy as np
import quantities as pq

from dlens_vx_v3 import halco, sta, hal, logger

from connection_setup import ConnectionSetup

from calix.common import algorithms, base, cadc
from calix.hagen import neuron_helpers
from calix.multicomp.neuron_icc_bias import ICCMADCCalib


log = logger.get("calix")
logger.set_loglevel(log, logger.LogLevel.DEBUG)


class TestICCCalib(ConnectionSetup):
    """
    Calibrate inter-compartment conductance and assert that standard deviation
    decreases compared to uncalibrated case.

    Calibrate the inter-compartment conductance and verify that the standard
    deviation of the measured total time constant is decreasing.
    """
    @staticmethod
    def set_config_to_most_common(
            neuron_configs: List[hal.NeuronConfig]) -> List[bool]:
        """
        Alter neuron configs so they use the most common inter-comaprtment
        division/multiplication setting and return neuron-indicies of most
        common setting in list.

        Inspect the configuration of all calibrated neurons and count the
        number of neuron configurations with i_bias_nmda
        division/multiplication enabled/disabled, respectively.
        Set the most commonly used configuration for all neurons.
        Additionally, return a list of bools marking the positions of the
        most common config in the original neuron config.

        :param neuron_configs: Neuron configurations.

        :return: List indicating bool positions of original most common neuron
            config
        """

        div_icc = [nc.enable_divide_multicomp_conductance_bias for nc in
                   neuron_configs]
        mul_icc = [nc.enable_multiply_multicomp_conductance_bias for nc in
                   neuron_configs]

        no_div_and_no_mul = [not(mul or div) for mul, div in
                             zip(mul_icc, div_icc)]
        cases = {"division": sum(div_icc), "multiplication": sum(mul_icc),
                 "None": sum(no_div_and_no_mul)}

        indicies = []
        case = max(cases, key=cases.get)
        for config in neuron_configs:
            if case == "division":
                config.enable_divide_multicomp_conductance_bias = True
                config.enable_multiply_multicomp_conductance_bias = False
                indicies = div_icc
            elif case == "multiplication":
                config.enable_divide_multicomp_conductance_bias = False
                config.enable_multiply_multicomp_conductance_bias = True
                indicies = mul_icc
            else:
                config.enable_divide_multicomp_conductance_bias = False
                config.enable_multiply_multicomp_conductance_bias = False
                indicies = no_div_and_no_mul

        return indicies

    def test_icc_calibration(self):
        """
        Calibrate inter-compartment conductance and assert that standard
        deviation decreases compared to uncalibrated case.

        Execute the inter-compartment calibration and measure the standard
        deviation of the total time constant. Then find the most common setting
        of inter-compartment conductance multplication/division
        enabled/disabled and set all neurons to that setting. Apply a mean
        CapMem value, which is calculated from the mean CapMem values after
        calibration, to all neurons.
        """

        # Calibrate CADC
        cadc.calibrate(self.connection)

        builder = sta.PlaybackProgramBuilder()
        builder, _ = neuron_helpers.configure_chip(builder)
        base.run(self.connection, builder)

        calibration = ICCMADCCalib(target=5. * pq.us)

        # calibrate the inter-compartment conductance
        calib_res = calibration.run(
            self.connection, algorithm=algorithms.NoisyBinarySearch()).\
            calibrated_parameters

        # reapply the neuron configs of the calibration, which were overwritten
        # during the postlude, which is called in run
        builder = sta.PlaybackProgramBuilder()
        for coord, config in zip(halco.iter_all(halco.NeuronConfigOnDLS),
                                 calibration.neuron_configs):
            builder.write(coord, config)

        # measure the total time constants of calibrated setup
        tau_tot_calib = calibration.measure_results(self.connection, builder)

        # set majority of div/mul inter-compartment conductance bias for all
        # neuron configs
        indicies_most_common = self.set_config_to_most_common(
            calibration.neuron_configs)

        # write the most common neuron config for all neurons
        builder = sta.PlaybackProgramBuilder()
        for coord, config in zip(halco.iter_all(halco.NeuronConfigOnDLS),
                                 calibration.neuron_configs):
            builder.write(coord, config)

        # set the mean bias value of the most common neuron configs for all
        # neurons
        calibration.configure_parameters(
            builder, int(np.mean(calib_res[indicies_most_common])))

        # measure the total time constants of uncalibrated setup
        tau_tot_no_calib = calibration.measure_results(
            self.connection, builder)

        # compare measurement results and assert that the standard deviation
        # of the total time constant between the neurons is decreasing
        self.assertLess(np.std(tau_tot_calib), np.std(tau_tot_no_calib) / 2,
                        "Starndard deviation did not decrease significantly "
                        + "after calibration:\n"
                        + f"tau_calib:\tf{np.std(tau_tot_calib)}\n"
                        + f"tau_no_calib:\t{np.std(tau_tot_no_calib)}")


if __name__ == "__main__":
    unittest.main()
