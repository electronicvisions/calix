import unittest
from itertools import product
import numpy as np
import quantities as pq

from dlens_vx_v2 import halco, hal

from calix.spiking.refractory_period import calculate_settings, Settings, \
    _clock_period, _clock_base_frequency


class RefractoryPeriodTest(unittest.TestCase):
    """
    Test whether the clocks and refractory counters are set correctly.

    :cvar min_tau_ref: Minimal possible refractory time.
    :cvar max_tau_ref: Maximal possible refractory time.
    :cvar min_holdoff_time: Minimal possible holdoff time.
    :cvar max_holdoff_time: Maximal possible holdoff time.
    """
    min_tau_ref: pq.Quantity = _clock_period(
        hal.CommonNeuronBackendConfig.ClockScale.min, _clock_base_frequency) \
        * hal.NeuronBackendConfig.RefractoryTime.min
    max_tau_ref: pq.Quantity = _clock_period(
        hal.CommonNeuronBackendConfig.ClockScale.max, _clock_base_frequency) \
        * hal.NeuronBackendConfig.RefractoryTime.max

    min_holdoff_time: pq.Quantity = 0 * pq.us
    max_holdoff_time: pq.Quantity = _clock_period(
        hal.CommonNeuronBackendConfig.ClockScale.max, _clock_base_frequency) \
        * (2 * hal.NeuronBackendConfig.ResetHoldoff.max + 1)

    def assert_holdoff_near_target(self,
                                   settings: Settings,
                                   target: pq.Quantity):
        """
        Assert that the holdoff period differes by at most one clock period.
        """
        clock_scale = np.ones(settings.input_clock.size) * settings.fast_clock
        clock_scale[settings.input_clock == 0] = settings.slow_clock

        clock_period = _clock_period(clock_scale,
                                     _clock_base_frequency).rescale(pq.us)

        # Lowest bit of refractory counter is always used for holdoff
        real_cycles = 2 * (hal.NeuronBackendConfig.ResetHoldoff.max
                           - settings.reset_holdoff) + 1
        calculated = real_cycles * clock_period

        # Factor of two since lowest bit of refractory counters is always used
        # for holdoff -> increase a bit due to rounding errors
        self.assertTrue(
            np.all(np.abs(calculated - target) <= 2.1 * clock_period),
            'The calculated values of the holdoff time are not near the '
            'provided target values.')

    def assert_ref_near_target(self, settings: Settings, target: pq.Quantity):
        """
        Assert that the refractory period  differes by at most one clock
        period.
        """

        clock_scale = np.ones(settings.input_clock.size) * settings.fast_clock
        clock_scale[settings.input_clock == 0] = settings.slow_clock

        clock_period = _clock_period(clock_scale,
                                     _clock_base_frequency).rescale(pq.us)
        calculated = settings.refractory_counters * clock_period

        # Factor of 1.1 due to rounding errors
        self.assertTrue(
            np.all(np.abs(calculated - target) <= 1.1 * clock_period),
            'The calculated values of the refractory time are not near the '
            'provided target values.')

    def test_ref_whole_range(self):
        """
        Test that the refractory period is larger or equal to the target time.
        """
        target = np.linspace(self.min_tau_ref.rescale(pq.ns).base,
                             self.max_tau_ref.rescale(pq.ns).base,
                             num=halco.NeuronConfigOnDLS.size) * pq.ns
        settings = calculate_settings(target, 1 * pq.us)

        self.assert_ref_near_target(settings, target)
        self.assert_holdoff_near_target(settings, 1 * pq.us)

    def test_scalar_refractory_target(self):
        '''
        Test a scalar target for the refractory time.
        '''
        settings = calculate_settings(10 * pq.us)

        # All neurons use the fast clock
        self.assertTrue(np.all(settings.input_clock == 1))

        # All neurons have the same counter value
        self.assertTrue(len(set(settings.refractory_counters)) == 1)

        # All neurons have the minimal holdoff time
        self.assertTrue(set(settings.reset_holdoff) == {15})

    def test_holdoff_whole_range(self):
        """
        Test that the holdoff period is larger or equal to the target time.
        """
        target = np.linspace(self.min_holdoff_time.rescale(pq.ns).base,
                             self.max_holdoff_time.rescale(pq.ns).base,
                             num=halco.NeuronConfigOnDLS.size,
                             dtype=int) * pq.ns
        settings = calculate_settings(10 * pq.us, target)

        self.assert_ref_near_target(settings, 10 * pq.us)
        self.assert_holdoff_near_target(settings, target)

    def test_edge_cases(self):
        # rescale needed since quantity is lost in `np.repeat` operation
        min_max_holdoff = [self.min_holdoff_time.rescale(pq.us),
                           self.max_holdoff_time.rescale(pq.us)]
        min_max_ref = [self.min_tau_ref.rescale(pq.us),
                       self.max_tau_ref.rescale(pq.us)]

        combinations = list(product(min_max_ref, min_max_holdoff))
        tau_ref_holdoff = \
            np.repeat(combinations,
                      halco.NeuronConfigOnDLS.size / len(combinations),
                      axis=0)

        settings = calculate_settings(tau_ref_holdoff[:, 0] * pq.us,
                                      tau_ref_holdoff[:, 1] * pq.us)

        self.assert_ref_near_target(settings,
                                    tau_ref_holdoff[:, 0] * pq.us)
        self.assert_holdoff_near_target(settings,
                                        tau_ref_holdoff[:, 1] * pq.us)


if __name__ == "__main__":
    unittest.main()
