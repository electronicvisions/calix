import unittest

import quantities as pq
import pyccalix
from dlens_vx_v3 import halco


class TypesTest(unittest.TestCase):
    """
    Tests the voltage and time types.
    """

    def test_time_in_us(self):
        time = pyccalix.TimeInS(10.)
        self.assertEqual(time.value(), 10.)
        self.assertEqual(time.as_quantity(), 10. * pq.s)

    def test_potential_in_volt(self):
        potential = pyccalix.PotentialInVolt(10.)
        self.assertEqual(potential.value(), 10.)
        self.assertEqual(potential.as_quantity(), 10. * pq.V)

    def test_per_neuron_time_constant(self):
        times = pyccalix.NeuronCalibTarget.PerNeuronTimeConstant()
        times[halco.AtomicNeuronOnDLS()] = 12.
        self.assertEqual(times[halco.AtomicNeuronOnDLS()], 12.)
        self.assertEqual(times.as_quantity()[0], 12. * pq.s)

    def test_tau_syn(self):
        times = pyccalix.NeuronCalibTarget.TauSyn()
        times[halco.AtomicNeuronOnDLS()][halco.SynapticInputOnNeuron()] = 12.
        self.assertEqual(times[
            halco.AtomicNeuronOnDLS()][halco.SynapticInputOnNeuron()], 12.)
        self.assertEqual(times.as_quantity()[0][0], 12. * pq.s)

    def test_hagen_per_nrn_time_const(self):
        times = pyccalix.HagenNeuronCalibTarget.PerNeuronTimeConstant()
        times[halco.AtomicNeuronOnDLS()] = 12.
        self.assertEqual(times[halco.AtomicNeuronOnDLS()], 12.)
        self.assertEqual(times.as_quantity()[0], 12. * pq.s)

    def test_hagen_tau_syn(self):
        times = pyccalix.HagenNeuronCalibTarget.TauSyn()
        times[halco.AtomicNeuronOnDLS()][halco.SynapticInputOnNeuron()] = 12.
        self.assertEqual(times[
            halco.AtomicNeuronOnDLS()][halco.SynapticInputOnNeuron()], 12.)
        self.assertEqual(times.as_quantity()[0][0], 12. * pq.s)

    def test_assignment(self):
        target = pyccalix.NeuronCalibTarget()
        with self.assertRaises(TypeError):
            target.tau_mem[halco.AtomicNeuronOnDLS()] = 20 * pq.us  # pylint: disable=unsupported-assignment-operation


if __name__ == "__main__":
    unittest.main()
