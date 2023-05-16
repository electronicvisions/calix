from __future__ import annotations

from pathlib import Path
import time
import threading
import unittest
import tempfile
from connection_setup import QuiggeldyConnectionSetup

from dlens_vx_v3 import hxcomm, sta, lola, halco, hal
import calix
from calix import calibrate


class TestQuiggeldy(QuiggeldyConnectionSetup):
    run_reset_chip_state = True

    @staticmethod
    def thread_reset_chip_state():
        builder, _ = sta.generate(sta.DigitalInit())
        builder.write(halco.ChipOnDLS(), lola.Chip())
        builder.block_until(halco.BarrierOnFPGA(), hal.Barrier.omnibus)
        program = builder.done()
        with hxcomm.ManagedConnection() as conn:
            while TestQuiggeldy.run_reset_chip_state:
                sta.run(conn, program)
                time.sleep(5)

    def test_concurrent_calibration(self):
        reset_chip_state = threading.Thread(
            target=TestQuiggeldy.thread_reset_chip_state)

        try:
            reset_chip_state.start()

            with tempfile.TemporaryDirectory() as temp_dir:
                target = calix.spiking.SpikingCalibTarget()
                target.neuron_target = \
                    calix.spiking.neuron.NeuronCalibTarget().DenseDefault
                calibrate(target, cache_paths=[Path(temp_dir)],
                          connection=self.connection)

        finally:
            TestQuiggeldy.run_reset_chip_state = False
            reset_chip_state.join()


if __name__ == "__main__":
    unittest.main()
