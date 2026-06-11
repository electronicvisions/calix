"""
Test the MADC characterization.
"""

import unittest
from connection_setup import ConnectionSetup

from calix.scripts.characterize_madc import main


class TestMADCCharacterization(ConnectionSetup):
    """
    Test that the script is executable.
    """
    def test_run(self):
        main(self.connection, start_voltage=0.1, stop_voltage=1, n_steps=3)


if __name__ == '__main__':
    unittest.main()
