import unittest
from dlens_vx_v2 import sta, hxcomm


class ConnectionSetup(unittest.TestCase):
    """
    Base class for hardware tests:
    Provides connection and ExperimentInit() initialization
    during setup and disconnects during teardown.

    :cvar conn_manager: Connection context manager.
    :cvar connection: Connection to chip to use.
    """

    conn_manager = hxcomm.ManagedConnection()
    connection = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.connection = cls.conn_manager.__enter__()

        # Initialize the chip
        builder, _ = sta.ExperimentInit().generate()
        sta.run(cls.connection, builder.done())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.conn_manager.__exit__()
