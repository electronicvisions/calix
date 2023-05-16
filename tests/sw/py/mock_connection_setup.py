import unittest
from dlens_vx_v3 import hxcomm
from calix.common import base


class ConnectionSetup(unittest.TestCase):
    """
    Base class for software tests:
    Provides a ZeroMockConnection, which returns zeros
    for all reads.

    :cvar conn_manager: Connection context manager.
    :cvar connection: Connection to chip to use.
    """

    conn_manager = hxcomm.ManagedZeroMockConnection()
    connection = None

    @classmethod
    def setUpClass(cls) -> None:
        cls.connection = base.StatefulConnection(
            cls.conn_manager.__enter__())

    @classmethod
    def tearDownClass(cls) -> None:
        cls.conn_manager.__exit__()
