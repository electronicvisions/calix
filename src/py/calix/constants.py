"""
Global constants used for calibrations.
"""
import quantities as pq

from calix.common import base


# Wait time after reconfiguring the CapMem, before stable voltages
# are expected:
capmem_level_off_time = 20000 * pq.us

# Reliable read range of the CADC
cadc_reliable_range = base.ParameterRange(30, 210)  # LSB
