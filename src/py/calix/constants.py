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

# Boundary values for membrane time constant fit function
tau_mem_range = base.ParameterRange(0.1 * pq.us, 1000 * pq.us)

# Boundary values for synaptic input time constant fit function
tau_syn_range = base.ParameterRange(0.1 * pq.us, 100 * pq.us)
