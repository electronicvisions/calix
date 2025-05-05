"""
Global constants used for calibrations.
"""
import quantities as pq

from calix.common import parameter_range


# Wait time after reconfiguring the CapMem, before stable voltages
# are expected:
capmem_level_off_time = 20000 * pq.us

# Reliable read range of the CADC
cadc_reliable_range = parameter_range.ParameterRange(30, 210)  # LSB

# Boundary values for membrane time constant fit function
tau_mem_range = parameter_range.ParameterRange(0.1 * pq.us, 1000 * pq.us)

# Boundary values for synaptic input time constant fit function
tau_syn_range = parameter_range.ParameterRange(0.1 * pq.us, 100 * pq.us)
