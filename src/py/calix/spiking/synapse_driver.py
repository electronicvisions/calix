"""
Provides a calibration class for the synapse driver STP offset
for the usual spiking mode.
"""

from typing import Union
import numpy as np
from dlens_vx_v1 import hal, halco, sta, hxcomm

from calix.common import helpers
import calix.hagen.synapse_driver as hagen_driver


class STPOffsetCalibration(hagen_driver.STPOffsetCalibration):
    """
    Calibrate synapse driver STP ramp offsets for usage in
    spiking mode.

    Since the hagen-mode DAC is not involved now, the STP voltages can
    use a higher dynamic range determined by v_charge and v_recover.
    This usually means a higher STP ramp current can be used compared
    to hagen mode.

    Drivers connected to different CapMem instances are calibrated to
    different targets since deviations in STP voltages and ramp currents
    (which are set for each CapMem instance individually) have higher
    variations than can be compensated with the STP offset.

    Requirements:
    * Neurons are calibrated for integration mode.
    * Synaptic events can reach the neurons, i.e. the synapse DAC bias
      is set and the `hal.ColumnCurrentSwitch`es allow currents from
      the synapses through.
    * Neuron membrane readout is connected to the CADCs (causal and acausal).

    :ivar v_stp: STP voltage to use during amplitude measurement,
        reached by setting v_charge and v_recover to this value.
        If an array of values is given, they are configured per quadrant.
    :ivar i_ramp: STP ramp current.
        If an array of values is given, they are configured per quadrant.
    """

    def __init__(self, v_stp: Union[int, np.ndarray] = 180,
                 i_ramp: Union[int, np.ndarray] = 600):
        super().__init__()

        self.default_syndrv_config = hal.SynapseDriverConfig()
        self.default_syndrv_config.enable_address_out = True
        self.default_syndrv_config.enable_receiver = True
        self.default_syndrv_config.row_mode_top = \
            hal.SynapseDriverConfig.RowMode.excitatory
        self.default_syndrv_config.row_mode_bottom = \
            hal.SynapseDriverConfig.RowMode.excitatory
        self.default_syndrv_config.enable_stp = True
        self.default_syndrv_config.row_address_compare_mask = 0b00000

        self.v_stp = v_stp
        self.i_ramp = i_ramp

    def prelude(self, connection: hxcomm.ConnectionHandle):
        """
        Set up STP voltages and currents.
        Measure target values as median of drivers of a CapMem block.
        """

        builder = sta.PlaybackProgramBuilder()
        builder = helpers.capmem_set_quadrant_cells(builder, config={
            halco.CapMemCellOnCapMemBlock.stp_v_charge_0: self.v_stp,
            halco.CapMemCellOnCapMemBlock.stp_v_recover_0: self.v_stp,
            halco.CapMemCellOnCapMemBlock.stp_i_ramp: self.i_ramp,
            halco.CapMemCellOnCapMemBlock.stp_i_calib: self.i_ramp,
            halco.CapMemCellOnCapMemBlock.stp_i_bias_comparator: 1022
        })
        sta.run(connection, builder.done())

        super().prelude(connection)

        # Use median value of CapMem block as a calibration target
        results = self.measure_results(connection, builder)
        block_results = np.median(
            hagen_driver.STPRampCalibration.reshape_syndrv_amplitudes(results),
            axis=1)
        self.target = np.empty(halco.SynapseDriverOnDLS.size)
        for coord in halco.iter_all(halco.SynapseDriverOnDLS):
            self.target[int(coord.toEnum())] = \
                block_results[int(coord.toCapMemBlockOnDLS().toEnum())]