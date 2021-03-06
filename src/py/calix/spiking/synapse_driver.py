"""
Provides a calibration class for the synapse driver STP offset
for the usual spiking mode.
"""

from typing import Union
import numpy as np
from dlens_vx_v3 import hal, halco, sta, hxcomm, logger

from calix import constants
from calix.common import base, helpers
import calix.hagen.synapse_driver as hagen_driver
from calix.hagen import multiplication


class STPMultiplication(multiplication.Multiplication):
    """
    Perform vector-matrix multiplications, but without hagen-mode
    pulse length encoding. The pulse length is instead determined
    by the STP voltage.

    Note that we set a static voltage for both v_recover and v_charge
    and this way leave no dynamic range for modulation of amplitudes
    depending on their timing. We only set a medium amplitude with
    a suitable voltage v_stp := v_recover = v_charge.
    """

    def preconfigure(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Call preconfigure from parent class, but disable
        hagen-mode encoding (use STP voltages instead).

        :param connection: Connection to the chip to run on.
        """

        super().preconfigure(connection)

        # read synapse driver config
        syndrv_tickets = list()
        builder = sta.PlaybackProgramBuilder()
        for coord in halco.iter_all(halco.SynapseDriverOnSynapseDriverBlock):
            syndrv_tickets.append(builder.read(
                halco.SynapseDriverOnDLS(
                    coord,
                    block=self.synram_coord.toSynapseDriverBlockOnDLS())))
        base.run(connection, builder)

        # write synapse driver config with hagen encoding disabled
        builder = sta.PlaybackProgramBuilder()
        for coord in halco.iter_all(halco.SynapseDriverOnSynapseDriverBlock):
            config = syndrv_tickets[coord.toEnum()].get()
            config.enable_stp = True
            config.enable_hagen_modulation = False
            config.enable_hagen_dac = False

            builder.write(
                halco.SynapseDriverOnDLS(
                    coord,
                    block=self.synram_coord.toSynapseDriverBlockOnDLS()),
                config)
        base.run(connection, builder)


class STPOffsetCalibration(base.Calibration):
    """
    Calibrate synapse driver STP ramp offsets for usage in
    spiking mode.

    Since the hagen-mode DAC is not involved now, the STP voltages can
    use a higher dynamic range determined by v_charge and v_recover.
    This usually means a different STP ramp current is used than in
    hagen mode.

    Drivers connected to different CapMem instances are calibrated to
    different targets since deviations in STP voltages and ramp currents
    (which are set for each CapMem instance individually) have higher
    variations than can be compensated with the STP offset.

    Note that this calibration measures synapse drivers' outputs on the
    synaptic input lines, not on the neurons. The neurons are reconfigured
    for this purpose. If they were calibrated, you will need to re-apply
    their calibration after the STP offsets are calibrated.

    Requirements:
    * CADCs and Synapse DAC biases are calibrated.

    :ivar v_stp: STP voltage to use during amplitude measurement,
        reached by setting v_charge and v_recover to this value.
        If an array of values is given, they are configured per quadrant.
    :ivar i_ramp: STP ramp current.
        If an array of values is given, they are configured per quadrant.
    :ivar measurement: Instance of measurement class, in order to measure
        the characteristic output amplitudes of each synapse driver.
    """

    def __init__(self, v_stp: Union[int, np.ndarray] = 180,
                 i_ramp: Union[int, np.ndarray] = 600):
        super().__init__(
            parameter_range=base.ParameterRange(
                0, hal.SynapseDriverConfig.Offset.max),
            n_instances=halco.SynapseDriverOnDLS.size,
            inverted=True)

        self.v_stp = v_stp
        self.i_ramp = i_ramp
        self.measurement = hagen_driver.SynapseDriverMeasurement()
        self.measurement.multiplication = STPMultiplication(signed_mode=False)

    def configure_parameters(self, builder: sta.PlaybackProgramBuilder,
                             parameters: np.ndarray
                             ) -> sta.PlaybackProgramBuilder:
        """
        Configure the synapse drivers to the given offsets.

        :param builder: Builder to append configuration to.
        :param parameters: Synapse driver offset settings to configure.

        :return: Builder with configuration instructions appended.
        """

        for coord in halco.iter_all(halco.SynapseDriverOnDLS):
            synapse_driver_config = hal.SynapseDriverConfig()
            synapse_driver_config.offset = int(
                parameters[int(coord.toEnum())])

            builder.write(coord, synapse_driver_config)

        return builder

    def measure_results(self, connection: hxcomm.ConnectionHandle,
                        builder: sta.PlaybackProgramBuilder
                        ) -> np.ndarray:
        """
        Read output amplitudes of synapse drivers.

        :param connection: Connection to the chip to run on.
        :param builder: Builder that is run before measuring.

        :return: Array of synapse drivers' output amplitudes.
        """

        base.run(connection, builder)
        results = self.measurement.measure_syndrv_amplitudes(
            connection,  # use any non-zero activation:
            activations=hal.PADIEvent.HagenActivation.max)

        return results

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
        builder = helpers.wait(builder, constants.capmem_level_off_time)

        builder = self.configure_parameters(
            builder, np.ones(halco.SynapseDriverOnDLS.size, dtype=int)
            * (hal.SynapseDriverConfig.Offset.max // 2))

        # Use median value of CapMem block as a calibration target
        results = self.measure_results(connection, builder)

        block_results = [
            list() for _ in halco.iter_all(halco.CapMemBlockOnDLS)]
        for coord, result in zip(
                halco.iter_all(halco.SynapseDriverOnDLS), results):
            block_results[int(coord.toCapMemBlockOnDLS().toEnum())].append(
                result)
        block_results = np.median(block_results, axis=1)

        self.target = np.empty(halco.SynapseDriverOnDLS.size)
        for coord in halco.iter_all(halco.SynapseDriverOnDLS):
            self.target[int(coord.toEnum())] = \
                block_results[int(coord.toCapMemBlockOnDLS().toEnum())]

        logger.get(
            "calix.spiking.synapse_driver.STPOffsetCalibration.prelude"
        ).DEBUG(
            "Deviation of synapse driver amplitudes before offset calib: "
            + f"{np.std(results):4.2f}")

    def postlude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Log standard deviation of amplitudes after calibration.

        :param connection: Connection to the chip to run on.
        """

        builder = sta.PlaybackProgramBuilder()
        results = self.measure_results(connection, builder)
        logger.get(
            "calix.spiking.synapse_driver.STPOffsetCalibration.postlude"
        ).INFO(
            "Deviation of synapse driver amplitudes after offset calib: "
            + f"{np.std(results):4.2f}")
