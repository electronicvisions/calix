"""
Provides a calibration class for the synapse driver STP offset
for the usual spiking mode.
"""

from dataclasses import dataclass
from typing import Union, Optional

import numpy as np

from dlens_vx_v3 import hal, halco, sta, hxcomm, logger

from pyccalix import STPCalibOptions
from calix import constants
from calix.common import algorithms, base, helpers
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
        syndrv_tickets = []
        builder = base.WriteRecordingPlaybackProgramBuilder()
        for coord in halco.iter_all(halco.SynapseDriverOnSynapseDriverBlock):
            syndrv_tickets.append(builder.read(
                halco.SynapseDriverOnDLS(
                    coord,
                    block=self.synram_coord.toSynapseDriverBlockOnDLS())))
        base.run(connection, builder)

        # write synapse driver config with hagen encoding disabled
        builder = base.WriteRecordingPlaybackProgramBuilder()
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


class STPOffsetCalib(base.Calib):
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

    def configure_parameters(
            self, builder: base.WriteRecordingPlaybackProgramBuilder,
            parameters: np.ndarray) \
            -> base.WriteRecordingPlaybackProgramBuilder:
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
                        builder: base.WriteRecordingPlaybackProgramBuilder
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

        builder = base.WriteRecordingPlaybackProgramBuilder()
        builder = helpers.capmem_set_quadrant_cells(builder, config={
            halco.CapMemCellOnCapMemBlock.stp_v_charge_0: self.v_stp,
            halco.CapMemCellOnCapMemBlock.stp_v_recover_0: self.v_stp,
            halco.CapMemCellOnCapMemBlock.stp_i_ramp: self.i_ramp,
            halco.CapMemCellOnCapMemBlock.stp_i_calib: self.i_ramp,
            halco.CapMemCellOnCapMemBlock.stp_i_bias_comparator:
            hal.CapMemCell.Value.max
        })
        builder = helpers.wait(builder, constants.capmem_level_off_time)

        builder = self.configure_parameters(
            builder, np.ones(halco.SynapseDriverOnDLS.size, dtype=int)
            * (hal.SynapseDriverConfig.Offset.max // 2))

        # Use median value of CapMem block as a calibration target
        results = self.measure_results(connection, builder)

        block_results = [
            [] for _ in halco.iter_all(halco.CapMemBlockOnDLS)]
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
            "calix.spiking.synapse_driver.STPOffsetCalib.prelude"
        ).DEBUG(
            "Deviation of synapse driver amplitudes before offset calib: "
            + f"{np.std(results):4.2f}")

    def postlude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Log standard deviation of amplitudes after calibration.

        :param connection: Connection to the chip to run on.
        """

        builder = base.WriteRecordingPlaybackProgramBuilder()
        results = self.measure_results(connection, builder)
        logger.get(
            "calix.spiking.synapse_driver.STPOffsetCalib.postlude"
        ).INFO(
            "Deviation of synapse driver amplitudes after offset calib: "
            + f"{np.std(results):4.2f}")


@dataclass
class STPCalibTarget(base.CalibTarget):
    """
    Target for STP calibration.

    The STP voltages, which affect the dynamic range of the amplitudes,
    are set here. They are currently not calibrated, but can be set per
    quadrant.

    :param v_charge_0: STP v_charge (fully modulated state) for
        voltage set 0, in CapMem LSB. You can choose the two voltage
        sets to be, e.g., depressing and facilitating. By default, we
        select voltage set 0 to be facilitating and voltage set 1 to
        be depressing.
    :param v_recover_0: STP v_recover (fully recovered state) for
        voltage set 0, in CapMem LSB. Note that a utilization of some
        0.2 happens before processing each event, so the voltage applied
        to the comparator never actually reaches v_recover.
    :param v_charge_1: STP v_charge (fully modulated state) for
        voltage set 1, in CapMem LSB.
    :param v_recover_1: STP v_recover (fully recovered state) for
        voltage set 1, in CapMem LSB.
    """

    v_charge_0 = 100
    v_recover_0 = 400
    v_charge_1 = 330
    v_recover_1 = 50


@dataclass
class STPCalibResult(base.CalibResult):
    """
    Result from STP calibration.
    """

    offsets: np.ndarray

    def apply(self, builder: sta.PlaybackProgramBuilder):
        """
        Apply the result in the given builder.

        Note that in the synapse driver config, no outputs are generated,
        and STP is disabled. Only the offset is configured by this calib.
        So STP is ready to be used, but disabled.

        :param builder: Builder to append instructions to.
        """

        config = {
            halco.CapMemCellOnCapMemBlock.stp_i_ramp: self.options.i_ramp,
            halco.CapMemCellOnCapMemBlock.stp_i_calib: self.options.i_ramp,
            halco.CapMemCellOnCapMemBlock.stp_v_charge_0:
            self.target.v_charge_0,
            halco.CapMemCellOnCapMemBlock.stp_v_charge_1:
            self.target.v_charge_1,
            halco.CapMemCellOnCapMemBlock.stp_v_recover_0:
            self.target.v_recover_0,
            halco.CapMemCellOnCapMemBlock.stp_v_recover_1:
            self.target.v_recover_1,
            halco.CapMemCellOnCapMemBlock.stp_i_bias_comparator:
            hal.CapMemCell.Value.max,
        }

        helpers.capmem_set_quadrant_cells(builder, config)
        helpers.wait(builder, constants.capmem_level_off_time)

        for coord in halco.iter_all(halco.SynapseDriverOnDLS):
            config = hal.SynapseDriverConfig()
            config.offset = hal.SynapseDriverConfig.Offset(
                self.offsets[int(coord.toEnum())])
            builder.write(coord, config)


def calibrate(connection: hxcomm.ConnectionHandle, *,
              target: Optional[STPCalibTarget] = None,
              options: Optional[STPCalibOptions] = None,
              ) -> STPCalibResult:
    """
    Calibrate all synapse drivers' STP offsets.

    The STP ramp current is set as provided in the options.
    The other STP voltages, as given in the options, are configured
    after calibration.

    Requirements:
    * CADCs and Synapse DAC biases are calibrated.

    :param connection: Connection to the chip to calibrate.
    :param target: Target parameters for calibration.
    :param options: Further options for calibration, including STP
        voltages.

    :return: STP calib result.
    """

    if target is None:
        target = STPCalibTarget()
    if options is None:
        options = STPCalibOptions()

    calib = STPOffsetCalib(i_ramp=options.i_ramp, v_stp=options.v_stp)
    calib_result = calib.run(connection, algorithm=algorithms.BinarySearch())

    result = STPCalibResult(
        target=target, options=options,
        offsets=calib_result.calibrated_parameters)

    builder = base.WriteRecordingPlaybackProgramBuilder()
    result.apply(builder)
    base.run(connection, builder)

    return result
