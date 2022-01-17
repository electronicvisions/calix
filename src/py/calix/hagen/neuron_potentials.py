"""
Provides functions to calibrate leak and reset potential.
"""

from typing import Union, List, Optional
import numpy as np
import quantities as pq
from dlens_vx_v2 import hal, sta, halco, logger, hxcomm

from calix.common import base, cadc, cadc_helpers, helpers
from calix.hagen import neuron_helpers
from calix import constants


class LeakPotentialCalibration(base.Calibration):
    """
    Calibrate the neurons' leak potentials to match specified CADC reads.

    The leak potential parameter is varied until all CADC channels match the
    median of the original reads or the provided target values.

    Requirements:
    * Neuron membrane readout is connected to the CADCs (causal and acausal).
    * Leak bias current is set not too low, such that the resting potential
      is affected by the leak potential. Note that this calibration has
      to be re-run after the leak bias current was changed, due to the
      OTA's characteristics.
    """

    def __init__(self, target: Union[int, np.ndarray] = None):
        super().__init__(
            parameter_range=base.ParameterRange(
                hal.CapMemCell.Value.min, hal.CapMemCell.Value.max),
            n_instances=halco.NeuronConfigOnDLS.size,
            inverted=False,
            errors=["Leak potential for neurons {0} has reached {1}. "
                    + "Please choose a higher target read or exclude "
                    + "them during usage.",
                    "Leak potential for neurons {0} has reached {1}. "
                    + "Please choose a lower target read or exclude "
                    + "them during usage."])
        self.target = target

    def prelude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        If no calibration target was provided during initialization,
        the current median resting potential is measured and used as a target.

        :param connection: Connection to the chip to run on.
        """

        if self.target is not None:
            return

        builder = sta.PlaybackProgramBuilder()
        self.target = int(np.median(
            self.measure_results(connection, builder)))
        logger.get(
            "calix.hagen.neuron_potentials"
            + ".LeakPotentialCalibration.prelude"
        ).DEBUG(f"Read target for v_leak calibration: {self.target}")

    def configure_parameters(self, builder: sta.PlaybackProgramBuilder,
                             parameters: np.ndarray
                             ) -> sta.PlaybackProgramBuilder:
        """
        Configure the given parameters (leak potential settings) in the given
        builder.

        :param builder: Builder to append configuration instructions to.
        :param parameters: v_leak setting for each neuron.

        :return: Builder with configuration appended.
        """

        builder = helpers.capmem_set_neuron_cells(
            builder,
            {halco.CapMemRowOnCapMemBlock.v_leak: parameters})

        builder = helpers.wait(builder, constants.capmem_level_off_time)
        return builder

    def measure_results(self, connection: hxcomm.ConnectionHandle,
                        builder: sta.PlaybackProgramBuilder
                        ) -> np.ndarray:
        """
        Measures the membrane potentials of all neurons.

        We use the CADCs to take only one measurement of the resting
        potential and assume this is an accurate representation
        of the leak potential.

        :param connection: Connection to a chip.
        :param builder: Builder to append read instructions to.

        :return: Array containing membrane potential CADC reads.
        """

        results = neuron_helpers.cadc_read_neuron_potentials(
            connection, builder)
        return results

    def postlude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Print statistics of the neurons' resting potentials.

        :Param connection: Connection to the chip to run on.
        """

        builder = sta.PlaybackProgramBuilder()
        reads = self.measure_results(connection, builder)
        logger.get(
            "calix.hagen.neuron_potentials"
            + ".LeakPotentialCalibration.postlude"
        ).INFO("Calibrated v_leak, CADC statistics: "
               + "{0:5.2f} +- {1:4.2f}".format(
                   np.mean(reads[self.result.success]),
                   np.std(reads[self.result.success])))


class ResetPotentialCalibration(base.Calibration):
    """
    Calibrate the neurons' reset target voltage such that it matches
    the leak potential.

    The refractory time is extended in order to measure the membrane
    potentials while the neuron is at reset potential. The CapMem
    settings which yield CADC reads which match the potential without reset
    are searched.

    Requirements:
    * Neuron membrane readout is connected to the CADCs (causal and acausal).
    * Reset bias current is set not too low, such that a resting potential
      is reached during the refractory period. Note that this calibration
      has to be re-run after the reset bias current was changed, due to
      the OTA's characteristics.
    * Target CADC read at reset potential is given, or membrane potential
      is not floating (i.e. the leak bias current is not too low). You
      may use the highnoise flag to indicate multiple measurements of the
      resting potential shall be taken in order to find the target
      reset potential.

    :ivar highnoise: Decides whether to expect high noise on the
        membrane potential to be present. Setting this to True results
        in multiple reads of the resting potential when finding the
        calibration target value.
    :ivar backend_configs: List containing neuron backend configs
        as read in the prelude of the calibration. Used to restore
        the original config in the postlude.
    :ivar common_backend_configs: List containing common neuron
        backend configs as read during the prelude. Used to restore
        the original config in the postlude.
    """

    def __init__(self, target: Union[int, np.ndarray] = None,
                 highnoise: bool = False):
        """
        Initialize the ResetPotentialCalibration class.

        :param target: Target CADC reads during reset. If not given,
            they are measured during prelude. If given, the highnoise
            option has no effect.
        """

        super().__init__(
            parameter_range=base.ParameterRange(
                hal.CapMemCell.Value.min, hal.CapMemCell.Value.max),
            n_instances=halco.NeuronConfigOnDLS.size,
            inverted=False,
            errors=["Reset potential for neurons {0} has reached {1}"] * 2)
        self.target = target
        self.highnoise = highnoise
        self.backend_configs: Optional[List[hal.NeuronBackendConfig]] = None
        self.common_backend_configs: Optional[List[
            hal.CommonNeuronBackendConfig]] = None

    def prelude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Measure a calibration target (the leak reads) if not provided.

        Select the slower clock in the digital neuron backend and
        set the refractory counter values in each neuron to keep the
        neuron refractory long enough so that the CADCs can
        read the potentials.
        The original neuron backend config is stored to be restored.

        :param connection: Connection to the chip to run on.
        """

        # Read the previous neuron backend configs
        builder = sta.PlaybackProgramBuilder()
        backend_tickets = list()
        for coord in halco.iter_all(halco.NeuronBackendConfigOnDLS):
            backend_tickets.append(builder.read(coord))

        common_backend_tickets = list()
        for coord in halco.iter_all(halco.CommonNeuronBackendConfigOnDLS):
            common_backend_tickets.append(builder.read(coord))

        # Reconfigure neuron backend
        config = hal.CommonNeuronBackendConfig()
        config.clock_scale_fast = 9  # slow clock to measure reset potential
        config.clock_scale_slow = 9
        for coord in halco.iter_all(halco.CommonNeuronBackendConfigOnDLS):
            builder.write(coord, config)

        config = hal.NeuronBackendConfig()
        config.refractory_time = hal.NeuronBackendConfig.RefractoryTime.max
        for coord in halco.iter_all(halco.NeuronBackendConfigOnDLS):
            builder.write(coord, config)

        # Obtain target for calibration
        if self.target is not None:
            base.run(connection, builder)
        elif not self.highnoise:
            self.target = neuron_helpers.cadc_read_neuron_potentials(
                connection, builder)
        else:
            self.target = np.empty(halco.NeuronConfigOnDLS.size, dtype=int)
            for synram in halco.iter_all(halco.SynramOnDLS):
                neuron_reads = \
                    neuron_helpers.cadc_read_neurons_repetitive(
                        connection, builder, synram=synram,
                        n_reads=200, wait_time=5000 * pq.us, reset=False)
                self.target[
                    halco.SynapseOnSynapseRow.size
                    * int(synram.toEnum()):halco.SynapseOnSynapseRow.size
                    * (int(synram.toEnum()) + 1)] = \
                    np.mean(neuron_reads, axis=0)

        # extract the previous neuron backend configs from tickets
        self.backend_configs = list()
        for ticket in backend_tickets:
            self.backend_configs.append(ticket.get())

        self.common_backend_configs = list()
        for ticket in common_backend_tickets:
            self.common_backend_configs.append(ticket.get())

    def configure_parameters(self, builder: sta.PlaybackProgramBuilder,
                             parameters: np.ndarray
                             ) -> sta.PlaybackProgramBuilder:
        """
        Configure the reset potentials of all neurons to the
        values given in the parameter array.

        :param builder: Builder to append configuration to.
        :param parameters: CapMem values of the reset potential.

        :return: Builder with configuration appended.
        """

        builder = helpers.capmem_set_neuron_cells(
            builder,
            {halco.CapMemRowOnCapMemBlock.v_reset: parameters})

        builder = helpers.wait(builder, constants.capmem_level_off_time)
        return builder

    def measure_results(self, connection: hxcomm.ConnectionHandle,
                        builder: sta.PlaybackProgramBuilder) -> np.ndarray:
        """
        Test the configured settings. The neurons are artificially reset and
        their membrane potentials are read using the CADCs.

        :param connection: Connection to a chip to run on.
        :param builder: Builder to append read instructions.

        :return: Membrane potentials of neurons during reset.
        """

        tickets = list()

        for synram in halco.iter_all(halco.SynramOnDLS):
            # trigger neuron resets
            builder = neuron_helpers.reset_neurons(builder, synram)
            builder = helpers.wait(builder, 10 * pq.us)

            # read membrane potentials
            builder, ticket = cadc_helpers.cadc_read_row(builder, synram)
            tickets.append(ticket)

        base.run(connection, builder)

        # Inspect read tickets
        results = neuron_helpers.inspect_read_tickets(tickets).flatten()

        return results

    def postlude(self, connection: hxcomm.ConnectionHandle) -> None:
        """
        Log statistics of the results.

        Restore original configuration of the neuron backends.

        :param connection: Connection to the chip to run on.
        """

        # Print results
        results = self.measure_results(
            connection, builder=sta.PlaybackProgramBuilder())

        logger.get(
            "calix.hagen.neuron_potentials"
            + ".ResetPotentialCalibration.postlude"
        ).INFO("Calibrated v_reset, CADC statistics: "
               + "{0:5.2f} +- {1:4.2f}".format(
                   np.mean(results[self.result.success]),
                   np.std(results[self.result.success])))

        # Set up original neuron backend config again
        builder = sta.PlaybackProgramBuilder()
        for coord, config in zip(halco.iter_all(
                halco.NeuronBackendConfigOnDLS), self.backend_configs):
            builder.write(coord, config)

        for coord, config in zip(
                halco.iter_all(halco.CommonNeuronBackendConfigOnDLS),
                self.common_backend_configs):
            builder.write(coord, config)
        base.run(connection, builder)


class BaselineCalibration(cadc.ChannelOffsetCalibration):
    """
    Calibrate all CADC channels offsets such that the neurons read a given
    value shortly after reset.

    Requirements:
    * Neuron membrane readout is connected to the CADCs (causal and acausal).
    * The potential shortly after a neuron reset is stable and does not
      quickly drift away. Also, it has to be in a suitable range for the
      CADCs, ideally near the middle of their dynamic range.
    """

    def __init__(self):
        # initialize CADC calib, value of reference voltage
        # `dynamic_range_mid` irrelevant for further calibration.
        super().__init__(dynamic_range_mid=0)

        # use center of CADC range as default target
        self.target = hal.CADCSampleQuad.Value.size // 2

    def prelude(self, connection: hxcomm.ConnectionHandle):
        pass

    def postlude(self, connection: hxcomm.ConnectionHandle):
        log = logger.get("calix.hagen.neuron_potentials."
                         + "BaselineCalibration.postlude")
        results = neuron_helpers.cadc_read_neuron_potentials(connection)
        log.INFO("Calibrated neuron baseline, CADC reads:",
                 f"{np.mean(results):5.2f} +- {np.std(results):3.2f}")

    def measure_results(self, connection: hxcomm.ConnectionHandle,
                        builder: sta.PlaybackProgramBuilder
                        ) -> np.ndarray:
        """
        Read all CADC channels at the neuron's baseline voltage.
        This read is executed multiple times to get a more accurate result.

        :param connection: Connection to the chip to run on.
        :param builder: Builder to append read instructions to.

        :return: Array of read values for both synrams.
        """

        n_reads = 20
        results = np.empty((n_reads, halco.CADCChannelConfigOnDLS.size))

        for run in range(n_reads):
            builder = neuron_helpers.reset_neurons(builder)
            builder = helpers.wait(builder, 30 * pq.us)
            results[run] = cadc_helpers.read_cadcs(connection, builder)
            builder = sta.PlaybackProgramBuilder()

        return results.mean(axis=0)
