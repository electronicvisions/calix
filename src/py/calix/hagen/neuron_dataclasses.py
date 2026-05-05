"""
Dataclasses for hagen neuron calib target and result.
"""

from typing import Dict
from dataclasses import dataclass, field

import numpy as np

from dlens_vx_v3 import sta, halco, hal, lola

from pyccalix import NeuronCalibOptions, HagenNeuronCalibTarget \
    as NeuronCalibTarget
from calix.common import base, helpers
from calix.hagen import neuron_helpers
from calix import constants


@dataclass
class NeuronCalibResult(base.CalibResult):
    """
    Result object of a neuron calibration.
    Holds calibrated parameters for all neurons and their calibration success.
    """

    neurons: Dict[halco.AtomicNeuronOnDLS, lola.AtomicNeuron]
    cocos: Dict  # some coordinate, some container
    success: Dict[halco.AtomicNeuronOnDLS, bool]

    def apply(self, builder: base.WriteRecordingPlaybackProgramBuilder):
        """
        Apply the calibration in the given builder.

        Configures neurons in a "default-working" state with
        calibration applied, just like after the calibration.

        :param builder: Builder or dumper to append configuration
            instructions to.
        """

        for neuron_coord, neuron in self.neurons.items():
            builder.write(neuron_coord, neuron)

        for coord, container in self.cocos.items():
            builder.write(coord, container)

        builder = helpers.wait(builder, constants.capmem_level_off_time)

    @property
    def success_mask(self) -> np.ndarray:
        """
        Convert the success dict to a boolean numpy mask.

        :return: Numpy array containing neuron calibration success,
            ordered matching the AtomicNeuronOnDLS enum.
        """

        success_mask = np.empty(halco.AtomicNeuronOnDLS.size, dtype=bool)
        for coord in halco.iter_all(halco.AtomicNeuronOnDLS):
            success_mask[coord.toEnum()] = self.success[coord]

        return success_mask


@dataclass
class CalibResultInternal:
    """
    Class providing numpy-array access to calibrated parameters.
    Used internally during calibration.
    """

    v_leak: np.ndarray = field(
        default_factory=lambda: np.empty(
            halco.NeuronConfigOnDLS.size, dtype=int))
    v_reset: np.ndarray = field(
        default_factory=lambda: np.empty(
            halco.NeuronConfigOnDLS.size, dtype=int))
    i_syn_exc_shift: np.ndarray = field(
        default_factory=lambda: np.empty(
            halco.NeuronConfigOnDLS.size, dtype=int))
    i_syn_inh_shift: np.ndarray = field(
        default_factory=lambda: np.empty(
            halco.NeuronConfigOnDLS.size, dtype=int))
    i_bias_leak: np.ndarray = field(
        default_factory=lambda: np.empty(
            halco.NeuronConfigOnDLS.size, dtype=int))
    i_bias_reset: np.ndarray = field(
        default_factory=lambda: np.empty(
            halco.NeuronConfigOnDLS.size, dtype=int))
    i_syn_exc_gm: np.ndarray = field(
        default_factory=lambda: np.empty(
            halco.NeuronConfigOnDLS.size, dtype=int))
    i_syn_inh_gm: np.ndarray = field(
        default_factory=lambda: np.empty(
            halco.NeuronConfigOnDLS.size, dtype=int))
    i_syn_exc_tau: np.ndarray = field(
        default_factory=lambda: np.empty(
            halco.NeuronConfigOnDLS.size, dtype=int))
    i_syn_inh_tau: np.ndarray = field(
        default_factory=lambda: np.empty(
            halco.NeuronConfigOnDLS.size, dtype=int))
    syn_bias_dac: np.ndarray = field(
        default_factory=lambda: np.empty(
            halco.CapMemBlockOnDLS.size, dtype=int))
    success: np.ndarray = field(
        default_factory=lambda: np.ones(
            halco.NeuronConfigOnDLS.size, dtype=bool))
    use_synin_small_capacitance: bool = True

    def to_atomic_neuron(self,
                         neuron_coord: halco.AtomicNeuronOnDLS
                         ) -> lola.AtomicNeuron:
        """
        Returns an AtomicNeuron with calibration applied.

        :param neuron_coord: Coordinate of requested neuron.

        :return: Complete AtomicNeuron configuration.
        """

        neuron_id = neuron_coord.toEnum().value()
        atomic_neuron = lola.AtomicNeuron()
        atomic_neuron.set_from(neuron_helpers.neuron_config_default())
        atomic_neuron.set_from(neuron_helpers.neuron_backend_config_default())

        anl = atomic_neuron.leak
        anl.v_leak = hal.CapMemCell.Value(self.v_leak[neuron_id])
        anl.i_bias = hal.CapMemCell.Value(self.i_bias_leak[neuron_id])

        anr = atomic_neuron.reset
        anr.v_reset = hal.CapMemCell.Value(self.v_reset[neuron_id])
        anr.i_bias = hal.CapMemCell.Value(self.i_bias_reset[neuron_id])

        anexc = atomic_neuron.excitatory_input
        anexc.i_shift_reference = hal.CapMemCell.Value(
            self.i_syn_exc_shift[neuron_id])
        anexc.i_bias_gm = hal.CapMemCell.Value(
            self.i_syn_exc_gm[neuron_id])
        anexc.i_bias_tau = hal.CapMemCell.Value(
            self.i_syn_exc_tau[neuron_id])
        anexc.enable_small_capacitance = self.use_synin_small_capacitance

        aninh = atomic_neuron.inhibitory_input
        aninh.i_shift_reference = hal.CapMemCell.Value(
            self.i_syn_inh_shift[neuron_id])
        aninh.i_bias_gm = hal.CapMemCell.Value(
            self.i_syn_inh_gm[neuron_id])
        aninh.i_bias_tau = hal.CapMemCell.Value(
            self.i_syn_inh_tau[neuron_id])
        aninh.enable_small_capacitance = self.use_synin_small_capacitance

        return atomic_neuron

    def to_neuron_calib_result(self, target: NeuronCalibTarget,
                               options: NeuronCalibOptions
                               ) -> NeuronCalibResult:
        """
        Conversion to NeuronCalibResult.
        The numpy arrays get merged into lola AtomicNeurons.

        :param target: Target parameters for calibration.
        :param options: Further options for calibration.

        :return: Equivalent NeuronCalibResult.
        """

        result = NeuronCalibResult(
            target=target, options=options,
            neurons={}, cocos={}, success={})

        # set neuron configuration, including CapMem
        for neuron_coord in halco.iter_all(halco.AtomicNeuronOnDLS):
            result.neurons[neuron_coord] = self.to_atomic_neuron(neuron_coord)

            neuron_id = neuron_coord.toEnum().value()
            result.success[neuron_coord] = self.success[neuron_id]

        # set global CapMem parameters
        dumper = sta.PlaybackProgramBuilderDumper()
        dumper = neuron_helpers.configure_integration(dumper)
        dumper = neuron_helpers.set_global_capmem_config(dumper)

        # set synapse DAC bias current
        dumper = helpers.capmem_set_quadrant_cells(
            dumper,
            {halco.CapMemCellOnCapMemBlock.syn_i_bias_dac: self.syn_bias_dac})

        cocolist = dumper.done().tolist()

        for coord, config in cocolist:
            # remove Timer-commands like waits, we only want to collect
            # coord/container pairs (and would have many entries for the
            # timer coordinate otherwise)
            if coord == halco.TimerOnDLS():
                continue
            result.cocos[coord] = config

        return result
