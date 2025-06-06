"""
Dataclasses for hagen neuron calib target and result.
"""

from typing import Dict, Optional, Union
from dataclasses import dataclass, field

import numpy as np
import quantities as pq

from dlens_vx_v3 import sta, halco, hal, lola

from pyccalix import NeuronCalibOptions
from calix.common import base, helpers
from calix.hagen import neuron_helpers
from calix import constants


@dataclass
class NeuronCalibTarget(base.CalibTarget):
    """
    Target parameters for the neuron calibration.

    :ivar target_leak_read: Target CADC read at resting potential
        of the membrane. Due to the low leak bias currents, the spread
        of resting potentials may be high even after calibration.
    :ivar tau_mem: Targeted membrane time constant while calibrating the
        synaptic inputs.
        Too short values can not be achieved with this calibration routine.
        The default value of 60 us should work.
        If a target_noise is given (default), this setting does not affect
        the final leak bias currents, as those are determined by
        reaching the target noise.
    :ivar tau_syn: Controls the synaptic input time constant.
        If set to 0 us, the minimum synaptic input time constant will be
        used, which means different synaptic input time constants per
        neuron. If a single different Quantity is given, it is used for
        all synaptic inputs of all neurons, excitatory and inhibitory.
        If an array of Quantities is given, it can be shaped
        (2, 512) for the excitatory and inhibitory synaptic input
        of each neuron. It can also be shaped (2,) for the excitatory
        and inhibitory synaptic input of all neurons, or shaped (512,)
        for both inputs per neuron.
    :ivar i_synin_gm: Target synaptic input OTA bias current.
        The amplitudes of excitatory inputs using this target current are
        measured, and the median of all neurons' amplitudes is taken as target
        for calibration of the synaptic input strengths.
        The inhibitory synaptic input gets calibrated to match the excitatory.
        Some 300 LSB are proposed here. Choosing high values yields
        higher noise and lower time constants on the neurons, choosing
        low values yields less gain in a multiplication.
    :ivar target_noise: Noise amplitude in an integration process to
        aim for when searching the optimum leak OTA bias current,
        given as the standard deviation of successive reads in CADC LSB.
        Higher noise settings mean longer membrane time constants but
        impact reproducibility.
        Set target_noise to None to skip optimization of noise amplitudes
        entirely. In this case, the original membrane time constant
        calibration is used for leak bias currents.
    :ivar synapse_dac_bias: Synapse DAC bias current that is desired.
        Can be lowered in order to reduce the amplitude of a spike
        at the input of the synaptic input OTA. This can be useful
        to avoid saturation when using larger synaptic time constants.
    """

    target_leak_read: Union[int, np.ndarray] = 120
    tau_mem: pq.Quantity = field(
        default_factory=lambda: 60 * pq.us)
    tau_syn: pq.Quantity = field(
        default_factory=lambda: 0.32 * pq.us)
    i_synin_gm: int = 450
    target_noise: Optional[float] = None
    synapse_dac_bias: int = hal.CapMemCell.Value.max


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
