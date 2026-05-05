#pragma once
#include "ccalix/calib_target.h"
#include "ccalix/cerealization.h"
#include "ccalix/neuron_calib_target.h"
#include "ccalix/types.h"
#include "halco/common/typed_array.h"
#include "halco/hicann-dls/vx/v3/neuron.h"
#include "haldls/vx/v3/cadc.h"
#include "haldls/vx/v3/capmem.h"
#include "hate/visibility.h"

namespace ccalix GENPYBIND_TAG_CCALIX {

/**
 * Target parameters for the neuron calibration.
 */
struct GENPYBIND(visible) HagenNeuronCalibTarget : public CalibTarget
{
	typedef halco::common::typed_array<
	    haldls::vx::v3::CADCSampleQuad::Value,
	    halco::hicann_dls::vx::v3::AtomicNeuronOnDLS>
	    Potential GENPYBIND(opaque(false));

	/**
	 * Construct calib target parameters.
	 */
	HagenNeuronCalibTarget();

	/**
	 * Target CADC read at resting potential of the membrane. Due to the low leak bias currents, the
	 * spread of resting potentials may be high even after calibration.
	 */
	Potential leak;

	typedef halco::common::typed_array<TimeInS, halco::hicann_dls::vx::v3::AtomicNeuronOnDLS>
	    PerNeuronTimeConstant GENPYBIND(opaque(false));

	/**
	 * Targeted membrane time constant while calibrating the synaptic inputs.
	 * Too short values can not be achieved with this calibration routine. The default value of 60
	 * us should work. If a target_noise is given (default), this setting does not affect the final
	 * leak bias currents, as those are determined by reaching the target noise.
	 */
	PerNeuronTimeConstant tau_mem;

	typedef halco::common::typed_array<TimeInS, halco::hicann_dls::vx::v3::SynapticInputOnNeuron>
	    TauSynOnNeuron GENPYBIND(opaque(false));
	typedef halco::common::typed_array<TauSynOnNeuron, halco::hicann_dls::vx::v3::AtomicNeuronOnDLS>
	    TauSyn GENPYBIND(opaque(false));

	/**
	 * Controls the synaptic input time constant. If set to 0 us, the minimum synaptic input time
	 * constant will be used, which means different synaptic input time constants per neuron.
	 */
	TauSyn tau_syn;

	/**
	 * Target synaptic input OTA bias current.
	 * The amplitudes of excitatory inputs using this target current are measured, and the median of
	 * all neurons' amplitudes is taken as target for calibration of the synaptic input strengths.
	 * The inhibitory synaptic input gets calibrated to match the excitatory. Some 300 LSB are
	 * proposed here. Choosing high values yields higher noise and lower time constants on the
	 * neurons, choosing low values yields less gain in a multiplication.
	 */
	haldls::vx::v3::CapMemCell::Value i_synin_gm;

	/**
	 * Noise amplitude in an integration process to aim for when searching the optimum leak OTA bias
	 * current, given as the standard deviation of successive reads in CADC LSB. Higher noise
	 * settings mean longer membrane time constants but impact reproducibility. Set target_noise to
	 * None to skip optimization of noise amplitudes entirely. In this case, the original membrane
	 * time constant calibration is used for leak bias currents.
	 */
	std::optional<double> target_noise;

	/**
	 * Synapse DAC bias current that is desired. Can be lowered in order to reduce the amplitude of
	 * a spike at the input of the synaptic input OTA. This can be useful to avoid saturation when
	 * using larger synaptic time constants.
	 */
	haldls::vx::v3::CapMemCell::Value synapse_dac_bias;

private:
	friend class cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // namespace ccalix

CCALIX_EXTERN_INSTANTIATE_CEREAL_SERIALIZE(ccalix::HagenNeuronCalibTarget)
