#pragma once
#include "ccalix/calib_target.h"
#include "ccalix/cerealization.h"
#include "ccalix/types.h"
#include "halco/hicann-dls/vx/v3/neuron.h"
#include "halco/hicann-dls/vx/v3/synapse.h"
#include "haldls/vx/v3/cadc.h"
#include "hate/visibility.h"
#include "lola/vx/v3/neuron.h"
#include <optional>

namespace ccalix GENPYBIND_TAG_CCALIX {

/**
 * Target parameters for the neuron calibration.
 */
struct GENPYBIND(visible) NeuronCalibTarget : public CalibTarget
{
	typedef halco::common::typed_array<
	    haldls::vx::v3::CADCSampleQuad::Value,
	    halco::hicann_dls::vx::v3::AtomicNeuronOnDLS>
	    Potential GENPYBIND(opaque(false));

	/**
	 * Construct calib target parameters.
	 */
	NeuronCalibTarget();

	/**
	 * Target CADC read at leak (resting) potential.
	 */
	Potential leak;

	/**
	 * Target CADC read at reset potential.
	 */
	Potential reset;

	/**
	 * Target CADC read near spike threshold.
	 */
	Potential threshold;

	typedef halco::common::typed_array<TimeInS, halco::hicann_dls::vx::v3::AtomicNeuronOnDLS>
	    PerNeuronTimeConstant GENPYBIND(opaque(false));

	/**
	 * Membrane time constant.
	 */
	PerNeuronTimeConstant tau_mem;

	typedef halco::common::typed_array<TimeInS, halco::hicann_dls::vx::v3::SynapticInputOnNeuron>
	    TauSynPerNeuron GENPYBIND(opaque(false));
	typedef halco::common::
	    typed_array<TauSynPerNeuron, halco::hicann_dls::vx::v3::AtomicNeuronOnDLS>
	        TauSyn GENPYBIND(opaque(false));

	/**
	 * Synaptic input time constant.
	 */
	TauSyn tau_syn;

	/**
	 * The synaptic input strength is NOT calibrated, as this array would already be the result
	 * of calibration. Instead, the values are set up per neuron and used during the later
	 * parts, i.e.synaptic input reference calibration.
	 */
	struct UncalibratedCubaSynapticInput
	{
		typedef halco::common::typed_array<
		    haldls::vx::v3::CapMemCell::Value,
		    halco::hicann_dls::vx::v3::SynapticInputOnNeuron>
		    ISyninGMOnNeuron GENPYBIND(opaque(false));
		typedef halco::common::
		    typed_array<ISyninGMOnNeuron, halco::hicann_dls::vx::v3::AtomicNeuronOnDLS>
		        ISyninGM GENPYBIND(opaque(false));

		/**
		 * Synaptic input strength as CapMem bias current.
		 */
		ISyninGM i_synin_gm;

		UncalibratedCubaSynapticInput() SYMBOL_VISIBLE;

	private:
		friend class cereal::access;
		template <typename Archive>
		void serialize(Archive& ar, std::uint32_t);
	};

	/**
	 * Both excitatory and inhibitory synaptic inputs are calibrated to a target measured as the
	 * median of all neurons at this setting.
	 */
	struct CalibratedCubaSynapticInput
	{
		typedef halco::common::typed_array<
		    haldls::vx::v3::CapMemCell::Value,
		    halco::hicann_dls::vx::v3::SynapticInputOnNeuron>
		    ISyninGM GENPYBIND(opaque(false));

		/**
		 * Synaptic input strength as CapMem bias current.
		 */
		ISyninGM i_synin_gm;

		CalibratedCubaSynapticInput() SYMBOL_VISIBLE;

	private:
		friend class cereal::access;
		template <typename Archive>
		void serialize(Archive& ar, std::uint32_t);
	};

	/**
	 * Both excitatory and inhibitory synaptic inputs are calibrated to a target measured as the
	 * median of all neurons at this setting.
	 */
	struct CalibratedMatchingCubaSynapticInput
	{
		/**
		 * Synaptic input strength as CapMem bias current.
		 */
		haldls::vx::v3::CapMemCell::Value i_synin_gm;

		CalibratedMatchingCubaSynapticInput() SYMBOL_VISIBLE;

	private:
		friend class cereal::access;
		template <typename Archive>
		void serialize(Archive& ar, std::uint32_t);
	};

	std::variant<
	    UncalibratedCubaSynapticInput,
	    CalibratedCubaSynapticInput,
	    CalibratedMatchingCubaSynapticInput>
	    cuba_synin;

	/**
	 * Targets for the conductance-based synaptic input.
	 * Values provided superseed the current-based targets.
	 */
	struct CalibratedCobaSynapticInput
	{
		typedef halco::common::
		    typed_array<std::optional<double>, halco::hicann_dls::vx::v3::SynapticInputOnNeuron>
		        PotentialOnNeuron GENPYBIND(opaque(false));
		typedef halco::common::
		    typed_array<PotentialOnNeuron, halco::hicann_dls::vx::v3::AtomicNeuronOnDLS>
		        Potential GENPYBIND(opaque(false));

		/**
		 * COBA synaptic input reversal potential.
		 *
		 * At this potential, the synaptic input strength will be zero. The distance between COBA
		 * reversal and reference potential determines the strength of the amplitude modulation.
		 * Note that in biological context, the difference between reference and reversal potentials
		 * is a scaling factor for the conductance achieved by an input event. Optional: If None,
		 * the synaptic input will use CUBA mode for the respective neuron. Given in CADC units. The
		 * values may exceed the dynamic range of leak and CADC. In this case, the calibration is
		 * performed at a lower, linearly interpolated value.
		 */
		Potential e_coba_reversal;

		/**
		 * COBA synaptic input reference potential.
		 *
		 * At this potential, the original CUBA synaptic input strength, given via i_synin_gm, is
		 * not modified by COBA modulation. Optional: If None, the midpoint between leak and
		 * threshold will be used. Given in CADC units. The values must be reachable by the leak
		 * term, and the dynamic range of the CADC must allow for measurement of synaptic input
		 * amplitudes on top of this potential. We recommend choosing a value between the leak and
		 * threshold.
		 */
		Potential e_coba_reference;

		CalibratedCobaSynapticInput() SYMBOL_VISIBLE;

	private:
		friend class cereal::access;
		template <typename Archive>
		void serialize(Archive& ar, std::uint32_t);
	};

	CalibratedCobaSynapticInput coba_synin;

	typedef halco::common::typed_array<
	    lola::vx::v3::AtomicNeuron::MembraneCapacitance::CapacitorSize,
	    halco::hicann_dls::vx::v3::AtomicNeuronOnDLS>
	    MembraneCapacitance GENPYBIND(opaque(false));

	/**
	 * Selected membrane capacitance.
	 * The available range is 0 to approximately 2.2 pF, represented as 0 to 63 LSB.
	 */
	MembraneCapacitance membrane_capacitance;

	/**
	 * Refractory time in us.
	 */
	PerNeuronTimeConstant refractory_time;

	/**
	 * Target length of the holdoff period. The holdoff period is the time at the end of the
	 * refractory period in which the clamping to the reset voltage is already released but new
	 * spikes can still not be generated.
	 */
	PerNeuronTimeConstant holdoff_time;

	/**
	 * Synapse DAC bias current that is desired.
	 * Can be lowered in order to reduce the amplitude of a spike at the input of the synaptic input
	 * OTA. This can be useful to avoid saturation when using larger synaptic time constants.
	 */
	haldls::vx::v3::CapMemCell::Value synapse_dac_bias;

	/**
	 * Time constant of the inter-compartment conductance. If the value is None, the
	 * inter-compartment conductance is not calibrated.
	 */
	std::optional<PerNeuronTimeConstant> tau_icc;

private:
	friend class cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // namespace ccalix

CCALIX_EXTERN_INSTANTIATE_CEREAL_SERIALIZE(ccalix::NeuronCalibTarget)
