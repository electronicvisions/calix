#include "ccalix/hagen_neuron_calib_target.h"

#include "ccalix/cerealization.tcc"
#include "cereal/types/halco/common/geometry.h"
#include "cereal/types/halco/common/typed_array.h"
#include <limits>
#include <cereal/types/optional.hpp>


namespace ccalix {

HagenNeuronCalibTarget::HagenNeuronCalibTarget() :
    leak([]() {
	    Potential v;
	    v.fill(Potential::value_type(120));
	    return v;
    }()),
    tau_mem([]() {
	    PerNeuronTimeConstant v;
	    v.fill(PerNeuronTimeConstant::value_type(60e-6));
	    return v;
    }()),
    tau_syn([]() {
	    TauSyn v;
	    for (auto& an : v) {
		    an.fill(TauSyn::value_type::value_type(0.32e-6));
	    }
	    return v;
    }()),
    i_synin_gm(450),
    target_noise(std::nullopt),
    synapse_dac_bias(haldls::vx::v3::CapMemCell::Value::max)
{
}

template <typename Archive>
void HagenNeuronCalibTarget::serialize(Archive& ar, std::uint32_t const)
{
	ar(CEREAL_NVP(leak));
	ar(CEREAL_NVP(tau_mem));
	ar(CEREAL_NVP(tau_syn));
	ar(CEREAL_NVP(i_synin_gm));
	ar(CEREAL_NVP(target_noise));
	ar(CEREAL_NVP(synapse_dac_bias));
}

} // namespace ccalix

EXPLICIT_INSTANTIATE_CEREAL_SERIALIZE(ccalix::HagenNeuronCalibTarget)
