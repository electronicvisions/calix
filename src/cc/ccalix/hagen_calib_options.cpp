#include "ccalix/hagen_calib_options.h"

#include "ccalix/cerealization.tcc"
#include <ostream>

namespace ccalix {

template <typename Archive>
void HagenCalibOptions::serialize(Archive& ar, std::uint32_t const)
{
	ar(CEREAL_NVP(cadc_options));
	ar(CEREAL_NVP(neuron_options));
	ar(CEREAL_NVP(neuron_disable_leakage));
	ar(CEREAL_NVP(synapse_driver_options));
}

std::ostream& operator<<(std::ostream& os, HagenCalibOptions const& options)
{
	return os << "HagenCalibOptions(" << options.cadc_options << ", " << options.neuron_options
	          << ", " << options.neuron_disable_leakage << ", " << options.synapse_driver_options
	          << ")";
}

} // namespace ccalix

EXPLICIT_INSTANTIATE_CEREAL_SERIALIZE(ccalix::HagenCalibOptions)
