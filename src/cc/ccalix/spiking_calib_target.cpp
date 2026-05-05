#include "ccalix/spiking_calib_target.h"

#include "ccalix/cerealization.tcc"
#include <cereal/types/optional.hpp>

namespace ccalix {

template <typename Archive>
void SpikingCalibTarget::serialize(Archive& ar, std::uint32_t const)
{
	ar(CEREAL_NVP(cadc_target));
	ar(CEREAL_NVP(neuron_target));
	ar(CEREAL_NVP(correlation_target));
	ar(CEREAL_NVP(stp_target));
}

} // namespace ccalix

EXPLICIT_INSTANTIATE_CEREAL_SERIALIZE(ccalix::SpikingCalibTarget)
