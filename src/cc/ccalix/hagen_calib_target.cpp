#include "ccalix/hagen_calib_target.h"

#include "ccalix/cerealization.tcc"

namespace ccalix {

HagenCalibTarget::HagenCalibTarget() : cadc_target(), neuron_target()
{
	cadc_target.dynamic_range.lower = CADCCalibTarget::DynamicRange::Value(150);
	cadc_target.dynamic_range.upper = CADCCalibTarget::DynamicRange::Value(500);
}

template <typename Archive>
void HagenCalibTarget::serialize(Archive& ar, std::uint32_t const)
{
	ar(CEREAL_NVP(cadc_target));
	ar(CEREAL_NVP(neuron_target));
}

} // namespace ccalix

EXPLICIT_INSTANTIATE_CEREAL_SERIALIZE(ccalix::HagenCalibTarget)
