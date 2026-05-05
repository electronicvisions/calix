#include "ccalix/hagen_synin_calib_target.h"

#include "ccalix/cerealization.tcc"
#include "cereal/types/halco/common/geometry.h"
#include "cereal/types/halco/common/typed_array.h"
#include <limits>

namespace ccalix {

HagenSyninCalibTarget::HagenSyninCalibTarget() : cadc_target(), synapse_dac_bias(800)
{
	cadc_target.dynamic_range.lower = CADCCalibTarget::DynamicRange::Value(150);
	cadc_target.dynamic_range.upper = CADCCalibTarget::DynamicRange::Value(340);
}

template <typename Archive>
void HagenSyninCalibTarget::serialize(Archive& ar, std::uint32_t const)
{
	ar(CEREAL_NVP(cadc_target));
	ar(CEREAL_NVP(synapse_dac_bias));
}

} // namespace ccalix

EXPLICIT_INSTANTIATE_CEREAL_SERIALIZE(ccalix::HagenSyninCalibTarget)
