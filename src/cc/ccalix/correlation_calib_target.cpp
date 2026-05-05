#include "ccalix/correlation_calib_target.h"

#include "ccalix/cerealization.tcc"
#include "cereal/types/halco/common/geometry.h"

namespace ccalix {

CorrelationCalibTarget::CorrelationCalibTarget() : amplitude(0.5), time_constant(TimeInS(5e-6)) {}

template <typename Archive>
void CorrelationCalibTarget::serialize(Archive& ar, std::uint32_t const)
{
	ar(CEREAL_NVP(amplitude));
	ar(CEREAL_NVP(time_constant));
}

} // namespace ccalix

EXPLICIT_INSTANTIATE_CEREAL_SERIALIZE(ccalix::CorrelationCalibTarget)
