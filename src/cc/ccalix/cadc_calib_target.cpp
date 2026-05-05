#include "ccalix/cadc_calib_target.h"

#include "ccalix/cerealization.tcc"
#include "cereal/types/halco/common/geometry.h"

namespace ccalix {

CADCCalibTarget::CADCCalibTarget() :
    dynamic_range(DynamicRange(DynamicRange::Value(70), DynamicRange::Value(550))),
    read_range(ReadRange(ReadRange::Value(20), ReadRange::Value(220)))
{
}

template <typename Archive>
void CADCCalibTarget::serialize(Archive& ar, std::uint32_t const)
{
	ar(CEREAL_NVP(dynamic_range));
	ar(CEREAL_NVP(read_range));
}

} // namespace ccalix

EXPLICIT_INSTANTIATE_CEREAL_SERIALIZE(ccalix::CADCCalibTarget)
