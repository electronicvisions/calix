#include "ccalix/synapse_driver_calib_options.h"

#include "ccalix/cerealization.tcc"
#include "cereal/types/halco/common/geometry.h"

namespace ccalix {

template <typename Archive>
void SynapseDriverCalibOptions::serialize(Archive& ar, std::uint32_t const)
{
	ar(CEREAL_NVP(offset_test_activation));
}

std::ostream& operator<<(std::ostream& os, SynapseDriverCalibOptions const& options)
{
	return os << "SynapseDriverCalibOptions(" << options.offset_test_activation << ")";
}

} // namespace ccalix

EXPLICIT_INSTANTIATE_CEREAL_SERIALIZE(ccalix::SynapseDriverCalibOptions)
