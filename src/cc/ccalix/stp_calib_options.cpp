#include "ccalix/stp_calib_options.h"

#include "ccalix/cerealization.tcc"
#include "cereal/types/halco/common/geometry.h"
#include "cereal/types/halco/common/typed_array.h"
#include "hate/join.h"
#include <ostream>

namespace ccalix {

STPCalibOptions::STPCalibOptions() : i_ramp(), v_stp()
{
	i_ramp.fill(ValuePerQuadrant::value_type(600));
	v_stp.fill(ValuePerQuadrant::value_type(180));
}

template <typename Archive>
void STPCalibOptions::serialize(Archive& ar, std::uint32_t const)
{
	ar(CEREAL_NVP(i_ramp));
	ar(CEREAL_NVP(v_stp));
}

std::ostream& operator<<(std::ostream& os, STPCalibOptions const& options)
{
	return os << "STPCalibOptions([" << hate::join(options.i_ramp, ", ") << "], ["
	          << hate::join(options.v_stp, ", ") << "])";
}

} // namespace ccalix

EXPLICIT_INSTANTIATE_CEREAL_SERIALIZE(ccalix::STPCalibOptions)
