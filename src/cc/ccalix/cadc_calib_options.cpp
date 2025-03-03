#include "ccalix/cadc_calib_options.h"

#include "ccalix/cerealization.tcc"
#include <ostream>

namespace ccalix {

template <typename Archive>
void CADCCalibOptions::serialize(Archive& ar, std::uint32_t const)
{
	ar(CEREAL_NVP(calibrate_offsets));
}

std::ostream& operator<<(std::ostream& os, CADCCalibOptions const& options)
{
	return os << "CADCCalibOptions(" << options.calibrate_offsets << ")";
}

} // namespace ccalix

EXPLICIT_INSTANTIATE_CEREAL_SERIALIZE(ccalix::CADCCalibOptions)
