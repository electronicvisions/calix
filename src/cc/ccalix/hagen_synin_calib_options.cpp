#include "ccalix/hagen_synin_calib_options.h"

#include "ccalix/cerealization.tcc"
#include <ostream>

namespace ccalix {

template <typename Archive>
void HagenSyninCalibOptions::serialize(Archive& ar, std::uint32_t const)
{
	ar(CEREAL_NVP(cadc_options));
	ar(CEREAL_NVP(synapse_driver_options));
}

std::ostream& operator<<(std::ostream& os, HagenSyninCalibOptions const& options)
{
	return os << "CADCCalibOptions(" << options.cadc_options << ", "
	          << options.synapse_driver_options << ")";
}

} // namespace ccalix

EXPLICIT_INSTANTIATE_CEREAL_SERIALIZE(ccalix::HagenSyninCalibOptions)
