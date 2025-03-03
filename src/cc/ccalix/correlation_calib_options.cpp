#include "ccalix/correlation_calib_options.h"

#include "ccalix/cerealization.tcc"
#include "cereal/types/halco/common/geometry.h"
#include "hate/join.h"
#include <ostream>

namespace ccalix {

std::ostream& operator<<(std::ostream& os, CorrelationCalibOptions::Branches const& value)
{
	switch (value) {
		case CorrelationCalibOptions::Branches::CAUSAL: {
			return os << "CAUSAL";
		}
		case CorrelationCalibOptions::Branches::ACAUSAL: {
			return os << "ACAUSAL";
		}
		case CorrelationCalibOptions::Branches::BOTH: {
			return os << "BOTH";
		}
		default: {
		}
	}
	throw std::logic_error("Branches type not implemented.");
}

void CorrelationCalibOptions::check() const
{
	if (time_constant_priority < 0 || time_constant_priority > 1) {
		throw std::out_of_range(
		    "Time constant priority " + std::to_string(time_constant_priority) +
		    " is not in the range from 0 to 1.");
	}
}

template <typename Archive>
void CorrelationCalibOptions::serialize(Archive& ar, std::uint32_t const)
{
	ar(CEREAL_NVP(branches));
	ar(CEREAL_NVP(v_res_meas));
	ar(CEREAL_NVP(v_reset));
	ar(CEREAL_NVP(calibrate_synapses));
	ar(CEREAL_NVP(time_constant_priority));
	ar(CEREAL_NVP(default_amp_calib));
	ar(CEREAL_NVP(default_time_calib));
}

std::ostream& operator<<(std::ostream& os, CorrelationCalibOptions const& options)
{
	os << "CorrelationCalibOptions(" << options.branches << ", " << options.v_res_meas << ", "
	   << options.v_reset << ", time_constant_priority: " << options.time_constant_priority
	   << ", default_amp_calib: " << options.default_amp_calib << ", " << options.default_time_calib
	   << ")";
	return os;
}

} // namespace ccalix

EXPLICIT_INSTANTIATE_CEREAL_SERIALIZE(ccalix::CorrelationCalibOptions)
