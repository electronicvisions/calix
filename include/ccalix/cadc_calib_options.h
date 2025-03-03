#pragma once
#include "ccalix/calib_options.h"
#include "ccalix/cerealization.h"
#include "hate/visibility.h"
#include <iosfwd>
#include <optional>

namespace ccalix GENPYBIND_TAG_CCALIX {

/**
 * Further configuration parameters for the CADC calibration, that are not directly calibration
 * targets.
 */
struct GENPYBIND(visible) CADCCalibOptions : public CalibOptions
{
	CADCCalibOptions() = default;

	/*
	 * Decide whether the individual channel offsets are calibrated. For standard use-cases,
	 * including neuron and correlation measurements, this should be enabled (default). Only in case
	 * the auto-calibrating correlation reset (cf. hal.CommonCorrelationConfig) is used, this should
	 * be disabled.
	 */
	bool calibrate_offsets{true};

	GENPYBIND(stringstream)
	friend std::ostream& operator<<(std::ostream& os, CADCCalibOptions const& options)
	    SYMBOL_VISIBLE;

private:
	friend class cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // namespace ccalix

CCALIX_EXTERN_INSTANTIATE_CEREAL_SERIALIZE(ccalix::CADCCalibOptions)
