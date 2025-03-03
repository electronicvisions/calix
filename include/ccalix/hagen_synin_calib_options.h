#pragma once
#include "ccalix/cadc_calib_options.h"
#include "ccalix/calib_options.h"
#include "ccalix/cerealization.h"
#include "ccalix/synapse_driver_calib_options.h"
#include "hate/visibility.h"
#include <iosfwd>
#include <optional>

namespace ccalix GENPYBIND_TAG_CCALIX {

/*
 * Further options for Hagen-mode calibrations with integration on synaptic input lines.
 */
struct GENPYBIND(visible) HagenSyninCalibOptions : public CalibOptions
{
	/**
	 * Further options for CADC calibration.
	 */
	CADCCalibOptions cadc_options;

	/**
	 * Further options for synapse driver calibration.
	 */
	SynapseDriverCalibOptions synapse_driver_options;

	HagenSyninCalibOptions() = default;

	GENPYBIND(stringstream)
	friend std::ostream& operator<<(std::ostream& os, HagenSyninCalibOptions const& options)
	    SYMBOL_VISIBLE;

private:
	friend class cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // namespace ccalix

CCALIX_EXTERN_INSTANTIATE_CEREAL_SERIALIZE(ccalix::HagenSyninCalibOptions)
