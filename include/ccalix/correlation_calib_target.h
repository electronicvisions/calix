#pragma once
#include "ccalix/calib_target.h"
#include "ccalix/cerealization.h"
#include "ccalix/types.h"
#include "hate/visibility.h"
#include <optional>

namespace ccalix GENPYBIND_TAG_CCALIX {

/**
 * Target parameters for correlation calibration.
 */
struct GENPYBIND(visible) CorrelationCalibTarget : public CalibTarget
{
	/**
	 * Target correlation amplitude (at delay 0) for all synapses, per correlated event. Feasible
	 * targets range from some 0.2 to 2.0, higher amplitudes will likely require adjusting
	 * v_res_meas.
	 */
	double amplitude;

	/**
	 * Target correlation time constant for all synapses. Feasible targets range from some 2 to 30
	 * us.
	 */
	TimeInS time_constant;

	CorrelationCalibTarget() SYMBOL_VISIBLE;

private:
	friend class cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // namespace ccalix

CCALIX_EXTERN_INSTANTIATE_CEREAL_SERIALIZE(ccalix::CorrelationCalibTarget)
