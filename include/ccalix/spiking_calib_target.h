#pragma once
#include "ccalix/cadc_calib_target.h"
#include "ccalix/calib_target.h"
#include "ccalix/cerealization.h"
#include "ccalix/correlation_calib_target.h"
#include "ccalix/neuron_calib_target.h"
#include "ccalix/stp_calib_target.h"
#include "hate/visibility.h"
#include <optional>

namespace ccalix GENPYBIND_TAG_CCALIX {

/*
 * Dataclass collecting target parameters for Spiking-mode calibrations with integration on
 * membranes.
 */
struct GENPYBIND(visible) SpikingCalibTarget : public CalibTarget
{
	/**
	 * Target parameters for CADC calibration.
	 */
	CADCCalibTarget cadc_target;

	/**
	 * Target parameters for neuron calibration.
	 */
	NeuronCalibTarget neuron_target;

	/**
	 * Target parameters for correlation calibration.
	 */
	std::optional<CorrelationCalibTarget> correlation_target{std::nullopt};

	/**
	 * Target parameters for STP calibration.
	 */
	STPCalibTarget stp_target;

	SpikingCalibTarget() = default;

private:
	friend class cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // namespace ccalix

CCALIX_EXTERN_INSTANTIATE_CEREAL_SERIALIZE(ccalix::SpikingCalibTarget)
