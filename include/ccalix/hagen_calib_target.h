#pragma once
#include "ccalix/cadc_calib_target.h"
#include "ccalix/calib_target.h"
#include "ccalix/cerealization.h"
#include "ccalix/hagen_neuron_calib_target.h"
#include "hate/visibility.h"

namespace ccalix GENPYBIND_TAG_CCALIX {

/*
 * Dataclass collecting target parameters for Hagen-mode calibrations with integration on membranes.
 */
struct GENPYBIND(visible) HagenCalibTarget : public CalibTarget
{
	/**
	 * Target parameters for CADC calibration.
	 */
	CADCCalibTarget cadc_target;

	/**
	 * Target parameters for neuron calibration.
	 */
	HagenNeuronCalibTarget neuron_target;

	HagenCalibTarget() SYMBOL_VISIBLE;

private:
	friend class cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // namespace ccalix

CCALIX_EXTERN_INSTANTIATE_CEREAL_SERIALIZE(ccalix::HagenCalibTarget)
