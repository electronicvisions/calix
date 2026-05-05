#pragma once
#include "ccalix/cadc_calib_target.h"
#include "ccalix/calib_target.h"
#include "ccalix/cerealization.h"
#include "ccalix/hagen_neuron_calib_target.h"
#include "hate/visibility.h"

namespace ccalix GENPYBIND_TAG_CCALIX {

struct GENPYBIND(visible) HagenSyninCalibTarget : public CalibTarget
{
	CADCCalibTarget cadc_target;

	haldls::vx::v3::CapMemCell::Value synapse_dac_bias;

	HagenSyninCalibTarget() SYMBOL_VISIBLE;

private:
	friend class cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // namespace ccalix

CCALIX_EXTERN_INSTANTIATE_CEREAL_SERIALIZE(ccalix::HagenSyninCalibTarget)
