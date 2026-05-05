#pragma once
#include "ccalix/cerealization.h"
#include "ccalix/genpybind.h"

namespace ccalix GENPYBIND_TAG_CCALIX {

/*
 * Data structure for collecting targets for higher-level calibration functions into one logical
 * unit. Targets are parameters that directly affect how a circuit is configured. They have a
 * standard range, where the circuits will work well. Exceeding the standard range may work better
 * for some instances (e.g., neurons) than others.
 */
class GENPYBIND(visible) CalibTarget
{
	friend class cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // namespace ccalix

CCALIX_EXTERN_INSTANTIATE_CEREAL_SERIALIZE(ccalix::CalibTarget)
