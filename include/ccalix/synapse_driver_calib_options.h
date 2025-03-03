#pragma once
#include "ccalix/calib_options.h"
#include "ccalix/cerealization.h"
#include "haldls/vx/v3/padi.h"
#include "haldls/vx/v3/synapse.h"
#include "hate/visibility.h"
#include <iosfwd>

namespace ccalix GENPYBIND_TAG_CCALIX {

/*
 * Further options for synapse driver calibration.
 */
struct GENPYBIND(visible) SynapseDriverCalibOptions : public CalibOptions
{
	/*
	 * Hagen-mode activation where amplitudes between different drivers are aligned using the
	 * individual DAC offsets.
	 */
	haldls::vx::v3::SynapseQuad::Label offset_test_activation{
	    haldls::vx::v3::SynapseQuad::Label(haldls::vx::v3::PADIEvent::HagenActivation(3))};

	SynapseDriverCalibOptions() = default;

	GENPYBIND(stringstream)
	friend std::ostream& operator<<(std::ostream& os, SynapseDriverCalibOptions const& options)
	    SYMBOL_VISIBLE;

private:
	friend class cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // namespace ccalix

CCALIX_EXTERN_INSTANTIATE_CEREAL_SERIALIZE(ccalix::SynapseDriverCalibOptions)
