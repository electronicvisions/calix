#pragma once
#include "ccalix/calib_target.h"
#include "ccalix/cerealization.h"
#include "ccalix/range.h"
#include "haldls/vx/v3/cadc.h"
#include "haldls/vx/v3/capmem.h"
#include "hate/visibility.h"
#include <optional>

namespace ccalix GENPYBIND_TAG_CCALIX {

/**
 * Target parameters for the CADC calibration.
 */
struct GENPYBIND(visible) CADCCalibTarget : public CalibTarget
{
	typedef Range<haldls::vx::v3::CapMemCell::Value> DynamicRange GENPYBIND(opaque(false));

	/**
	 * CapMem settings (LSB) at the minimum and maximum of the desired dynamic
	 * range. By default, the full dynamic range of the CADC is used, which corresponds to some 0.15
	 * to 1.05 V. The voltages are configured as `stp_v_charge_0`, which gets connected to the CADCs
	 * via the CapMem debug readout.
	 */
	DynamicRange dynamic_range;

	typedef Range<haldls::vx::v3::CADCSampleQuad::Value> ReadRange GENPYBIND(opaque(false));

	/**
	 * Target CADC reads at the lower and upper end of the dynamic range.
	 */
	ReadRange read_range;

	CADCCalibTarget() SYMBOL_VISIBLE;

private:
	friend class cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // namespace ccalix

CCALIX_EXTERN_INSTANTIATE_CEREAL_SERIALIZE(ccalix::CADCCalibTarget)
