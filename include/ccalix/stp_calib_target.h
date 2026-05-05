#pragma once
#include "ccalix/calib_target.h"
#include "ccalix/cerealization.h"
#include "halco/common/typed_array.h"
#include "halco/hicann-dls/vx/v3/capmem.h"
#include "haldls/vx/v3/capmem.h"
#include "hate/visibility.h"
#include <optional>

namespace ccalix GENPYBIND_TAG_CCALIX {

/*
 * Target for STP calibration.
 *
 * The STP voltages, which affect the dynamic range of the amplitudes, are set here. They are
 * currently not calibrated, but can be set per quadrant.
 */
struct GENPYBIND(visible) STPCalibTarget : public CalibTarget
{
	typedef halco::common::
	    typed_array<haldls::vx::v3::CapMemCell::Value, halco::hicann_dls::vx::v3::CapMemBlockOnDLS>
	        ValuePerQuadrant GENPYBIND(opaque(false));

	STPCalibTarget() SYMBOL_VISIBLE;

	/*
	 * STP v_charge (fully modulated state) for voltage set 0, in CapMem LSB. You can choose the two
	 * voltage sets to be, e.g., depressing and facilitating. By default, we select voltage set 0 to
	 * be facilitating and voltage set 1 to be depressing.
	 */
	ValuePerQuadrant v_charge_0;

	/*
	 * STP v_recover (fully recovered state) for voltage set 0, in CapMem LSB. Note that a
	 * utilization of some 0.2 happens before processing each event, so the voltage applied to the
	 * comparator never actually reaches v_recover.
	 */
	ValuePerQuadrant v_recover_0;

	/*
	 * @param v_charge_1 STP v_charge (fully modulated state) for voltage set 1, in CapMem LSB.
	 */
	ValuePerQuadrant v_charge_1;

	/*
	 * @param v_recover_1 STP v_recover (fully recovered state) for voltage set 1, in CapMem LSB.
	 */
	ValuePerQuadrant v_recover_1;

private:
	friend class cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // namespace ccalix

CCALIX_EXTERN_INSTANTIATE_CEREAL_SERIALIZE(ccalix::STPCalibTarget)
