#include "ccalix/stp_calib_target.h"

#include "ccalix/cerealization.tcc"
#include "cereal/types/halco/common/geometry.h"
#include "cereal/types/halco/common/typed_array.h"

namespace ccalix {

STPCalibTarget::STPCalibTarget() : v_charge_0(), v_recover_0(), v_charge_1(), v_recover_1()
{
	v_charge_0.fill(haldls::vx::v3::CapMemCell::Value(100));
	v_recover_0.fill(haldls::vx::v3::CapMemCell::Value(400));
	v_charge_1.fill(haldls::vx::v3::CapMemCell::Value(330));
	v_recover_1.fill(haldls::vx::v3::CapMemCell::Value(50));
}

template <typename Archive>
void STPCalibTarget::serialize(Archive& ar, std::uint32_t const)
{
	ar(CEREAL_NVP(v_charge_0));
	ar(CEREAL_NVP(v_recover_0));
	ar(CEREAL_NVP(v_charge_1));
	ar(CEREAL_NVP(v_recover_1));
}

} // namespace ccalix

EXPLICIT_INSTANTIATE_CEREAL_SERIALIZE(ccalix::STPCalibTarget)
