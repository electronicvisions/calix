#include "ccalix/translation/madc_calib_param.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/polymorphic.hpp>

namespace GENPYBIND_TAG_CCALIX ccalix {

MadcCalibParam::MadcCalibParam(trafo::Linear const& transformation) : translation(transformation) {}

std::unique_ptr<CalibratedParameter> MadcCalibParam::copy() const
{
	return std::make_unique<MadcCalibParam>(*this);
}

std::unique_ptr<CalibratedParameter> MadcCalibParam::move()
{
	return std::make_unique<MadcCalibParam>(std::move(*this));
}

std::ostream& MadcCalibParam::print(std::ostream& os) const
{
	return os << "MADC (" << translation << ")";
}

bool MadcCalibParam::is_equal_to(CalibratedParameter const& other) const
{
	return translation == static_cast<MadcCalibParam const&>(other).translation;
}


}
CEREAL_REGISTER_TYPE(ccalix::MadcCalibParam);
CEREAL_REGISTER_DYNAMIC_INIT(MadcCalibParam);