#include "ccalix/translation/cadc_calib_param.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/polymorphic.hpp>

namespace GENPYBIND_TAG_CCALIX ccalix {

CadcCalibParam::CadcCalibParam(trafo::Linear const& transformation) : translation(transformation) {}

std::unique_ptr<CalibratedParameter> CadcCalibParam::copy() const
{
	return std::make_unique<CadcCalibParam>(*this);
}

std::unique_ptr<CalibratedParameter> CadcCalibParam::move()
{
	return std::make_unique<CadcCalibParam>(std::move(*this));
}

std::ostream& CadcCalibParam::print(std::ostream& os) const
{
	return os << "CadcCalibParam(" << translation << ")";
}

bool CadcCalibParam::is_equal_to(CalibratedParameter const& other) const
{
	return translation == static_cast<CadcCalibParam const&>(other).translation;
}

}
CEREAL_REGISTER_TYPE(ccalix::CadcCalibParam);
CEREAL_REGISTER_DYNAMIC_INIT(CadcCalibParam);