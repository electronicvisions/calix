#include "ccalix/translation/cadc_translation.h"
#include "ccalix/trafo/linear.h"
#include "ccalix/translation/cadc_calib_param.h"
#include "ccalix/translation/cadc_on_dls.h"
#include "ccalix/translation/parameter_translation.h"
#include <memory>
#include <cereal/archives/binary.hpp>
#include <cereal/types/polymorphic.hpp>

namespace GENPYBIND_TAG_CCALIX ccalix {

CadcTranslation::CadcTranslation(CadcOnDLS cadc, trafo::Linear const& transformation) :
    translation(transformation), cadc(cadc)
{
}

// basically returns the transformation
void CadcTranslation::translate(
    ModelParameters const& targets,
    CalibratedParameters& calibrated_parameters /*, JITGraphExecutor& executor*/) const
{
	if (targets.contains(cadc)) {
		calibrated_parameters.set(cadc, CadcCalibParam(translation));
	}
}

std::unique_ptr<ParameterTranslation> CadcTranslation::copy() const
{
	return std::make_unique<CadcTranslation>(*this);
}

std::unique_ptr<ParameterTranslation> CadcTranslation::move()
{
	return std::make_unique<CadcTranslation>(std::move(*this));
}

std::ostream& CadcTranslation::print(std::ostream& os) const
{
	return os << "CADC(" << translation << ")";
}

bool CadcTranslation::is_equal_to(ParameterTranslation const& other) const
{
	return translation == static_cast<CadcTranslation const&>(other).translation &&
	       cadc == static_cast<CadcTranslation const&>(other).cadc;
}


} // namespace ccalix
CEREAL_REGISTER_TYPE(ccalix::CadcTranslation);
CEREAL_REGISTER_DYNAMIC_INIT(CadcTranslation);