#include "ccalix/translation/madc_translation.h"
#include "ccalix/trafo/linear.h"
#include "ccalix/translation/madc_calib_param.h"
#include "ccalix/translation/madc_on_dls.h"
#include "ccalix/translation/parameter_translation.h"
#include <memory>
#include <cereal/archives/binary.hpp>
#include <cereal/types/polymorphic.hpp>

namespace GENPYBIND_TAG_CCALIX ccalix {

MadcTranslation::MadcTranslation(MadcOnDLS madc, trafo::Linear const& transformation) :
    madc(madc), translation(transformation)
{
}

// basically returns the transformation
void MadcTranslation::translate(
    ModelParameters const& targets,
    CalibratedParameters& calibrated_parameters /*,JITGraphExecutor& executor*/) const
{
	if (targets.contains(madc)) {
		calibrated_parameters.set(madc, MadcCalibParam(translation));
	}
}

std::unique_ptr<ParameterTranslation> MadcTranslation::copy() const
{
	return std::make_unique<MadcTranslation>(*this);
}

std::unique_ptr<ParameterTranslation> MadcTranslation::move()
{
	return std::make_unique<MadcTranslation>(std::move(*this));
}

std::ostream& MadcTranslation::print(std::ostream& os) const
{
	return os << "MADC(" << translation << ")";
}

bool MadcTranslation::is_equal_to(ParameterTranslation const& other) const
{
	return translation == static_cast<MadcTranslation const&>(other).translation &&
	       madc == static_cast<MadcTranslation const&>(other).madc;
}

} // namespace ccalix
CEREAL_REGISTER_TYPE(ccalix::MadcTranslation);
CEREAL_REGISTER_DYNAMIC_INIT(MadcTranslation);