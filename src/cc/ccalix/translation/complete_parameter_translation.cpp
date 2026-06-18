#include "ccalix/translation/complete_parameter_translation.h"
#include "ccalix/translation/cadc_calib_param.h"
#include "ccalix/translation/cadc_on_dls.h"
#include "ccalix/translation/madc_on_dls.h"
#include "ccalix/translation/model_cadc.h"
#include "ccalix/translation/model_entity_on_dls.h"
#include "ccalix/translation/model_madc.h"
#include <memory>
#include <sstream>
#include <vector>
#include <cereal/archives/binary.hpp>
#include <cereal/types/polymorphic.hpp>

namespace GENPYBIND_TAG_CCALIX ccalix {

CompleteParameterTranslation::CompleteParameterTranslation(
    std::vector<CadcTranslation> const& cadc_translations,
    std::vector<MadcTranslation> const& madc_translations) :
    cadc(cadc_translations), madc(madc_translations)
{
}

void CompleteParameterTranslation::translate(
    ModelParameters const& targets, CalibratedParameters& parameters) const
{
	for (auto const& cadc_translation : cadc) {
		// If the target is a CadcCalibParam, we can use the CADC translation
		// This assumes that CadcTranslation is a valid type for the target
		// and that it has been initialized properly
		// Translate using CADC translation
		cadc_translation.translate(targets, parameters);
	}
	// Translate using each MADC translation
	for (auto const& madc_translation : madc) {
		madc_translation.translate(targets, parameters);
	}
}

std::unique_ptr<ParameterTranslation> CompleteParameterTranslation::copy() const
{
	return std::make_unique<CompleteParameterTranslation>(*this);
}

std::unique_ptr<ParameterTranslation> CompleteParameterTranslation::move()
{
	return std::make_unique<CompleteParameterTranslation>(std::move(*this));
}

std::ostream& CompleteParameterTranslation::print(std::ostream& os) const
{
	for (auto const& translation : cadc) {
		os << translation << " ";
	}
	for (auto const& translation : madc) {
		os << translation << " ";
	}
	return os;
}

bool CompleteParameterTranslation::is_equal_to(ParameterTranslation const& other) const
{
	return cadc == static_cast<CompleteParameterTranslation const&>(other).cadc &&
	       madc == static_cast<CompleteParameterTranslation const&>(other).madc;
}

std::string to_portable_binary(
    CompleteParameterTranslation const& generator) /*SYMBOL_VISIBLE GENPYBIND(visible)*/
{
	// Implementation of serialization to a portable binary format
	std::stringstream is;
	cereal::BinaryOutputArchive archive(is);
	archive(generator);
	return is.str();
}

CompleteParameterTranslation from_portable_binary(
    std::string const& text) /*SYMBOL_VISIBLE GENPYBIND(visible)*/
{
	// deserialization from binary archive
	CompleteParameterTranslation translation;
	std::stringstream os(text);
	cereal::BinaryInputArchive input_archive(os);
	input_archive(translation);
	return translation;
}


} // namespace ccalix
CEREAL_REGISTER_TYPE(ccalix::CompleteParameterTranslation);
CEREAL_REGISTER_DYNAMIC_INIT(CompleteParameterTranslation);