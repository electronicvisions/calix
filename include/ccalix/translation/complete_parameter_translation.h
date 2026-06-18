#pragma once
#include "ccalix/cerealization.h"
#include "ccalix/genpybind.h"
#include "ccalix/translation/cadc_translation.h"
#include "ccalix/translation/madc_translation.h"
#include <vector>
#include <cereal/archives/json.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>

namespace GENPYBIND_TAG_CCALIX ccalix {

struct GENPYBIND(visible) CompleteParameterTranslation : public ParameterTranslation
{
	std::vector<CadcTranslation> cadc; // translation for all (1024) channels
	std::vector<MadcTranslation> madc; // translations for all (two) channels

	CompleteParameterTranslation() = default;

	CompleteParameterTranslation(
	    std::vector<CadcTranslation> const& cadc_translations,
	    std::vector<MadcTranslation> const& madc_translations);

	// translates all model parameters at once
	// returns all calibrated parameters at once
	virtual void translate(
	    ModelParameters const& targets, CalibratedParameters& parameters) const override;


	// serialization support
	friend class cereal::access;
	template <typename Archiver>
	void serialize(Archiver& ar)
	{
		ar(cereal::base_class<ParameterTranslation>(this));
		ar(cadc);
		ar(madc);
	}

	virtual std::unique_ptr<ParameterTranslation> copy() const override;

	virtual std::unique_ptr<ParameterTranslation> move() override;

	virtual std::ostream& print(std::ostream& os) const override;


protected:
	virtual bool is_equal_to(ParameterTranslation const& other) const override;
};

std::string to_portable_binary(CompleteParameterTranslation const&);
CompleteParameterTranslation from_portable_binary(std::string const&);

GENPYBIND_MANUAL({
	parent.def("to_portable_binary", [](ccalix::CompleteParameterTranslation const& t) {
		return pybind11::bytes(ccalix::to_portable_binary(t));
	});
	parent.def("from_portable_binary", &ccalix::from_portable_binary);
})

} // namespace ccalix
CEREAL_FORCE_DYNAMIC_INIT(CompleteParameterTranslation);