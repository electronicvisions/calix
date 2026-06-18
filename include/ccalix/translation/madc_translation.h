#pragma once
#include "ccalix/cerealization.h"
#include "ccalix/genpybind.h"
#include "ccalix/trafo/linear.h"
#include "ccalix/translation/madc_on_dls.h"
#include "ccalix/translation/parameter_translation.h"
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>

namespace GENPYBIND_TAG_CCALIX ccalix {

struct GENPYBIND(visible) MadcTranslation : public ParameterTranslation
{
	MadcTranslation() = default;

	MadcTranslation(MadcOnDLS madc, trafo::Linear const& transformation);

	virtual void translate(
	    ModelParameters const& targets, CalibratedParameters& calibrated_parameters) const override;


	// if both channels must be represented, a second channel and trafo can be added
	MadcOnDLS madc;
	trafo::Linear translation;

	virtual std::unique_ptr<ParameterTranslation> copy() const override;

	virtual std::unique_ptr<ParameterTranslation> move() override;

	virtual std::ostream& print(std::ostream& os) const override;


protected:
	virtual bool is_equal_to(ParameterTranslation const& other) const override;

private:
	// serialization support
	friend class cereal::access;
	template <typename Archiver>
	void serialize(Archiver& ar)
	{
		// serialize transformations
		ar(cereal::base_class<ParameterTranslation>(this));
		ar(madc);
		ar(translation);
	}
};

} // namespace ccalix
CEREAL_FORCE_DYNAMIC_INIT(MadcTranslation);