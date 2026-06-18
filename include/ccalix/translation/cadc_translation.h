#pragma once
#include "ccalix/cerealization.h"
#include "ccalix/genpybind.h"
#include "ccalix/trafo/linear.h"
#include "ccalix/translation/cadc_on_dls.h"
#include "ccalix/translation/parameter_translation.h"
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>

namespace GENPYBIND_TAG_CCALIX ccalix {

struct GENPYBIND(visible) CadcTranslation : public ParameterTranslation
{
	CadcTranslation() = default;

	CadcTranslation(CadcOnDLS cadc, trafo::Linear const& transformation);

	virtual void translate(
	    ModelParameters const& targets, CalibratedParameters& calibrated_parameters) const override;


	trafo::Linear translation;
	CadcOnDLS cadc;


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
		ar(cadc);
		ar(translation);
	}
};
} // namespace ccalix
CEREAL_FORCE_DYNAMIC_INIT(CadcTranslation);