#pragma once
#include "ccalix/cerealization.h"
#include "ccalix/genpybind.h"
#include "ccalix/trafo/linear.h"
#include "ccalix/translation/calibrated_parameter.h"
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>


namespace GENPYBIND_TAG_CCALIX ccalix {


struct GENPYBIND(visible) MadcCalibParam : public CalibratedParameter
{
	MadcCalibParam() = default;

	MadcCalibParam(trafo::Linear const& transformation);


	trafo::Linear translation;

	virtual std::unique_ptr<CalibratedParameter> copy() const override;

	virtual std::unique_ptr<CalibratedParameter> move() override;

	virtual std::ostream& print(std::ostream& os) const override;

protected:
	virtual bool is_equal_to(CalibratedParameter const& other) const override;

private:
	// serialization support for polymorphism
	friend class cereal::access;
	template <typename Archiver>
	void serialize(Archiver& ar)
	{
		ar(cereal::base_class<CalibratedParameter>(this));
		ar(translation);
	}
};

} // namespace ccalix
CEREAL_FORCE_DYNAMIC_INIT(MadcCalibParam);