#pragma once
#include "ccalix/cerealization.h"
#include "ccalix/genpybind.h"
#include "ccalix/translation/calibrated_parameter.h"
#include "ccalix/translation/model_entity_on_dls.h"
#include "ccalix/translation/model_parameter.h"
#include "dapr/unordered_map.h"
#include <map>
#include <memory>
#include <vector>

namespace GENPYBIND_TAG_CCALIX ccalix {
struct GENPYBIND(inline_base("*")) ParameterTranslation
    : public dapr::Property<ParameterTranslation>
{
	// set of model parameters
	typedef dapr::UnorderedMap<ModelEntityOnDLS, ModelParameter> ModelParameters
	    GENPYBIND(opaque(false));


	/**
	 * Calibrated set of hardware parameters.
	 * In general they can't be combined to form a more complete
	 * calibration, since the interface for querying doesn't expose
	 * dependencies.
	 */
	// stored by pointer for polymorphism
	// key pointer poses problems (to be fixed)
	typedef dapr::UnorderedMap<ModelEntityOnDLS, CalibratedParameter> CalibratedParameters
	    GENPYBIND(opaque(false));


	// translatiion function, to be overridden by derived classes
	virtual void translate(
	    ModelParameters const& targets, CalibratedParameters& calibrated_parameters) const = 0;

private:
	// serialization support
	friend class cereal::access;
	template <typename Archiver>
	void serialize(Archiver& /*ar*/)
	{
		// nothing to serialize, just there for polymorphism
	}
};


} // namespace ccalix