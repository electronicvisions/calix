#pragma once
#include "ccalix/cerealization.h"
#include "ccalix/genpybind.h"
#include "dapr/property.h"
#include <map>
#include <vector>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
namespace GENPYBIND_TAG_CCALIX ccalix {

struct GENPYBIND(inline_base("*")) CalibratedParameter : public dapr::Property<CalibratedParameter>
{
private:
	// serialization support for polymorphism
	friend class cereal::access;
	template <typename Archiver>
	void serialize(Archiver& /*ar*/)
	{
		// dapr::Property has no serializable state, nothing to do
	}
};


} // namespace ccalix