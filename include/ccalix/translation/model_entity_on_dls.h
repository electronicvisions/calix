#pragma once
#include "ccalix/cerealization.h"
#include "ccalix/genpybind.h"
#include "dapr/hashable.h"
#include "dapr/property.h"
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>

namespace GENPYBIND_TAG_CCALIX ccalix {

struct GENPYBIND(inline_base("*")) ModelEntityOnDLS
    : public dapr::Property<ModelEntityOnDLS>
    , public dapr::Hashable
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