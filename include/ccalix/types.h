#pragma once
#include "ccalix/genpybind.h"
#include "halco/common/geometry.h"

namespace ccalix GENPYBIND_TAG_CCALIX {

struct GENPYBIND(inline_base("*")) PotentialInVolt
    : public halco::common::detail::BaseType<PotentialInVolt, double>
{
	constexpr explicit PotentialInVolt(value_type const value = 0.) GENPYBIND(implicit_conversion) :
	    base_t(value)
	{
	}

	GENPYBIND_MANUAL({
		parent.def("as_quantity", [&parent](GENPYBIND_PARENT_TYPE const& self) {
			auto pq = pybind11::module_::import("quantities");
			return pq.attr("Quantity")(self.value(), pq.attr("V"));
		});
	})
};

} // namespace ccalix
