#pragma once
#include "ccalix/genpybind.h"
#include <stdexcept>

namespace GENPYBIND_TAG_CCALIX ccalix {

class OutsideDomainException : public std::runtime_error
{
public:
	OutsideDomainException(std::string const& msg = "Value outside of domain.") :
	    std::runtime_error(msg)
	{
	}
};

GENPYBIND_MANUAL({
	parent->py::template register_exception<ccalix::OutsideDomainException>(
	    parent, "OutsideDomainException");
});
} // namespace ccalix
