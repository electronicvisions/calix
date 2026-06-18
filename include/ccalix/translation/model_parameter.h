#pragma once

#include "ccalix/genpybind.h"
#include "dapr/hashable.h"
#include "dapr/property.h"

namespace GENPYBIND_TAG_CCALIX ccalix {

struct GENPYBIND(inline_base("*")) ModelParameter : public dapr::Property<ModelParameter>
{};

} // namespace ccalix