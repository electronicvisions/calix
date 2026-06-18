#pragma once
#include "ccalix/genpybind.h"
#include "ccalix/translation/model_parameter.h"
#include "dapr/empty_property.h"
#include "dapr/empty_property.tcc"
#include <map>
#include <vector>
#include <boost/functional/hash.hpp>

namespace GENPYBIND_TAG_CCALIX ccalix {

struct GENPYBIND(inline_base("*EmptyProperty*")) ModelMadc
    : public dapr::EmptyProperty<ModelMadc, ModelParameter>
{};

} // namespace ccalix