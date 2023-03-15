#pragma once

#if defined(__has_include)
#if __has_include(<genpybind.h>)
#include <genpybind.h>
#else
#define GENPYBIND(...)
#define GENPYBIND_MANUAL(...)
#endif
#else
#define GENPYBIND(...)
#define GENPYBIND_MANUAL(...)
#endif

#define GENPYBIND_TAG_CCALIX GENPYBIND(tag(ccalix))
#define GENPYBIND_MODULE GENPYBIND(module)
