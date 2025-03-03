#pragma once
#include "ccalix/genpybind.h"

namespace ccalix GENPYBIND_TAG_CCALIX {

/*
 * Data structure for collecting other configuration parameters for higher-level calibration
 functions.

 * These options are not targets in the sense that they are more technical parameters. They may
 still affect the result, though.

 * The available choices (ranges) will be clear from the expected data types. For example, boolean
 switches can allow to perform the calibration differently, or a priority setting can be applied to
 some targets, at the cost of accuracy at other targets.
 */
struct GENPYBIND(visible) CalibOptions
{};

} // namespace ccalix
