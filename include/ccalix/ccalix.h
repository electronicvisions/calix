#include "ccalix/genpybind.h"
#include "ccalix/helpers.h"

GENPYBIND_MANUAL({
	parent.attr("__variant__") = "pybind11";
	parent->py::module::import("pyhalco_hicann_dls_vx_v3");
})
