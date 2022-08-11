#include "ccalix/genpybind.h"
#include "ccalix/hagen/multiplication.h"
#include "ccalix/helpers.h"
#include "ccalix/spiking/correlation_measurement.h"

GENPYBIND_MANUAL({
	parent.attr("__variant__") = "pybind11";
	parent->py::module::import("pyhalco_hicann_dls_vx_v3");
	parent->py::module::import("pyhaldls_vx_v3");
	parent->py::module::import("pystadls_vx_v3");
})
