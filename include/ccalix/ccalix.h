#include "ccalix/cadc_calib_options.h"
#include "ccalix/calib_options.h"
#include "ccalix/correlation_calib_options.h"
#include "ccalix/genpybind.h"
#include "ccalix/hagen/multiplication.h"
#include "ccalix/hagen_calib_options.h"
#include "ccalix/hagen_synin_calib_options.h"
#include "ccalix/helpers.h"
#include "ccalix/neuron_calib_options.h"
#include "ccalix/spiking/correlation_measurement.h"
#include "ccalix/spiking_calib_options.h"
#include "ccalix/stp_calib_options.h"
#include "ccalix/synapse_driver_calib_options.h"
#include "haldls/vx/pickle.h"
#include "hate/type_list.h"

GENPYBIND_MANUAL({
	parent.attr("__variant__") = "pybind11";
	parent->py::module::import("pyhalco_hicann_dls_vx_v3");
	parent->py::module::import("pyhaldls_vx_v3");
	parent->py::module::import("pystadls_vx_v3");
})

namespace ccalix::detail {

static std::vector<std::string> const pickle_type_names = {
    "CADCCalibOptions",          "CorrelationCalibOptions", "HagenCalibOptions",
    "HagenSyninCalibOptions",    "NeuronCalibOptions",      "SpikingCalibOptions",
    "SynapseDriverCalibOptions", "STPCalibOptions"};

typedef hate::type_list<
    ccalix::CADCCalibOptions,
    ccalix::CorrelationCalibOptions,
    ccalix::HagenCalibOptions,
    ccalix::HagenSyninCalibOptions,
    ccalix::NeuronCalibOptions,
    ccalix::SpikingCalibOptions,
    ccalix::SynapseDriverCalibOptions,
    ccalix::STPCalibOptions>
    pickle_types;

} // namespace ccalix::detail

GENPYBIND_MANUAL({
	::haldls::vx::AddPickle<::ccalix::detail::pickle_types>::apply(
	    parent, ::ccalix::detail::pickle_type_names);
})
