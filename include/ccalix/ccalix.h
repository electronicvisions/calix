#include "ccalix/cadc_calib_options.h"
#include "ccalix/calib_options.h"
#include "ccalix/correlation_calib_options.h"
#include "ccalix/correlation_calib_target.h"
#include "ccalix/genpybind.h"
#include "ccalix/hagen/multiplication.h"
#include "ccalix/hagen_calib_options.h"
#include "ccalix/hagen_calib_target.h"
#include "ccalix/hagen_neuron_calib_target.h"
#include "ccalix/hagen_synin_calib_options.h"
#include "ccalix/hagen_synin_calib_target.h"
#include "ccalix/helpers.h"
#include "ccalix/neuron_calib_options.h"
#include "ccalix/neuron_calib_target.h"
#include "ccalix/spiking/correlation_measurement.h"
#include "ccalix/spiking_calib_options.h"
#include "ccalix/spiking_calib_target.h"
#include "ccalix/stp_calib_options.h"
#include "ccalix/stp_calib_target.h"
#include "ccalix/synapse_driver_calib_options.h"
#include "ccalix/types.h"
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
    "CalibTarget",
    "CADCCalibOptions",
    "CorrelationCalibOptions",
    "HagenCalibOptions",
    "HagenSyninCalibOptions",
    "NeuronCalibOptions",
    "SpikingCalibOptions",
    "SynapseDriverCalibOptions",
    "STPCalibOptions",
    "CADCCalibTarget",
    "CorrelationCalibTarget",
    "HagenCalibTarget",
    "HagenSyninCalibTarget",
    "HagenNeuronCalibTarget",
    "NeuronCalibTarget",
    "SpikingCalibTarget",
    "STPCalibTarget"};

typedef hate::type_list<
    ccalix::CalibTarget,
    ccalix::CADCCalibOptions,
    ccalix::CorrelationCalibOptions,
    ccalix::HagenCalibOptions,
    ccalix::HagenSyninCalibOptions,
    ccalix::NeuronCalibOptions,
    ccalix::SpikingCalibOptions,
    ccalix::SynapseDriverCalibOptions,
    ccalix::STPCalibOptions,
    ccalix::CADCCalibTarget,
    ccalix::CorrelationCalibTarget,
    ccalix::HagenCalibTarget,
    ccalix::HagenSyninCalibTarget,
    ccalix::HagenNeuronCalibTarget,
    ccalix::NeuronCalibTarget,
    ccalix::SpikingCalibTarget,
    ccalix::STPCalibTarget>
    pickle_types;

} // namespace ccalix::detail

GENPYBIND(postamble)
GENPYBIND_MANUAL({
	::haldls::vx::AddPickle<::ccalix::detail::pickle_types>::apply(
	    parent, ::ccalix::detail::pickle_type_names);

	{
		auto attr = parent.attr("NeuronCalibTarget").attr("PerNeuronTimeConstant");
		auto ism = pybind11::is_method(attr);
		attr.attr("as_quantity") = pybind11::cpp_function(
		    [](ccalix::NeuronCalibTarget::PerNeuronTimeConstant& self) {
			    auto pq = pybind11::module_::import("quantities");
			    return pq.attr("Quantity")(halco::common::detail::to_numpy(self), pq.attr("s"));
		    },
		    ism);
		auto attr_syn = parent.attr("NeuronCalibTarget").attr("TauSyn");
		auto ism_syn = pybind11::is_method(attr_syn);
		attr_syn.attr("as_quantity") = pybind11::cpp_function(
		    [](ccalix::NeuronCalibTarget::TauSyn& self) {
			    auto pq = pybind11::module_::import("quantities");
			    return pq.attr("Quantity")(halco::common::detail::to_numpy(self), pq.attr("s"));
		    },
		    ism_syn);
	}

	{
		auto attr = parent.attr("HagenNeuronCalibTarget").attr("PerNeuronTimeConstant");
		auto ism = pybind11::is_method(attr);
		attr.attr("as_quantity") = pybind11::cpp_function(
		    [](ccalix::NeuronCalibTarget::PerNeuronTimeConstant& self) {
			    auto pq = pybind11::module_::import("quantities");
			    return pq.attr("Quantity")(halco::common::detail::to_numpy(self), pq.attr("s"));
		    },
		    ism);
		auto attr_syn = parent.attr("HagenNeuronCalibTarget").attr("TauSyn");
		auto ism_syn = pybind11::is_method(attr_syn);
		attr_syn.attr("as_quantity") = pybind11::cpp_function(
		    [](ccalix::NeuronCalibTarget::TauSyn& self) {
			    auto pq = pybind11::module_::import("quantities");
			    return pq.attr("Quantity")(halco::common::detail::to_numpy(self), pq.attr("s"));
		    },
		    ism_syn);
	}
})
