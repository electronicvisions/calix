#pragma once
#include "ccalix/cadc_calib_options.h"
#include "ccalix/calib_options.h"
#include "ccalix/cerealization.h"
#include "ccalix/neuron_calib_options.h"
#include "ccalix/synapse_driver_calib_options.h"
#include "hate/visibility.h"
#include <optional>

namespace ccalix GENPYBIND_TAG_CCALIX {

/*
 * Further options for Hagen-mode calibrations with integration on membranes.
 */
struct GENPYBIND(visible) HagenCalibOptions : public CalibOptions
{
	/**
	 * Further options for CADC calibration.
	 */
	CADCCalibOptions cadc_options;

	/**
	 * Further options for neuron calibration.
	 */
	NeuronCalibOptions neuron_options;


	/**
	 * Decide whether the neuron leak bias currents are set to zero after calibration. This is done
	 * by default, which disables leakage entirely. Note that even if the leak bias is set to zero,
	 * some pseudo-leakage may occur through the synaptic input OTAs.
	 */
	bool neuron_disable_leakage{true};

	/**
	 * Further options for synapse driver calibration.
	 */
	SynapseDriverCalibOptions synapse_driver_options;

	HagenCalibOptions() = default;

	GENPYBIND(stringstream)
	friend std::ostream& operator<<(std::ostream& os, HagenCalibOptions const& options)
	    SYMBOL_VISIBLE;

private:
	friend class cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // namespace ccalix

CCALIX_EXTERN_INSTANTIATE_CEREAL_SERIALIZE(ccalix::HagenCalibOptions)
