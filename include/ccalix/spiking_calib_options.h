#pragma once
#include "ccalix/cadc_calib_options.h"
#include "ccalix/calib_options.h"
#include "ccalix/cerealization.h"
#include "ccalix/correlation_calib_options.h"
#include "ccalix/neuron_calib_options.h"
#include "ccalix/stp_calib_options.h"
#include "hate/visibility.h"
#include <iosfwd>
#include <optional>

namespace ccalix GENPYBIND_TAG_CCALIX {

/*
 * Data class containing further options for spiking calibration.
 */
struct GENPYBIND(visible) SpikingCalibOptions : public CalibOptions
{
	CADCCalibOptions cadc_options;
	NeuronCalibOptions neuron_options;
	CorrelationCalibOptions correlation_options;
	STPCalibOptions stp_options;
	std::optional<bool> refine_potentials = std::nullopt;

	/*
	 * @param cadc_options Further options for CADC calibration.
	 * @param neuron_options Further options for neuron calibration.
	 * @param correlation_options Further options for correlation calibration.
	 * @param stp_options Further options for STP calibration.
	 * @param refine_potentials Switch whether after the neuron calibration, the CADCs and neuron
	 * potentials are calibrated again. This mitigates CapMem crosstalk effects. By default,
	 * refinement is only performed if COBA mode is disabled.
	 */
	SpikingCalibOptions(
	    CADCCalibOptions cadc_options = CADCCalibOptions(),
	    NeuronCalibOptions neuron_options = NeuronCalibOptions(),
	    CorrelationCalibOptions correlation_options = CorrelationCalibOptions(),
	    STPCalibOptions stp_options = STPCalibOptions(),
	    std::optional<bool> refine_potentials = std::nullopt);

	GENPYBIND(stringstream)
	friend std::ostream& operator<<(std::ostream& os, SpikingCalibOptions const& options)
	    SYMBOL_VISIBLE;

private:
	friend class cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // namespace ccalix

CCALIX_EXTERN_INSTANTIATE_CEREAL_SERIALIZE(ccalix::SpikingCalibOptions)
