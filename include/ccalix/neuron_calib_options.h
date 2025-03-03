#pragma once
#include "ccalix/calib_options.h"
#include "ccalix/cerealization.h"
#include "halco/hicann-dls/vx/v3/neuron.h"
#include "hate/visibility.h"
#include <optional>

namespace ccalix GENPYBIND_TAG_CCALIX {

/*
 * Further configuration parameters for neuron calibration.
 */
struct GENPYBIND(visible) NeuronCalibOptions : public CalibOptions
{
	/*
	 * Coordinate of the neuron to be connected to a readout pad, i.e. can be observed using an
	 * oscilloscope. The selected neuron is connected to the upper pad (channel 0), the lower pad
	 * (channel 1) always shows the CADC ramp of quadrant 0. When using the MADC, select halco.
	 * SourceMultiplexerOnReadoutSourceSelection(0) for the neuron and mux 1 for the CADC ramps. If
	 * None is given, the readout is not configured.
	 */
	std::optional<halco::hicann_dls::vx::v3::AtomicNeuronOnDLS> readout_neuron{std::nullopt};

	NeuronCalibOptions() = default;

	GENPYBIND(stringstream)
	friend std::ostream& operator<<(std::ostream& os, NeuronCalibOptions const& options)
	    SYMBOL_VISIBLE;

private:
	friend class cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // namespace ccalix

CCALIX_EXTERN_INSTANTIATE_CEREAL_SERIALIZE(ccalix::NeuronCalibOptions)
