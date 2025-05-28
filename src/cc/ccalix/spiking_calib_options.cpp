#include "ccalix/spiking_calib_options.h"

#include "ccalix/cerealization.tcc"
#include <ostream>
#include <cereal/types/optional.hpp>

namespace ccalix {

SpikingCalibOptions::SpikingCalibOptions(
    CADCCalibOptions cadc_options,
    NeuronCalibOptions neuron_options,
    CorrelationCalibOptions correlation_options,
    STPCalibOptions stp_options,
    std::optional<bool> refine_potentials) :
    cadc_options(cadc_options),
    neuron_options(neuron_options),
    correlation_options(correlation_options),
    stp_options(stp_options),
    refine_potentials(refine_potentials)
{
}

template <typename Archive>
void SpikingCalibOptions::serialize(Archive& ar, std::uint32_t const)
{
	ar(CEREAL_NVP(cadc_options));
	ar(CEREAL_NVP(neuron_options));
	ar(CEREAL_NVP(correlation_options));
	ar(CEREAL_NVP(stp_options));
	ar(CEREAL_NVP(refine_potentials));
}

std::ostream& operator<<(std::ostream& os, SpikingCalibOptions const& options)
{
	os << "CADCCalibOptions(" << options.cadc_options << ", " << options.neuron_options << ", "
	   << options.correlation_options << options.stp_options << ", refine_potentials: ";
	if (options.refine_potentials) {
		os << *options.refine_potentials;
	} else {
		os << "None";
	}
	os << ")";
	return os;
}

} // namespace ccalix

EXPLICIT_INSTANTIATE_CEREAL_SERIALIZE(ccalix::SpikingCalibOptions)
