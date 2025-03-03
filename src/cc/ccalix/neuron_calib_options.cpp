#include "ccalix/neuron_calib_options.h"

#include "ccalix/cerealization.tcc"
#include "cereal/types/halco/common/geometry.h"
#include <ostream>
#include <cereal/types/optional.hpp>

namespace ccalix {

template <typename Archive>
void NeuronCalibOptions::serialize(Archive& ar, std::uint32_t const)
{
	ar(CEREAL_NVP(readout_neuron));
}

std::ostream& operator<<(std::ostream& os, NeuronCalibOptions const& options)
{
	os << "CADCCalibOptions(readout_neuron: ";
	if (options.readout_neuron) {
		os << *options.readout_neuron;
	} else {
		os << "None";
	}
	os << ")";
	return os;
}

} // namespace ccalix

EXPLICIT_INSTANTIATE_CEREAL_SERIALIZE(ccalix::NeuronCalibOptions)
