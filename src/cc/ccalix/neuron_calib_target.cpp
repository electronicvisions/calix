#include "ccalix/neuron_calib_target.h"

#include "ccalix/cerealization.tcc"
#include "cereal/types/halco/common/geometry.h"
#include "cereal/types/halco/common/typed_array.h"
#include <limits>
#include <cereal/types/optional.hpp>
#include <cereal/types/variant.hpp>

namespace ccalix {

NeuronCalibTarget::UncalibratedCubaSynapticInput::UncalibratedCubaSynapticInput() :
    i_synin_gm([]() {
	    ISyninGM v;
	    for (auto& an : v) {
		    an.fill(ISyninGM::value_type::value_type(500));
	    }
	    return v;
    }())
{
}

template <typename Archive>
void NeuronCalibTarget::UncalibratedCubaSynapticInput::serialize(Archive& ar, std::uint32_t const)
{
	ar(CEREAL_NVP(i_synin_gm));
}


NeuronCalibTarget::CalibratedCubaSynapticInput::CalibratedCubaSynapticInput() :
    i_synin_gm([]() {
	    ISyninGM v;
	    v.fill(ISyninGM::value_type(500));
	    return v;
    }())
{
}

template <typename Archive>
void NeuronCalibTarget::CalibratedCubaSynapticInput::serialize(Archive& ar, std::uint32_t const)
{
	ar(CEREAL_NVP(i_synin_gm));
}


NeuronCalibTarget::CalibratedMatchingCubaSynapticInput::CalibratedMatchingCubaSynapticInput() :
    i_synin_gm(500)
{
}

template <typename Archive>
void NeuronCalibTarget::CalibratedMatchingCubaSynapticInput::serialize(
    Archive& ar, std::uint32_t const)
{
	ar(CEREAL_NVP(i_synin_gm));
}


NeuronCalibTarget::CalibratedCobaSynapticInput::CalibratedCobaSynapticInput() :
    e_coba_reversal([]() {
	    Potential v;
	    for (auto& an : v) {
		    an.fill(std::nullopt);
	    }
	    return v;
    }()),
    e_coba_reference([]() {
	    Potential v;
	    for (auto& an : v) {
		    an.fill(std::nullopt);
	    }
	    return v;
    }())
{
}

template <typename Archive>
void NeuronCalibTarget::CalibratedCobaSynapticInput::serialize(Archive& ar, std::uint32_t const)
{
	ar(CEREAL_NVP(e_coba_reversal));
	ar(CEREAL_NVP(e_coba_reference));
}


NeuronCalibTarget::NeuronCalibTarget() :
    leak([]() {
	    Potential v;
	    v.fill(Potential::value_type(80));
	    return v;
    }()),
    reset([]() {
	    Potential v;
	    v.fill(Potential::value_type(70));
	    return v;
    }()),
    threshold([]() {
	    Potential v;
	    v.fill(Potential::value_type(125));
	    return v;
    }()),
    tau_mem([]() {
	    PerNeuronTimeConstant v;
	    v.fill(PerNeuronTimeConstant::value_type(10e-6));
	    return v;
    }()),
    tau_syn([]() {
	    TauSyn v;
	    for (auto& an : v) {
		    an.fill(TauSyn::value_type::value_type(10e-6));
	    }
	    return v;
    }()),
    cuba_synin(CalibratedMatchingCubaSynapticInput()),
    coba_synin(),
    membrane_capacitance([]() {
	    MembraneCapacitance m;
	    m.fill(MembraneCapacitance::value_type(63));
	    return m;
    }()),
    refractory_time([]() {
	    PerNeuronTimeConstant v;
	    v.fill(PerNeuronTimeConstant::value_type(2e-6));
	    return v;
    }()),
    holdoff_time([]() {
	    PerNeuronTimeConstant v;
	    v.fill(PerNeuronTimeConstant::value_type(0.));
	    return v;
    }()),
    synapse_dac_bias(600),
    tau_icc(std::nullopt)
{
}

template <typename Archive>
void NeuronCalibTarget::serialize(Archive& ar, std::uint32_t const)
{
	ar(CEREAL_NVP(leak));
	ar(CEREAL_NVP(reset));
	ar(CEREAL_NVP(threshold));
	ar(CEREAL_NVP(tau_mem));
	ar(CEREAL_NVP(tau_syn));
	ar(CEREAL_NVP(cuba_synin));
	ar(CEREAL_NVP(coba_synin));
	ar(CEREAL_NVP(membrane_capacitance));
	ar(CEREAL_NVP(refractory_time));
	ar(CEREAL_NVP(holdoff_time));
	ar(CEREAL_NVP(synapse_dac_bias));
}

} // namespace ccalix

EXPLICIT_INSTANTIATE_CEREAL_SERIALIZE(ccalix::NeuronCalibTarget)
