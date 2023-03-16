#include "ccalix/hagen/multiplication.h"

#include "halco/hicann-dls/vx/v3/timing.h"
#include "haldls/vx/v3/timer.h"
#include <sstream>
#include <stdexcept>
#include <pybind11/numpy.h>


namespace ccalix::hagen::multiplication {

namespace detail {
haldls::vx::v3::SpikePack1ToChip prepare_event(
    const haldls::vx::v3::SynapseQuad::Label address,
    const halco::hicann_dls::vx::v3::SynapseDriverOnSynapseDriverBlock target_driver,
    const halco::hicann_dls::vx::v3::SynramOnDLS synram_coord,
    const size_t synram_selection_bit)
{
	halco::hicann_dls::vx::v3::SpikeLabel label;

	// select target PADI bus
	label.set_spl1_address(
	    halco::hicann_dls::vx::SPL1Address(target_driver.toPADIBusOnPADIBusBlock()));

	// select target synram
	label.set_neuron_label(
	    halco::hicann_dls::vx::NeuronLabel(synram_coord << synram_selection_bit));

	// select target driver on the PADI bus
	label.set_row_select_address(
	    halco::hicann_dls::vx::PADIRowSelectAddress(target_driver.toSynapseDriverOnPADIBus()));

	// set address sent to the driver (MSB + hagen activation)
	label.set_synapse_label(address);

	haldls::vx::v3::SpikePack1ToChip::labels_type labels;
	labels[0] = label;
	return haldls::vx::v3::SpikePack1ToChip(labels);
}
} // namespace detail

void send_vectors(
    stadls::vx::v3::PlaybackProgramBuilder& builder,
    const pybind11::array_t<uint_fast16_t>& vector,
    const size_t num_sends,
    const size_t wait_period,
    const halco::hicann_dls::vx::v3::SynramOnDLS synram_coord,
    const size_t synram_selection_bit)
{
	if (vector.size() != halco::hicann_dls::vx::v3::SynapseRowOnSynram::size) {
		std::stringstream ss;
		ss << "Length of vector (" << vector.size()
		   << ") does not match number of synapse rows in hemisphere ("
		   << halco::hicann_dls::vx::v3::SynapseRowOnSynram::size << ").";
		throw std::runtime_error(ss.str());
	}

	const size_t num_loops = (num_sends > 1 and wait_period <= 1) ? 1 : num_sends;
	const size_t num_copies = (num_sends > 1 and wait_period <= 1) ? num_sends : 1;

	size_t entry_counter = 0;
	builder.write(halco::hicann_dls::vx::v3::TimerOnDLS(), haldls::vx::v3::Timer());

	stadls::vx::v3::PlaybackProgramBuilder vector_builder;
	for (size_t i = 0; i < num_loops; ++i) {
		for (size_t row = 0; row < halco::hicann_dls::vx::v3::SynapseRowOnSynram::size; ++row) {
			uint_fast16_t entry = vector.at(row);
			if (entry == 0)
				continue;

			// send event on a different address in order to
			// select one of the two rows connected to a driver
			if ((row % halco::hicann_dls::vx::v3::SynapseRowOnSynapseDriver::size) == 0)
				entry += 32;

			vector_builder.write(
			    halco::hicann_dls::vx::v3::SpikePack1ToChipOnDLS(),
			    detail::prepare_event(
			        halco::hicann_dls::vx::v3::SynapseLabel(entry),
			        halco::hicann_dls::vx::v3::SynapseDriverOnSynapseDriverBlock(
			            row / halco::hicann_dls::vx::v3::SynapseRowOnSynapseDriver::size),
			        synram_coord, synram_selection_bit));

			// wait only if needed:
			if (wait_period > 1) {
				vector_builder.block_until(
				    halco::hicann_dls::vx::v3::TimerOnDLS(),
				    haldls::vx::v3::Timer::Value(wait_period * entry_counter));
			}
			entry_counter += 1;
		}
	}

	for (size_t i = 0; i < num_copies; ++i)
		builder.copy_back(vector_builder);
}

} // namespace ccalix::hagen::multiplication
