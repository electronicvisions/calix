#include "ccalix/spiking/correlation_measurement.h"

#include "halco/hicann-dls/vx/v3/cadc.h"
#include "halco/hicann-dls/vx/v3/padi.h"
#include "haldls/vx/v3/padi.h"
#include "haldls/vx/v3/synapse.h"

#include <pybind11/numpy.h>


namespace ccalix::spiking::correlation_measurement {

std::vector<stadls::vx::v3::ContainerTicket> read_correlation(
    stadls::vx::v3::PlaybackProgramBuilder& builder,
    const halco::hicann_dls::vx::v3::SynapseQuadColumnOnDLS quad,
    const halco::hicann_dls::vx::v3::SynramOnDLS synram)
{
	std::vector<stadls::vx::v3::ContainerTicket> tickets;

	for (auto const row :
	     halco::common::iter_all<halco::hicann_dls::vx::v3::SynapseRowOnSynram>()) {
		// causal read
		auto const synapse_quad = halco::hicann_dls::vx::v3::SynapseQuadOnSynram(quad, row);
		auto coord = halco::hicann_dls::vx::v3::CADCSampleQuadOnSynram(
		    synapse_quad, halco::hicann_dls::vx::v3::CADCChannelType::causal,
		    halco::hicann_dls::vx::v3::CADCReadoutType::trigger_read);
		auto coord_on_synram = halco::hicann_dls::vx::v3::CADCSampleQuadOnDLS(coord, synram);

		tickets.push_back(builder.read(coord_on_synram));

		// acausal read
		coord = halco::hicann_dls::vx::v3::CADCSampleQuadOnSynram(
		    synapse_quad, halco::hicann_dls::vx::v3::CADCChannelType::acausal,
		    halco::hicann_dls::vx::v3::CADCReadoutType::buffered);
		coord_on_synram = halco::hicann_dls::vx::v3::CADCSampleQuadOnDLS(coord, synram);

		tickets.push_back(builder.read(coord_on_synram));
	}

	return tickets;
}

void reset_correlation(
    stadls::vx::v3::PlaybackProgramBuilder& builder,
    const halco::hicann_dls::vx::v3::SynapseQuadColumnOnDLS quad,
    const halco::hicann_dls::vx::v3::SynramOnDLS synram)
{
	for (auto const row :
	     halco::common::iter_all<halco::hicann_dls::vx::v3::SynapseRowOnSynram>()) {
		auto coord = halco::hicann_dls::vx::v3::CorrelationResetOnDLS(
		    halco::hicann_dls::vx::v3::SynapseQuadOnSynram(quad, row), synram);
		builder.write(coord, haldls::vx::v3::CorrelationReset());
	}
}

pybind11::array_t<uint_fast16_t> evaluate_correlation(
    std::vector<stadls::vx::v3::ContainerTicket> tickets)
{
	auto results = pybind11::array_t<uint_fast16_t>(
	    {halco::hicann_dls::vx::v3::EntryOnQuad::size,
	     halco::hicann_dls::vx::v3::SynapseRowOnSynram::size,
	     halco::hicann_dls::vx::v3::CADCChannelType::size});

	uint_fast16_t* results_ptr = static_cast<uint_fast16_t*>(results.request().ptr);

	size_t const channels_per_synapse = halco::hicann_dls::vx::v3::CADCChannelType::size;
	for (size_t ticket_id = 0; ticket_id < tickets.size(); ++ticket_id) {
		auto ticket = tickets.at(ticket_id);

		auto const& result = dynamic_cast<haldls::vx::v3::CADCSampleQuad const&>(ticket.get());
		for (auto const entry_on_quad :
		     halco::common::iter_all<halco::hicann_dls::vx::v3::EntryOnQuad>()) {
			size_t index = results.index_at(
			    entry_on_quad.toEnum(), ticket_id / channels_per_synapse,
			    ticket_id % channels_per_synapse);
			results_ptr[index] = result.get_sample(entry_on_quad);
		}
	}

	return results;
}

void send_prepulse(
    stadls::vx::v3::PlaybackProgramBuilder& builder,
    const halco::hicann_dls::vx::v3::SynramOnDLS synram,
    const haldls::vx::v3::SynapseQuad::Label address)
{
	haldls::vx::v3::PADIEvent padi_event;
	auto fire_bus = padi_event.get_fire_bus();
	for (auto coord : halco::common::iter_all<halco::hicann_dls::vx::v3::PADIBusOnPADIBusBlock>())
		fire_bus[coord] = true;
	padi_event.set_fire_bus(fire_bus);
	padi_event.set_event_address(address);

	builder.write(synram.toPADIEventOnDLS(), padi_event);
}

void send_postpulse(
    stadls::vx::v3::PlaybackProgramBuilder& builder,
    const halco::hicann_dls::vx::v3::SynapseQuadColumnOnDLS quad,
    const halco::hicann_dls::vx::v3::SynramOnDLS synram)
{
	auto coord = halco::hicann_dls::vx::v3::NeuronResetQuadOnDLS(quad, synram);
	builder.write(coord, haldls::vx::v3::NeuronResetQuad());
}


} // namespace ccalix::spiking::correlation_measurement
