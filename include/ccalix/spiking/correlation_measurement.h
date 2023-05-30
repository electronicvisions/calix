#pragma once

#include "ccalix/genpybind.h"
#include "halco/hicann-dls/vx/v3/synapse.h"
#include "halco/hicann-dls/vx/v3/synram.h"
#include "haldls/vx/v3/cadc.h"
#include "haldls/vx/v3/synapse.h"
#include "stadls/vx/v3/container_ticket.h"
#include "stadls/vx/v3/playback_program.h"
#include "stadls/vx/v3/playback_program_builder.h"

#include <pybind11/numpy.h>


namespace ccalix GENPYBIND_TAG_CCALIX {
namespace spiking GENPYBIND_MODULE {
namespace correlation_measurement {

/**
 * Read CADCs in given quad column.
 *
 * Returns a list of tickets for each row.
 *
 * @param builder Builder to append reads to.
 * @param quad Quad coordinate to be read.
 * @param synram Synram to be used.
 * @return List of read tickets, ordered [causal row 0, acausal row 0, causal row 1, ...]
 */
SYMBOL_VISIBLE GENPYBIND(visible) std::vector<stadls::vx::v3::ContainerTicket> read_correlation(
    stadls::vx::v3::PlaybackProgramBuilder& builder,
    const halco::hicann_dls::vx::v3::SynapseQuadColumnOnDLS quad,
    const halco::hicann_dls::vx::v3::SynramOnDLS synram);

/**
 * Reset all synapse correlations in given quad.
 *
 * @param builder Builder to append instructions to.
 * @param quad Quad column to be reset.
 * @param synram Target synram coordinate.
 */
SYMBOL_VISIBLE GENPYBIND(visible) void reset_correlation(
    stadls::vx::v3::PlaybackProgramBuilder& builder,
    const halco::hicann_dls::vx::v3::SynapseQuadColumnOnDLS quad,
    const halco::hicann_dls::vx::v3::SynramOnDLS synram);

/**
 * Evaluate correlation reads in given list of tickets.
 *
 * @param List of read tickets, as returned by the read_correlation() function.
 * @return Numpy array containing all reads. It will be
 *      shaped (4, 256, 2) for the entries in a quad, the rows,
 *      and the causal/acausal correlation.
 */
SYMBOL_VISIBLE GENPYBIND(visible) pybind11::array_t<uint_fast16_t> evaluate_correlation(
    std::vector<stadls::vx::v3::ContainerTicket> tickets);

/**
  * Send a PADI event to all drivers, i.e. an STDP prepulse to all
  * synapses.

  * @param builder Builder to append instructions to.
  * @param synram Synram coordinate of synapses to stimulate.
  * @param address Address to be sent to synapses.
 */
SYMBOL_VISIBLE GENPYBIND(visible) void send_prepulse(
    stadls::vx::v3::PlaybackProgramBuilder& builder,
    const halco::hicann_dls::vx::v3::SynramOnDLS synram,
    const haldls::vx::v3::SynapseQuad::Label address);

/**
 * Reset the given quad of neurons, sending an STDP postpulse to synapses.
 *
 * @param builder Builder to append instructions to.
 * @param quad Quad column to reset neurons in.
 * @param synram Target synram coordinate.
 */
SYMBOL_VISIBLE GENPYBIND(visible) void send_postpulse(
    stadls::vx::v3::PlaybackProgramBuilder& builder,
    const halco::hicann_dls::vx::v3::SynapseQuadColumnOnDLS quad,
    const halco::hicann_dls::vx::v3::SynramOnDLS synram);


} // namespace correlation_measurement
} // namespace spiking
} // namespace ccalix
