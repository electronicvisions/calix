#pragma once

#include "ccalix/genpybind.h"
#include "halco/hicann-dls/vx/v3/synapse_driver.h"
#include "halco/hicann-dls/vx/v3/synram.h"
#include "haldls/vx/v3/event.h"
#include "haldls/vx/v3/synapse.h"
#include "stadls/vx/v3/playback_program_builder.h"
#include <pybind11/numpy.h>


namespace ccalix GENPYBIND_TAG_CCALIX {
namespace hagen GENPYBIND_MODULE {
namespace multiplication {

/**
 * Generate events for the given vector in hagen mode.
 *
 * @param builder Builder to append writes to
 * @param vector Array containing the input vector
 * @param num_sends Number of repeats of the whole vector
 * @param wait_period Wait time between two successive events
 * @param synram_coord Coordinate of synapse array to target with the events
 * @param synram_selection_bit Determines which bit in the event label selects the synram
 */
SYMBOL_VISIBLE GENPYBIND(visible) void send_vectors(
    stadls::vx::v3::PlaybackProgramBuilder& builder,
    const pybind11::array_t<uint_fast16_t>& vector,
    const size_t num_sends,
    const size_t wait_period,
    const halco::hicann_dls::vx::v3::SynramOnDLS synram_coord,
    const size_t synram_selection_bit);

namespace detail {
/**
 * Return a spike pack to chip, containing an event reaching the desired
 * synapse driver on the desired synram.
 *
 * @param address Address that is sent to the driver. The MSB reaches the synapses, the lower 5 bit
 * encode the desired activation.
 * @param target_driver Coordinate of target synapse driver.
 * @param synram_coord Coordinate of target synapse array.
 * @param synram_selection_bit Bit position that selects synapse array.
 *
 * @return Spike packet to chip.
 */
haldls::vx::v3::SpikePack1ToChip prepare_event(
    const haldls::vx::v3::SynapseQuad::Label address,
    const halco::hicann_dls::vx::v3::SynapseDriverOnSynapseDriverBlock target_driver,
    const halco::hicann_dls::vx::v3::SynramOnDLS synram_coord,
    const size_t synram_selection_bit);
} // namespace detail

} // namespace multiplication
} // namespace hagen
} // namespace ccalix
