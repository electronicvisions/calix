#pragma once

#include "ccalix/genpybind.h"
#include "halco/hicann-dls/vx/v3/capmem.h"
#include "stadls/vx/v3/dumper.h"
#include "stadls/vx/v3/playback_program_builder.h"
#include <pybind11/numpy.h>


namespace ccalix GENPYBIND_TAG_CCALIX {
namespace helpers GENPYBIND_MODULE {

/**
 * Attach write commands for a single, full row of non-global capmem cells to a given builder.
 *
 * @tparam builder_t Type of the builder in use
 * @param builder Builder writes are appended to
 * @param row Capmem row to be written
 * @param values Array of payload data to be written to the given capmem row
 */
template <typename builder_t>
GENPYBIND(visible)
void write_capmem_row(
    builder_t& builder,
    const halco::hicann_dls::vx::v3::CapMemRowOnCapMemBlock row,
    const pybind11::array_t<uint_fast16_t>& values);

extern template SYMBOL_VISIBLE void write_capmem_row(
    stadls::vx::v3::PlaybackProgramBuilder&,
    const halco::hicann_dls::vx::v3::CapMemRowOnCapMemBlock,
    const pybind11::array_t<uint_fast16_t>&);

extern template SYMBOL_VISIBLE void write_capmem_row(
    stadls::vx::v3::PlaybackProgramBuilderDumper&,
    const halco::hicann_dls::vx::v3::CapMemRowOnCapMemBlock,
    const pybind11::array_t<uint_fast16_t>&);

} // namespace helpers
} // namespace ccalix
