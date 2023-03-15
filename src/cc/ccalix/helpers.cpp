#include "ccalix/helpers.h"

#include "halco/hicann-dls/vx/v3/capmem.h"
#include "haldls/vx/v3/capmem.h"
#include <sstream>
#include <stdexcept>
#include <pybind11/numpy.h>

using namespace halco::common;
using namespace halco::hicann_dls::vx::v3;
using namespace haldls::vx::v3;
using namespace stadls::vx::v3;

namespace ccalix::helpers {

template <typename builder_t>
void write_capmem_row(
    builder_t& builder,
    const halco::hicann_dls::vx::v3::CapMemRowOnCapMemBlock row,
    const pybind11::array_t<uint_fast16_t>& values)
{
	const size_t num_values = values.size();
	const size_t num_capmem_columns = NeuronConfigOnDLS::size;

	if (num_values != num_capmem_columns) {
		std::stringstream ss;
		ss << "Number of values (" << num_values << ") does not match number of capmem columns ("
		   << num_capmem_columns << ").";
		throw std::runtime_error(ss.str());
	}

	for (auto const column : iter_all<NeuronConfigOnDLS>()) {
		const auto coord = CapMemCellOnDLS(
		    CapMemCellOnCapMemBlock(
		        column.toNeuronConfigOnNeuronConfigBlock().toCapMemColumnOnCapMemBlock(), row),
		    column.toNeuronConfigBlockOnDLS().toCapMemBlockOnDLS());
		const auto value = CapMemCell(CapMemCell::Value(values.at(column.toEnum())));
		builder.write(coord, value);
	}
};

template void write_capmem_row(
    stadls::vx::v3::PlaybackProgramBuilder&,
    const halco::hicann_dls::vx::v3::CapMemRowOnCapMemBlock,
    const pybind11::array_t<uint_fast16_t>&);

template void write_capmem_row(
    stadls::vx::v3::PlaybackProgramBuilderDumper&,
    const halco::hicann_dls::vx::v3::CapMemRowOnCapMemBlock,
    const pybind11::array_t<uint_fast16_t>&);

} // namespace ccalix::helpers
