#pragma once
#include "ccalix/calib_options.h"
#include "ccalix/cerealization.h"
#include "halco/hicann-dls/vx/v3/capmem.h"
#include "haldls/vx/v3/capmem.h"
#include "hate/visibility.h"

namespace ccalix GENPYBIND_TAG_CCALIX {

/*
 * Set bias parameters for the STP circuitry.
 *
 * All parameters can be configured per quadrant.
 */
struct GENPYBIND(visible) STPCalibOptions : public CalibOptions
{
	typedef halco::common::
	    typed_array<haldls::vx::v3::CapMemCell::Value, halco::hicann_dls::vx::v3::CapMemBlockOnDLS>
	        ValuePerQuadrant GENPYBIND(opaque(false));

	/*
	 * Ramp current for STP pulse width modulation, in CapMem LSB.
	 */
	ValuePerQuadrant i_ramp;

	/*
	 * Voltage (STP state) where all drivers' amplitudes are equalized at, in CapMem LSB. Should be
	 * chosen between v_charge and v_recover.
	 */
	ValuePerQuadrant v_stp;

	STPCalibOptions() SYMBOL_VISIBLE;

	GENPYBIND(stringstream)
	friend std::ostream& operator<<(std::ostream& os, STPCalibOptions const& options)
	    SYMBOL_VISIBLE;

private:
	friend class cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};

} // namespace ccalix

CCALIX_EXTERN_INSTANTIATE_CEREAL_SERIALIZE(ccalix::STPCalibOptions)
