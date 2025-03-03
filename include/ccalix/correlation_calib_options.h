#pragma once
#include "ccalix/cadc_calib_options.h"
#include "ccalix/calib_options.h"
#include "ccalix/cerealization.h"
#include "ccalix/correlation_calib_options.h"
#include "ccalix/neuron_calib_options.h"
#include "ccalix/stp_calib_options.h"
#include "ccalix/types.h"
#include "haldls/vx/v3/synapse.h"
#include "hate/visibility.h"
#include <iosfwd>
#include <optional>

namespace ccalix GENPYBIND_TAG_CCALIX {

/*
 * Further options for correlation calibration.
 */
struct GENPYBIND(visible) CorrelationCalibOptions : public CalibOptions
{
	enum class GENPYBIND(visible) Branches
	{
		CAUSAL,
		ACAUSAL,
		BOTH
	};

	/*
	 * Correlation traces to consider during calibration. Use the Enum type CorrelationBranches to
	 * select from causal, acausal or both.
	 */
	Branches branches{Branches::BOTH};

	/*
	 * Reset voltage for the measurement capacitors. Affects the achievable amplitudes In case your
	 * desired amplitudes cannot be reached at sensible CapMem currents, consider increasing
	 * v_res_meas. However, this also increases problems observed on some synapses, like traces no
	 * longer behaving exponentially. Generally, you should set this as low as possible - we
	 * recommend some 0.9 V.
	 */
	PotentialInVolt v_res_meas{PotentialInVolt(0.9)};

	/*
	 * Reset voltage for the accumulation capacitors in each synapse. Controls the baseline
	 * measurement when the sensors' accumulation capacitors were reset and no correlated events
	 * were recorded. The baseline read should be near the upper end of the reliable range of the
	 * CADC, which is the case when v_reset is set to some 1.85 V. (There's a source follower
	 * circuit in the correlation readout.)
	 */
	PotentialInVolt v_reset{PotentialInVolt(1.85)};

	/*
	 * Decide whether individual synapses' calibration bits shall be calibrated. This requires hours
	 * of runtime and may not improve the usability significantly.
	 */
	bool calibrate_synapses{false};

	/*
	 * Priority given to time constant during individual calibration of synapses. Has to be in the
	 * range from 0 to 1, the remaining priority is given to the amplitude.
	 */
	double time_constant_priority{0.3};

	/*
	 * Amplitude calibration setting used for all synapses when calibrating CapMem bias currents.
	 * Should allow headroom for adjusting amplitudes in both directions if individual synapses are
	 * to be calibrated. Otherwise, a value of 0 is recommended. Set sensibly by default.
	 */
	haldls::vx::v3::SynapseCorrelationCalibQuad::AmpCalib default_amp_calib{0};

	/*
	 * Time constant calibration setting used for all synapses when calibrating CapMem bias
	 * currents. Should allow headroom for adjusting amplitudes in both directions if individual
	 * synapses are to be calibrated. Set sensibly by default.
	 */
	haldls::vx::v3::SynapseCorrelationCalibQuad::TimeCalib default_time_calib{0};

	CorrelationCalibOptions() = default;

	/**
	 * Check if given parameters are in a valid range.
	 */
	void check() const SYMBOL_VISIBLE;

	GENPYBIND(stringstream)
	friend std::ostream& operator<<(std::ostream& os, CorrelationCalibOptions const& options)
	    SYMBOL_VISIBLE;

private:
	friend class cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t);
};


std::ostream& operator<<(std::ostream& os, CorrelationCalibOptions::Branches const& value)
    SYMBOL_VISIBLE;

} // namespace ccalix

CCALIX_EXTERN_INSTANTIATE_CEREAL_SERIALIZE(ccalix::CorrelationCalibOptions)
