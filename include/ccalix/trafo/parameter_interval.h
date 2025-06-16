#pragma once
#include "ccalix/genpybind.h"
#include <iosfwd>
#include <cereal/cereal.hpp>

namespace ccalix {
template <class T>
struct ParameterInterval
{
	// Constructor
	ParameterInterval() = default;
	ParameterInterval(T lower_limit, T upper_limit);

	// Check if parameter is in Interval
	bool contains(T const& parameter) const;

	// Get for limit Values
	T const& get_lower() const;
	T const& get_upper() const;

	// Equality-Operator and Inequality-Operator
	bool operator==(ParameterInterval const& other) const = default;
	bool operator!=(ParameterInterval const& other) const = default;

private:
	T m_lower_limit{};
	T m_upper_limit{};
};

// Print Interval
template <typename T>
std::ostream& operator<<(std::ostream& os, ParameterInterval<T> const& interval);


typedef ParameterInterval<double> ParameterIntervalDouble GENPYBIND(opaque(false));


} // namespace ccalix

namespace cereal {
/**
 * Serialization support for ParameterInterval.
 */
template <class Archive, class T>
inline void serialize(Archive& ar, ccalix::ParameterInterval<T>& interval)
{
	auto lower = interval.get_lower();
	auto upper = interval.get_upper();
	ar(cereal::make_nvp("lower", lower), cereal::make_nvp("upper", upper));
	interval = ccalix::ParameterInterval<T>(lower, upper);
}

} // namespace cereal

#include "ccalix/trafo/parameter_interval.tcc"