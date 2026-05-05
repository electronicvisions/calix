#pragma once
#include "ccalix/cerealization.h"
#include "ccalix/genpybind.h"
#include <ostream>
#include <stdexcept>
#include <tuple>

namespace ccalix GENPYBIND_TAG_CCALIX {

template <typename T>
struct Range
{
private:
	T check_lower(T const& lower, T const& upper)
	{
		if (lower > upper) {
			throw std::invalid_argument("Range lower > upper.");
		}
		return lower;
	}

	friend class cereal::access;
	template <typename Archive>
	void serialize(Archive& ar, std::uint32_t)
	{
		ar(lower);
		ar(upper);
	}

public:
	Range(T const& lower, T const& upper) : lower(check_lower(lower, upper)), upper(upper) {}

	typedef T Value;

	Value lower;
	Value upper;

	std::tuple<Value, Value> to_tuple() const
	{
		return {lower, upper};
	}

	friend std::ostream& operator<<(std::ostream& os, Range const& value)
	{
		return os << "Range(" << value.lower << ", " << value.upper << ")";
	}
};

} // namespace ccalix
