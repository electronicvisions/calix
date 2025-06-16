#include "ccalix/trafo/transformation.h"
#include <cmath>
#include <cstddef>
#include <log4cxx/logger.h>


namespace GENPYBIND_TAG_CCALIX ccalix {
namespace trafo {

Transformation::~Transformation() {}

Transformation::Transformation()
{
	// default domain is implemented in the child classes, because it varies depending on the trafo
}

void Transformation::set_domain(BoxType const& domain)
{
	// Create a multi-dimensional box type domain
	m_domain = domain;
	// get_reverse_domain function depends on transformation and is therefore abstract
	m_reverse_domain = get_reverse_domain();
}

bool Transformation::in_domain(std::vector<FloatType> const& val) const
{
	if (val.size() != m_domain.size()) {
		throw std::invalid_argument("Value vector size must match domain boundaries.");
	}
	for (size_t i = 0; i < val.size(); ++i) {
		if (val[i] < m_domain[i].get_lower() || val[i] > m_domain[i].get_upper()) {
			return false;
		}
	}
	return true;
}

bool Transformation::in_reverse_domain(std::vector<FloatType> const& val) const
{
	if (val.size() != m_reverse_domain.size()) {
		throw std::invalid_argument("Value vector size must match reverse domain boundaries.");
	}
	for (size_t i = 0; i < val.size(); ++i) {
		if (val[i] < m_reverse_domain[i].get_lower() || val[i] > m_reverse_domain[i].get_upper()) {
			return false;
		}
	}
	return true;
}

BoxType Transformation::get_domain() const
{
	return m_domain;
}


std::vector<FloatType> Transformation::respect_domain(
    std::vector<FloatType> const& val, OutsideDomainBehavior outside_domain_behavior) const
{
	if (!in_domain(val)) {
		return respect_domain_impl(val, outside_domain_behavior, m_domain);
	}
	return val; // if in domain, return the value as is
}

std::vector<FloatType> Transformation::respect_reverse_domain(
    std::vector<FloatType> const& val, OutsideDomainBehavior outside_domain_behavior) const
{
	// if the value is not in the reverse domain, apply the respect_domain_impl
	if (!in_reverse_domain(val)) {
		return respect_domain_impl(val, outside_domain_behavior, m_reverse_domain);
	}
	return val; // if in domain, return the value as is
}

/**
 * Does not support non BoxType domains.
 */
std::vector<FloatType> Transformation::respect_domain_impl(
    std::vector<FloatType> const& val,
    OutsideDomainBehavior outside_domain_behavior,
    BoxType const& domain) const
{
	std::vector<FloatType> result = val;
	for (size_t i = 0; i < val.size(); ++i) {
		switch (outside_domain_behavior) {
			// CLIP clips only those values which are outside the domain
			// NOT the whole vector
			case OutsideDomainBehavior::CLIP: {
				if (val[i] < domain[i].get_lower()) {
					result[i] = domain[i].get_lower();
				} else if (val[i] > domain[i].get_upper()) {
					result[i] = domain[i].get_upper();
				}
				break;
			}
			case OutsideDomainBehavior::THROW: {
				if (val[i] < domain[i].get_lower() || val[i] > domain[i].get_upper()) {
					throw std::out_of_range("Value is outside the domain.");
				}
				break;
			}
			case OutsideDomainBehavior::IGNORE: {
				// Do nothing, ignore the value
				break;
			}
			default:
				throw std::logic_error("OutsideDomainBehavior not implemented.");
		}
	}
	return result;
}

} // namespace trafo
} // namespace ccalix
