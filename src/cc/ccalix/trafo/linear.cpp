#include "ccalix/trafo/linear.h"
#include <cmath>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_poly.h>
#include <log4cxx/logger.h>

namespace GENPYBIND_TAG_CCALIX ccalix {

namespace trafo {

Linear::Linear(std::vector<FloatType> const& coefficients, FloatType constant_term) :
    m_coefficients(coefficients), m_constant_term(constant_term)
{
	BoxType domain;
	// initialize box domain with the default calix min/max values
	for (size_t i = 0; i < coefficients.size(); ++i) {
		domain.push_back((IntervalType(calix_domain_min, calix_domain_max)));
	}
	set_domain(domain);
}

Linear::~Linear() {}

/**
 * Apply the linear transformation to a vector of inputs.
 */
std::vector<FloatType> Linear::apply(
    std::vector<FloatType> const& in, OutsideDomainBehavior outside_domain_behavior) const
{
	// throw error if input size does not match coefficients size
	if (in.size() != m_coefficients.size()) {
		throw std::invalid_argument("Input size does not match transformation coefficients size.");
	}
	FloatType result = m_constant_term; // Start with the constant term
	// return scalar product of input and coefficients (+constant term)

	std::vector<FloatType> const effective_in =
	    !in_domain(in) ? respect_domain(in, outside_domain_behavior) : in;
	for (size_t i = 0; i < effective_in.size(); ++i) {
		// in case of invalid input, throw an error
		if (std::isnan(effective_in[i]) || std::isinf(effective_in[i])) {
			throw std::invalid_argument("Input contains NaN or Inf values.");
		}
		// Linear transformation always returns 1 dimensional output
		result = result + m_coefficients[i] * effective_in[i];
	}
	return respect_reverse_domain({result}, outside_domain_behavior);
}

bool Linear::is_invertible() const
{
	return Linear::get_dimensionality() == 1 &&
	       m_coefficients[0] != 0; // A linear transformation is invertible if it has exactly one
	                               // dimension and a non-zero coefficient
}

std::vector<FloatType> Linear::reverse_apply(
    std::vector<FloatType> const& in, OutsideDomainBehavior outside_domain_behavior) const
{
	if (!is_invertible()) {
		throw std::runtime_error("Linear transformation is not invertible.");
	}

	std::vector<FloatType> const effective_in =
	    !in_reverse_domain(in) ? respect_reverse_domain(in, outside_domain_behavior) : in;

	// Reverse the (1D-) linear transformation
	std::vector<FloatType> result(effective_in.size());
	return respect_reverse_domain(
	    {(effective_in[0] - m_constant_term) / m_coefficients[0]}, outside_domain_behavior);
}

std::ostream& Linear::print(std::ostream& os) const
{
	os << "Linear( ";
	os << m_constant_term; // Print the constant term
	for (auto const& coeff : m_coefficients) {
		os << " + " << coeff << "*x_" << coeff;
	}
	os << " )";
	return os;
}

std::vector<FloatType> const& Linear::get_coefficients() const
{
	return m_coefficients;
}

FloatType Linear::get_constant_term() const
{
	return m_constant_term;
}

size_t Linear::get_dimensionality() const
{
	return m_coefficients.size(); // Return the size of the coefficients vector
}

/**
 * For Linear using IntervalType would be sufficient, however keeping the type
 * consistent is required for compatibility with other trafos.
 */
BoxType Linear::get_reverse_domain() const
{
	// for Linear min: if coeff is negative -> set value max
	// if positive -> set value min
	// for Linear max the other way around
	std::vector<FloatType> min, max;
	// create a box type for the reverse domain
	BoxType rev_domain;
	for (auto const& coeff : m_coefficients) {
		if (coeff < 0) {
			min.push_back(m_domain[0].get_upper());
			max.push_back(m_domain[0].get_lower());
		} else {
			min.push_back(m_domain[0].get_lower());
			max.push_back(m_domain[0].get_upper());
		}
	}
	for (size_t i = 0; i < min.size(); ++i) {
		rev_domain.push_back((IntervalType(min[i], max[i])));
	}
	return rev_domain;
}

std::unique_ptr<Transformation> Linear::move()
{
	return std::make_unique<Linear>(std::move(*this));
}

std::unique_ptr<Transformation> Linear::copy() const
{
	return std::make_unique<Linear>(*this);
}

bool Linear::is_equal_to(Transformation const& other) const
{
	if (auto other_linear = dynamic_cast<Linear const*>(&other); other_linear != nullptr) {
		return m_coefficients == other_linear->m_coefficients &&
		       m_constant_term == other_linear->m_constant_term;
	}
	return false;
}

} // namespace trafo
} // namespace ccalix
