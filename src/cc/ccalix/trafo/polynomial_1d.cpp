#include "ccalix/trafo/polynomial_1d.h"
#include <cmath>


namespace GENPYBIND_TAG_CCALIX ccalix {
namespace trafo {

Polynomial1D::Polynomial1D(DataType const& coeff, DataType const& offset) :
    Polynomial(coeff, offset)
{
	BoxType domain;
	// one interval per input dimension; a 1D polynomial consumes exactly one.
	// Set here rather than in the base, where get_reverse_domain() is still pure.
	domain.push_back(IntervalType(calix_domain_min, calix_domain_max));
	set_domain(domain);
}

Polynomial1D::~Polynomial1D() {}

std::vector<FloatType> Polynomial1D::apply(
    std::vector<FloatType> const& in, OutsideDomainBehavior /*outside_domain_behavior*/) const
{
	FloatType ret = 0;
	// Iterate the coefficients, not the offsets: every exponent that has a
	// coefficient contributes, and an exponent with no offset entry is taken to
	// have offset 0. An offset entry without a matching coefficient contributes
	// nothing.
	for (auto const& pair : m_coefficients) {
		auto const offset = m_offset.find(pair.first);
		FloatType const shift = offset != m_offset.end() ? offset->second[0] : 0;
		ret += pair.second[0] * std::pow(in[0] - shift, pair.first);
	}
	return {ret};
}

bool Polynomial1D::is_invertible() const
{
	return false;
}

std::vector<FloatType> Polynomial1D::reverse_apply(
    std::vector<FloatType> const& /*in*/, OutsideDomainBehavior /*outside_domain_behavior*/) const
{
	throw std::runtime_error("Polynomial1D is not invertible.");
}

std::ostream& Polynomial1D::print(std::ostream& os) const
{
	os << "Polynomial1D(";
	for (auto const& pair : m_coefficients) {
		os << "Exponent: " << pair.first << ", Coefficients: ";
		for (auto const& coeff : pair.second) {
			os << coeff << " ";
		}
		os << "\n";
	}
	os << ")";
	return os;
}
size_t Polynomial1D::get_dimensionality() const
{
	return 1;
}

std::unique_ptr<Transformation> Polynomial1D::copy() const
{
	return std::make_unique<Polynomial1D>(*this);
}

std::unique_ptr<Transformation> Polynomial1D::move()
{
	return std::make_unique<Polynomial1D>(std::move(*this));
}

BoxType Polynomial1D::get_reverse_domain() const
{
	// Polynomial1D does not have a meaningful reverse domain
	return BoxType();
}

bool Polynomial1D::in_reverse_domain(std::vector<FloatType> const& /*val*/) const
{
	// Polynomial1D is not invertible, so its reverse domain is empty and no
	// value lies within it. The inherited implementation would instead reject
	// every query as a size mismatch against the empty domain.
	return false;
}

bool Polynomial1D::is_equal_to(Transformation const& other) const
{
	if (auto* other_poly = dynamic_cast<Polynomial1D const*>(&other)) {
		return m_coefficients == other_poly->m_coefficients && m_offset == other_poly->m_offset;
	}
	return false;
}


} // namespace trafo
} // namespace ccalix