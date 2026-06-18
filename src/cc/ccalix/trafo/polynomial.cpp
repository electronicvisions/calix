#include "ccalix/trafo/polynomial.h"


namespace GENPYBIND_TAG_CCALIX ccalix {
namespace trafo {

Polynomial::Polynomial(DataType const& coeff, DataType const& offset) :
    m_coefficients(coeff), m_offset(offset)
{
	/**
	 * The domain is deliberately NOT set here. set_domain() calls the pure
	 * virtual get_reverse_domain(), and during this constructor the dynamic
	 * type is still Polynomial, so that call would abort with "pure virtual
	 * method called". Concrete subclasses set the domain in their own
	 * constructor, where the override exists and the dimensionality is known.
	 */
}

Polynomial::~Polynomial() {}

Polynomial::DataType const& Polynomial::get_data() const
{
	return m_coefficients;
}

size_t Polynomial::get_dimensionality() const
{
	/**
	 * Every exponent carries one coefficient per dimension, so any entry's
	 * vector length gives the dimensionality
	 */
	if (m_coefficients.empty()) {
		return 0;
	}
	return m_coefficients.begin()->second.size();
}

} // namespace trafo
} // namespace ccalix
