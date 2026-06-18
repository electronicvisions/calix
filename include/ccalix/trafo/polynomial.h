#pragma once

#include "ccalix/cerealization.h"
#include "ccalix/genpybind.h"
#include "ccalix/trafo/transformation.h"
#include <iosfwd>
#include <map>
#include <vector>
#include <cereal/types/array.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/vector.hpp>


namespace GENPYBIND_TAG_CCALIX ccalix {
namespace trafo {
/**
 * Abstract base for polynomial transformations of the form
 * f(x_i) = sum_i(coeff_i * (x_i - offset_i) ** exp_i).
 *
 * Holds the shared data model only. There is no evaluation algorithm valid
 * for every polynomial, so apply(), reverse_apply(), is_invertible(),
 * get_reverse_domain(), print(), copy(), move() and is_equal_to() stay pure
 * and are implemented by concrete subclasses, each of which also fixes its
 * own input/output arity.
 */
class GENPYBIND(visible) Polynomial : public Transformation
{
public:
	/**
	 * first argument denotes exponent, second argument contains coefficients or offset for
	 * that exponent
	 */
	using DataType = std::map<FloatType, std::vector<FloatType>>;


	Polynomial() = default;
	/**
	 * set default outer domain and coefficients, set offset (optional)
	 */
	Polynomial(DataType const& coeff, DataType const& offset = DataType());

	virtual ~Polynomial();

	/**
	 * Number of input dimensions, taken as the length of a coefficient vector.
	 */
	virtual size_t get_dimensionality() const override;


	/**
	 * Get the stored coefficients.
	 */
	DataType const& get_data() const;

protected:
	DataType m_coefficients; //!< coefficients for the polynomial
	DataType m_offset;       //!< offset for the polynomial, default is zero

private:
	friend class cereal::access;
	template <typename Archiver>
	void serialize(Archiver& ar)
	{
		using namespace cereal;
		// serialize base class + coefficients (+ offset)
		ar(base_class<Transformation>(this), m_coefficients, m_offset);
	}
};

} // namespace trafo
} // namespace ccalix