#pragma once

#include "ccalix/cerealization.h"
#include "ccalix/genpybind.h"
#include "ccalix/trafo/polynomial.h"
#include <iosfwd>
#include <map>
#include <vector>
#include <cereal/types/array.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/vector.hpp>


namespace GENPYBIND_TAG_CCALIX ccalix {
namespace trafo {
/**
 * polynomial of form f(x)= sum_i(coeff_i * (x - offset_i) ** exp_i)
 */
class GENPYBIND(visible) Polynomial1D : public Polynomial
{
public:
	Polynomial1D() = default;
	/**
	 * set default outer domain and coefficients, set offset (optional)
	 */
	Polynomial1D(DataType const& coeff, DataType const& offset = DataType());

	virtual ~Polynomial1D();

	virtual std::vector<FloatType> apply(
	    std::vector<FloatType> const& in,
	    OutsideDomainBehavior outside_domain_behavior = OutsideDomainBehavior::CLIP) const override;

	virtual bool is_invertible() const override;

	virtual std::vector<FloatType> reverse_apply(
	    std::vector<FloatType> const& in,
	    OutsideDomainBehavior outside_domain_behavior = OutsideDomainBehavior::CLIP) const override;

	virtual std::ostream& print(std::ostream& os) const override;

	/**
	 * A 1D polynomial consumes exactly one input.
	 */
	virtual size_t get_dimensionality() const override;

	/**
	 * Always false: a non-invertible transformation has an empty reverse domain,
	 * so no value lies within it.
	 */
	virtual bool in_reverse_domain(std::vector<FloatType> const& val) const override;

	virtual std::unique_ptr<Transformation> copy() const override;
	virtual std::unique_ptr<Transformation> move() override;

protected:
	BoxType get_reverse_domain() const override;

	virtual bool is_equal_to(Transformation const& other) const override;

private:
	friend class cereal::access;
	template <typename Archiver>
	void serialize(Archiver& ar)
	{
		using namespace cereal;
		// coefficients and offset are serialized by the base class
		ar(base_class<Polynomial>(this));
	}
};

} // namespace trafo
} // namespace ccalix