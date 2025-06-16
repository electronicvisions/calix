#pragma once

#include "ccalix/cerealization.h"
#include "ccalix/genpybind.h"
#include "ccalix/trafo/transformation.h"
#include <iosfwd>
#include <vector>
#include <cereal/types/base_class.hpp>
#include <cereal/types/vector.hpp>


namespace GENPYBIND_TAG_CCALIX ccalix {
namespace trafo {

class GENPYBIND(visible) Linear : public Transformation
{
public:
	Linear() = default;
	/**
	 * Set default outer domain and coefficients.
	 */
	Linear(std::vector<FloatType> const& coefficients, FloatType constant_term);

	virtual ~Linear();

	virtual std::vector<FloatType> apply(
	    std::vector<FloatType> const& in,
	    OutsideDomainBehavior outside_domain_behavior = OutsideDomainBehavior::CLIP) const override;

	virtual bool is_invertible() const override;

	virtual std::vector<FloatType> reverse_apply(
	    std::vector<FloatType> const& in,
	    OutsideDomainBehavior outside_domain_behavior = OutsideDomainBehavior::CLIP) const override;

	virtual std::ostream& print(std::ostream& os) const override;

	virtual size_t get_dimensionality() const override;

	std::vector<FloatType> const& get_coefficients() const;
	FloatType get_constant_term() const;

	virtual std::unique_ptr<Transformation> move() override;
	virtual std::unique_ptr<Transformation> copy() const override;

protected:
	virtual bool is_equal_to(Transformation const& other) const override;

	BoxType get_reverse_domain() const override;

private:
	std::vector<FloatType> m_coefficients; // coefficients for the linear transformation
	FloatType m_constant_term{};           // constant term of the linear transformation

	friend class cereal::access;
	template <typename Archiver>
	void serialize(Archiver& ar)
	{
		using namespace cereal;
		// serialize base class + coefficients + constant term
		ar(base_class<Transformation>(this), m_coefficients, m_constant_term);
	}
};

} // namespace trafo
} // namespace ccalix