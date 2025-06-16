#pragma once

#include "ccalix/cerealization.h"
#include "ccalix/genpybind.h"
#include "ccalix/trafo/outside_domain_exception.h"
#include "ccalix/trafo/parameter_interval.h"
#include "dapr/property.h"
#include "hate/visibility.h"
#include <iosfwd>
#include <limits>
#include <vector>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>

namespace GENPYBIND_TAG_CCALIX ccalix {
namespace trafo {

typedef double FloatType;
typedef GENPYBIND(opaque(false)) ParameterInterval<FloatType> IntervalType;
typedef std::vector<IntervalType> BoxType; // used for multidimensional domain
const FloatType calix_domain_min = -std::numeric_limits<FloatType>::min();
const FloatType calix_domain_max = std::numeric_limits<FloatType>::max();

/**
 * Base class of all n-dimensional Transformations, similar to calibtic/trafo::Transformation.
 */
class GENPYBIND(inline_base("*")) Transformation : public dapr::Property<Transformation>
{
public:
	virtual ~Transformation();

	/**
	 * Behavior when the transformation of a value outside of the domain is
	 * requested.
	 */
	enum class OutsideDomainBehavior
	{
		THROW, //!< throw an OutsideDomainException
		CLIP,  //!< clip: use the domain boundary that is closer to the value
		IGNORE //!< ignore the domain boundaries
	};

	/**
	 * Sets the domain to calix's min/max.
	 */
	Transformation();

	/**
	 * Applies the transformation to @param in.
	 * @param outside_domain_behavior determines the behavior if @param in is
	 * outside of the stored domain.
	 */
	virtual std::vector<FloatType> apply(
	    std::vector<FloatType> const& in,
	    OutsideDomainBehavior outside_domain_behavior = OutsideDomainBehavior::THROW) const = 0;

	/**
	 * Checks if the transformation is invertible.
	 */
	virtual bool is_invertible() const = 0;

	/**
	 * Applies the transformation in reverse.
	 * Need to call is_invertible() first.
	 * Heavily depends on specific transformation.
	 * Implementation is required -> throw error if not invertible.
	 */
	virtual std::vector<FloatType> reverse_apply(
	    std::vector<FloatType> const& in,
	    OutsideDomainBehavior outside_domain_behavior = OutsideDomainBehavior::THROW) const = 0;

	/**
	 * Sets the domain and also calculates and sets the reverse domain.
	 */
	void set_domain(BoxType const& domain);

	/**
	 * Checks if @param val is within @param domain.
	 * Can be overridden in the child classes to support non box type domains.
	 */
	virtual bool in_domain(std::vector<FloatType> const& val) const;
	/**
	 * Like this domain and reverse domain of different types are supported.
	 */
	virtual bool in_reverse_domain(std::vector<FloatType> const& val) const;

	/**
	 * Returns number of dimensions.
	 */
	virtual size_t get_dimensionality() const = 0;

	/**
	 * Returns domain.
	 */
	BoxType get_domain() const;

	/**
	 * Used by set_domain function to calculate the reverse domain.
	 * Depends on specific transformation.
	 */
	virtual BoxType get_reverse_domain() const = 0;

protected:
	/**
	 * Returns a value within the domain.
	 * Default implementation supports only BoxType, can be overridden.
	 */
	virtual std::vector<FloatType> respect_domain(
	    std::vector<FloatType> const& val, OutsideDomainBehavior outside_domain_behavior) const;

	/**
	 * Returns a value within the reverse domain.
	 */
	virtual std::vector<FloatType> respect_reverse_domain(
	    std::vector<FloatType> const& val, OutsideDomainBehavior outside_domain_behavior) const;

	BoxType m_domain;
	BoxType m_reverse_domain; // must also be box type in case of n->m trafo

private:
	/**
	 * Currently only supports BoxType domains, may be changed in the future.
	 */
	virtual std::vector<FloatType> respect_domain_impl(
	    std::vector<FloatType> const& val,
	    OutsideDomainBehavior outside_domain_behavior,
	    BoxType const& domain) const;
	// serialization support
	friend class cereal::access;
	template <typename Archiver>
	void serialize(Archiver& ar)
	{
		// serialize Domain
		ar(m_domain);
		ar(m_reverse_domain);
	}
};

} // namespace trafo
} // namespace ccalix