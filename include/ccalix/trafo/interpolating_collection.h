#pragma once

#include "ccalix/cerealization.h"
#include "ccalix/genpybind.h"
#include "ccalix/trafo/linear.h"
#include "ccalix/trafo/transformation.h"
#include "dapr/map.h"
#include <iosfwd>
#include <map>
#include <vector>
#include <cereal/types/array.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/vector.hpp>


namespace GENPYBIND_TAG_CCALIX ccalix {
namespace trafo {
/**
 * Set of transformations of dimensionality n->m, positioned in a space of dimensionality n+1
 */
class GENPYBIND(visible) InterpolatingCollection : public Transformation
{
public:
	/**
	 * vector of (discrete) 2d transformations
	 * key denotes the position in n-dimensional space, value is the transformation
	 * implementation is possible for arbitrary dimensionality
	 */
	typedef dapr::Map<std::vector<FloatType>, Transformation> TransformationMap
	    GENPYBIND(opaque(false));
	typedef std::vector<TransformationMap> Transformations;


	InterpolatingCollection() = default;
	/**
	 * set default outer domain and coefficients, set offset (optional)
	 */
	InterpolatingCollection(Transformations const& transformations);

	virtual ~InterpolatingCollection();
	/**
	 * projects intersection at given output to n-1 dimensions space and finds intersection
	 * between projections (using interpolation) to yield a vector of calibrated parameter
	 * values
	 */
	virtual std::vector<FloatType> apply(
	    std::vector<FloatType> const& in,
	    OutsideDomainBehavior outside_domain_behavior = OutsideDomainBehavior::CLIP) const override;

	/**
	 * input is index of collection, returns positions of all transformations in the collection
	 */
	std::vector<std::vector<FloatType>> get_positions(int collection_index) const;

	Transformations const& get_transformations() const;

	virtual size_t get_dimensionality() const override;

protected:
	/**
	 * provides default impl. but may want to override for some cases
	 */
	virtual std::map<std::vector<FloatType>, std::vector<FloatType>> project(
	    std::vector<FloatType> const& in,
	    dapr::Map<std::vector<FloatType>, Transformation> const& trafo) const;

	virtual std::vector<std::vector<FloatType>> interpolate(
	    std::vector<std::pair<std::vector<FloatType>, std::vector<FloatType>>> const& projections)
	    const = 0;

	virtual std::vector<FloatType> find_intersection(
	    std::vector<std::map<std::vector<FloatType>, std::vector<FloatType>>> projections)
	    const = 0;

	std::ostream& print(std::ostream& os) const GENPYBIND(hidden) override;

private:
	Transformations m_transformations;
	friend class cereal::access;
	template <typename Archiver>
	void serialize(Archiver& ar)
	{
		using namespace cereal;
		// serialize base class + transformations
		ar(m_transformations);
	}
};

} // namespace trafo
} // namespace ccalix