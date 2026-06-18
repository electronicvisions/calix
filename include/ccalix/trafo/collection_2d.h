#pragma once

#include "ccalix/cerealization.h"
#include "ccalix/genpybind.h"
#include "ccalix/trafo/interpolating_collection.h"
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
class GENPYBIND(visible) Collection2D : public InterpolatingCollection
{
public:
	typedef dapr::Map<std::vector<FloatType>, Transformation> TransformationMap
	    GENPYBIND(opaque(false));
	typedef std::vector<TransformationMap> Transformations;

	/**
	 * Tolerance on the determinant of two direction vectors below which the lines
	 * are treated as parallel, so that no unique intersection exists.
	 *
	 * The comparison is absolute, not relative to the magnitude of the direction
	 * vectors, so the value is only meaningful for data of roughly unit scale.
	 * It is exposed so callers can see which scale the intersection search
	 * assumes.
	 */
	static constexpr FloatType parallel_determinant_tolerance = 1e-12;


	Collection2D(Transformations const& transformations) : InterpolatingCollection(transformations)
	{
	}

	~Collection2D();

	virtual BoxType get_reverse_domain() const override;

	/**
	 * Collection2D is not invertible, as it is a collection of transformations.
	 */
	virtual bool is_invertible() const override;

	/**
	 * Always false: a non-invertible transformation has an empty reverse domain,
	 * so no value lies within it.
	 */
	virtual bool in_reverse_domain(std::vector<FloatType> const& val) const override;

	virtual std::vector<FloatType> reverse_apply(
	    std::vector<FloatType> const& in,
	    OutsideDomainBehavior outside_domain_behavior =
	        OutsideDomainBehavior::THROW) const override;

	virtual std::unique_ptr<Transformation> copy() const override;

	virtual std::unique_ptr<Transformation> move() override;


protected:
	virtual std::vector<std::vector<FloatType>> interpolate(
	    std::vector<std::pair<std::vector<FloatType>, std::vector<FloatType>>> const& projections)
	    const override;

	virtual std::vector<FloatType> find_intersection(
	    std::vector<std::map<std::vector<FloatType>, std::vector<FloatType>>> projections)
	    const override;

	virtual std::ostream& print(std::ostream& os) const GENPYBIND(hidden) override;

	virtual bool is_equal_to(Transformation const& other) const override;

	/**
	 * finds the closest pair of points in the two sets
	 * used in find_intersection
	 */
	std::pair<
	    std::pair<std::vector<FloatType>, std::vector<FloatType>>,
	    std::pair<std::vector<FloatType>, std::vector<FloatType>>>
	closest_pair(std::vector<std::map<std::vector<FloatType>, std::vector<FloatType>>> const&
	                 projections) const;

	/**
	 * finds the closest position along a coordinate
	 * used in closest_pair to find closest I and V values
	 * returns the key of the closest point in the positions vector
	 */
	FloatType closest_along_coordinate(
	    std::pair<std::vector<FloatType>, std::vector<FloatType>> const& point,
	    std::vector<FloatType> positions) const;

	/**
	 * intersects two linear transformations
	 */
	std::pair<FloatType, FloatType> intersect(
	    std::pair<std::vector<std::vector<FloatType>>, std::vector<std::vector<FloatType>>> lines)
	    const;


private:
	friend class cereal::access;
	template <typename Archiver>
	void serialize(Archiver& ar)
	{
		using namespace cereal;
		ar(cereal::base_class<InterpolatingCollection>(this));
	}
};

} // namespace trafo
} // namespace ccalix