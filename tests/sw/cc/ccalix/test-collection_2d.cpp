#include <gtest/gtest.h>

#include "ccalix/trafo/collection_2d.h"
#include <map>
#include <utility>
#include <vector>

namespace {

using Collection2D = ccalix::trafo::Collection2D;
using ccalix::trafo::FloatType;

typedef std::map<std::vector<FloatType>, std::vector<FloatType>> Curve;

/**
 * find_intersection() is protected; expose it so the algorithm can be exercised
 * directly, without apply()/project() and their input slicing and domain
 * clipping. An empty Transformations suffices: find_intersection() never reads
 * m_transformations.
 */
struct Testable : public Collection2D
{
	Testable() : Collection2D(Transformations{}) {}
	using Collection2D::find_intersection;
};

/**
 * Build one sampled curve (position -> value), as project() would produce.
 */
Curve curve(std::vector<std::pair<FloatType, FloatType>> const& points)
{
	Curve sampled;
	for (auto const& [position, value] : points) {
		sampled[{position}] = {value};
	}
	return sampled;
}

/**
 * The two curves are intersected in a common frame (cf. interpolate()):
 * x is curve 0's position and curve 1's value, y is curve 0's value and
 * curve 1's position, i.e. curve 1 is used transposed. The result is returned
 * as {y, x}.
 */

} // namespace


TEST(Collection2DFindIntersection, InteriorCrossing)
{
	// curve 0 is y = 4 - x, curve 1 transposed is y = x - 1; they cross at
	// (2.5, 1.5), strictly inside both sampled position ranges [0, 4].
	Testable collection;
	std::vector<FloatType> const result = collection.find_intersection(
	    {curve({{0., 4.}, {1., 3.}, {2., 2.}, {3., 1.}, {4., 0.}}),
	     curve({{0., 1.}, {1., 2.}, {2., 3.}, {3., 4.}, {4., 5.}})});

	ASSERT_EQ(result.size(), 2u);
	// asymmetric on purpose: a symmetric expectation would hide a swap of the
	// two returned coordinates
	EXPECT_DOUBLE_EQ(result[0], 1.5);
	EXPECT_DOUBLE_EQ(result[1], 2.5);
}

TEST(Collection2DFindIntersection, CrossingOutsideSampledRange)
{
	// Both curves are constant: y = 13 and, transposed, x = 15. The lines do
	// cross, at (15, 13), but that is outside the sampled positions [0, 9], so
	// the algorithm declines to extrapolate.
	std::vector<std::pair<FloatType, FloatType>> points_0;
	std::vector<std::pair<FloatType, FloatType>> points_1;
	for (size_t position = 0; position < 10; ++position) {
		points_0.push_back({static_cast<FloatType>(position), 13.});
		points_1.push_back({static_cast<FloatType>(position), 15.});
	}

	Testable collection;
	EXPECT_TRUE(collection.find_intersection({curve(points_0), curve(points_1)}).empty());
}

TEST(Collection2DFindIntersection, CrossingOnSamplePoint)
{
	// y = x and, transposed, y = 4 - x cross exactly on a sampled position, at
	// (2, 2). The bounds check is inclusive, so this must still be accepted.
	Testable collection;
	std::vector<FloatType> const result = collection.find_intersection(
	    {curve({{0., 0.}, {1., 1.}, {2., 2.}, {3., 3.}, {4., 4.}}),
	     curve({{0., 4.}, {1., 3.}, {2., 2.}, {3., 1.}, {4., 0.}})});

	ASSERT_EQ(result.size(), 2u);
	EXPECT_DOUBLE_EQ(result[0], 2.);
	EXPECT_DOUBLE_EQ(result[1], 2.);
}

TEST(Collection2DFindIntersection, ParallelCurves)
{
	// y = x and, transposed, y = x - 1 are parallel and distinct, so no
	// intersection exists. An empty result is expected; in particular the
	// parallel case must not escape as an exception.
	Testable collection;
	std::vector<FloatType> result;
	ASSERT_NO_THROW(
	    result = collection.find_intersection(
	        {curve({{0., 0.}, {1., 1.}, {2., 2.}, {3., 3.}, {4., 4.}}),
	         curve({{0., 1.}, {1., 2.}, {2., 3.}, {3., 4.}, {4., 5.}})}));

	EXPECT_TRUE(result.empty());
}
