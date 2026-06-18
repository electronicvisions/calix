#include "ccalix/trafo/collection_2d.h"
#include <cmath>
#include <sstream>
#include <log4cxx/logger.h>

namespace GENPYBIND_TAG_CCALIX ccalix {
namespace trafo {

namespace {

/**
 * Get whether two lines {start_point, direction} are parallel. Parallelism is a
 * property of the direction vectors, so their determinant is tested against
 * Collection2D::parallel_determinant_tolerance.
 */
bool parallel(
    std::vector<std::vector<FloatType>> const& line_1,
    std::vector<std::vector<FloatType>> const& line_2)
{
	return std::abs(line_1[1][0] * line_2[1][1] - line_1[1][1] * line_2[1][0]) <
	       Collection2D::parallel_determinant_tolerance;
}

} // namespace

Collection2D::~Collection2D() {}


std::ostream& Collection2D::print(std::ostream& os) const
{
	os << "Collection2D(";
	for (auto const& trafo : get_transformations()) {
		for (auto const& pair : trafo) {
			os << "  Position: [";
			for (auto const& val : pair.first) {
				os << val << " ";
			}
			os << "]" << std::endl;
			os << "  Value: " << pair.second << std::endl;
		}
	}
	os << ")";
	return os;
}

std::vector<FloatType> Collection2D::find_intersection(
    std::vector<std::map<std::vector<FloatType>, std::vector<FloatType>>> projections) const
{
	auto logger = log4cxx::Logger::getLogger("ccalix.trafo.Collection2D.find_intersection()");
	// if projections is not of size 2, throw an error
	if (projections.size() != 2) {
		throw std::invalid_argument(
		    "Expected vector of size 2, got " + std::to_string(projections.size()));
	}
	// repeat until intersection is found or projections are empty
	// if projections are empty, return empty vector
	// search in order of closest neighborhood
	while (projections[0].size() > 0 && projections[1].size() > 0) {
		std::stringstream dump;
		for (auto const& proj : projections) {
			for (auto const& pair : proj) {
				dump << "\n  Position: [";
				for (auto const& val : pair.first) {
					dump << val << " ";
				}
				dump << "]";
				for (auto const& val : pair.second) {
					dump << "\n  Value: " << val;
				}
			}
		}
		LOG4CXX_DEBUG(logger, "Projections of this iteration:" << dump.str());

		// if projections are empty, return empty vector
		// find closest pair of points in the two sets
		auto const& closest = closest_pair(projections);
		// find closest points in projections[0] and projections[1]
		auto it0 = projections[0].find(closest.first.first);
		auto it1 = projections[1].find(closest.second.first);
		// check if there is a next value in projections[0] and projections[1]
		if (std::next(it0) != projections[0].end()) {
			// value after current point in projections[0]
			auto const& next_value_0 = *std::next(it0);
			// interpolate linearly between the points
			std::vector<std::vector<FloatType>> line_1 =
			    interpolate(std::vector<std::pair<std::vector<FloatType>, std::vector<FloatType>>>{
			        {{closest.first.first, closest.first.second},
			         {next_value_0.first, next_value_0.second}}});
			// intersect with next point if present
			if (std::next(it1) != projections[1].end()) {
				// value after current point in projections[1]
				auto const& next_value_1 = *std::next(it1);
				// interpolate linearly between the points
				// first and second are switched to cast to same coordinate system
				std::vector<std::vector<FloatType>> line_2 = interpolate(
				    std::vector<std::pair<std::vector<FloatType>, std::vector<FloatType>>>{
				        {{closest.second.second, closest.second.first},
				         {next_value_1.second, next_value_1.first}}});
				// if the two lines are not parallel, we can find the intersection
				if (!parallel(line_1, line_2)) {
					std::pair<FloatType, FloatType> intersection =
					    intersect(std::make_pair(line_1, line_2));
					// check if intersection is in domain of both transformations
					if (intersection.first >= closest.first.first[0] &&
					    intersection.first <= next_value_0.first[0] &&
					    intersection.second >= closest.second.first[0] &&
					    intersection.second <= next_value_1.first[0]) {
						return {intersection.second, intersection.first};
					}
				}
			}
			// intersect with previous point in projections[1] if present
			if (it1 != projections[1].begin()) {
				// if there is no next value in projections[1], use the previous value
				auto const& prev_value_1 = *std::prev(it1);
				// interpolate linearly between the points
				// first and second are switched to cast to same coordinate system
				std::vector<std::vector<FloatType>> line_2 = interpolate(
				    std::vector<std::pair<std::vector<FloatType>, std::vector<FloatType>>>{
				        {{closest.second.second, closest.second.first},
				         {prev_value_1.second, prev_value_1.first}}});
				// if the two lines are not parallel, we can find the intersection
				if (!parallel(line_1, line_2)) {
					std::pair<FloatType, FloatType> intersection =
					    intersect(std::make_pair(line_1, line_2));
					// check if intersection is in domain of both transformations
					if (intersection.first >= closest.first.first[0] &&
					    intersection.first <= next_value_0.first[0] &&
					    intersection.second >= prev_value_1.first[0] &&
					    intersection.second <= closest.second.first[0]) {
						return {intersection.second, intersection.first};
					}
				}
			}
		}
		if (it0 != projections[0].begin()) {
			// value before current point in projections[0]
			auto const& prev_value_0 = *std::prev(it0);
			// interpolate linearly between the points
			std::vector<std::vector<FloatType>> line_1 =
			    interpolate(std::vector<std::pair<std::vector<FloatType>, std::vector<FloatType>>>{
			        {{prev_value_0.first, prev_value_0.second},
			         {closest.first.first, closest.first.second}}});
			// intersect with previous point in projections[1] if present
			if (it1 != projections[1].begin()) {
				// value before current point in projections[1]
				auto const& prev_value_1 = *std::prev(it1);
				// interpolate linearly between the points
				// first and second are switched to cast to same coordinate system
				std::vector<std::vector<FloatType>> line_2 = interpolate(
				    std::vector<std::pair<std::vector<FloatType>, std::vector<FloatType>>>{
				        {{prev_value_1.second, prev_value_1.first},
				         {closest.second.second, closest.second.first}}});
				// if the two lines are not parallel, we can find the intersection
				if (!parallel(line_1, line_2)) {
					std::pair<FloatType, FloatType> intersection =
					    intersect(std::make_pair(line_1, line_2));
					// check if intersection is in domain of both transformations
					if (intersection.first >= prev_value_0.first[0] &&
					    intersection.first <= closest.first.first[0] &&
					    intersection.second >= prev_value_1.first[0] &&
					    intersection.second <= closest.second.first[0]) {
						return {intersection.second, intersection.first};
					}
				}
			}
			// intersect with next point in projections[1] if present
			if (std::next(it1) != projections[1].end()) {
				// if there is no previous value in projections[1], use the next value
				auto const& next_value_1 = *std::next(it1);
				// interpolate linearly between the points
				// first and second are switched to cast to same coordinate system
				std::vector<std::vector<FloatType>> line_2 = interpolate(
				    std::vector<std::pair<std::vector<FloatType>, std::vector<FloatType>>>{
				        {{next_value_1.second, next_value_1.first},
				         {closest.second.second, closest.second.first}}});
				// if the two lines are not parallel, we can find the intersection
				if (!parallel(line_1, line_2)) {
					std::pair<FloatType, FloatType> intersection =
					    intersect(std::make_pair(line_1, line_2));
					// check if intersection is in domain of both transformations
					if (intersection.first >= prev_value_0.first[0] &&
					    intersection.first <= closest.first.first[0] &&
					    intersection.second >= closest.second.first[0] &&
					    intersection.second <= next_value_1.first[0]) {
						return {intersection.second, intersection.first};
					}
				}
			}
		}
		// if intersection is not found, remove the closest points from the projections
		projections[0].erase(closest.first.first);
		projections[1].erase(closest.second.first);
	}

	return {}; // return empty vector if no intersection is found
}

std::pair<
    std::pair<std::vector<FloatType>, std::vector<FloatType>>,
    std::pair<std::vector<FloatType>, std::vector<FloatType>>>
Collection2D::closest_pair(
    std::vector<std::map<std::vector<FloatType>, std::vector<FloatType>>> const& projections) const
{
	// vector of pairs of points with closest I values
	std::vector<std::pair<
	    std::pair<std::vector<FloatType>, std::vector<FloatType>>,
	    std::pair<std::vector<FloatType>, std::vector<FloatType>>>>
	    closest_I;
	// vector of pairs of points with closest V values
	std::vector<std::pair<
	    std::pair<std::vector<FloatType>, std::vector<FloatType>>,
	    std::pair<std::vector<FloatType>, std::vector<FloatType>>>>
	    closest_V;
	// vector of sorted I positions
	std::vector<FloatType> I_positions;
	// vector of sorted V positions
	std::vector<FloatType> V_positions;
	for (auto& pair : projections[0]) {
		I_positions.push_back(pair.first[0]);
	}

	for (auto& pair : projections[1]) {
		V_positions.push_back(pair.first[0]);
	}
	// pairs are arranged such that point from projections[0] is first
	for (auto& point : projections[0]) {
		FloatType closest = closest_along_coordinate(point, V_positions);
		closest_I.push_back(std::make_pair(
		    point, std::make_pair(
		               std::vector<FloatType>{closest},
		               projections[1].at(std::vector<FloatType>{closest}))));
	}
	for (auto& point : projections[1]) {
		FloatType closest = closest_along_coordinate(point, I_positions);
		// switched
		closest_V.push_back(std::make_pair(
		    std::make_pair(
		        std::vector<FloatType>{closest},
		        projections[0].at(std::vector<FloatType>{closest})),
		    point));
	}
	std::pair<
	    std::pair<std::vector<FloatType>, std::vector<FloatType>>,
	    std::pair<std::vector<FloatType>, std::vector<FloatType>>>
	    best;
	// compute the distance for each pair of points in both sets and return pair with smallest
	// distance
	FloatType closest_distance = std::numeric_limits<FloatType>::max();
	for (auto& pair : closest_I) {
		FloatType distance = std::sqrt(
		    std::pow(pair.first.second[0] - pair.second.first[0], 2) +
		    std::pow(pair.first.first[0] - pair.second.second[0], 2));
		// print distance
		if (distance < closest_distance) {
			closest_distance = distance;
			best = std::make_pair(pair.first, pair.second);
			// print best
		}
	}
	for (auto& pair : closest_V) {
		FloatType distance = std::sqrt(
		    std::pow(pair.first.second[0] - pair.second.first[0], 2) +
		    std::pow(pair.first.first[0] - pair.second.second[0], 2));
		// print distance
		if (distance < closest_distance) {
			closest_distance = distance;
			best = std::make_pair(pair.first, pair.second);
		}
	}
	return best;
}

FloatType Collection2D::closest_along_coordinate(
    std::pair<std::vector<FloatType>, std::vector<FloatType>> const& point,
    std::vector<FloatType> positions) const
{
	bool end_condition = false;
	// current candidate for closest I value
	FloatType curr;
	// previous candidate for closest I value
	FloatType closest = positions.at(0);
	while (positions.size() > 1 && !end_condition) {
		// binary search closest I position to point.second
		curr = positions.at(positions.size() / 2);
		// if difference is smaller than current closest, update closest
		if (std::abs(point.second[0] - curr) < std::abs(point.second[0] - closest)) {
			closest = curr;
		}
		// if current is smaller than point.second, remove all elements larger than current
		if (curr < point.second[0]) {
			positions.erase(positions.begin() + positions.size() / 2, positions.end());
		}
		// if current is larger than point.second, remove all elements smaller than current
		else if (curr > point.second[0]) {
			positions.erase(positions.begin(), positions.begin() + positions.size() / 2 + 1);
		}
		// in the (unlikely) case that current is equal to point.second, we can stop
		else {
			end_condition = true;
		}
	}
	return closest;
}
// interpolates linearly between two points
std::vector<std::vector<FloatType>> Collection2D::interpolate(
    std::vector<std::pair<std::vector<FloatType>, std::vector<FloatType>>> const& projections) const
{
	// check if map has exactly two entries
	if (projections.size() != 2) {
		throw std::runtime_error(
		    "Expected two points for interpolation, got " + std::to_string(projections.size()));
	}
	// Extract the two points
	std::pair<std::vector<FloatType>, std::vector<FloatType>> point1 = *projections.begin();
	std::pair<std::vector<FloatType>, std::vector<FloatType>> point2 =
	    *std::next(projections.begin());
	// starting point is point1, directional vector is point2-point1
	std::vector<FloatType> direction;
	std::vector<FloatType> start_point;
	// for 2 dimensions point1.first and second are of size 1 anyway
	direction.push_back(point2.first[0] - point1.first[0]);
	direction.push_back(point2.second[0] - point1.second[0]);
	start_point.push_back(point1.first[0]);
	start_point.push_back(point1.second[0]);
	// Return the starting point and direction vector
	return {start_point, direction};
}

// intersect function is called only when two linear transformations are not parallel
std::pair<FloatType, FloatType> Collection2D::intersect(
    std::pair<std::vector<std::vector<FloatType>>, std::vector<std::vector<FloatType>>> lines) const
{
	auto const& a1 = lines.first[0];
	auto const& b1 = lines.first[1];
	auto const& a2 = lines.second[0];
	auto const& b2 = lines.second[1];

	FloatType x1 = a1[0], y1 = a1[1];
	FloatType dx1 = b1[0], dy1 = b1[1];
	FloatType x2 = a2[0], y2 = a2[1];
	FloatType dx2 = b2[0], dy2 = b2[1];

	FloatType det = dx1 * dy2 - dy1 * dx2;
	if (std::abs(det) < Collection2D::parallel_determinant_tolerance) {
		throw std::runtime_error("Lines are parallel or coincident, no unique intersection.");
	}

	FloatType rhs1 = x2 - x1;
	FloatType rhs2 = y2 - y1;

	FloatType t1 = (rhs1 * dy2 - rhs2 * dx2) / det;
	[[maybe_unused]] FloatType t2 = (dx1 * rhs2 - dy1 * rhs1) / det;

	// If you want the intersection point:
	FloatType ix = x1 + t1 * dx1;
	FloatType iy = y1 + t1 * dy1;
	return {ix, iy};
}

BoxType Collection2D::get_reverse_domain() const
{
	// return empty for now
	return BoxType();
}

bool Collection2D::is_invertible() const
{
	// Collection2D is not invertible, as it is a collection of transformations
	return false;
}

bool Collection2D::in_reverse_domain(std::vector<FloatType> const& /*val*/) const
{
	// Collection2D is not invertible, so its reverse domain is empty and no value
	// lies within it. The inherited implementation would instead reject every
	// query as a size mismatch against the empty domain, accepting only the empty
	// vector and answering true for it.
	return false;
}
std::vector<FloatType> Collection2D::reverse_apply(
    const std::vector<FloatType>& /*in*/, OutsideDomainBehavior /*outside_domain_behavior*/) const
{
	// Collection2D is not invertible, as it is a collection of transformations
	throw std::runtime_error("Collection2D is not invertible, cannot reverse apply.");
}


std::unique_ptr<Transformation> Collection2D::copy() const
{
	return std::make_unique<Collection2D>(*this);
}
std::unique_ptr<Transformation> Collection2D::move()
{
	return std::make_unique<Collection2D>(std::move(*this));
}
bool Collection2D::is_equal_to(Transformation const& other) const
{
	if (auto* collection_other = dynamic_cast<Collection2D const*>(&other)) {
		return get_transformations() == collection_other->get_transformations() &&
		       get_domain() == collection_other->get_domain();
	}
	return false;
}

} // namespace trafo
} // namespace ccalix