#include "ccalix/trafo/interpolating_collection.h"

namespace GENPYBIND_TAG_CCALIX ccalix {
namespace trafo {

InterpolatingCollection::InterpolatingCollection(Transformations const& transformations) :
    m_transformations(transformations)
{
	/**
	 * One default interval per input dimension. apply() consumes
	 * get_dimensionality() values from each collection in turn, so the domain
	 * describes the concatenation of those inputs; it must not grow with the
	 * number of sample positions a collection happens to hold.
	 */
	size_t dimensionality = 0;
	for (auto const& transformation : m_transformations) {
		if (transformation.empty()) {
			continue;
		}
		dimensionality += transformation.begin()->second.get_dimensionality();
	}
	m_domain.resize(dimensionality, IntervalType(calix_domain_min, calix_domain_max));
}

InterpolatingCollection::~InterpolatingCollection(){};

std::vector<FloatType> InterpolatingCollection::apply(
    std::vector<FloatType> const& in, OutsideDomainBehavior /*outside_domain_behavior*/) const
{
	std::vector<std::map<std::vector<FloatType>, std::vector<FloatType>>> projections;
	std::vector<FloatType> new_in = in;
	for (auto const& transformation : m_transformations) {
		auto const& first_pair = *transformation.begin();
		size_t dim = first_pair.second.get_dimensionality();
		// project takes the first dim values of in and applies the transformation
		// this is to ensure that apply/reverse_apply gets the correct number of dimensions
		projections.push_back(
		    project(std::vector<FloatType>(new_in.begin(), new_in.begin() + dim), transformation));
		// remove the first dim values from new_in to prepare for the next transformation
		new_in.erase(new_in.begin(), new_in.begin() + dim);
	}
	return find_intersection(projections);
}

std::map<std::vector<FloatType>, std::vector<FloatType>> InterpolatingCollection::project(
    std::vector<FloatType> const& in,
    dapr::Map<std::vector<FloatType>, Transformation> const& trafo) const
{
	std::map<std::vector<FloatType>, std::vector<FloatType>> out;
	for (auto const& pair : trafo) {
		out[pair.first] = pair.second.apply(in);
	}
	return out;
}

std::ostream& InterpolatingCollection::print(std::ostream& os) const
{
	os << "InterpolatingCollection(";
	int counter = 1;
	for (auto const& trafo : get_transformations()) {
		os << "Collection " << counter++ << ":\n";
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


std::vector<std::vector<FloatType>> InterpolatingCollection::get_positions(
    int collection_index) const
{
	std::vector<std::vector<FloatType>> positions;
	if (collection_index < 0 || static_cast<size_t>(collection_index) >= m_transformations.size()) {
		throw std::out_of_range("Invalid collection index");
	}
	auto const& transformation = m_transformations[collection_index];
	for (auto const& pair : transformation) {
		positions.push_back(pair.first);
	}
	return positions;
}

InterpolatingCollection::Transformations const& InterpolatingCollection::get_transformations() const
{
	return m_transformations;
}

size_t InterpolatingCollection::get_dimensionality() const
{
	return m_transformations.empty() ? 0
	                                 : m_transformations[0].begin()->second.get_dimensionality();
}

} // namespace trafo
} // namespace ccalix
