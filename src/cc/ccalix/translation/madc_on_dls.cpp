#include "ccalix/translation/madc_on_dls.h"
#include "hate/type_index.h"
#include <boost/functional/hash.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/polymorphic.hpp>

namespace GENPYBIND_TAG_CCALIX ccalix {

// need to hash type and channel together to ensure uniqueness
MadcOnDLS::MadcOnDLS(ChannelType const& channel) : channel(channel) {}

std::unique_ptr<ModelEntityOnDLS> MadcOnDLS::copy() const
{
	return std::make_unique<MadcOnDLS>(*this);
}

std::unique_ptr<ModelEntityOnDLS> MadcOnDLS::move()
{
	return std::make_unique<MadcOnDLS>(std::move(*this));
}

std::ostream& MadcOnDLS::print(std::ostream& os) const
{
	return os << "MADC(" << channel << ")";
}


bool MadcOnDLS::is_equal_to(ModelEntityOnDLS const& other) const
{
	return channel == static_cast<MadcOnDLS const&>(other).channel;
}

size_t MadcOnDLS::hash() const
{
	std::size_t value = std::hash<std::string>()(hate::full_name<MadcOnDLS>());
	boost::hash_combine(value, channel);
	return value;
}

}
CEREAL_REGISTER_TYPE(ccalix::MadcOnDLS);
CEREAL_REGISTER_POLYMORPHIC_RELATION(ccalix::ModelEntityOnDLS, ccalix::MadcOnDLS);
CEREAL_REGISTER_DYNAMIC_INIT(MadcOnDLS);
