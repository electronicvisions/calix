#include "ccalix/translation/cadc_on_dls.h"
#include "hate/type_index.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/polymorphic.hpp>

namespace GENPYBIND_TAG_CCALIX ccalix {

// hashes the type name
CadcOnDLS::CadcOnDLS(ChannelType const& channel) : channel(channel) {}

std::unique_ptr<ModelEntityOnDLS> CadcOnDLS::copy() const
{
	return std::make_unique<CadcOnDLS>(*this);
}

std::unique_ptr<ModelEntityOnDLS> CadcOnDLS::move()
{
	return std::make_unique<CadcOnDLS>(std::move(*this));
}

std::ostream& CadcOnDLS::print(std::ostream& os) const
{
	return os << "CADC(" << channel << ")";
}

bool CadcOnDLS::is_equal_to(ModelEntityOnDLS const& other) const
{
	return channel == static_cast<CadcOnDLS const&>(other).channel;
}

size_t CadcOnDLS::hash() const
{
	std::size_t value = std::hash<std::string>()(hate::full_name<CadcOnDLS>());
	boost::hash_combine(value, channel);
	return value;
}


}
CEREAL_REGISTER_TYPE(ccalix::CadcOnDLS);
CEREAL_REGISTER_DYNAMIC_INIT(CadcOnDLS);