#include "ccalix/calib_target.h"

#include "ccalix/cerealization.tcc"

namespace ccalix {

template <typename Archive>
void CalibTarget::serialize(Archive&, std::uint32_t const)
{
}

} // namespace ccalix

EXPLICIT_INSTANTIATE_CEREAL_SERIALIZE(ccalix::CalibTarget)
