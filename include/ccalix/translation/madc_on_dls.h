#pragma once
#include "ccalix/cerealization.h"
#include "ccalix/genpybind.h"
#include "ccalix/translation/model_entity_on_dls.h"
#include "cereal/types/halco/common/geometry.h"
#include "halco/hicann-dls/vx/readout.h"
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>

namespace GENPYBIND_TAG_CCALIX ccalix {

struct GENPYBIND(visible) MadcOnDLS : public ModelEntityOnDLS
{
	using ChannelType = halco::hicann_dls::vx::SourceMultiplexerOnReadoutSourceSelection;


	ChannelType channel;

	MadcOnDLS() = default;

	MadcOnDLS(ChannelType const& channel);

	virtual std::unique_ptr<ModelEntityOnDLS> copy() const override;

	virtual std::unique_ptr<ModelEntityOnDLS> move() override;

	virtual std::ostream& print(std::ostream& os) const override;


protected:
	virtual bool is_equal_to(ModelEntityOnDLS const& other) const override;

	virtual size_t hash() const override;


private:
	// serialization support
	friend class cereal::access;
	template <typename Archiver>
	void serialize(Archiver& ar)
	{
		ar(cereal::base_class<ModelEntityOnDLS>(this));
		ar(channel);
	}
};

} // namespace ccalix
CEREAL_FORCE_DYNAMIC_INIT(MadcOnDLS);
