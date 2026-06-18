#include "ccalix/trafo/linear.h"
#include "ccalix/translation/cadc_calib_param.h"
#include "ccalix/translation/complete_parameter_translation.h"
#include "ccalix/translation/madc_calib_param.h"
#include "ccalix/translation/model_cadc.h"
#include "ccalix/translation/model_madc.h"
#include "dapr/empty_property.h"
#include "dapr/empty_property.tcc"
#include <sstream>
#include <cereal/archives/binary.hpp>
#include <cereal/types/polymorphic.hpp>
#include <gtest/gtest.h>

TEST(CompleteParameterTranslation, IntegrationTest)
{
	using namespace ccalix;

	ParameterTranslation::ModelParameters targets;
	ParameterTranslation::CalibratedParameters calibrated_parameters;

	// ------------- CADC ----------------
	trafo::Linear::data_type coeff_cadc1 = {{{2.0}, {3.0}}}; // y = 2x + 3
	trafo::Linear::data_type coeff_cadc2 = {{{4.0}, {5.0}}}; // y = 4x + 5
	trafo::Linear linearTrafoCADC1(coeff_cadc1);             // CADC channel 1 trafo
	trafo::Linear linearTrafoCADC2(coeff_cadc2);             // CADC channel 2 trafo

	// CADC channel 1
	halco::hicann_dls::vx::CADCChannelConfigOnDLS::enum_type e1(1);
	halco::hicann_dls::vx::CADCChannelConfigOnDLS cadc_config1(e1);

	// CADC channel 2
	halco::hicann_dls::vx::CADCChannelConfigOnDLS::enum_type e2(2);
	halco::hicann_dls::vx::CADCChannelConfigOnDLS cadc_config2(e2);

	// create targets
	targets.set(CadcOnDLS(cadc_config1), ModelCadc()); // set target for channel 1
	targets.set(CadcOnDLS(cadc_config2), ModelCadc()); // set target for channel 2

	// create translations
	CadcTranslation translation1(CadcOnDLS(cadc_config1), linearTrafoCADC1);
	CadcTranslation translation2(CadcOnDLS(cadc_config2), linearTrafoCADC2);

	//----------- MADC ----------------

	trafo::Linear::data_type coeff_madc1 = {{{1.0}, {2.0}}}; // y = 1x + 2
	trafo::Linear::data_type coeff_madc2 = {{{3.0}, {4.0}}}; // y = 3x + 4
	trafo::Linear linearTrafoMADC1(coeff_madc1);             // MADC
	trafo::Linear linearTrafoMADC2(coeff_madc2);             // MADC

	// MADC channel 1
	halco::hicann_dls::vx::SourceMultiplexerOnReadoutSourceSelection madc_config1(0);
	halco::hicann_dls::vx::SourceMultiplexerOnReadoutSourceSelection madc_config2(1);

	// create targets
	targets.set(MadcOnDLS(madc_config1), ModelMadc()); // set target for MADC channel 1
	targets.set(MadcOnDLS(madc_config2), ModelMadc()); // set target for MADC channel 2

	// create translations
	MadcTranslation madc_translation1(MadcOnDLS(madc_config1), linearTrafoMADC1);
	MadcTranslation madc_translation2(MadcOnDLS(madc_config2), linearTrafoMADC2);

	//---------- CompleteParameterTranslation -----------
	CompleteParameterTranslation complete_translation(
	    {translation1, translation2},          // CADC translations
	    {madc_translation1, madc_translation2} // MADC translations
	);
	// Translate the targets
	complete_translation.translate(targets, calibrated_parameters);

	// Check the results
	EXPECT_TRUE(calibrated_parameters.contains(CadcOnDLS(cadc_config1)));
	EXPECT_TRUE(calibrated_parameters.contains(CadcOnDLS(cadc_config2)));
	EXPECT_TRUE(calibrated_parameters.contains(MadcOnDLS(madc_config1)));
	EXPECT_TRUE(calibrated_parameters.contains(MadcOnDLS(madc_config2)));

	// Check the values
	EXPECT_EQ(
	    dynamic_cast<const CadcCalibParam*>(&calibrated_parameters.get(CadcOnDLS(cadc_config1)))
	        ->translation,
	    linearTrafoCADC1);
	EXPECT_EQ(
	    dynamic_cast<const CadcCalibParam*>(&calibrated_parameters.get(CadcOnDLS(cadc_config2)))
	        ->translation,
	    linearTrafoCADC2);
	EXPECT_EQ(
	    dynamic_cast<const MadcCalibParam*>(&calibrated_parameters.get(MadcOnDLS(madc_config1)))
	        ->translation,
	    linearTrafoMADC1);
	EXPECT_EQ(
	    dynamic_cast<const MadcCalibParam*>(&calibrated_parameters.get(MadcOnDLS(madc_config2)))
	        ->translation,
	    linearTrafoMADC2);

	// check reverse transformation
	std::vector<double> input = {5};
	EXPECT_EQ(linearTrafoCADC1.reverse_apply(linearTrafoCADC1.apply(input)), input);

	// serialization
	{
		std::stringstream os;
		{
			cereal::BinaryOutputArchive archive(os);
			archive(calibrated_parameters);
		}

		ParameterTranslation::CalibratedParameters loaded_parameters;
		cereal::BinaryInputArchive input_archive(os);
		input_archive(loaded_parameters);

		EXPECT_EQ(calibrated_parameters, loaded_parameters);

		for (auto const& param : loaded_parameters) {
			std::cout << "Loaded parameter: " << param.first << " with value: " << param.second
			          << std::endl;
			EXPECT_EQ(param.second, calibrated_parameters.get(param.first));
		}
	}

	// Check if the loaded parameters match the original calibrated parameters


	// ModelParameters.set(MadcOnDLS(channel), MadcCalibParam(trafo::Linear()));
}
