#include "ccalix/trafo/linear.h"
#include <limits>
#include <stdexcept>
#include <gtest/gtest.h>

using Linear = ccalix::trafo::Linear;
using OutsideDomainBehavior = ccalix::trafo::Transformation::OutsideDomainBehavior;
using Interval = ccalix::ParameterInterval<double>;

TEST(LinearTransformation, Apply)
{
	std::vector<double> coeff = {2.0}; // coefficients
	double constant = 3.0;             // constant term, y = 2x + 3
	Linear lin(coeff, constant);
	auto result = lin.apply({5.0});
	ASSERT_EQ(result.size(), 1u);
	EXPECT_DOUBLE_EQ(result[0], 13.0);
}

TEST(LinearTransformation, ApplyMultidimensional)
{
	// y = 2*x0 + 3*x1 + 1
	Linear lin({2.0, 3.0}, 1.0);
	auto result = lin.apply({1.0, 2.0});
	ASSERT_EQ(result.size(), 1u);
	EXPECT_DOUBLE_EQ(result[0], 9.0);
}

TEST(LinearTransformation, ApplySizeMismatch)
{
	Linear lin({2.0, 3.0}, 1.0);
	EXPECT_THROW(lin.apply({1.0}), std::invalid_argument);
}

TEST(LinearTransformation, ApplyNaN)
{
	Linear lin({2.0}, 3.0);
	EXPECT_THROW(lin.apply({std::numeric_limits<double>::quiet_NaN()}), std::invalid_argument);
}

TEST(LinearTransformation, ApplyInf)
{
	Linear lin({2.0}, 3.0);
	EXPECT_THROW(lin.apply({std::numeric_limits<double>::infinity()}), std::invalid_argument);
}

TEST(LinearTransformation, ApplyClipAboveDomain)
{
	// y = x, domain [-10, 10]: input 15 clips to 10
	Linear lin({1.0}, 0.0);
	lin.set_domain({Interval(-10.0, 10.0)});
	auto result = lin.apply({15.0}, OutsideDomainBehavior::CLIP);
	ASSERT_EQ(result.size(), 1u);
	EXPECT_DOUBLE_EQ(result[0], 10.0);
}

TEST(LinearTransformation, ApplyClipBelowDomain)
{
	// y = x, domain [-10, 10]: input -15 clips to -10
	Linear lin({1.0}, 0.0);
	lin.set_domain({Interval(-10.0, 10.0)});
	auto result = lin.apply({-15.0}, OutsideDomainBehavior::CLIP);
	ASSERT_EQ(result.size(), 1u);
	EXPECT_DOUBLE_EQ(result[0], -10.0);
}

TEST(LinearTransformation, ApplyThrowOutsideDomain)
{
	// y = x, domain [-10, 10]: input 15 throws
	Linear lin({1.0}, 0.0);
	lin.set_domain({Interval(-10.0, 10.0)});
	EXPECT_THROW(lin.apply({15.0}, OutsideDomainBehavior::THROW), std::out_of_range);
}

TEST(LinearTransformation, ApplyIgnoreOutsideDomain)
{
	// y = x, domain [-10, 10]: input 15 used as-is
	Linear lin({1.0}, 0.0);
	lin.set_domain({Interval(-10.0, 10.0)});
	auto result = lin.apply({15.0}, OutsideDomainBehavior::IGNORE);
	ASSERT_EQ(result.size(), 1u);
	EXPECT_DOUBLE_EQ(result[0], 15.0);
}

TEST(LinearTransformation, IsInvertible1D)
{
	Linear lin({2.0}, 3.0);
	EXPECT_TRUE(lin.is_invertible());
}

TEST(LinearTransformation, IsInvertibleZeroCoeff)
{
	Linear lin({0.0}, 3.0);
	EXPECT_FALSE(lin.is_invertible());
}

TEST(LinearTransformation, IsInvertibleMultiDim)
{
	Linear lin({2.0, 3.0}, 1.0);
	EXPECT_FALSE(lin.is_invertible());
}

TEST(LinearTransformation, ReverseApplyNotInvertible)
{
	Linear lin({2.0, 3.0}, 1.0);
	EXPECT_THROW(lin.reverse_apply({9.0}), std::runtime_error);
}

TEST(LinearTransformation, GetCoefficients)
{
	Linear lin({2.0, 3.0}, 1.0);
	auto const& coeff = lin.get_coefficients();
	ASSERT_EQ(coeff.size(), 2u);
	EXPECT_DOUBLE_EQ(coeff[0], 2.0);
	EXPECT_DOUBLE_EQ(coeff[1], 3.0);
}

TEST(LinearTransformation, GetConstantTerm)
{
	Linear lin({2.0}, 5.0);
	EXPECT_DOUBLE_EQ(lin.get_constant_term(), 5.0);
}

TEST(LinearTransformation, GetDimensionality)
{
	Linear lin({2.0, 3.0, 4.0}, 1.0);
	EXPECT_EQ(lin.get_dimensionality(), 3u);
}

TEST(LinearTransformation, Copy)
{
	Linear lin({2.0, 3.0}, 1.0);
	auto copy_ptr = lin.copy();
	auto* copy_lin = dynamic_cast<Linear*>(copy_ptr.get());
	ASSERT_NE(copy_lin, nullptr);
	EXPECT_EQ(copy_lin->get_coefficients(), lin.get_coefficients());
	EXPECT_DOUBLE_EQ(copy_lin->get_constant_term(), lin.get_constant_term());
	EXPECT_EQ(copy_lin->get_dimensionality(), lin.get_dimensionality());
}