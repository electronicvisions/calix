#include <gtest/gtest.h>

#include "ccalix/range.h"
#include <stdexcept>

TEST(Range, General)
{
	EXPECT_THROW(ccalix::Range<int>(1, 0), std::invalid_argument);

	ccalix::Range<int> range(0, 1);

	EXPECT_EQ(range.lower, 0);
	EXPECT_EQ(range.upper, 1);

	EXPECT_EQ(range.to_tuple(), (std::tuple<int, int>{0, 1}));

	std::stringstream ss;
	ss << range;
	EXPECT_EQ(ss.str(), "Range(0, 1)");
}
