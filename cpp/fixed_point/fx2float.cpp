#include <iostream>
#include <bitset>

float fixedPointToFloat(
	int64_t &fixedPoint,
	const uint32_t &integerBit,
	const uint32_t &decimalBit,
	const bool &bSigned)
{
	
	int32_t signBit = 0;
	if (bSigned)
	{
		signBit = (fixedPoint >> (integerBit + decimalBit)) & 1;
		if (signBit)
		{
			fixedPoint = (~fixedPoint + 1) & (((int64_t)1 << (integerBit + decimalBit)) - 1);
		}
	}

	
	return signBit ? fixedPoint / -static_cast<float>(1 << decimalBit) : fixedPoint / static_cast<float>(1 << decimalBit);
}

int main()
{	

	// uint64_t test = 0x000000000001b332; // s2.14
	// uint64_t test = 0x0000000000006666; // s1.15
	int64_t test = 0x00000001E00A6894; // s1.15
	// 1DD604A62
	// 1E00A6894
	float res = fixedPointToFloat(test, 12, 25, true);
	std::cout << res << " ====================\n";

	int ret = system("pause");
	return 0;
}
