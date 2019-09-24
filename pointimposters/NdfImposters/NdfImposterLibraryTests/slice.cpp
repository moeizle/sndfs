#include "slice.h"


slice::slice()
{
}


slice::~slice()
{
}

bool slice::intersects(std::vector<slice>& slices, float angleIncrement, float radiusIncrement, int sliceIndx)
{
	//first check if modification will cause intersection
	float side1 = angle + angleIncrement - 0.5*(radius+radiusIncrement);
	float side2 = angle + angleIncrement + 0.5*(radius+radiusIncrement);

	side1 = std::fmod(side1, (2 * 3.14f));
	side2 = std::fmod(side2, (2 * 3.14f));

	if (side1 < 0)
		side1 = side1 + 2 * 3.14;
	if (side2 < 0)
		side2 = side2 + 2 * 3.14;

	float s1, s2;
	for (int i = 0; i < slices.size(); i++)
	{
		if (i != sliceIndx)
		{
			s1 = slices[i].angle - 0.5*slices[i].radius;
			s2 = slices[i].angle + 0.5*slices[i].radius;

			//check if side1 or side2 are between s1 and s2
			if ((side1 <= s2 && side1 >= s1) || (side2 <= s2 && side2 >= s1))
			{
				return true;

			}
		}
	}

	return false;
}