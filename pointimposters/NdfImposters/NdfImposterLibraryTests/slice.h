#pragma once
#include <vector>
#include <glm/vec2.hpp>
#include <glm/gtc/matrix_transform.hpp>

class slice
{
public:
	slice();
	~slice();
    
	float angle;
	float radius;
	glm::vec3 color;

	bool intersects(std::vector<slice>& slices, float angleIncrement, float radiusIncrement, int sliceIndx);

};

