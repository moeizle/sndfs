#pragma once

#include <vector>
#include <glm/vec2.hpp>
#include <glm/gtc/matrix_transform.hpp>

class disk
{
public:
	disk();
	~disk();

	float radius;
	glm::vec3 color;
};

