#pragma once

#include <glm/vec2.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <vector>

class region
{
public:
	region();
	~region();

	glm::vec4 color;
	int lod;
	std::vector<float> avgNDF;
	std::vector<glm::vec2> selectedPixels;
};

