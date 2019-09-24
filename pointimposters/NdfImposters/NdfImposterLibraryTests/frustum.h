#pragma once
#include <vector>
#include <glm/vec2.hpp>
#include <glm/gtc/matrix_transform.hpp>
class frustum
{

	

public:

	struct plane
	{
		glm::vec3 p, n;
		plane(glm::vec3 inp, glm::vec3 inn)
		{
			p = inp;
			n = inn;
		}
	};
	std::vector<plane> planes;
	frustum();
	~frustum();
};

