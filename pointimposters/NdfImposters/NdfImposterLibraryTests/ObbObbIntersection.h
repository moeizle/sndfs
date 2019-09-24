#pragma once
#include<vector>

#include <glm/vec2.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

class ObbObbIntersection
{
public:
	ObbObbIntersection();
	~ObbObbIntersection();
	bool TestIntersection3D(std::vector<glm::vec3> obbA,std::vector<glm::vec3> obbB);

private:
	int WhichSide(std::vector<glm::vec3> S, glm::vec3 D, glm::vec3 P);
	void computeMesh(std::vector<glm::vec3> obbA, std::vector<glm::vec3>& nA, std::vector<int>& fA, std::vector<std::pair<int, int>>& eA);
};

