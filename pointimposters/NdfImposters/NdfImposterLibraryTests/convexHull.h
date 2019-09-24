#pragma once

// Implementation of Andrew's monotone chain 2D convex hull algorithm.
// Asymptotic complexity: O(n log n).
// Practical performance: 0.5-1.0 seconds for n=1000000 on a 1GHz machine.
#include <algorithm>
#include <vector>
#include <glm/vec2.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace std;

typedef double coord_t;         // coordinate type
typedef double coord2_t;  // must be big enough to hold 2*max(|coordinate|)^2

class convexHull
{
public:
	


	
	
	convexHull();
	~convexHull();
	coord2_t cross(const glm::vec2 &O, const glm::vec2 &A, const glm::vec2 &B);
	vector<glm::vec2> convex_hull(vector<glm::vec2> P);
};

