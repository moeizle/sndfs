#include "convexHull.h"


convexHull::convexHull()
{
}


convexHull::~convexHull()
{
}




// 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
// Returns a positive value, if OAB makes a counter-clockwise turn,
// negative for clockwise turn, and zero if the glm::vec2s are collinear.
coord2_t convexHull::cross(const glm::vec2 &O, const glm::vec2 &A, const glm::vec2 &B)
{
	return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
}

// Returns a list of glm::vec2s on the convex hull in counter-clockwise order.
// Note: the last glm::vec2 in the returned list is the same as the first one.
vector<glm::vec2> convexHull::convex_hull(vector<glm::vec2> P)
{
	int n = P.size(), k = 0;
	if (n == 1) 
		return P;
	
	vector<glm::vec2> H(2 * n);

	// Sort points lexicographically
	std::sort(P.begin(), P.end(), [](glm::vec2 &left, glm::vec2 &right)
	{
		return left.x < right.x || (left.x == right.x && left.y < right.y);
	});

	// Build lower hull
	for (int i = 0; i < n; ++i) 
	{
		while (k >= 2 && cross(H[k - 2], H[k - 1], P[i]) <= 0) k--;
		H[k++] = P[i];
	}

	// Build upper hull
	for (int i = n - 2, t = k + 1; i >= 0; i--) 
	{
		while (k >= t && cross(H[k - 2], H[k - 1], P[i]) <= 0) k--;
		H[k++] = P[i];
	}

	H.resize(k - 1);
	return H;
}