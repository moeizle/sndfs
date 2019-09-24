#include "ObbObbIntersection.h"

// code taken from: http://www.geometrictools.com/Documentation/MethodOfSeparatingAxes.pdf

ObbObbIntersection::ObbObbIntersection()
{
}


ObbObbIntersection::~ObbObbIntersection()
{
}

bool ObbObbIntersection::TestIntersection3D(std::vector<glm::vec3> C0, std::vector<glm::vec3> C1)
{
	std::vector<glm::vec3> N0, N1;
	std::vector<int> F0, F1;
	std::vector<std::pair<int, int>> E0, E1;
	computeMesh(C0, N0,F0,E0);
	computeMesh(C1, N1,F1,E1);
	glm::vec3 n,a,b;
	int side0, side1;

	for (int i = 0; i < F0.size(); i++)
	{
		if (WhichSide(C1,N0[i],C0[F0[i]])>0)
		{
			return false;
		}
	}

	for (int i = 0; i < F1.size(); i++)
	{
		if (WhichSide(C0,N1[i],C1[F1[i]])>0)
		{
			return false;
		}
	}

	for (int i = 0; i < E0.size(); i++)
	{
		a = C0[E0[i].second] - C0[E0[i].first];
		for (int j = 0; j < E1.size(); j++)
		{
			b = C1[E1[j].second] - C1[E1[j].first];
			n = glm::cross(a,b);
			n = glm::normalize(n);

			side0 = WhichSide(C0, n, C0[E0[i].first]);
			if (side0 == 0)
			{
				continue;
			}

			side1 = WhichSide(C1, n, C0[E0[i].first]);
			if ( side1== 0)
			{
				continue;
			}

			if (side0*side1 < 0)
				return false;
		}
	}

	return true;
}

int ObbObbIntersection::WhichSide(std::vector<glm::vec3> S, glm::vec3 D, glm::vec3 P)
{
	// S v e r t i c e s a r e p r o j e c t e d t o t h e fo rm P+t ∗D. Re t u r n v a l u e i s +1 i f a l l t > 0 ,
	// −1 i f a l l t < 0 , 0 o t h e r w i s e , i n which c a s e t h e l i n e s p l i t s t h e p o l y g o n .
	int positive = 0, negative= 0;
	float t;

	for(int i = 0; i < S.size(); i++)
	{
		t = glm::dot(D, S[i]-P);
		if(t > 0) 
			positive++; 
		else if(t < 0) 
			negative++;
		if(positive && negative) 
			return 0;
	}
	
	return(positive ? +1 : -1);
}

void ObbObbIntersection::computeMesh(std::vector<glm::vec3> obbA, std::vector<glm::vec3>& nA, std::vector<int>& fA, std::vector<std::pair<int, int>>& eA)
{
	glm::vec3 n, a, b;

	nA.clear();
	fA.clear();
	eA.clear();

	a = obbA[1] - obbA[0];
	b = obbA[3] - obbA[0];
	n = glm::cross(a, b);
	n=glm::normalize(n);
	nA.push_back(n);
	fA.push_back(0);
	

	a = obbA[4] - obbA[1];
	b = obbA[2] - obbA[1];
	n = glm::cross(a, b);
	n = glm::normalize(n);
	nA.push_back(n);
	fA.push_back(1);

	a = obbA[7] - obbA[4];
	b = obbA[5] - obbA[4];
	n = glm::cross(a, b);
	n = glm::normalize(n);
	nA.push_back(n);
	fA.push_back(4);

	a = obbA[0] - obbA[7];
	b = obbA[6] - obbA[7];
	n = glm::cross(a, b);
	n = glm::normalize(n);
	nA.push_back(n);
	fA.push_back(7);

	a = obbA[5] - obbA[2];
	b = obbA[3] - obbA[2];
	n = glm::cross(a, b);
	n = glm::normalize(n);
	nA.push_back(n);
	fA.push_back(2);

	a = obbA[0] - obbA[1];
	b = obbA[4] - obbA[1];
	n = glm::cross(a, b);
	n = glm::normalize(n);
	nA.push_back(n);
	fA.push_back(1);


	eA.push_back(std::make_pair(0, 1));
	eA.push_back(std::make_pair(1, 2));
	eA.push_back(std::make_pair(2, 3));
	eA.push_back(std::make_pair(3, 0));

	eA.push_back(std::make_pair(1, 4));
	eA.push_back(std::make_pair(4, 5));
	eA.push_back(std::make_pair(5, 2));
	eA.push_back(std::make_pair(2, 1));

	eA.push_back(std::make_pair(4, 7));
	eA.push_back(std::make_pair(7, 6));
	eA.push_back(std::make_pair(6, 5));
	eA.push_back(std::make_pair(5, 4));

	eA.push_back(std::make_pair(0, 3));
	eA.push_back(std::make_pair(3, 6));
	eA.push_back(std::make_pair(6, 7));
	eA.push_back(std::make_pair(7, 0));

	eA.push_back(std::make_pair(3, 2));
	eA.push_back(std::make_pair(2, 5));
	eA.push_back(std::make_pair(5, 6));
	eA.push_back(std::make_pair(6, 3));

	eA.push_back(std::make_pair(0, 1));
	eA.push_back(std::make_pair(1, 4));
	eA.push_back(std::make_pair(4, 7));
	eA.push_back(std::make_pair(7, 0));
}