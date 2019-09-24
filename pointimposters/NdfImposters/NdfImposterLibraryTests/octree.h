#pragma once
#include <vector>
#include <glm/vec2.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "frustum.h"
#include <queue>
#include <iostream>
#include <deque>

class octree
{
	

	//struct edge
	//{
	//	int p_indx, c_indx;
	//	edge(int p, int c)
	//	{
	//		p_indx = p;
	//		c_indx = c;
	//	}
	//};


public:
	struct node
	{
		//std::vector<int> indicies;
		std::vector<glm::vec4> Points;
		glm::vec3 extent;
		glm::vec3 center;
		std::vector<int> children;
		int parent;
		int id;
		__int64 nodeCount;

		//constructors
		node(glm::vec3 c, glm::vec3 in_extent)
		{
			center = c;
			extent = in_extent;
			id = 0;
			parent = -1;
			nodeCount = 0;
		}
		node(glm::vec3 c, glm::vec3 in_extent, std::vector<glm::vec4>& in_points)
		{
			center = c;
			extent = in_extent;
			Points = in_points;
			id = 0;
			parent = -1;
			nodeCount = 0;
		}
	};

	struct less_than_z
	{
		inline bool operator() (const node& n1, const node& n2)
		{
			return (n1.center.z < n2.center.z);
		}
	};

	std::vector<node> nodes;
	std::vector<int> leaves;
	bool exceedMemory;

	float pixel_dim; 
	__int64 node_max_memory_consumption;

	octree(glm::vec3 c, glm::vec3 extent, int points_count, float in_pixel_dim, __int64 in_node_max_memory_consumption);
	octree();
	~octree();
	
	void init(glm::vec3 c, glm::vec3 extent, int points_count, float in_pixel_dim, __int64 in_node_max_memory_consumption);
	void init(glm::vec3 c, glm::vec3 extent, int points_count, float in_pixel_dim, __int64 in_node_max_memory_consumption, unsigned int maxLevels);
	int generate_octree(int node_indx,  std::vector<glm::vec4>& points,bool preCompute);
	int generate_octree(int node_indx, int level, int maxLevels);
	inline bool termination_condition(int node_indx);
	inline bool inside_node(glm::vec3& c, glm::vec3& extent, glm::vec4& p);
	inline void get_children(int p_indx, std::vector<int>& children);
	void get_visible_indicies(int node_indx,frustum& frustums, std::vector<glm::vec3>& points, std::vector<int>& visible_indicies);
	void get_visible_indicies_2(int node_indx, frustum& frustums, std::vector<glm::vec3>& points, std::vector<int>& visible_indicies);
	bool frustum_intersects_node(frustum frustums, node nod);
	void getPvertex(glm::vec3& p, glm::vec3 normal, glm::vec3 c, glm::vec3 extent);
	void getNvertex(glm::vec3& n, glm::vec3 normal, glm::vec3 c, glm::vec3 extent);
	void get_bf_traversal(std::vector<int>& order);
	void get_leaves();
	void populate_leaves(std::vector<glm::vec4>& points);
	void placeParticle(int node_indx, glm::vec4 p,bool preCompute);
	void sort_nodes_backToFront(std::vector<int>& nodeIndecies, glm::mat3 GlobalRotationMatrix);
	void sort_nodes_frontToBack(std::vector<int>& nodeIndecies, glm::mat3 GlobalRotationMatrix);
	void groupSmallChildren(__int64 max_node_memory_consumption);
	void allocateStorage();
	void getLeavesWithCommonAncestor(int ind, std::vector<int>& l);
	int getNodeLevel(node& n);
};

