#include "octree.h"

octree::octree(glm::vec3 c, glm::vec3 extent, int points_count,float in_pixel_dim, __int64 in_node_max_memory_consumption)
{
	//getnerate an octree with the root node having the following properties
	nodes.push_back(node(c, extent));
	pixel_dim = in_pixel_dim;
	node_max_memory_consumption = in_node_max_memory_consumption;
	exceedMemory = false;
}

octree::octree()
{
	pixel_dim = 0;
	node_max_memory_consumption = 0;
	exceedMemory = false;
}
octree::~octree()
{}

void octree::init(glm::vec3 c, glm::vec3 extent, int points_count, float in_pixel_dim, __int64 in_node_max_memory_consumption)
{
	//getnerate an octree with the root node having the following properties
	nodes.push_back(node(c, extent));
	pixel_dim = in_pixel_dim;
	node_max_memory_consumption = in_node_max_memory_consumption;
	exceedMemory = false;
}

void octree::init(glm::vec3 c, glm::vec3 extent, int points_count, float in_pixel_dim, __int64 in_node_max_memory_consumption, unsigned int maxLevels)
{
	//clear tree
	nodes.clear();
	leaves.clear();
	
	//getnerate an octree with the root node having the following properties
	nodes.push_back(node(c, extent));
	pixel_dim = in_pixel_dim;
	node_max_memory_consumption = in_node_max_memory_consumption;
	exceedMemory = false;

	nodes.reserve(8^(maxLevels+1)-1);
	generate_octree(0, 0, maxLevels);
}

void octree::placeParticle(int node_indx, glm::vec4 p, bool preCompute)
{
	glm::vec3 c;
	if (nodes[node_indx].children.size() > 0)
	{
		c = nodes[node_indx].center;
		if (p.x >= c.x && p.y >= c.y && p.z >= c.z)  //1,1,1
		{
			placeParticle(nodes[node_indx].children[0], p, preCompute);
		}
		else if (p.x <= c.x && p.y >= c.y && p.z >= c.z)  //-1,1,1
		{
			placeParticle(nodes[node_indx].children[1], p, preCompute);
		}
		else if (p.x <= c.x && p.y <= c.y && p.z >= c.z) //-1,-1,1
		{
			placeParticle(nodes[node_indx].children[2], p, preCompute);
		}
		else if (p.x >= c.x && p.y <= c.y && p.z >= c.z) //1,-1,1
		{
			placeParticle(nodes[node_indx].children[3], p, preCompute);
		}
		else if (p.x >= c.x && p.y >= c.y && p.z <= c.z)  //1,1,-1
		{
			placeParticle(nodes[node_indx].children[4], p, preCompute);
		}
		else if (p.x >= c.x && p.y <= c.y && p.z <= c.z) //1,-1,-1
		{
			placeParticle(nodes[node_indx].children[5], p, preCompute);
		}
		else if (p.x <= c.x && p.y <= c.y && p.z <= c.z) //-1,-1,-1
		{
			placeParticle(nodes[node_indx].children[6], p, preCompute);
		}
		else if (p.x <= c.x && p.y >= c.y && p.z <= c.z)  //-1,1,-1
		{
			placeParticle(nodes[node_indx].children[7], p, preCompute);
		}
		
	}
	else
	{
		if (preCompute)
		{
			nodes[node_indx].nodeCount++;
			//if (__int64(nodes[node_indx].nodeCount * 4 * 4) > node_max_memory_consumption)
			//{
			//	exceedMemory = true;
			//	
			//	//add a new level at node_indx
			//	generate_octree(node_indx, 0, 2);
			//	
			//	//reset nodecounts
			//	get_leaves();
			//	for (int i = 0; i < leaves.size(); i++)
			//	{
			//		nodes[leaves[i]].nodeCount = 0;
			//	}

			//	return;
			//}
		}
		else
			nodes[node_indx].Points.push_back(p);
	}
}
void octree::groupSmallChildren(__int64 max_node_memory_consumption)
{
	bool finished = false;
	int count;
	std::vector<int> tbe;
	bool areLeaves;

	while (!finished)
	{
		finished = true;

		tbe.clear();

		for (int i = 0; i < nodes.size(); i++)
		{
			//if node has children
			if (nodes[i].children.size()>0)
			{
				areLeaves = true;
				//if children are leaves

				for (int j = 0; j < nodes[i].children.size(); j++)
				{
					if (nodes[nodes[i].children[j]].children.size()>0)
					{
						areLeaves = false;
						break;
					}
				}

				if (areLeaves)
				{
					//sum up their counters
					count = 0;
					for (int j = 0; j < nodes[i].children.size(); j++)
					{
						count += nodes[nodes[i].children[j]].nodeCount;
					}

					if (__int64(count * 4 * 4) < node_max_memory_consumption)
					{
						nodes[i].nodeCount = count;
						tbe.insert(tbe.end(), nodes[i].children.begin(), nodes[i].children.end());
						nodes[i].children.clear();
					}
				}
			}
		}

		if (tbe.size() > 0)
		{
			std::sort(tbe.begin(), tbe.end());
			for (int i = tbe.size() - 1; i >= 0; i--)
			{
				nodes.erase(nodes.begin() + tbe[i]);

				for (int j = 0; j < nodes.size(); j++)                                   //inefficient, better for children to be a vector of pointers to children nodes
				{
					for (int k = 0; k < nodes[j].children.size(); k++)
					{
						if (nodes[j].children[k] > tbe[i])
							nodes[j].children[k]--;
					}

					if (nodes[j].parent>tbe[i])
						nodes[j].parent--;
				}
			}
			finished = false;
		}

	}
}
void octree::allocateStorage()
{
	get_leaves();
	for (int i = 0; i < leaves.size(); i++)
	{
		nodes[leaves[i]].Points.reserve(nodes[leaves[i]].nodeCount);
	}
}
int octree::generate_octree(int node_indx, std::vector<glm::vec4>& points,bool preCompute)
{
	////starting with the root node containing all points
	////recursively do the following:
	////1-> if termination condition met, add all primitives of parent to current node, return
	////2-> else add all parent points within 'min' and 'max' of the current node to the node
	////3->      divide the node to 8 nodes and call generate octree recursively

	//for (int i = 0; i < points.size();i++)
	//{
	//	if (inside_node(nodes[node_indx].center, nodes[node_indx].extent, points[i]))
	//	{
	//		//nodes[node_indx].indicies.push_back(nodes[node_indx].indicies.size());
	//		if (!preCompute)
	//			nodes[node_indx].Points.push_back(points[i]);
	//		else
	//			nodes[node_indx].nodeCount++;
	//	}
	//}

	//
	//if (termination_condition(node_indx))
	//{
	//	//if (nodes[node_indx].indicies.size() > 0)
	//	//{
	//	//	//then this node is a leaf
	//	//	leaves.push_back(node_indx);

	//	//	//populate the leaf with nodes
	//	//	for (int i = 0; i < nodes[node_indx].indicies.size(); i++)
	//	//	{
	//	//		nodes[node_indx].Points.push_back(points[nodes[node_indx].indicies[i]]);
	//	//	}

	//	//	//clear indicies, they are not used anymore
	//	//	nodes[node_indx].indicies.clear();
	//	//}

	//	return 0;
	//}
	//	

	////divide the node to eight sub nodes and recursively call the operation
	//glm::vec3 new_center;
	//float new_extent = 0.5f*nodes[node_indx].extent;

	////1st cube x,y,z
	////add node
	//new_center = nodes[node_indx].center + new_extent*glm::vec3(1, 1, 1);
	//nodes.push_back(node(new_center, new_extent));
	//nodes[node_indx].children.push_back(nodes.size() - 1);
	//nodes[nodes.size() - 1].parent = node_indx;
	//nodes[nodes.size() - 1].id = nodes.size() - 1;
	////add edge
	////edges.push_back(edge(node_indx, nodes.size() - 1));

	////call the recursive function
	//generate_octree(nodes.size() - 1,  nodes[node_indx].Points);


	////2nd cube x,y,-z
	//new_center = nodes[node_indx].center + new_extent*glm::vec3(1, 1, -1);
	//nodes.push_back(node(new_center, new_extent));
	//nodes[node_indx].children.push_back(nodes.size() - 1);
	//nodes[nodes.size() - 1].parent = node_indx;
	//nodes[nodes.size() - 1].id = nodes.size() - 1;
	////add edge
	////edges.push_back(edge(node_indx, nodes.size() - 1));

	////call the recursive function
	//generate_octree(nodes.size() - 1,  nodes[node_indx].Points);

	////3rd cube x,-y,-z
	//new_center = nodes[node_indx].center + new_extent*glm::vec3(1, -1, -1);
	//nodes.push_back(node(new_center, new_extent));
	//nodes[node_indx].children.push_back(nodes.size() - 1);
	//nodes[nodes.size() - 1].parent = node_indx;
	//nodes[nodes.size() - 1].id = nodes.size() - 1;
	////add edge
	////edges.push_back(edge(node_indx, nodes.size() - 1));

	////call the recursive function
	//generate_octree(nodes.size() - 1,  nodes[node_indx].Points);

	////4th cube -x,-y,-z
	//new_center = nodes[node_indx].center + new_extent*glm::vec3(-1, -1, -1);
	//nodes.push_back(node(new_center, new_extent));
	//nodes[node_indx].children.push_back(nodes.size() - 1);
	//nodes[nodes.size() - 1].parent = node_indx;
	//nodes[nodes.size() - 1].id = nodes.size() - 1;
	////add edge
	////edges.push_back(edge(node_indx, nodes.size() - 1));

	////call the recursive function
	//generate_octree(nodes.size() - 1, nodes[node_indx].Points);

	////5th cube -x,y,z
	//new_center = nodes[node_indx].center + new_extent*glm::vec3(-1, 1, 1);
	//nodes.push_back(node(new_center, new_extent));
	//nodes[node_indx].children.push_back(nodes.size() - 1);
	//nodes[nodes.size() - 1].parent = node_indx;
	//nodes[nodes.size() - 1].id = nodes.size() - 1;
	////add edge
	////edges.push_back(edge(node_indx, nodes.size() - 1));

	////call the recursive function
	//generate_octree(nodes.size() - 1,  nodes[node_indx].Points);

	////6th cube -x,-y,z
	//new_center = nodes[node_indx].center + new_extent*glm::vec3(-1, -1, 1);
	//nodes.push_back(node(new_center, new_extent));
	//nodes[node_indx].children.push_back(nodes.size() - 1);
	//nodes[nodes.size() - 1].parent = node_indx;
	//nodes[nodes.size() - 1].id = nodes.size() - 1;
	////add edge
	////edges.push_back(edge(node_indx, nodes.size() - 1));

	////call the recursive function
	//generate_octree(nodes.size() - 1,  nodes[node_indx].Points);

	////7th cube -x,y,-z
	//new_center = nodes[node_indx].center + new_extent*glm::vec3(-1, 1, -1);
	//nodes.push_back(node(new_center, new_extent));
	//nodes[node_indx].children.push_back(nodes.size() - 1);
	//nodes[nodes.size() - 1].parent = node_indx;
	//nodes[nodes.size() - 1].id = nodes.size() - 1;
	////add edge
	////edges.push_back(edge(node_indx, nodes.size() - 1));

	////call the recursive function
	//generate_octree(nodes.size() - 1,  nodes[node_indx].Points);

	////8th cube x,-y,z
	//new_center = nodes[node_indx].center + new_extent*glm::vec3(1, -1, 1);
	//nodes.push_back(node(new_center, new_extent));
	//nodes[node_indx].children.push_back(nodes.size() - 1);
	//nodes[nodes.size() - 1].parent = node_indx;
	//nodes[nodes.size() - 1].id = nodes.size() - 1;
	////add edge
	////edges.push_back(edge(node_indx, nodes.size() - 1));

	////call the recursive function
	//generate_octree(nodes.size() - 1,  nodes[node_indx].Points);


	//if (nodes[node_indx].children.size() > 0)
	//{
	//	//nodes[node_indx].indicies.clear();
	//	nodes[node_indx].Points.clear();
	//}

	return 0;
}
int octree::generate_octree(int node_indx, int level, int maxLevels)
{
	//starting with the root node containing all points
	//recursively do the following:
	//1-> if termination condition met, add all primitives of parent to current node, return
	//2-> else add all parent points within 'min' and 'max' of the current node to the node
	//3->      divide the node to 8 nodes and call generate octree recursively



	if (level == maxLevels)
	{
		return 0;
	}

	level++;

	//divide the node to eight sub nodes and recursively call the operation
	glm::vec3 new_center;
	glm::vec3 new_extent = 0.5f*nodes[node_indx].extent;

	//1st cube x,y,z
	//add node
	new_center = nodes[node_indx].center + new_extent*glm::vec3(1, 1, 1);
	nodes.push_back(node(new_center, new_extent));
	nodes[node_indx].children.push_back(nodes.size() - 1);
	nodes[nodes.size() - 1].parent = node_indx;
	nodes[nodes.size() - 1].id = nodes.size() - 1;
	//add edge
	//edges.push_back(edge(node_indx, nodes.size() - 1));

	//call the recursive function
	generate_octree(nodes.size() - 1, level, maxLevels);

	//5th cube -x,y,z
	new_center = nodes[node_indx].center + new_extent*glm::vec3(-1, 1, 1);
	nodes.push_back(node(new_center, new_extent));
	nodes[node_indx].children.push_back(nodes.size() - 1);
	nodes[nodes.size() - 1].parent = node_indx;
	nodes[nodes.size() - 1].id = nodes.size() - 1;
	//add edge
	//edges.push_back(edge(node_indx, nodes.size() - 1));

	//call the recursive function
	generate_octree(nodes.size() - 1, level, maxLevels);

	//6th cube -x,-y,z
	new_center = nodes[node_indx].center + new_extent*glm::vec3(-1, -1, 1);
	nodes.push_back(node(new_center, new_extent));
	nodes[node_indx].children.push_back(nodes.size() - 1);
	nodes[nodes.size() - 1].parent = node_indx;
	nodes[nodes.size() - 1].id = nodes.size() - 1;
	//add edge
	//edges.push_back(edge(node_indx, nodes.size() - 1));

	//call the recursive function
	generate_octree(nodes.size() - 1, level, maxLevels);


	//8th cube x,-y,z
	new_center = nodes[node_indx].center + new_extent*glm::vec3(1, -1, 1);
	nodes.push_back(node(new_center, new_extent));
	nodes[node_indx].children.push_back(nodes.size() - 1);
	nodes[nodes.size() - 1].parent = node_indx;
	nodes[nodes.size() - 1].id = nodes.size() - 1;
	//add edge
	//edges.push_back(edge(node_indx, nodes.size() - 1));

	//call the recursive function
	generate_octree(nodes.size() - 1, level, maxLevels);

	//2nd cube x,y,-z
	new_center = nodes[node_indx].center + new_extent*glm::vec3(1, 1, -1);
	nodes.push_back(node(new_center, new_extent));
	nodes[node_indx].children.push_back(nodes.size() - 1);
	nodes[nodes.size() - 1].parent = node_indx;
	nodes[nodes.size() - 1].id = nodes.size() - 1;
	//add edge
	//edges.push_back(edge(node_indx, nodes.size() - 1));

	//call the recursive function
	generate_octree(nodes.size() - 1, level, maxLevels);

	//3rd cube x,-y,-z
	new_center = nodes[node_indx].center + new_extent*glm::vec3(1, -1, -1);
	nodes.push_back(node(new_center, new_extent));
	nodes[node_indx].children.push_back(nodes.size() - 1);
	nodes[nodes.size() - 1].parent = node_indx;
	nodes[nodes.size() - 1].id = nodes.size() - 1;
	//add edge
	//edges.push_back(edge(node_indx, nodes.size() - 1));

	//call the recursive function
	generate_octree(nodes.size() - 1, level, maxLevels);

	//4th cube -x,-y,-z
	new_center = nodes[node_indx].center + new_extent*glm::vec3(-1, -1, -1);
	nodes.push_back(node(new_center, new_extent));
	nodes[node_indx].children.push_back(nodes.size() - 1);
	nodes[nodes.size() - 1].parent = node_indx;
	nodes[nodes.size() - 1].id = nodes.size() - 1;
	//add edge
	//edges.push_back(edge(node_indx, nodes.size() - 1));

	//call the recursive function
	generate_octree(nodes.size() - 1, level, maxLevels);

	

	//7th cube -x,y,-z
	new_center = nodes[node_indx].center + new_extent*glm::vec3(-1, 1, -1);
	nodes.push_back(node(new_center, new_extent));
	nodes[node_indx].children.push_back(nodes.size() - 1);
	nodes[nodes.size() - 1].parent = node_indx;
	nodes[nodes.size() - 1].id = nodes.size() - 1;
	//add edge
	//edges.push_back(edge(node_indx, nodes.size() - 1));

	//call the recursive function
	generate_octree(nodes.size() - 1, level, maxLevels);






	return 0;
}
inline bool octree::termination_condition(int node_indx)
{
	//for now, let it just be a condition on the number of nodes
	if (nodes[node_indx].Points.size() < 1)   //cell is empty
	{
		////delete node from tree
		//{
		//	//remove it from the children of the parent
		//	for (int i = 0; i < nodes[nodes[node_indx].parent].children.size(); i++)
		//	{
		//		if (nodes[nodes[node_indx].parent].children[i] == node_indx)
		//		{
		//			nodes[nodes[node_indx].parent].children.erase(nodes[nodes[node_indx].parent].children.begin() + i);
		//			break;
		//		}
		//	}
		//	//remove the node
		//	nodes.erase(nodes.begin() + node_indx);
		//}

		return true;
	}

	//if (2.0f*nodes[node_indx].extent <= pixel_dim)             
	//{
	//	return true;
	//}

	if (__int64(nodes[node_indx].Points.size() * 4 * 4) <= node_max_memory_consumption)
	{
		return true;
	}

	return false;
}

inline bool octree::inside_node(glm::vec3& c, glm::vec3& extent, glm::vec4& p)
{
	if  (((p.x-p.w) > c.x + extent.x) || ((p.y-p.w) > c.y + extent.y) || ((p.z-p.w) > c.z + extent.z))
		return false;
	if (((p.x+p.w) < c.x - extent.x) || ((p.y+p.w) < c.y - extent.y) || ((p.z+p.w) < c.z - extent.z))
		return false;

	return true;
}

inline void octree::get_children(int p_indx, std::vector<int>&outchildren)
{
	//new
	outchildren = nodes[p_indx].children;
	return;
	//end new
	
	//outchildren.clear();
	//for (int i = 0; i < edges.size(); i++)
	//{
	//	if (edges[i].p_indx == p_indx)
	//		outchildren.push_back(edges[i].c_indx);
	//}
}

void octree::get_visible_indicies(int node_indx,frustum& frustums, std::vector<glm::vec3>& points, std::vector<int>& visible_indicies)
{
	////traverse the tree in dfs manner
	//std::vector<int> children;
	//get_children(node_indx, children);
	//bool intersects;

	////if the furstum intersects a child, 
	////then if it's a leaf, add all its points to indicies
	////else call its children

	//intersects = frustum_intersects_node(frustums, nodes[node_indx]);

	//if (intersects)
	//{
	//	if (children.size() == 0)
	//	{
	//		visible_indicies.insert(visible_indicies.end(), nodes[node_indx].indicies.begin(), nodes[node_indx].indicies.end());
	//	}
	//	else
	//	{
	//		for (int i = 0; i < children.size(); i++)
	//		{
	//			get_visible_indicies(children[i], frustums, points, visible_indicies);
	//		}
	//	}
	//}
}
void octree::get_visible_indicies_2(int node_indx, frustum& frustums, std::vector<glm::vec3>& points, std::vector<int>& visible_indicies)
{
	bool inside;
	glm::vec3 vec;

	for (int i = 0; i < points.size(); i++)
	{
		inside = true;
		for (int j = 0; j < frustums.planes.size();j++)
		{
			vec = points[i] - frustums.planes[j].p;
			if (glm::dot(vec, frustums.planes[j].n) < 0)
			{
				inside = false;
				break;
			}
		}
		if (inside)
			visible_indicies.push_back(i);
	}
}

bool octree::frustum_intersects_node(frustum frustums, node nod)
{
	float dist;
	glm::vec3 vec;
	glm::vec3 P, N;

	////with pvertex
	//for (int i = 0; i < frustums.planes.size(); i++)
	//{
	//	//debug
	//	//frustums.planes[i].n *= -1.0f;
	//	//end debug

	//	//with pvertex
	//	getPvertex(P, frustums.planes[i].n, nod.center,nod.extent);
	//	//getNvertex(N, frustums.planes[i].n, nod.center,nod.extent);
	//	vec = P - frustums.planes[i].p;
	//	if (glm::dot(vec, frustums.planes[i].n) < 0)
	//		return false;        //it is outside
	//	//vec = N - frustums.planes[i].p;
	//	//if (glm::dot(vec, frustums.planes[i].n) < 0)
	//	//	return true;         //it intersects the frusutm but not fully inside
	//}
	return true;  //it is fully inside


	////without pvertex
	//glm::vec3 c1 = nod.center + nod.extent*glm::vec3(1, 1, 1);
	//glm::vec3 c2 = nod.center + nod.extent*glm::vec3(1, 1, -1);
	//glm::vec3 c3 = nod.center + nod.extent*glm::vec3(1, -1, 1);
	//glm::vec3 c4 = nod.center + nod.extent*glm::vec3(-1, 1, 1);
	//glm::vec3 c5 = nod.center + nod.extent*glm::vec3(1, -1, -1);
	//glm::vec3 c6 = nod.center + nod.extent*glm::vec3(-1, 1, -1);
	//glm::vec3 c7 = nod.center + nod.extent*glm::vec3(-1, -1, 1);
	//glm::vec3 c8 = nod.center + nod.extent*glm::vec3(-1, -1, -1);

	//std::vector<glm::vec3> c;
	//c.push_back(c1);
	//c.push_back(c2);
	//c.push_back(c3);
	//c.push_back(c4);
	//c.push_back(c5);
	//c.push_back(c6);
	//c.push_back(c7);
	//c.push_back(c8);

	//bool in;

	//for (int i = 0; i < c.size(); i++)
	//{
	//	//if a corner is inside all planes, return true
	//	in=true;
	//	for (int j = 0; j < frustums.planes.size(); j++)
	//	{
	//		vec = c[i] - frustums.planes[j].p;
	//		if (glm::dot(vec, frustums.planes[j].n) < -0.0001)
	//		{
	//			in = false;
	//			break;
	//		}
	//	}
	//	if (in)
	//	{
	//		return true;
	//	}
	//}
	//return false;
}
void octree::getPvertex(glm::vec3& p, glm::vec3 normal, glm::vec3 c, glm::vec3 extent)
{
	//glm::vec3 nmin = c - extent*glm::vec3(1, 1, 1);
	//glm::vec3 nmax = c + extent*glm::vec3(1, 1, 1);
	//p = nmin;
	//if (normal.x >= 0)
	//	p.x = nmax.x;
	//if (normal.y >= 0)
	//	p.y = nmax.y;
	//if (normal.z >= 0)
	//	p.z = nmax.z;
}
void octree::getNvertex(glm::vec3& n, glm::vec3 normal, glm::vec3 c, glm::vec3 extent)
{
	//glm::vec3 nmin = c - extent*glm::vec3(1, 1, 1);
	//glm::vec3 nmax = c + extent*glm::vec3(1, 1, 1);
	//n = nmax;
	//if (normal.x >= 0)
	//	n.x = nmin.x;
	//if (normal.y >= 0)
	//	n.y = nmin.y;
	//if (normal.z >= 0)
	//	n.z = nmin.z;
}
void octree::get_bf_traversal(std::vector<int>&order)
{
	std::queue<int> q;
	std::vector<int> children;

	order.clear();

	q.push(0);

	int c;

	while (!q.empty())
	{
		c = q.front();
		order.push_back(c);
		q.pop();
		
		get_children(c, children);
		for (int i = 0; i < children.size(); i++)
			q.push(children[i]);
	}
}
void octree::get_leaves()
{
	std::vector<int> children;
	leaves.clear();

	for (int i = 0; i < nodes.size(); i++)
	{
		if (nodes[i].children.size() == 0 && nodes[i].nodeCount>0)
			leaves.push_back(i);
	}
}

void octree::populate_leaves( std::vector<glm::vec4>& points)
{
	//for (int i = 0; i < leaves.size(); i++)
	//{
	//	for (int j = 0; j < nodes[leaves[i]].indicies.size(); j++)
	//		nodes[leaves[i]].Points.push_back(points[nodes[leaves[i]].indicies[j]]);

	//	nodes[leaves[i]].indicies.clear();
	//}
}
void octree::sort_nodes_backToFront(std::vector<int>& nodeIndecies, glm::mat3 GlobalRotationMatrix)
{
	std::vector<std::pair<int,glm::vec3>> N;

	for (int i = 0; i < nodeIndecies.size(); i++)
	{
		N.push_back(std::make_pair(nodeIndecies[i],nodes[nodeIndecies[i]].center));
		N[i].second = GlobalRotationMatrix*N[i].second;
	}

	std::sort(N.begin(), N.end(), [](std::pair<int, glm::vec3> &left, std::pair<int, glm::vec3> &right)
	{
		return left.second.z < right.second.z;
	});

	for (int i = 0; i < N.size(); i++)
	{
		nodeIndecies[i] = N[i].first;
	}
}

void octree::sort_nodes_frontToBack(std::vector<int>& nodeIndecies, glm::mat3 GlobalRotationMatrix)
{
	std::vector<std::pair<int, glm::vec3>> N;

	for (int i = 0; i < nodeIndecies.size(); i++)
	{
		N.push_back(std::make_pair(nodeIndecies[i], nodes[nodeIndecies[i]].center));
		N[i].second = GlobalRotationMatrix*N[i].second;
	}

	std::sort(N.begin(), N.end(), [](std::pair<int, glm::vec3> &left, std::pair<int, glm::vec3> &right)
	{
		return left.second.z > right.second.z;
	});

	for (int i = 0; i < N.size(); i++)
	{
		nodeIndecies[i] = N[i].first;
	}
}
int octree::getNodeLevel(node& n)
{
	int level = 0;

	int p = n.parent;
	while (p != -1)
	{
		p = nodes[p].parent;
		level++;
	}

	return level;
}

void octree::getLeavesWithCommonAncestor(int ind, std::vector<int>& l)
{
	//get all leaves that have 'ind' as an ancestor
	l.clear();
	std::queue<int> tbt;
	tbt.push(ind);
	
	while (tbt.size() > 0)
	{
		//get first element in queue
		int node_indx = tbt.front();
		tbt.pop();

		if (nodes[node_indx].children.size() > 0)
		{
			//add it's children to queue
			for (int i = 0; i < nodes[node_indx].children.size(); i++)
			{
				tbt.push(nodes[node_indx].children[i]);
			}
		}
		else
		{
			l.push_back(node_indx);
		}
	}
}