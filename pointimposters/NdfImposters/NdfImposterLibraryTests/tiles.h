#pragma once
#include <vector>
#include <glm/vec2.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "frustum.h"
#include "octree.h"
class tiles
{

public:
	struct tile{
		glm::vec3 c;
		float w, h;
		tile()
		{}
		tile(glm::vec3 center, float width, float height)
		{
			c = center;
			w = width;
			h = height;
		}
	};
	std::vector<tile> T;
	std::vector<int> visible;
	glm::vec2 visible_tiles_size;

	tiles();
	~tiles();
	void tile_image(glm::vec3 center, float width, float height, float t_w, float t_h);
	void intersect_with_camera(glm::vec3 cam_c, float cam_window_w, float cam_window_h);
	void intersect_with_camera(glm::vec2 lod_blc, glm::vec2 cam_blc, float cam_w, float cam_h, int tile_w, int tile_h, int lodw, int lodh);
	void intersect_with_range(glm::vec3 s, glm::vec3 e);
	void update_visible_tiles_size(glm::vec3 bl, glm::vec3 tr, glm::vec2& w_extents, glm::vec2& h_extents);
	bool inside(glm::vec3 p, glm::vec3 max, glm::vec3 min);
	void compute_frustums_for_visible_tiles(glm::vec3 camPos, std::vector<frustum>& frustums, float nearPlane, float farPlane, glm::vec3 p, glm::vec3 &l, glm::vec3 &u);

	//void get_blc_and_trc_of_viible_tiles(glm::vec3& blc, glm::vec3& trc);
	//void get_blc_and_trc_of_viible_tiles(glm::vec3& blc, glm::vec3& trc, int& blc_tile_indx, int& trc_tile_indx);

	void get_blc_and_trc_of_viible_tiles(glm::vec3& blc, glm::vec3& trc, int lod_w, int lod_h);
	
};


