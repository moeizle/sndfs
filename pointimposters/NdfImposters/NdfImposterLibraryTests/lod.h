#pragma once
#include <math.h>
#include <algorithm>
#include "tiles.h"
#include "ObbObbIntersection.h"

class lod
{
	int max_levels;
	float initial_camera_distance;

	float max_res_w, max_res_h;
	float w_orig, h_orig;
	
	int tileW, tileH;
	float pixelPerObj;

	void compute_tiles();
	glm::vec3 modelExtent;

	float lodDelta;

public:
	float initial_lod;
	float max_cam_dist, min_cam_dist;


	std::vector<tiles> myTiles;

	lod(int max_w, int max_h, float initial_cam_dist, float initial_w, float initial_h,int tile_w,int tile_h,glm::vec3 mExtent,float pixelPerObj,float lDelta);

	float get_lod(float cam_dist);
	float get_cam_dist(float level_of_detail);
	void set_cam_spectrum();
	void get_lod_width_and_hight(float lod, int& lod_w, int& lod_h);
	glm::vec2 lod::pixel2obj(glm::vec2 pixel_coordinates, float lod);
	glm::vec2 lod::obj2pixel(glm::vec2 obj_coordinates, float lod);
	void get_visible_in_ceil_and_floor(float lod, int w, int h, glm::vec3 camOffset, float OrigR, float OrigL, float OrigT, float OrigB, glm::vec3 camPosi,std::vector<glm::vec3>& obb);
	void get_visible_tiles_in_ceil(float lod, int ceilLod, int w, int h, glm::vec2 pixel_coords, std::vector<glm::vec3>& obb);
	void get_visible_tiles_in_floor(int ceilLod, int floorLod, std::vector<glm::vec3>& obb);
	void constructTileAabb(float lod, int indx, std::vector<glm::vec3>& tileAabb);

	lod();
	~lod();
};