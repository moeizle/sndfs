#include "tiles.h"


tiles::tiles()
{
	visible_tiles_size.x = 0;
	visible_tiles_size.y = 0;
}
tiles::~tiles()
{
}

void tiles::tile_image(glm::vec3 center, float width, float height, float t_w, float t_h)
{
	//given an image of center and width and height, tile the image using tiles of t_w and t_h
	T.clear();

	//get vectors for width and height of image
	glm::vec3 img_w_e = center + glm::vec3(width / 2.0, height / 2.0, 0);
	glm::vec3 img_w_s = center + glm::vec3(-width / 2.0, height / 2.0, 0);
	glm::vec3 img_w = img_w_e - img_w_s;

	glm::vec3 img_h_e = center + glm::vec3(-width / 2.0, -height / 2.0, 0);
	glm::vec3 img_h_s = center + glm::vec3(-width / 2.0, height / 2.0, 0);
	glm::vec3 img_h = img_h_e - img_h_s;

	float tile_count_in_x, tile_count_in_y, wt, ht;
	tile t;

	tile_count_in_x = std::ceil(width / t_w);
	tile_count_in_y = std::ceil(height / t_h);

	//new
	//if (int(tile_count_in_x) % 2 != 0)
	//{
	//	tile_count_in_x++;
	//}
	//if (int(tile_count_in_y) % 2 != 0)
	//{
	//	tile_count_in_y++;
	//}
	//end new


	for (float i = 0; i < tile_count_in_x; i++)
	{
		for (float j = 0; j < tile_count_in_y; j++)
		{
			t.c = img_h_e + glm::vec3((i*t_w) + 0.5f*t_w, (j*t_h) + 0.5f*t_h, 0);
			t.h = t_h;
			t.w = t_w;

			T.push_back(t);
			return;
		}
	}



	//for (float i = t_w / 2.0; i < glm::length(img_w)+t_w; i = i + t_w)
	//{
	//	wt = float(i / glm::length(img_w));
	//	for (float j = t_h / 2.0; j < glm::length(img_h)+t_h; j = j + t_h)
	//	{
	//		ht = float(j / glm::length(img_h));

	//		//compute center of tile
	//		t.c = img_w_s + wt*(img_w) + ht*(img_h);
	//		t.h = t_h;
	//		t.w = t_w;

	//		T.push_back(t);
	//	}
	//}
}

void tiles::intersect_with_camera(glm::vec3 cam_c, float cam_window_w, float cam_window_h)
{
	//intersect the tiles in 'T' with the camera passed to this function, and put the visible tiles in 'visible'

	visible.clear();

	glm::vec3 t_max;
	glm::vec3 t_min;

	glm::vec3 cambl, cambr, camtl, camtr;
	glm::vec3 cam_max, cam_min;

	glm::vec2 w_extents = glm::vec2(0, 0);
	glm::vec2 h_extents = glm::vec2(0, 0);


	cam_max = camtr = cam_c + glm::vec3(cam_window_w / 2.0, cam_window_h / 2.0, 0);
	cam_min = cambl = cam_c - glm::vec3(cam_window_w / 2.0, cam_window_h / 2.0, 0);
	cambr = cam_c + glm::vec3(cam_window_w / 2.0, -cam_window_h / 2.0, 0);
	camtl = cam_c + glm::vec3(-cam_window_w / 2.0, cam_window_h / 2.0, 0);

	//for each tile, if any of the camera corners are inside the tile, then its visible
	glm::vec3 c1, c2, c3, c4;
	bool in;
	for (int i = 0; i < T.size(); i++)
	{
		t_max = T[i].c + glm::vec3(T[i].w / 2.0, T[i].h / 2.0, 0);
		t_min = T[i].c - glm::vec3(T[i].w / 2.0, T[i].h / 2.0, 0);

		c1 = t_max;
		c2 = t_min;
		c3 = T[i].c + glm::vec3(T[i].w / 2.0, -T[i].h / 2.0, 0);
		c4 = T[i].c + glm::vec3(-T[i].w / 2.0, T[i].h / 2.0, 0);

		in = false;

		//check if camera is inside tile

		if (inside(camtr, t_max, t_min))
		{
			visible.push_back(i);
			in = true;
		}
		else if (inside(cambr, t_max, t_min))
		{
			visible.push_back(i);
			in = true;
		}
		else if (inside(camtl, t_max, t_min))
		{
			visible.push_back(i);
			in = true;
		}
		else if (inside(cambl, t_max, t_min))
		{
			visible.push_back(i);
			in = true;
		}

		//check if tile is inside camera
		if (!in)
		{
			if (inside(c1, cam_max, cam_min))
			{
				visible.push_back(i);
			}
			else if (inside(c2, cam_max, cam_min))
			{
				visible.push_back(i);
			}
			else if (inside(c3, cam_max, cam_min))
			{
				visible.push_back(i);
			}
			else if (inside(c4, cam_max, cam_min))
			{
				visible.push_back(i);
			}
		}
	}

	//update visible_tiles_size
	visible_tiles_size.x = glm::length(w_extents);
	visible_tiles_size.y = glm::length(h_extents);
}
void tiles::intersect_with_range(glm::vec3 s, glm::vec3 e)
{
	glm::vec3 t_max;
	glm::vec3 t_min;

	glm::vec3 c1, c2, c3, c4;
	bool in;

	glm::vec3 r1, r2, r3, r4;
	r1 = s;
	r2 = e;
	r3 = glm::vec3(s.x, e.y, s.z);
	r4 = glm::vec3(e.x, s.y, s.z);

	visible.clear();

	for (int i = 0; i < T.size(); i++)
	{
		t_max = T[i].c + glm::vec3(T[i].w / 2.0, T[i].h / 2.0, 0);
		t_min = T[i].c - glm::vec3(T[i].w / 2.0, T[i].h / 2.0, 0);

		c1 = t_max;
		c2 = t_min;
		c3 = T[i].c + glm::vec3(T[i].w / 2.0, -T[i].h / 2.0, 0);
		c4 = T[i].c + glm::vec3(-T[i].w / 2.0, T[i].h / 2.0, 0);

		in = false;

		//check if range is inside tile

		if (inside(r1, t_max, t_min))
		{
			visible.push_back(i);
			in = true;
		}
		else if (inside(r2, t_max, t_min))
		{
			visible.push_back(i);
			in = true;
		}
		else if (inside(r3, t_max, t_min))
		{
			visible.push_back(i);
			in = true;
		}
		else if (inside(r4, t_max, t_min))
		{
			visible.push_back(i);
			in = true;
		}

		//check if tile is inside camera
		if (!in)
		{
			if (inside(c1, e, s))
			{
				visible.push_back(i);
			}
			else if (inside(c2, e, s))
			{
				visible.push_back(i);
			}
			else if (inside(c3, e, s))
			{
				visible.push_back(i);
			}
			else if (inside(c4, e, s))
			{
				visible.push_back(i);
			}
		}
	}
}
void tiles::intersect_with_camera(glm::vec2 lod_blc, glm::vec2 cam_blc, float cam_w, float cam_h, int tile_w, int tile_h, int lodw, int lodh)
{
	glm::ivec2 pixel_ind;
	glm::ivec2 tile_indx_2d;

	int tile_indx;
	int num_tiles_in_h = lodw / tile_w;

	visible.clear();
	for (int i = cam_blc.x; i < cam_blc.x + cam_w + tile_w; i = i + tile_w)
	{
		for (int j = cam_blc.y; j < cam_blc.y + cam_h + tile_h; j = j + tile_h)
		{
			pixel_ind = glm::ivec2(glm::vec2(i, j) - lod_blc);
			if (pixel_ind.x >= 0 && pixel_ind.y >= 0 && pixel_ind.x<lodw && pixel_ind.y<lodh)
			{
				tile_indx_2d = glm::ivec2(pixel_ind.x / tile_w, pixel_ind.y / tile_h);
				tile_indx = tile_indx_2d.y*num_tiles_in_h + tile_indx_2d.x;
				visible.push_back(tile_indx);
			}
		}
	}
}
void tiles::update_visible_tiles_size(glm::vec3 bl, glm::vec3 tr, glm::vec2& w_extents, glm::vec2& h_extents)
{
	if (bl.x < w_extents.x)
	{
		w_extents.x = bl.x;
	}
	if (bl.y < h_extents.x)
	{
		h_extents.x = bl.y;
	}
	if (tr.x > w_extents.y)
	{
		w_extents.y = tr.x;
	}
	if (tr.y < h_extents.y)
	{
		h_extents.y = tr.y;
	}
}
bool tiles::inside(glm::vec3 p, glm::vec3 max, glm::vec3 min)
{
	return ((p.x < max.x) && (p.x >= min.x) && (p.y < max.y) && (p.y >= min.y));
}
void tiles::compute_frustums_for_visible_tiles(glm::vec3 camPos, std::vector<frustum>& frustums, float nearPlane, float farPlane, glm::vec3 origp, glm::vec3 &origl, glm::vec3 &u)
{
	//create frustums for the visible tiles (view independent)	
	tile t;
	glm::vec3 c, X, Y, Z, nc, fc, ntl, ntr, nbl, nbr, ftl, ftr, fbl, fbr, a, b, n, point, vec, p, l;
	frustum f;

	frustums.clear();

	//create a camera at each tile, and extrat the planes from it
	for (int i = 0; i < visible.size(); i++)
	{
		//get the visible tile
		t = T[visible[i]];

		f.planes.clear();

		//get vector from camera position to the tile center
		vec = glm::vec3(t.c.x, t.c.y, t.c.z) - origp;

		//set camera position and target to match the tile
		p = origp + vec;
		l = origl + vec;

		// compute the Z axis of camera
		// this axis points in the opposite direction from
		// the looking direction
		Z = p - l;
		Z = glm::normalize(Z);

		// X axis of camera with given "up" vector and Z axis
		X = glm::cross(u, Z);
		X = glm::normalize(X);

		// the real "up" vector is the cross product of Z and X
		Y = glm::cross(Z, X);

		// compute the centers of the near and far planes
		nc = p - Z * nearPlane;
		fc = p - Z * farPlane;

		// compute the 4 corners of the frustum on the near plane
		ntl = nc + Y * 0.5f*t.h - X * 0.5f*t.w;
		ntr = nc + Y * 0.5f*t.h + X * 0.5f*t.w;
		nbl = nc - Y * 0.5f*t.h - X * 0.5f*t.w;
		nbr = nc - Y * 0.5f*t.h + X * 0.5f*t.w;

		// compute the 4 corners of the frustum on the far plane
		ftl = fc + Y * 0.5f*t.h - X * 0.5f*t.w;
		ftr = fc + Y * 0.5f*t.h + X * 0.5f*t.w;
		fbl = fc - Y * 0.5f*t.h - X * 0.5f*t.w;
		fbr = fc - Y * 0.5f*t.h + X * 0.5f*t.w;

		//compute planes of frustum
		//top plane
		a = ntr - ntl;
		b = ftl - ntl;
		n = glm::cross(a, b);
		n = glm::normalize(n);

		f.planes.push_back(frustum::plane(ntr, -n));

		//bottom plane
		f.planes.push_back(frustum::plane(fbr, n));

		//left plane
		a = ntl - nbl;
		b = fbl - nbl;

		n = glm::cross(a, b);
		n = glm::normalize(n);

		f.planes.push_back(frustum::plane(nbl, -n));

		//right plane
		f.planes.push_back(frustum::plane(nbr, n));

		//near plane
		a = ntl - ntr;
		b = nbr - ntr;

		n = glm::cross(a, b);
		n = glm::normalize(n);

		f.planes.push_back(frustum::plane(ntr, -n));

		//far plane
		f.planes.push_back(frustum::plane(ftr, n));

		//old
		////create a point at the center of the camera plane of the frustum
		//c = glm::vec3(t.c.x, t.c.y, p.z);

		////near plane
		//point = c+glm::vec3(0,0,nearPlane);
		//n = glm::vec3(0, 0, -1);
		//f.planes.push_back(frustum::plane(point, n));

		////far plane
		//point = c + glm::vec3(0, 0, farPlane);
		//n = glm::vec3(0, 0, 1);
		//f.planes.push_back(frustum::plane(point, n));

		////left plane
		//point = c + 0.5f*t.w*glm::vec3(-1, 0, 0);
		//n = glm::vec3(1, 0, 0);
		//f.planes.push_back(frustum::plane(point, n));

		////right plane
		//point = c + 0.5f*t.w*glm::vec3(1, 0, 0);
		//n = glm::vec3(-1, 0, 0);
		//f.planes.push_back(frustum::plane(point, n));

		////top plane
		//point = c + 0.5f*t.h*glm::vec3(0, 1, 0);
		//n = glm::vec3(0, -1, 0);
		//f.planes.push_back(frustum::plane(point, n));

		////bottom plane
		//point = c + 0.5f*t.h*glm::vec3(0, -1, 0);
		//n = glm::vec3(0, 1, 0);
		//f.planes.push_back(frustum::plane(point, n));
		//end old

		frustums.push_back(f);
	}
}
//void tiles::get_blc_and_trc_of_viible_tiles(glm::vec3& blc, glm::vec3& trc)
//{
//	blc = glm::vec3(10000000, 10000000, 10000000);
//	trc = glm::vec3(-10000000, -10000000, -10000000);
//	for (int i = 0; i < visible.size(); i++)
//	{
//		T[i].h = 0 / 0;
//	}
//}

void tiles::get_blc_and_trc_of_viible_tiles(glm::vec3& blc, glm::vec3& trc, int lod_w, int lod_h)
{
	if (T.size() == 0)
		return;

	blc = glm::vec3(10000000, 10000000, 10000000);
	trc = glm::vec3(-10000000, -10000000, -10000000);
	int tiles_in_h = lod_h / T[0].h;
	int tiles_in_w = lod_w / T[0].w;
	int blc_indx = -1, trc_indx = -1;

	for (int i = 0; i < visible.size(); i++)
	{
		glm::ivec2 twoDindx = glm::ivec2(visible[i] % tiles_in_w, visible[i] / tiles_in_w);
		glm::vec3 tileCenter = T[0].c + glm::vec3(twoDindx.x*T[0].w, twoDindx.y*T[0].h, 0);

		if (tileCenter.x - 0.5*T[0].w<blc.x) 
			blc.x = tileCenter.x - 0.5*T[0].w;
		if (tileCenter.y - 0.5*T[0].h<blc.y)
			blc.y = tileCenter.y - 0.5*T[0].h;
		if (tileCenter.x + 0.5*T[0].w>trc.x)
			trc.x = tileCenter.x + 0.5*T[0].w;
		if (tileCenter.y + 0.5*T[0].h>trc.y)
			trc.y = tileCenter.y + 0.5*T[0].h;
	}
}

//void tiles::get_blc_and_trc_of_viible_tiles(glm::vec3& blc, glm::vec3& trc, int& blc_tile_indx, int& trc_tile_indx)
//{
//	blc = glm::vec3(10000000, 10000000, 10000000);
//	trc = glm::vec3(-10000000, -10000000, -10000000);
//	blc_tile_indx = trc_tile_indx= -1;
//	for (int i = 0; i < visible.size(); i++)
//	{
//		if (((T[visible[i]].c.y - .5f*T[visible[i]].h) < blc.y) && ((T[visible[i]].c.x - .5f*T[visible[i]].w) < blc.x))
//		{
//			blc.y = (T[visible[i]].c.y - .5f*T[visible[i]].h);
//			blc.x = (T[visible[i]].c.x - .5f*T[visible[i]].w);
//			blc_tile_indx = visible[i];
//		}
//		if (((T[visible[i]].c.y + .5f*T[visible[i]].h) > trc.y) && ((T[visible[i]].c.x + .5f*T[visible[i]].w) > trc.x))
//		{
//			trc.y = (T[visible[i]].c.y + .5f*T[visible[i]].h);
//			trc.x = (T[visible[i]].c.x + .5f*T[visible[i]].w);
//			trc_tile_indx = visible[i];
//		}
//
//
//		blc.z = trc.z = T[visible[i]].c.z;
//	}
//}


