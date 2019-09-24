#include "lod.h"



lod::lod()
{
}


lod::lod(int max_w, int max_h, float initial_cam_dist, float initial_w, float initial_h, int tile_w, int tile_h, glm::vec3 mExtent, float pixelPerObj, float lDelta)
{
	//first set the maximum level of the mip-map
	max_levels = log2f( std::min(max_w, max_h))-log2(tile_w);

	this->pixelPerObj = pixelPerObj;


	max_res_w = max_w;
	max_res_h = max_h;

	w_orig = initial_w;
	h_orig = initial_h;

	tileW = tile_w;
	tileH = tile_h;

	modelExtent = mExtent;

	lodDelta = lDelta;

	//set cam spectrum
	set_cam_spectrum();

	initial_camera_distance = initial_cam_dist;
	initial_lod = get_lod(initial_camera_distance);

	//now we compute all the tiling of all lods
	compute_tiles();
}

lod::~lod()
{
}

float lod::get_lod(float cam_dist)
{

	return  float (log2(cam_dist/min_cam_dist));

}

float lod::get_cam_dist(float level_of_detail)
{

	return min_cam_dist*pow(2, level_of_detail);

}

void lod::set_cam_spectrum()
{

	min_cam_dist = 1;
	max_cam_dist = min_cam_dist*pow(2, max_levels); 

}

void lod::get_lod_width_and_hight(float lod, int& lod_w, int& lod_h)
{
	//lod_w = std::ceil(max_res_w /pow(2,std::floor(lod+0.5)));
	//lod_h = std::ceil(max_res_h /pow(2,std::floor(lod+0.5)));

	lod_w = std::floor(max_res_w / pow(2, lod ));
	lod_h = std::floor(max_res_h / pow(2, lod ));
}

void lod::compute_tiles()
{
	//construct lod pyramid such that camera is at center of all lods
	int w, h;
	tiles::tile t;
	myTiles.clear();

	for (int i = 0; i <= max_levels; i++)
	{
		tiles lodTiles;
		myTiles.push_back(lodTiles);

		get_lod_width_and_hight(i, w, h);
		
		if (w >= tileW && h >= tileH)
		{
			t.c = glm::vec3(-0.5f*w + 0.5f*tileW, -0.5f*h+ 0.5f*tileH, 0);
			t.h = tileH;
			t.w = tileW;
			myTiles[i].T.push_back(t);
		}
	}
}

glm::vec2 lod::pixel2obj(glm::vec2 pixel_coordinates, float lod)
{
	//1 pixel in initial lod covers an area of (modelExtent.x/intial_w)*(modelExtent.y/initial_h)
	//we map that to lod at hand, it should corresond to less area in lower lod and bigger area in higher lod
	//int lw, lh;
	//get_lod_width_and_hight(initial_lod, lw, lh);

	////map pixel coordiantes to initial
	//pixel_coordinates = glm::vec2(pixel_coordinates.x*std::pow(2, lod - initial_lod), pixel_coordinates.y*std::pow(2, lod - initial_lod));

	////map pixel coordinates in initial lod to object space
	//glm::vec2 obj = glm::vec2(pixel_coordinates.x *( modelExtent.x / float(lw)), pixel_coordinates.y *(modelExtent.y / float(lh)));
	//obj = glm::vec2(obj.x*pow(2, initial_lod - lod), obj.y*pow(2, initial_lod - lod));

	glm::vec2 obj;
	//new
	{
		int lw, lh;
		get_lod_width_and_hight(initial_lod, lw, lh);
		glm::vec2 objPerPixel_inInitialLod = glm::vec2(modelExtent.x / lw, modelExtent.y / lh);
		glm::vec2 objPerPixel_inLod = glm::vec2(pow(2, lod - initial_lod)*objPerPixel_inInitialLod.x, pow(2, lod - initial_lod)*objPerPixel_inInitialLod.y);
		obj = glm::vec2(objPerPixel_inLod.x*pixel_coordinates.x, objPerPixel_inLod.y*pixel_coordinates.y);
	}
	//end new

	//newer
	//{
	//	int lw, lh;
	//	get_lod_width_and_hight(initial_lod, lw, lh);
	//	glm::vec2 objPerPixel_inInitialLod = glm::vec2(modelExtent.x / w_orig, modelExtent.y/ h_orig);
	//	glm::vec2 objPerPixel_inLod = glm::vec2(pow(2, lod - initial_lod)*objPerPixel_inInitialLod.x, pow(2, lod - initial_lod)*objPerPixel_inInitialLod.y);
	//	obj = glm::vec2(objPerPixel_inLod.x*pixel_coordinates.x, objPerPixel_inLod.y*pixel_coordinates.y);
	//}
	//end newer


	return obj;
}
glm::vec2 lod::obj2pixel(glm::vec2 obj_coordinates, float lod)
{
	//int lw, lh;
	//get_lod_width_and_hight(initial_lod, lw, lh);

	////get pixel coordinates in initial lod of given obj_coordinates
	//glm::vec2 pixel_coordinates = glm::vec2(obj_coordinates.x*(float(lw) / modelExtent.x), obj_coordinates.y*(float(lh) / modelExtent.y));

	////map pixel coordinates in lod 0 to pixel coordinates in given lod
	//pixel_coordinates = glm::vec2(pixel_coordinates.x*std::pow(2, initial_lod- lod), pixel_coordinates.y*std::pow(2, initial_lod - lod));



	//new
	int lw, lh;
	get_lod_width_and_hight(initial_lod, lw, lh);
	glm::vec2 pixelPerObj_inInitialLod = glm::vec2(lw/modelExtent.x, lh/modelExtent.y);
	glm::vec2 pixelPerObj_inLod = glm::vec2(pow(2,  initial_lod-lod)*pixelPerObj_inInitialLod.x, pow(2, initial_lod-lod)*pixelPerObj_inInitialLod.y);
	glm::vec2 pixel_coordinates = glm::vec2(pixelPerObj_inLod.x*obj_coordinates.x, pixelPerObj_inLod.y*obj_coordinates.y);
	//end new

	return pixel_coordinates;
}

void lod::get_visible_in_ceil_and_floor(float lod, int w, int h, glm::vec3 camOffset,float OrigR,float OrigL,float OrigT,float OrigB,glm::vec3 camPosi, std::vector<glm::vec3>& obb)
{
	glm::vec2 pixel_coords;
	glm::vec2 obj_coords;

	int ceilLod = std::floor(lod + 1);
	int floorLod = std::floor(lod-lodDelta);

	//erase what was visible before
	for (int i = 0; i <= max_levels;i++)
		myTiles[i].visible.clear();


	//get pixel coordinates in 'lod', this is the pixel coordinate of the center of the window in space of 'lod'
	{
		pixel_coords = obj2pixel(glm::vec2(camOffset), lod);
	}
	//new
	{
		glm::vec2 objPerPixel = glm::vec2((OrigR - OrigL) /w, (OrigT - OrigB) / h);
		pixel_coords = glm::vec2(camPosi.x/objPerPixel.x,camPosi.y/objPerPixel.y);
		//glm::vec2 camTrcInInitialLod = glm::vec2(camPosi) + glm::vec2(OrigR, OrigT);
		//camTrcInInitialLod.x *= 1.0f / objPerPixel.x;
		//camTrcInInitialLod.y *= 1.0f / objPerPixel.y;

		//glm::vec2 camBlcInInitialLod = glm::vec2(camPosi) + glm::vec2(OrigL, OrigB);
		//camBlcInInitialLod.x *= 1.0f / objPerPixel.x;
		//camBlcInInitialLod.y *= 1.0f / objPerPixel.y;
	}
	
	//get visible tiles in ceil and floor
	get_visible_tiles_in_ceil(lod, ceilLod, w, h, pixel_coords,obb);
	get_visible_tiles_in_floor(ceilLod,floorLod,obb);
}
void lod::get_visible_tiles_in_ceil(float lod, int ceilLod, int w, int h, glm::vec2 pixel_coords, std::vector<glm::vec3>& obb)
{
	glm::vec2 pc;
	glm::vec2 tileIndx2d;
	int tileIndx;
	int tiles_in_w, tiles_in_h;
	int lodw, lodh;
	int curw, curh;
	ObbObbIntersection Intersects;
	std::vector < glm::vec3> tileAabb;


	get_lod_width_and_hight(ceilLod, lodw, lodh);
	get_lod_width_and_hight(lod, curw, curh);

	tiles_in_w = lodw / tileW;
	tiles_in_h = lodh / tileH;

	
	//get pixel coordinate of bottom left corner of visible screen
	pixel_coords -= glm::vec2(0.5*w,0.5*h);

	//get visible tiles in the ceil
	//map the pixel coordinate from 'lod' to floor(lod+1)
	pixel_coords.x *= pow(2, lod - ceilLod);
	pixel_coords.y *= pow(2, lod - ceilLod);

	//get pixel coords relative to the bottom left corner of the lod
	pixel_coords.x -= (myTiles[ceilLod].T[0].c.x-0.5f*tileW);
	pixel_coords.y -= (myTiles[ceilLod].T[0].c.y-0.5f*tileH);

	//compute tiles in width and height of ceil lod

	//now get the visible tiles in the lod
	pc = pixel_coords;
	glm::ivec2 start_2dIndx = glm::ivec2(std::floor(pc.x / tileW), std::floor(pc.y / tileH));
	start_2dIndx = glm::ivec2(std::max(0,start_2dIndx.x),std::max(0,start_2dIndx.y));

	pc = pixel_coords + glm::vec2(w*pow(2, lod - ceilLod), h*pow(2, lod - ceilLod));
	glm::ivec2 end_2dIndx = glm::ivec2(std::floor(pc.x / tileW), std::floor(pc.y / tileH));
	end_2dIndx = glm::ivec2(std::min(tiles_in_w-1,end_2dIndx.x),std::min(tiles_in_h-1,end_2dIndx.y));
	
	for (int i = start_2dIndx.x; i <= end_2dIndx.x; i ++)
	{
		for (int j = start_2dIndx.y; j <= end_2dIndx.y; j++)
		{
			//the tile is visible to the user, but if it doesn't contain any data, we won't 
			//add it to visible tiles to save space in the cache
			constructTileAabb(ceilLod, j*tiles_in_w + i, tileAabb);
			if (Intersects.TestIntersection3D(obb, tileAabb))
			{
				myTiles[ceilLod].visible.push_back(j*tiles_in_w + i);
			}
		}
	}
}
void lod::get_visible_tiles_in_floor(int ceilLod, int floorLod, std::vector<glm::vec3>& obb)
{
	int tileIndxC,tileIndxF;
	glm::vec2 tileIndx2dC,tileIndx2dF;
	int lodwC, lodhC;
	int tiles_in_wC, tiles_in_hC;

	int lodwF, lodhF;
	int tiles_in_wF, tiles_in_hF;

	float factor = pow(2, ceilLod - floorLod);

	ObbObbIntersection Intersects;
	std::vector < glm::vec3> tileAabb;

	get_lod_width_and_hight(ceilLod, lodwC, lodhC);

	tiles_in_wC = lodwC / tileW;
	tiles_in_hC = lodhC / tileH;

	get_lod_width_and_hight(floorLod, lodwF, lodhF);

	tiles_in_wF = lodwF / tileW;
	tiles_in_hF = lodhF / tileH;

	//look at the visible tiles in the ceil and for each tile get four corresponding tiles in the floor level
	for (int i = 0; i < myTiles[ceilLod].visible.size(); i++)
	{
		//get 2d index of the tile in the ceil level
		tileIndxC = myTiles[ceilLod].visible[i];
		tileIndx2dC = glm::vec2(tileIndxC%tiles_in_wC,tileIndxC/tiles_in_wC);

		

		//in the floor level you just multiply by
		tileIndx2dF = glm::vec2(tileIndx2dC.x * factor, tileIndx2dC.y * factor);
		tileIndxF = tileIndx2dF.y*tiles_in_wF + tileIndx2dF.x;
		//the tile is visible to the user, but if it doesn't contain any data, we won't 
		//add it to visible tiles to save space in the cache
		//constructTileAabb(floorLod, tileIndxF, tileAabb);
		//if (Intersects.TestIntersection3D(obb, tileAabb))
		{
			myTiles[floorLod].visible.push_back(tileIndxF);
		}
		

		tileIndx2dF = glm::vec2(tileIndx2dC.x * factor + 1, tileIndx2dC.y * factor);
		tileIndxF = tileIndx2dF.y*tiles_in_wF + tileIndx2dF.x;
		//the tile is visible to the user, but if it doesn't contain any data, we won't 
		//add it to visible tiles to save space in the cache
		//constructTileAabb(floorLod, tileIndxF, tileAabb);
		//if (Intersects.TestIntersection3D(obb, tileAabb))
		{
			myTiles[floorLod].visible.push_back(tileIndxF);
		}

		tileIndx2dF = glm::vec2(tileIndx2dC.x * factor, tileIndx2dC.y * factor + 1);
		tileIndxF = tileIndx2dF.y*tiles_in_wF + tileIndx2dF.x;
		//the tile is visible to the user, but if it doesn't contain any data, we won't 
		//add it to visible tiles to save space in the cache
		//constructTileAabb(floorLod, tileIndxF, tileAabb);
		//if (Intersects.TestIntersection3D(obb, tileAabb))
		{
			myTiles[floorLod].visible.push_back(tileIndxF);
		}

		tileIndx2dF = glm::vec2(tileIndx2dC.x * factor + 1, tileIndx2dC.y * factor + 1);
		tileIndxF = tileIndx2dF.y*tiles_in_wF + tileIndx2dF.x;
		//the tile is visible to the user, but if it doesn't contain any data, we won't 
		//add it to visible tiles to save space in the cache
		//constructTileAabb(floorLod, tileIndxF, tileAabb);
		//if (Intersects.TestIntersection3D(obb, tileAabb))
		{
			myTiles[floorLod].visible.push_back(tileIndxF);
		}

	}
}

void lod::constructTileAabb(float LOD, int indx, std::vector<glm::vec3>& tileAabb)
{
	//construct an Aabb for the given tile
	int lw, lh;
	glm::vec2 objCoord;

	tileAabb.clear();
	get_lod_width_and_hight(LOD, lw, lh);

	int tiles_in_w = lw / tileW;

	glm::ivec2 Indx2d = glm::ivec2(indx%tiles_in_w,indx/tiles_in_w);

	glm::vec3 TileC = myTiles[LOD].T[0].c +glm::vec3(Indx2d.x*tileW,Indx2d.y*tileH,0);

	int zNear = 1000000000, zFar = -1000000000;

	objCoord = pixel2obj(glm::vec2(TileC.x - 0.5*tileW, TileC.y - 0.5*tileH),LOD);
	tileAabb.push_back(glm::vec3(objCoord.x,objCoord.y,zNear));
	objCoord = pixel2obj(glm::vec2(TileC.x + 0.5*tileW, TileC.y - 0.5*tileH), LOD);
	tileAabb.push_back(glm::vec3(objCoord.x, objCoord.y, zNear));
	objCoord = pixel2obj(glm::vec2(TileC.x + 0.5*tileW, TileC.y + 0.5*tileH), LOD);
	tileAabb.push_back(glm::vec3(objCoord.x, objCoord.y, zNear));
	objCoord = pixel2obj(glm::vec2(TileC.x - 0.5*tileW, TileC.y + 0.5*tileH), LOD);
	tileAabb.push_back(glm::vec3(objCoord.x, objCoord.y, zNear));
	objCoord = pixel2obj(glm::vec2(TileC.x + 0.5*tileW, TileC.y - 0.5*tileH), LOD);
	tileAabb.push_back(glm::vec3(objCoord.x, objCoord.y, zFar));
	objCoord = pixel2obj(glm::vec2(TileC.x + 0.5*tileW, TileC.y + 0.5*tileH), LOD);
	tileAabb.push_back(glm::vec3(objCoord.x, objCoord.y, zFar));
	objCoord = pixel2obj(glm::vec2(TileC.x - 0.5*tileW, TileC.y + 0.5*tileH), LOD);
	tileAabb.push_back(glm::vec3(objCoord.x, objCoord.y, zFar));
	objCoord = pixel2obj(glm::vec2(TileC.x - 0.5*tileW, TileC.y - 0.5*tileH), LOD);
	tileAabb.push_back(glm::vec3(objCoord.x, objCoord.y, zFar));


}