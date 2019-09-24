#version 430


uniform ivec2 histogramDiscretizations;

int HistogramWidth = histogramDiscretizations.x;
int HistogramHeight = histogramDiscretizations.y;

uniform int tile_w;
uniform int tile_h;

uniform int win_w, win_h;

uniform vec2 blc_s, blc_l, trc_s;

uniform int floor_w, floor_h;
uniform int ceil_w, ceil_h;

uniform float lod;

uniform float selectedPixelsSize;

layout(binding = 1, rgba32f) uniform image2D floorLevel;
layout(binding = 2, rgba32f) uniform image2D ceilLevel;

layout(std430) buffer NdfVoxelData
{
	coherent float histograms[];
} ndfVoxelData;

layout(std430) buffer sampleCount
{
	float sample_count[]; // This is the important name (in the shader).
};

layout(std430) buffer avgNDF
{
	float avg_NDF[]; // This is the important name (in the shader).
};

layout(std430) buffer binAreas
{
	double bArea[]; // This is the important name (in the shader).
};

layout(std430) buffer similarityLimitsF
{
	float sim_limitsf[]; // This is the important name (in the shader).
};

layout(std430) buffer selectedPixels
{
	float sPixels[]; // This is the important name (in the shader).
};

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;


int calculate_lookup(int lod_w, int lod_h, inout int tile_sample_count, bool levelFlag, vec2 Coordinate, float F_lod, float E_lod, vec2 offset, inout float du, inout float dv, inout ivec2 pixel_in_exact)
{
	//if (levelFlag)
	//int x = calculate_lookup_pixel(lod_w,lod_h,tile_sample_count, levelFlag, Coordinate, F_lod, E_lod,offset, offset);
	//mohamed's lookup
	int num_tiles_in_w = lod_w / tile_w;
	int num_tiles_in_h = lod_h / tile_h;

	////get spatialCoordinate in image space
	//ivec2 blc_tile_indx = ivec2(int(lod_blc.x) / num_tiles_in_h, int(lod_blc.x) % num_tiles_in_h);
	//ivec2 blc_pixel_coordinate = ivec2(blc_tile_indx.x*tile_w, blc_tile_indx.y*tile_w);

	////to transform coordinate to image space, we add coordiante to blc pixel coordinate (blc mapped to image space) and add the offset of bottomleftconer from blc
	//ivec2 PixelCoordIn_F_Lod = visible_region_blc + blc_pixel_coordinate + ivec2(lod_blc.y, lod_blc.z) + ivec2(Coordinate);

	ivec2 PixelCoordIn_F_Lod = ivec2(Coordinate - blc_s + blc_l);

	//now we get pixel coordinate in exact lod
	//ivec2 PixelCoordIn_E_Lod = ivec2(PixelCoordIn_F_Lod.x*pow(2, F_lod - E_lod), PixelCoordIn_F_Lod.y*pow(2, F_lod - E_lod));

	//new
	//ivec2 center_F_Lod = ivec2((lod_w*pow(2, E_lod - F_lod)) / 2, (lod_w*pow(2, E_lod - F_lod)) / 2);
	//ivec2 center_E_Lod = ivec2(lod_w / 2, lod_w / 2);
	//vec2 dir = vec2(PixelCoordIn_F_Lod - center_F_Lod);
	//ivec2 PixelCoordIn_E_Lod = center_E_Lod + ivec2((F_lod / E_lod)*dir);
	//end new

	//newer
	du = fract(PixelCoordIn_F_Lod.x*pow(2, F_lod - E_lod));
	dv = fract(PixelCoordIn_F_Lod.y*pow(2, F_lod - E_lod));
	ivec2 PixelCoordIn_E_Lod = ivec2(PixelCoordIn_F_Lod.x*pow(2, F_lod - E_lod), PixelCoordIn_F_Lod.y*pow(2, F_lod - E_lod));
	pixel_in_exact = PixelCoordIn_E_Lod;
	//end newer


	//add offset to image in PixelCoordIN_E_Lod
	uvec2 spatialCoordinate = uvec2(PixelCoordIn_E_Lod + offset);

	//the offset may get us a pixel outside the iamge

	if ((spatialCoordinate.x >= 0) && (spatialCoordinate.x < lod_w) && (spatialCoordinate.y >= 0) && (spatialCoordinate.y < lod_h))
	{
		//get tile of the pixel
		ivec2 tileindx2D = ivec2(spatialCoordinate.x / tile_w, spatialCoordinate.y / tile_h);
		int tile_indx = tileindx2D.y*num_tiles_in_w + tileindx2D.x;
		vec2 withinTileOffset = vec2(spatialCoordinate.x%tile_w, spatialCoordinate.y%tile_h);

		//get tile coordiantes in page texture

		ivec2  tileCoords_InPageTex = ivec2(tile_indx% num_tiles_in_w, tile_indx / num_tiles_in_w);

		//read physical texture coordinates from page texture
		vec4 tileCoords_InPhysTex;

		if (levelFlag)
			tileCoords_InPhysTex = imageLoad(floorLevel, tileCoords_InPageTex);
		else
			tileCoords_InPhysTex = imageLoad(ceilLevel, tileCoords_InPageTex);


		//debug
		if ((tileCoords_InPhysTex.x < 0))
		{
			return -2;
		}
		//end debug

		////debug
		//if (tileCoords_InPhysTex.z == 0)   
		//{
		//	if (tileCoords_InPhysTex.z < 0)
		//		return -5;
		//}
		////end debug

		tile_sample_count = int(tileCoords_InPhysTex.z);
		//tileCoords_InPhysTex.x *= tile_w;
		//tileCoords_InPhysTex.y *= tile_h;

		//location in ndf tree is the physical texture location + within tile offset
		//ivec2 Pixelcoord_InPhysTex = ivec2(tileCoords_InPhysTex.x + withinTileOffset.x, tileCoords_InPhysTex.y + withinTileOffset.y);
		int Pixelcoord_InPhysTex = int((tileCoords_InPhysTex.x*tile_w*tile_h) + (withinTileOffset.y*tile_w + withinTileOffset.x));

		//const unsigned int lookup = unsigned int((Pixelcoord_InPhysTex.y * VolumeResolutionX + Pixelcoord_InPhysTex.x) * HistogramHeight * HistogramWidth);
		int lookup = int(Pixelcoord_InPhysTex * HistogramHeight * HistogramWidth);
		return lookup;
	}
	else
	{
		return -1;
	}
	//end mohamed's lookup
}

void main()
{
	int binIndx=int(gl_GlobalInvocationID.x);
	if (bArea[binIndx] == 0.0)
	{
		avg_NDF[binIndx] = 0.0f;
		return;
	}
	//loop over all pixels in selected pixels and compute bin 'binindx' in avg ndf
	vec2 p;
	int lookup;
	float f_du, f_dv;
	ivec2 garbage;
	int sc;
	//float samples;
	int lookups[5] = { -1, -1, -1, -1, -1 };
	float samples[5] = { 0.0f,0.0f,0.0f,0.0f,0.0f };
	float vals[5] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	float interpolatedVal=0.0f;
	float floor_val,ceil_val;

	float val=0.0f;

	for (int i = 0; i < selectedPixelsSize*2; i=i+2)
	{
		//get pixel
		p = vec2(sPixels[i], win_h-sPixels[i + 1]);

		//calculate lookup and sample count , and ndf bin value for floor pixels
		//lookup = calculate_lookup(floor_w, floor_h, sc , true, p, lod, floor(lod), vec2(0, 0), f_du, f_dv, garbage);
		lookups[0] = calculate_lookup(floor_w, floor_h, sc, true, p, lod, floor(lod), vec2(0, 0), f_du, f_dv, garbage);
		samples[0] = sample_count[lookups[0] / (HistogramWidth*HistogramHeight)];
		vals[0] = (ndfVoxelData.histograms[lookups[0] + binIndx] / samples[0]) ;

		lookups[1] = calculate_lookup(floor_w, floor_h, sc, true, p, lod, floor(lod), vec2(1, 0), f_du, f_dv, garbage);
		samples[1] = sample_count[lookups[1] / (HistogramWidth*HistogramHeight)];
		vals[1] = (ndfVoxelData.histograms[lookups[1] + binIndx] / samples[1]);

		lookups[2] = calculate_lookup(floor_w, floor_h, sc, true, p, lod, floor(lod), vec2(0, 1), f_du, f_dv, garbage);
		samples[2] = sample_count[lookups[2] / (HistogramWidth*HistogramHeight)];
		vals[2] = (ndfVoxelData.histograms[lookups[2] + binIndx] / samples[2]) ;

		lookups[3] = calculate_lookup(floor_w, floor_h, sc, true, p, lod, floor(lod), vec2(1, 1), f_du, f_dv, garbage);
		samples[3] = sample_count[lookups[3] / (HistogramWidth*HistogramHeight)];
		vals[3] = (ndfVoxelData.histograms[lookups[3] + binIndx] / samples[3]) ;

		floor_val = f_du         *f_dv         *vals[3] +
			        f_du         *(1.0f - f_dv)*vals[1] +
			        (1.0f - f_du)*f_dv         *vals[2] +
			        (1.0f - f_du)*(1.0f - f_dv)*vals[0];

		ceil_val = (vals[0] + vals[1] + vals[2] + vals[3]) / 4.0f;
		
		//add normalized ndf bins
		val+= mix(floor_val, ceil_val, fract(lod));
		
		//set color of selected pixels to red
		sim_limitsf[2 + int(win_w*p.y + p.x)] = -2.0f;	
	}

	avg_NDF[binIndx] = val/selectedPixelsSize;
}