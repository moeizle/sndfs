#version 430

//#define SINGLE_PRECISION
//#define HIGH_RES

uniform ivec2 histogramDiscretizations;
uniform ivec2 spatialDiscretizations;

const float PI = 3.141592f;

int HistogramWidth = histogramDiscretizations.x;
int HistogramHeight = histogramDiscretizations.y;
int VolumeResolutionX = spatialDiscretizations.x;
int VolumeResolutionY = spatialDiscretizations.y;


layout(binding = 2, rgba32f) uniform volatile image2D currentLevel;
layout(binding = 3, rgba32f) uniform image2D downsamplingLevel;

//layout(binding = 4) uniform sampler2D echo_tex;

uniform float lod;
uniform float dLod;

uniform int cur_w;
uniform int downsampled_w;
uniform int cur_h;
uniform int downsampled_h;
uniform int tile_w;
uniform int tile_h;
uniform int phys_tex_dim;

uniform int circularPattern;
uniform float sampleW;

uniform int colorMapSize;


uniform int cachedRayCasting;


uniform vec2 blc_s, blc_l, trc_s;

layout(std430) buffer NdfVoxelData
{
	coherent float histograms[];
} ndfVoxelData;


layout(std430) buffer sampleCount
{
	float sample_count[]; // This is the important name (in the shader).
};

layout(std430) buffer colorMap
{
	float color_map[]; // This is the important name (in the shader).
};

layout(std430) buffer ndfOverlay
{
	float ndf_overlay[]; // This is the important name (in the shader).
};

layout(std430) buffer preIntegratedBins
{
	float binColor[]; // This is the important name (in the shader).
};

layout(std430) buffer binAreas
{
	double bArea[]; // This is the important name (in the shader).
};


int calculate_lookup_pixel(int lod_w, int lod_h, inout int tile_sample_count, bool levelFlag, vec2 Coordinate, float F_lod, float E_lod, vec2 offset, vec2 rayoffset)

{
	int num_tiles_in_w = lod_w / tile_w;
	int num_tiles_in_h = lod_h / tile_h;


	vec2 PixelCoordIn_F_Lod = vec2(Coordinate - blc_s + blc_l) + rayoffset;


	//now we get pixel coordinate in exact lod
	ivec2 PixelCoordIn_E_Lod = ivec2(PixelCoordIn_F_Lod.x*pow(2, F_lod - E_lod), (PixelCoordIn_F_Lod.y*pow(2, F_lod - E_lod)));

	//add offset to image in PixelCoordIN_E_Lod
	uvec2 spatialCoordinate = uvec2(PixelCoordIn_E_Lod + offset);


	////the offset may get us a pixel outside the iamge
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
			tileCoords_InPhysTex = imageLoad(currentLevel, tileCoords_InPageTex);
		else
			tileCoords_InPhysTex = imageLoad(downsamplingLevel, tileCoords_InPageTex);

		//update sample count
		//i would like to update that count only once per tile, so I'll update it only when 'w' is zero
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



// has to be recompiled each time the size changes - consider the local size division
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()
{

	float unit;


	unit = 1.0;// / double(maxSamplingRuns * multiSampling * multiSampling);
	


	const double unitHighRes = 1.0;// / double(maxSamplingRuns);


	uvec2 spatialCoordinate = uvec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);

	int lookup;
	int lookups[5] = { -1, -1, -1, -1, -1 };
	int tile_sample_count[5] = { 0, 0, 0, 0, 0 };


	//1-spatial coordinate is the coordinate of the pixel in the lod that we want to downsample to
	//2- now, depending on LOD that we are at (lod), vs the LOD that we downsample too (dLod), we know that each pixel with coordinates 'spatialCoordinate', corresponds to (2*2^(downsampleLOD-currentLOD)) pixels 
	//in the current LOD.
	//3- we loop over these pixels in the current lod, and calculate an average ndf.
	//4- get color code for this average ndf (we need some descriptor of NDF)
	//5- put this color in the overlayNDFcolorsssbo.

	float avgNDF[64];
	float samples;
	for (int i = 0; i < pow(2, floor(dLod) - floor(lod)); i++)
	{
		for (int j = 0; j < pow(2, floor(dLod) - floor(lod)); j++)
		{
			//spatial coordinate is in the higher( downsampled), dlod
			//we calculate all the pixels that correspond to it, on our current lod, lod. 
			lookup = calculate_lookup_pixel(cur_w, cur_h, tile_sample_count[0], true, vec2(spatialCoordinate), floor(dLod), floor(lod), vec2(i, j), vec2(0, 0));
			//if (lookup >= 0)
			{
				if (i == 0 && j == 0)  //intialize samples and avgndf to value at 0,0
				{
					for (int k = 0; k < 64; k++)
					{
						avgNDF[k] = ndfVoxelData.histograms[lookup + k] / sample_count[lookup / (8 * 8)];
					}
				}
				else
				{
					for (int k = 0; k < 64; k++)
					{
						avgNDF[k] += ndfVoxelData.histograms[lookup + k]/sample_count[lookup / (8 * 8)];
					}
				}
			}
		}
	}


	vec3 color = vec3(0, 0, 0);
#if 1  //old
	
	vec3 kColor;
	float factor = colorMapSize / 64.0f;
	ivec2 transfer;

	const vec3 leftColor = vec3(.5, 0, 0);// vec3(0.35f, 0.65f, 0.8f);
	const vec3 rightColor = vec3(0f, 0.5f, 0f);
	const vec3 bottomColor = vec3(0.5f, 0.5f, 0.5f);
	const vec3 topColor = vec3(0.0f, 0.0f, 0.5f);

	//normalize avgNDF
	for (int k = 0; k < 64; k++)
	{
		avgNDF[k] = avgNDF[k] / (pow(2, floor(dLod) - floor(lod))* pow(2, floor(dLod) - floor(lod)));

		//get color for average ndf
#if 1 //color from color map
		kColor = vec3(color_map[3 * int(k*factor)], color_map[3 * int(k*factor) + 1], color_map[3 * int(k*factor) + 2]);
		color = color + (avgNDF[k] * kColor);
#else
		transfer = ivec2(k % 8, k / 8);
		color = color + avgNDF[k]*(0.5f * leftColor * (1.0f - transfer.x) + 0.5f * rightColor * transfer.x +
			0.5f * bottomColor * (1.0f - transfer.y) + 0.5f * topColor * transfer.y);
#endif

		//debug
		//color = vec3(0, 1, 0);
	}

#else  //new

	double binArea, binCDF, binNDF;
	float totalNumberOfSamples;
	vec3 binNTF;
	float ndfSample = 0.0f;
	totalNumberOfSamples = (pow(2, floor(dLod) - floor(lod))* pow(2, floor(dLod) - floor(lod)));

	for (int histogramY = 0; histogramY < HistogramHeight; histogramY++)
	{
		for (int histogramX = 0; histogramX < HistogramWidth; histogramX++)
		{

			int binIndex = histogramY * HistogramWidth + histogramX;
			ndfSample = avgNDF[binIndex];

			vec3 centralShadingPreComputed = vec3(binColor[binIndex * 3], binColor[binIndex * 3 + 1], binColor[binIndex * 3 + 2]);


			if ((bArea[binIndex] != 0.0) && (bArea[binIndex] == bArea[binIndex]))
				binArea = bArea[binIndex];
			else
				binArea = 0.0;

			if (binArea != 0.0f)
			{
				binCDF = double(ndfSample / (totalNumberOfSamples));
				binNDF = binCDF / binArea;// / area;
				binNTF = centralShadingPreComputed;// *area;
				color += float(binNDF)* binNTF;// centralShading;// / maxSamplingRuns;
			}
		}
	}
#endif 

	//put color in overlay ndf ssbo
	for (int i = 0; i < pow(2, floor(dLod) - floor(lod)); i++)
	{
		for (int j = 0; j < pow(2, floor(dLod) - floor(lod)); j++)
		{
			//spatial coordinate is in the higher( downsampled), dlod
			//we calculate all the pixels that correspond to it, on our current lod, lod. 
			lookup = calculate_lookup_pixel(cur_w, cur_h, tile_sample_count[0], true, vec2(spatialCoordinate), floor(dLod), floor(lod), vec2(i, j), vec2(0, 0));
			ndf_overlay[(3 * lookup / (8 * 8))  ] = color.x;
			ndf_overlay[(3 * lookup / (8 * 8))+1] = color.y;
			ndf_overlay[(3 * lookup / (8 * 8))+2] = color.z;
		}
	}


}