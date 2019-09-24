#version 430

//#define SINGLE_PRECISION
//#define HIGH_RES

uniform ivec2 viewDiscretizations;
uniform ivec2 histogramDiscretizations;
uniform ivec2 spatialDiscretizations;

uniform int maxSamplingRuns;
uniform int multiSamplingRate;
uniform int samplingRunIndex;

int HistogramWidth = histogramDiscretizations.x;
int HistogramHeight = histogramDiscretizations.y;
int ViewWidth = viewDiscretizations.x;
int ViewHeight = viewDiscretizations.y;
int VolumeResolutionX = spatialDiscretizations.x;
int VolumeResolutionY = spatialDiscretizations.y;

layout(binding = 0, rg8) uniform image2D tex;
//layout(binding = 1, rgba16ui) uniform uimage2D tileTex;
layout(binding = 2, rgba32f) uniform volatile image2D floorLevel;
layout(binding = 3, rgba32f) uniform image2D ceilLevel;

//layout(binding = 4) uniform sampler2D echo_tex;

uniform float lod;

uniform int floor_w;
uniform int ceil_w;
uniform int tile_w;
uniform int tile_h;
uniform int phys_tex_dim;

uniform int cachedRayCasting;

uniform ivec3 lod_blc;
uniform ivec3 lod_trc;

uniform ivec3 ceil_blc;
uniform ivec3 floor_blc;

uniform ivec2 visible_region_blc;
uniform ivec2 visible_region_trc;

uniform float viewportWidth;

uniform vec2 blc_s, blc_l, trc_s;
uniform vec2 Ceil_blc_s, Ceil_blc_l, Ceil_trc_s;
uniform vec2 f_blc_s, f_blc_l, f_trc_s;

uniform vec2 rayOffset;

layout(std430, binding = 0) buffer NdfVoxelData
{
	coherent float histograms[];
} ndfVoxelData;

layout(std430, binding = 1) buffer NdfVoxelDataHighRes
{
	coherent float histograms[];
} ndfVoxelDataHighRes;

layout(std430, binding = 2) buffer shaderData
{
	float samples[]; // This is the important name (in the shader).
};

float linearKernel(float x)
{
	return max(0.0f, 1.0f - abs(x));
}

float gaussKernel(float x)
{
	float numerator = 0.185f * exp(-(x*x) / 0.4f);
	const float denominator = sqrt(0.2f) / sqrt(2.0f * 3.14159265359);

	return numerator / denominator;
}

/*float guassKernel(float x) {
const float mean = 0.0;
const float variance = 0.4;
const float scale = std::sqrt(variance) * std::sqrt(2.0f * 3.14159265359f); // = 1.0 / f(0)

float diff_squared = (x - mean) * (x - mean);
float exponent = -(diff_squared / (2.0f * variance));
float numerator = std::exp(exponent);
float denominator = std::sqrt(variance) * std::sqrt(2.0f * 3.14159265359f);

return scale * (numerator / denominator);
};*/

void splatReconstructionKernel(vec2 normalPosition, unsigned int dataOffset, const float unit)
{
	const vec2 histogramScale = 1.0f / vec2(float(HistogramWidth * 2 - 1), float(HistogramHeight - 1));
	const vec2 quantizedRay = vec2(normalPosition.x * float(HistogramWidth * 2 - 1), normalPosition.y * float(HistogramHeight - 1));

	const int histogramIndex = int(quantizedRay.y) * HistogramWidth * 2 + int(quantizedRay.x);

	//const float basisFunctionScale = 32.0f;//2.0f;//0.25f;

	if (histogramIndex > 0 && histogramIndex < HistogramHeight * HistogramWidth * 2)
	{
		for (int yHist = 0; yHist < HistogramHeight; ++yHist)
		{
			for (int xHist = 0; xHist < HistogramWidth * 2; ++xHist)
			{
				vec2 histogramPosition = histogramScale * vec2(float(xHist), float(yHist));

				vec2 distance = histogramPosition - normalPosition;

#if 0 // bilinear
				float sampleStrength = 1.0f * linearKernel(distance.x) * linearKernel(distance.y);
#endif

#if 1 // gaussian
				float sampleStrength = 0.025f * gaussKernel(distance.x) * gaussKernel(distance.y);
#endif

				ndfVoxelData.histograms[dataOffset + histogramIndex] += sampleStrength * unit;
			}
		}
	}
}
int calculate_lookup_pixel(int lod_w, inout int tile_sample_count, bool levelFlag, vec2 Coordinate, float F_lod, float E_lod, vec2 offset, vec2 rayoffset)
{
	//mohamed's lookup
	int num_tiles_in_h = lod_w / tile_w;
	//ivec2 bin_2d_indx = ivec2(bin_indx%HistogramWidth*2, bin_indx / HistogramWidth*2);

	vec2 PixelCoordIn_F_Lod = vec2(Coordinate - Ceil_blc_s + Ceil_blc_l) + rayoffset;// +vec2(bin_2d_indx.x / float(HistogramWidth*2), bin_2d_indx.y / float(HistogramHeight));

	//now we get pixel coordinate in exact lod
	ivec2 PixelCoordIn_E_Lod = ivec2(PixelCoordIn_F_Lod.x*pow(2, F_lod - E_lod), (PixelCoordIn_F_Lod.y*pow(2, F_lod - E_lod)));

	//end newer


	//add offset to image in PixelCoordIN_E_Lod
	uvec2 spatialCoordinate = uvec2(PixelCoordIn_E_Lod + offset);// +ivec2(bin_2d_indx.x / (HistogramWidth*2 / 2), bin_2d_indx.y / (HistogramWidth*2 / 2)));


	////the offset may get us a pixel outside the iamge
	if ((spatialCoordinate.x >= 0) && (spatialCoordinate.x < lod_w) && (spatialCoordinate.y >= 0) && (spatialCoordinate.y < lod_w))
	{
		//get tile of the pixel
		ivec2 tileindx2D = ivec2(spatialCoordinate.x / tile_w, spatialCoordinate.y / tile_w);
		int tile_indx = tileindx2D.x*num_tiles_in_h + tileindx2D.y;
		vec2 withinTileOffset = vec2(spatialCoordinate.x%tile_w, spatialCoordinate.y%tile_w);

		//get tile coordiantes in page texture

		ivec2  tileCoords_InPageTex = ivec2(tile_indx% num_tiles_in_h, tile_indx / num_tiles_in_h);

		//read physical texture coordinates from page texture
		vec4 tileCoords_InPhysTex;

		if (levelFlag)
			tileCoords_InPhysTex = imageLoad(floorLevel, tileCoords_InPageTex);
		else
			tileCoords_InPhysTex = imageLoad(ceilLevel, tileCoords_InPageTex);

		//update sample count
		//i would like to update that count only once per tile, so I'll update it only when 'w' is zero
		tile_sample_count = int(tileCoords_InPhysTex.z);

		//if ((tileCoords_InPhysTex.w == 0) && (tileCoords_InPhysTex.z < maxSamplingRuns))
		//{


		//	if ((levelFlag) && (samplingRunIndex % 4 == 0))
		//	{
		//		tileCoords_InPhysTex.w = 1;   //prevent other threads from writing to it
		//		tileCoords_InPhysTex.z++;
		//		imageStore(floorLevel, tileCoords_InPageTex, tileCoords_InPhysTex);
		//	}
		//	else if (!levelFlag)
		//	{
		//		tileCoords_InPhysTex.w = 1;   //prevent other threads from writing to it
		//		tileCoords_InPhysTex.z++;
		//		imageStore(ceilLevel, tileCoords_InPageTex, tileCoords_InPhysTex);
		//	}
		//}

		//debug
		//if ((tileCoords_InPhysTex.x < 0) || (tileCoords_InPhysTex.y < 0))
		//{
		//	return -2;
		//}
		//end debug


		tileCoords_InPhysTex.x *= tile_w;
		tileCoords_InPhysTex.y *= tile_h;

		//location in ndf tree is the physical texture location + within tile offset
		ivec2 Pixelcoord_InPhysTex = ivec2(tileCoords_InPhysTex.x + withinTileOffset.x, tileCoords_InPhysTex.y + withinTileOffset.y);

		//const unsigned int lookup = unsigned int((Pixelcoord_InPhysTex.y * VolumeResolutionX + Pixelcoord_InPhysTex.x) * HistogramHeight * HistogramWidth);
		int lookup = int((Pixelcoord_InPhysTex.y * phys_tex_dim + Pixelcoord_InPhysTex.x) * HistogramHeight * HistogramWidth);
		return lookup;
	}
	else
	{
		return -1;
	}
	//end mohamed's lookup
}

void add_sample(int binIndex, float s, int lookup)
{
	//update histogram central
	float h = float(binIndex) / 2.0f;
	int sample_precision = int(pow(10, 5));
	int sample_digits = int(pow(10, 9));

	//get old sample
	float os = ndfVoxelData.histograms[lookup + int(floor(h))];

	vec2 components = unpackHalf2x16(floatBitsToUint(os));

	if (fract(h) > 0.49f)
	{
		//update b2
		components.y += s;
	}
	else
	{
		//update b1
		components.x += s;
	}

	ndfVoxelData.histograms[lookup + int(floor(h))] = uintBitsToFloat(packHalf2x16(components));
}



// has to be recompiled each time the size changes - consider the local size division
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main()
{
	// NOTE: constant has to be equal to multisamling rate on host
	const int multiSampling = multiSamplingRate;

	const double unit = 1.0;// / double(maxSamplingRuns * multiSampling * multiSampling);

	const double unitHighRes = 1.0;// / double(maxSamplingRuns);


	uvec2 spatialCoordinate = uvec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);

	//partrick's lookup
	//const unsigned int lookup = (spatialCoordinate.y * VolumeResolutionX + spatialCoordinate.x) * HistogramHeight * HistogramWidth;


	//mohamed's lookup
	//if a pixel is outside the visible region, it need not be rendered.
	//if ((spatialCoordinate.x < visible_region_blc.x) || (spatialCoordinate.x >= visible_region_trc.x) || (spatialCoordinate.y < visible_region_blc.y) || (spatialCoordinate.y >= visible_region_trc.y))
	//	return;

	if ((spatialCoordinate.x < blc_s.x) || (spatialCoordinate.x >= trc_s.x) || (spatialCoordinate.y < blc_s.y) || (spatialCoordinate.y >= trc_s.y))
		return;

	int lookup;
	int lookups[5] = { -1, -1, -1, -1, -1 };
	int tile_sample_count[5] = { 0, 0, 0, 0, 0 };


	vec2 rayOffset = vec2(mod(samples[samplingRunIndex], 128.0f) / 128.0f, (floor(samples[samplingRunIndex] / 128.0f)) / 128.0f);
	rayOffset.xy *= 1.0f / viewportWidth;
	//calculate rayoffset


	// FIXME: race condition
	//const unsigned int downLookup = (spatialCoordinate.y / 2) * VolumeResolutionX + (spatialCoordinate.x / 2)) * HistogramHeight * HistogramWidth;

	ivec2 multisampleSpatialCoordinate = ivec2(spatialCoordinate.xy);
	vec2 newRay = imageLoad(tex, multisampleSpatialCoordinate).xy;

	vec2 quantizedRay = vec2(newRay.x * float((HistogramWidth * 2) - 1), newRay.y * float(HistogramHeight - 1));

	// NOTE: the total amount of energy will only be available if the max. number of samples is reached.
	// before that the renderer has to upscale the ndf samples according to the total energy.
	const int histogramIndexCentral = int(quantizedRay.y) * HistogramWidth * 2 + int(quantizedRay.x);
	if (histogramIndexCentral < 0 || histogramIndexCentral >= HistogramHeight * HistogramWidth * 2) {
		return;
	}

	// TODO: use epanechnikov kernel instead of piecewise linear (supposed to be better than gaussian wrt MSE)

	dvec2 bilinearWeights = dvec2(double(fract(quantizedRay.x)), double(fract(quantizedRay.y)));

	//we now know what pixel we are looking at in the floor level, and where the sample is
	//we should now just know it's corresponding pixel in the ceil

	//now we need spatial coordinate to be in the ceil level
	if ((spatialCoordinate.x < Ceil_blc_s.x) || (spatialCoordinate.x >= Ceil_trc_s.x) || (spatialCoordinate.y < Ceil_blc_s.y) || (spatialCoordinate.y >= Ceil_trc_s.y))
		return;

	//new
	if (lod - int(lod) == 0)
	{

	}
	else
	{
		//for the sample in the given pixel, map the sample to different pixels in floor and ciel lods

		//sample floor and ceil
		int ceil_lookup = calculate_lookup_pixel(ceil_w, tile_sample_count[4], false, vec2(spatialCoordinate), ceil(lod), ceil(lod), vec2(0, 0), vec2(0, 0));
		if (ceil_lookup >= 0)
		{
			//float val;
			rayOffset.xy = vec2(0.1, 0.1);
			rayOffset.xy *= 1.0f / viewportWidth;
			int child_1_lookup = calculate_lookup_pixel(floor_w, tile_sample_count[0], true, vec2(spatialCoordinate), ceil(lod), int(lod), vec2(0, 0), -viewportWidth*rayOffset);

			rayOffset.xy = vec2(0.1, .9);
			rayOffset.xy *= 1.0f / viewportWidth;
			int child_2_lookup = calculate_lookup_pixel(floor_w, tile_sample_count[0], true, vec2(spatialCoordinate), ceil(lod), int(lod), vec2(0, 0), -viewportWidth*rayOffset);

			rayOffset.xy = vec2(.9, 0.1);
			rayOffset.xy *= 1.0f / viewportWidth;
			int child_3_lookup = calculate_lookup_pixel(floor_w, tile_sample_count[0], true, vec2(spatialCoordinate), ceil(lod), int(lod), vec2(0, 0), -viewportWidth*rayOffset);

			rayOffset.xy = vec2(.9, .9);
			rayOffset.xy *= 1.0f / viewportWidth;
			int child_4_lookup = calculate_lookup_pixel(floor_w, tile_sample_count[0], true, vec2(spatialCoordinate), ceil(lod), int(lod), vec2(0, 0), -viewportWidth*rayOffset);

			if ((child_1_lookup >= 0) && (child_2_lookup >= 0) && (child_3_lookup >= 0) && (child_4_lookup >= 0))
			{
				float ceil_os, os, h;
				vec2 ceil_components, components_sum;


				int i = histogramIndexCentral;

				h = float(i) / 2.0f;
				//get old sample in the ceil
				ceil_os = ndfVoxelData.histograms[ceil_lookup + int(floor(h))];
				ceil_components = unpackHalf2x16(floatBitsToUint(ceil_os));


				if (fract(h) > 0.49f)
				{
					//get sum of old samples in the floor
					components_sum = vec2(0, 0);
					os = ndfVoxelData.histograms[child_1_lookup + int(floor(h))];
					components_sum.y += (unpackHalf2x16(floatBitsToUint(os))).y;

					os = ndfVoxelData.histograms[child_2_lookup + int(floor(h))];
					components_sum.y += (unpackHalf2x16(floatBitsToUint(os))).y;

					os = ndfVoxelData.histograms[child_3_lookup + int(floor(h))];
					components_sum.y += (unpackHalf2x16(floatBitsToUint(os))).y;

					os = ndfVoxelData.histograms[child_4_lookup + int(floor(h))];
					components_sum.y += (unpackHalf2x16(floatBitsToUint(os))).y;
				}
				else
				{
					components_sum = vec2(0, 0);
					os = ndfVoxelData.histograms[child_1_lookup + int(floor(h))];
					components_sum.x += (unpackHalf2x16(floatBitsToUint(os))).x;

					os = ndfVoxelData.histograms[child_2_lookup + int(floor(h))];
					components_sum.x += (unpackHalf2x16(floatBitsToUint(os))).x;

					os = ndfVoxelData.histograms[child_3_lookup + int(floor(h))];
					components_sum.x += (unpackHalf2x16(floatBitsToUint(os))).x;

					os = ndfVoxelData.histograms[child_4_lookup + int(floor(h))];
					components_sum.x += (unpackHalf2x16(floatBitsToUint(os))).x;
				}

				//set the ceil value to the sum of the floor bins and what ever was there in the ceil
				ndfVoxelData.histograms[ceil_lookup + int(floor(h))] = uintBitsToFloat(packHalf2x16(components_sum + ceil_components));
			}
		}

	}

	//end new

	// TODO: on the fly eBRDF splatting - even sparse approximation?


}