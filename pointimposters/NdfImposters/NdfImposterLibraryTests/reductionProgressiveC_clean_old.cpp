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

layout(binding = 0, rg8  ) uniform image2D tex;
//layout(binding = 1, rgba16ui) uniform uimage2D tileTex;
layout(binding = 2, rgba32f  ) uniform volatile image2D floorLevel;
layout(binding = 3, rgba32f  ) uniform image2D ceilLevel;

//layout(binding = 4) uniform sampler2D echo_tex;

uniform float lod;

uniform int floor_w;
uniform int ceil_w;
uniform int tile_w;
uniform int tile_h;
uniform int phys_tex_dim;

uniform ivec3 lod_blc;
uniform ivec3 lod_trc;

uniform ivec3 ceil_blc;
uniform ivec3 floor_blc;

uniform ivec2 visible_region_blc;
uniform ivec2 visible_region_trc;

uniform float viewportWidth;

uniform vec2 blc_s, blc_l, trc_s;

layout(std430, binding = 0) buffer NdfVoxelData 
{
	coherent float histograms[];
} ndfVoxelData;

layout(std430, binding = 1) buffer NdfVoxelDataHighRes 
{
	coherent float histograms[];
} ndfVoxelDataHighRes;

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
	const vec2 histogramScale = 1.0f / vec2(float(HistogramWidth-1), float(HistogramHeight-1));
	const vec2 quantizedRay = vec2(normalPosition.x * float(HistogramWidth-1), normalPosition.y * float(HistogramHeight-1));

	const int histogramIndex = int(quantizedRay.y) * HistogramWidth +int(quantizedRay.x);

	//const float basisFunctionScale = 32.0f;//2.0f;//0.25f;

	if(histogramIndex > 0 && histogramIndex < HistogramHeight * HistogramWidth) 
	{
		for(int yHist = 0; yHist < HistogramHeight; ++yHist) 
		{
			for(int xHist = 0; xHist < HistogramWidth; ++xHist) 
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
int calculate_lookup_pixel(int lod_w, inout int tile_sample_count, bool levelFlag, vec2 Coordinate, float F_lod, float E_lod, vec2 offset,vec2 rayoffset)
{
	//mohamed's lookup
	int num_tiles_in_h = lod_w / tile_w;
	//ivec2 bin_2d_indx = ivec2(bin_indx%HistogramWidth, bin_indx / HistogramWidth);

	vec2 PixelCoordIn_F_Lod = vec2(Coordinate - blc_s + blc_l)+rayoffset;// +vec2(bin_2d_indx.x / float(HistogramWidth), bin_2d_indx.y / float(HistogramHeight));

	//now we get pixel coordinate in exact lod
	ivec2 PixelCoordIn_E_Lod = ivec2((PixelCoordIn_F_Lod.x*pow(2, F_lod - E_lod)),(PixelCoordIn_F_Lod.y*pow(2, F_lod - E_lod)));

	//end newer


	//add offset to image in PixelCoordIN_E_Lod
	uvec2 spatialCoordinate = uvec2(PixelCoordIn_E_Lod + offset);// +ivec2(bin_2d_indx.x / (HistogramWidth / 2), bin_2d_indx.y / (HistogramWidth / 2)));


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

		if ((tileCoords_InPhysTex.w == 0) && (tileCoords_InPhysTex.z < maxSamplingRuns))
		{


			if ((levelFlag) && (samplingRunIndex % 4 == 0))
			{
				tileCoords_InPhysTex.w = 1;   //prevent other threads from writing to it
				tileCoords_InPhysTex.z++;
				imageStore(floorLevel, tileCoords_InPageTex, tileCoords_InPhysTex);
			}
			else if (!levelFlag)
			{
				tileCoords_InPhysTex.w = 1;   //prevent other threads from writing to it
				tileCoords_InPhysTex.z++;
				imageStore(ceilLevel, tileCoords_InPageTex, tileCoords_InPhysTex);
			}
		}

		//debug
		if ((tileCoords_InPhysTex.x < 0) || (tileCoords_InPhysTex.y < 0))
		{
			return -2;
		}
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
// has to be recompiled each time the size changes - consider the local size division
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
  
void main()
{
	// NOTE: constant has to be equal to multisamling rate on host
	const int multiSampling = multiSamplingRate;

	const double unit = 1.0 / double(maxSamplingRuns * multiSampling * multiSampling);

	const double unitHighRes = 1.0 / double(maxSamplingRuns);

	 
	uvec2 spatialCoordinate = uvec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);

	//partrick's lookup
	//const unsigned int lookup = (spatialCoordinate.y * VolumeResolutionX + spatialCoordinate.x) * HistogramHeight * HistogramWidth;


	//mohamed's lookup
	//if a pixel is outside the visible region, it need not be rendered.
	//if ((spatialCoordinate.x < visible_region_blc.x) || (spatialCoordinate.x >= visible_region_trc.x) || (spatialCoordinate.y < visible_region_blc.y) || (spatialCoordinate.y >= visible_region_trc.y))
	//	return;

	if ((spatialCoordinate.x < blc_s.x) || (spatialCoordinate.x >= trc_s.x) || (spatialCoordinate.y < blc_s.y) || (spatialCoordinate.y >= trc_s.y))
		return;

	unsigned int lookup;
	int lookups[5] = { -1, -1, -1, -1, -1 };
	int tile_sample_count[5] = { 0, 0, 0, 0, 0 };

	if (lod - int(lod) == 0)
	{

		int num_tiles_in_h = floor_w / tile_w;


		//map pixel to lod space
		ivec2 coordinate = ivec2( spatialCoordinate- blc_s + blc_l);
		ivec2 tileindx_2d = ivec2(coordinate.x/tile_w,coordinate.y/tile_h);
		int tile_indx = tileindx_2d.x*num_tiles_in_h + tileindx_2d.y;
		vec2 withinTileOffset = vec2(coordinate.x%tile_w, coordinate.y%tile_h);
	
		
		//get tile coordiantes in page texture

		ivec2  tileCoords_InPageTex = ivec2(tile_indx% num_tiles_in_h, tile_indx / num_tiles_in_h);

		//read physical texture coordinates from page texture
		vec4 tileCoords_InPhysTex = imageLoad(floorLevel, tileCoords_InPageTex);

		//update sample count
		//i would like to update that count only once per tile, so I'll update it if the flag (tile_coord_inphystex.w) =false
		if ((tileCoords_InPhysTex.w == 0) && (tileCoords_InPhysTex.z<maxSamplingRuns))
		{
			tileCoords_InPhysTex.z++;
			tileCoords_InPhysTex.w= 1;   //prevent other threads from writing to it
			imageStore(floorLevel, tileCoords_InPageTex, tileCoords_InPhysTex);
		}


		tileCoords_InPhysTex.x *= tile_w;
		tileCoords_InPhysTex.y *= tile_h;

		
		//location in ndf tree is the physical texture location + within tile offset
		ivec2 Pixelcoord_InPhysTex = ivec2(tileCoords_InPhysTex.x + withinTileOffset.x, tileCoords_InPhysTex.y + withinTileOffset.y);

		//const unsigned int lookup = unsigned int((Pixelcoord_InPhysTex.y * VolumeResolutionX + Pixelcoord_InPhysTex.x) * HistogwramHeight * HistogramWidth);
		lookup = unsigned int((Pixelcoord_InPhysTex.y * phys_tex_dim + Pixelcoord_InPhysTex.x) * HistogramHeight * HistogramWidth);
		lookups[0] = int(lookup);
		tile_sample_count[0] = int(tileCoords_InPhysTex.z);
		//end mohamed's lookup
	}
	else
	{
		//get five lookups 
		//four for floor lod
		//lookups[0] = calculate_lookup(floor_w, tile_sample_count[0], true, vec2(spatialCoordinate), lod, int(lod), vec2(0, 0));
		//lookups[1] = calculate_lookup(floor_w, tile_sample_count[1], true, vec2(spatialCoordinate), lod, int(lod), vec2(0, 1));
		//lookups[2] = calculate_lookup(floor_w, tile_sample_count[2], true, vec2(spatialCoordinate), lod, int(lod), vec2(1, 0));
		//lookups[3] = calculate_lookup(floor_w, tile_sample_count[3], true, vec2(spatialCoordinate), lod, int(lod), vec2(1, 1));

		//lookups[4] = calculate_lookup(ceil_w, tile_sample_count[4], false, vec2(spatialCoordinate), lod, ceil(lod), vec2(0, 0));
	}


	//calculate rayoffset
	float squareRootMaxSamplesF = sqrt(float(maxSamplingRuns));
	int squareRootMaxSamples = int(squareRootMaxSamplesF);

	// prime in residue class wraps around and introduces a pseudo random sampling pattern that reduces aliasing
	//const float prime = 3163.0f;
	const float prime = 149.0f;
	float sampleIndex = float(samplingRunIndex) * prime;

	// offset by half dimension ensures that the first ray is in the center of the pixel
	float horizontalSampleIndex = mod((sampleIndex + squareRootMaxSamplesF * 0.5f), squareRootMaxSamplesF);
	float verticalSampleIndex = mod(((sampleIndex / squareRootMaxSamplesF) + squareRootMaxSamplesF * 0.5f), squareRootMaxSamplesF);

	// NOTE: this dependes on total samples and resolution
	const float smoothingFactor = 1.0f; // 1.0f = no smoothing, 0.0f = center ray only
	const float rayOffsetStrength = smoothingFactor / (squareRootMaxSamplesF * viewportWidth);
	vec2 rayOffset = rayOffsetStrength * vec2(horizontalSampleIndex, verticalSampleIndex);
	//calculate rayoffset


	// FIXME: race condition
	//const unsigned int downLookup = (spatialCoordinate.y / 2) * VolumeResolutionX + (spatialCoordinate.x / 2)) * HistogramHeight * HistogramWidth;

	for (int y = 0; y < multiSampling; ++y)
	{
		for (int x = 0; x < multiSampling; ++x)
		{
			ivec2 multisampleSpatialCoordinate = ivec2(spatialCoordinate.xy) * multiSampling + ivec2(x, y);
			vec2 newRay = imageLoad(tex, multisampleSpatialCoordinate).xy;

			vec2 quantizedRay = vec2(newRay.x * float(HistogramWidth - 1), newRay.y * float(HistogramHeight - 1));

			// NOTE: the total amount of energy will only be available if the max. number of samples is reached.
			// before that the renderer has to upscale the ndf samples according to the total energy.
			const int histogramIndexCentral = int(quantizedRay.y) * HistogramWidth + int(quantizedRay.x);
			const int histogramIndexHorizontal = int(quantizedRay.y) * HistogramWidth + int(quantizedRay.x) + 1;
			const int histogramIndexVertical = (int(quantizedRay.y) + 1) * HistogramWidth + int(quantizedRay.x);
			const int histogramIndexHorizontalVertical = (int(quantizedRay.y) + 1) * HistogramWidth + int(quantizedRay.x) + 1;

			// TODO: use epanechnikov kernel instead of piecewise linear (supposed to be better than gaussian wrt MSE)

			dvec2 bilinearWeights = dvec2(double(fract(quantizedRay.x)), double(fract(quantizedRay.y)));

			// NOTE: the total amount of energy will only be available if the max. number of samples is reached.
			// before that the renderer has to upscale the ndf samples according to the total energy.

			//old
			//if (histogramIndexCentral > 0 && histogramIndexCentral < HistogramHeight * HistogramWidth)
			//{
			//	ndfVoxelData.histograms[lookup + histogramIndexCentral] += float((1.0 - bilinearWeights.x) * (1.0 - bilinearWeights.y) * unit);
			//}
			//if (histogramIndexHorizontal > 0 && histogramIndexHorizontal < HistogramHeight * HistogramWidth)
			//{ 
			//	ndfVoxelData.histograms[lookup + histogramIndexHorizontal] += float((bilinearWeights.x) * (1.0 - bilinearWeights.y) * unit);
			//}
			//if (histogramIndexVertical > 0 && histogramIndexVertical < HistogramHeight * HistogramWidth)
			//{
			//	ndfVoxelData.histograms[lookup + histogramIndexVertical] += float((1.0 - bilinearWeights.x) * (bilinearWeights.y) * unit);
			//}
			//if (histogramIndexHorizontalVertical > 0 && histogramIndexHorizontalVertical < HistogramHeight * HistogramWidth)
			//{
			//	ndfVoxelData.histograms[lookup + histogramIndexHorizontalVertical] += float((bilinearWeights.x) * (bilinearWeights.y) * unit);
			//}
			//end old

			//new
			if (lod - int(lod) == 0)
			{
				//if a tile reached its maximum samples, return
				if (tile_sample_count[0] >= maxSamplingRuns)
					return;

				//else fill with probability
				if (histogramIndexCentral > 0 && histogramIndexCentral < HistogramHeight * HistogramWidth)
				{
					ndfVoxelData.histograms[lookup + histogramIndexCentral] += float(unit);
				}

				//if (histogramIndexCentral > 0 && histogramIndexCentral < HistogramHeight * HistogramWidth)
				//{
				//	ndfVoxelData.histograms[lookup + histogramIndexCentral] += float((1.0 - bilinearWeights.x) * (1.0 - bilinearWeights.y) * unit);
				//}
				//if (histogramIndexHorizontal > 0 && histogramIndexHorizontal < HistogramHeight * HistogramWidth)
				//{
				//	ndfVoxelData.histograms[lookup + histogramIndexHorizontal] += float((bilinearWeights.x) * (1.0 - bilinearWeights.y) * unit);
				//}
				//if (histogramIndexVertical > 0 && histogramIndexVertical < HistogramHeight * HistogramWidth)
				//{
				//	ndfVoxelData.histograms[lookup + histogramIndexVertical] += float((1.0 - bilinearWeights.x) * (bilinearWeights.y) * unit);
				//}
				//if (histogramIndexHorizontalVertical > 0 && histogramIndexHorizontalVertical < HistogramHeight * HistogramWidth)
				//{
				//	ndfVoxelData.histograms[lookup + histogramIndexHorizontalVertical] += float((bilinearWeights.x) * (bilinearWeights.y) * unit);
				//}
			}
			else
			{
				//for the sample in the given pixel, map the sample to different pixels in floor and ciel lods
				


				//sample floor 4 times
				{
					int c1 = int(quantizedRay.y) * HistogramWidth + int(quantizedRay.x);
					lookups[0] = calculate_lookup_pixel(floor_w, tile_sample_count[0], true, vec2(spatialCoordinate), ceil(lod), int(lod), vec2(0, 0), -rayOffset);

					if ( (c1 > 0 && c1 < HistogramHeight * HistogramWidth) && (lookups[0]>0))
					{
						ndfVoxelData.histograms[lookups[0]+c1] += float(unit);
					}


					//lookups[1] = calculate_lookup_pixel(floor_w, tile_sample_count[1], true, vec2(spatialCoordinate), ceil(lod), int(lod), vec2(1, 0), c1);

					//if ((c1 > 0 && c1 < HistogramHeight * HistogramWidth) && (lookups[1]>0))
					//{
					//	ndfVoxelData.histograms[lookups[1] + c1] += float(unit);
					//}


					//lookups[2] = calculate_lookup_pixel(floor_w, tile_sample_count[2], true, vec2(spatialCoordinate), ceil(lod), int(lod), vec2(0, 1), c1);

					//if ((c1 > 0 && c1 < HistogramHeight * HistogramWidth) && (lookups[2]>0))
					//{
					//	ndfVoxelData.histograms[lookups[2] + c1] += float(unit);
					//}


					//lookups[3] = calculate_lookup_pixel(floor_w, tile_sample_count[3], true, vec2(spatialCoordinate), ceil(lod), int(lod), vec2(1, 1), c1);

					//if ((c1 > 0 && c1 < HistogramHeight * HistogramWidth) && (lookups[3]>0))
					//{
					//	ndfVoxelData.histograms[lookups[3] + c1] += float(unit);
					//}

					//if (tile_sample_count[0] < maxSamplingRuns && histogramIndexCentral > 0 && histogramIndexCentral < HistogramHeight * HistogramWidth)
					//{
					//	ndfVoxelData.histograms[lookups[0] + histogramIndexCentral] += float((1.0 - bilinearWeights.x) * (1.0 - bilinearWeights.y) * unit);
					//}
					//if (tile_sample_count[0] < maxSamplingRuns && histogramIndexHorizontal > 0 && histogramIndexHorizontal < HistogramHeight * HistogramWidth)
					//{
					//	ndfVoxelData.histograms[lookups[0] + histogramIndexHorizontal] += float((bilinearWeights.x) * (1.0 - bilinearWeights.y) * unit);
					//}
					//if (tile_sample_count[0] < maxSamplingRuns && histogramIndexVertical > 0 && histogramIndexVertical < HistogramHeight * HistogramWidth)
					//{
					//	ndfVoxelData.histograms[lookups[0] + histogramIndexVertical] += float((1.0 - bilinearWeights.x) * (bilinearWeights.y) * unit);
					//}
					//if (tile_sample_count[0] < maxSamplingRuns && histogramIndexHorizontalVertical > 0 && histogramIndexHorizontalVertical < HistogramHeight * HistogramWidth)
					//{
					//	ndfVoxelData.histograms[lookups[0] + histogramIndexHorizontalVertical] += float((bilinearWeights.x) * (bilinearWeights.y) * unit);
					//}


					lookups[4] = calculate_lookup_pixel(ceil_w, tile_sample_count[4], false, vec2(spatialCoordinate), ceil(lod), ceil(lod), vec2(0, 0), -rayOffset);

					if ( (c1 > 0 && c1 < HistogramHeight * HistogramWidth) && (lookups[4]>0))
					{
						ndfVoxelData.histograms[lookups[4]+c1] += float(unit);
					}

					//if (tile_sample_count[4] < maxSamplingRuns && histogramIndexCentral > 0 && histogramIndexCentral < HistogramHeight * HistogramWidth)
					//{
					//	ndfVoxelData.histograms[lookups[4] + histogramIndexCentral] += float((1.0 - bilinearWeights.x) * (1.0 - bilinearWeights.y) * unit);
					//}
					//if (tile_sample_count[4] < maxSamplingRuns && histogramIndexHorizontal > 0 && histogramIndexHorizontal < HistogramHeight * HistogramWidth)
					//{
					//	ndfVoxelData.histograms[lookups[4] + histogramIndexHorizontal] += float((bilinearWeights.x) * (1.0 - bilinearWeights.y) * unit);
					//}
					//if (tile_sample_count[4] < maxSamplingRuns && histogramIndexVertical > 0 && histogramIndexVertical < HistogramHeight * HistogramWidth)
					//{
					//	ndfVoxelData.histograms[lookups[4] + histogramIndexVertical] += float((1.0 - bilinearWeights.x) * (bilinearWeights.y) * unit);
					//}
					//if (tile_sample_count[4] < maxSamplingRuns && histogramIndexHorizontalVertical > 0 && histogramIndexHorizontalVertical < HistogramHeight * HistogramWidth)
					//{
					//	ndfVoxelData.histograms[lookups[4] + histogramIndexHorizontalVertical] += float((bilinearWeights.x) * (bilinearWeights.y) * unit);
					//}
				}

			}
			//end new

			//debug
			//ndfVoxelData.histograms[lookup] = 1.0;
			//end debug
		}
	}

	// TODO: on the fly eBRDF splatting - even sparse approximation?


}