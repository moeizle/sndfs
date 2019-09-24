#version 430

//#define SINGLE_PRECISION
//#define HIGH_RES

uniform ivec2 viewDiscretizations;
uniform ivec2 histogramDiscretizations;
uniform ivec2 spatialDiscretizations;

uniform int maxSamplingRuns;
uniform int multiSamplingRate;
uniform int highestSampleCount;
uniform int samplescount;

int HistogramWidth = histogramDiscretizations.x;
int HistogramHeight = histogramDiscretizations.y;
int ViewWidth = viewDiscretizations.x;
int ViewHeight = viewDiscretizations.y;
int VolumeResolutionX = spatialDiscretizations.x;
int VolumeResolutionY = spatialDiscretizations.y;

uniform vec3 viewSpaceLightDir;
uniform int renderMode;
uniform sampler2D normalTransferSampler;

layout(binding = 0, rgba32f) uniform image2D tex;
//layout(binding = 1, rgba16ui) uniform uimage2D tileTex;
layout(binding = 2, rgba32f) uniform volatile image2D floorLevel;
layout(binding = 3, rgba32f) uniform image2D ceilLevel;

//layout(binding = 4) uniform sampler2D echo_tex;

const float PI = 3.141592f;

uniform float lod;

uniform int floor_w;
uniform int ceil_w;
uniform int floor_h;
uniform int ceil_h;
uniform int tile_w;
uniform int tile_h;
uniform int phys_tex_dim;

uniform int cachedRayCasting;

uniform float specularExp;

uniform ivec3 lod_blc;
uniform ivec3 lod_trc;

uniform ivec3 ceil_blc;
uniform ivec3 floor_blc;

uniform ivec2 visible_region_blc;
uniform ivec2 visible_region_trc;


uniform vec2 sPos;


uniform float viewportWidth;

uniform vec2 blc_s, blc_l, trc_s;

layout(std430) buffer NdfVoxelData
{
	coherent float histograms[];
} ndfVoxelData;

//layout(std430, binding = 1) buffer NdfVoxelDataHighRes
//{
//	coherent float histograms[];
//} ndfVoxelDataHighRes;


layout(std430) buffer sampleCount
{
	float sample_count[]; // This is the important name (in the shader).
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



void splatReconstructionKernel(vec2 normalPosition, unsigned int dataOffset, const float unit)
{
	const vec2 histogramScale = 1.0f / vec2(float(HistogramWidth - 1), float(HistogramHeight - 1));
	const vec2 quantizedRay = vec2(normalPosition.x * float(HistogramWidth  - 1), normalPosition.y * float(HistogramHeight - 1));

	const int histogramIndex = int(quantizedRay.y) * HistogramWidth  + int(quantizedRay.x);

	//const float basisFunctionScale = 32.0f;//2.0f;//0.25f;

	if (histogramIndex > 0 && histogramIndex < HistogramHeight * HistogramWidth )
	{
		for (int yHist = 0; yHist < HistogramHeight; ++yHist)
		{
			for (int xHist = 0; xHist < HistogramWidth ; ++xHist)
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
			tileCoords_InPhysTex = imageLoad(floorLevel, tileCoords_InPageTex);
		else
			tileCoords_InPhysTex = imageLoad(ceilLevel, tileCoords_InPageTex);

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


mat3 rotationMatrix(vec3 axis, float angle)
{
	axis = normalize(axis);
	float s = sin(angle);
	float c = cos(angle);
	float oc = 1.0f - c;

	return mat3(oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s,
		oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z - axis.x * s,
		oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c);
}

vec3 blinnPhong(vec3 normal, vec3 light, vec3 view, vec3 diffuseColor, vec3 specularColor, float specularExponent)
{
	vec3 halfVector = normalize(light + view);

	float diffuseIntensity = 1.0f * max(0.0f, -dot(normal, -light));
	float diffuseWeight = 1.0f;
	//float diffuseIntensity = max(0.0f, abs(dot(normal, light)));
	//float diffuseIntensity = max(0.0f, dot(normal, -light));
	//float specularIntensity = max(0.0f, pow(dot(normal, halfVector), specularExponent));

	float nDotHalf = abs(dot(normal, halfVector));
	float specularIntensity = 1.0f * max(0.0f, pow(nDotHalf, specularExponent));
	float specularWeight = 1.0f;
	//specularIntensity *= 0.0f;
	//diffuseIntensity *= 0.0f;

	const float ambientWeight = 0.0f;//0.25f;
	const vec3 ambientColor = vec3(1.0f, 0.0f, 0.0f);

	//return specularIntensity * specularColor;
	//return specularIntensity * vec3(0.0f, 0.5f, 0.0f) + min(1.0f, (1.0f - specularIntensity)) * vec3(0.5f, 0.0f, 0.0f);
	//return specularIntensity * specularColor + ambientIntensity * ambientColor;
	return (diffuseIntensity * diffuseColor * diffuseWeight + specularIntensity * specularColor * specularWeight + ambientWeight * ambientColor) / (diffuseWeight + specularWeight + ambientWeight);
	//return vec3(1.0f, 1.0f, 1.0f) - 0.25f * (diffuseIntensity * diffuseColor - specularIntensity * specularColor + ambientIntensity * ambientColor);
}
vec3 computeColor(vec2 newRay)
{
	const vec3 offset = vec3(0.4f, 0.0f, 0.0f);
	//samplePosition = mod(samplePosition + offset, vec3(1.0f, 1.0f, 1.0f));

	// TODO: add depth map and parallax mapping? Better rotation...
	// TODO: normal of tangent frame using depth map
	vec3 viewView;

	//outColor = vec4(0.5f, 0.0f, 0.0f, 1.0f);
	//return;

	vec3 shading = vec3(0.0f, 0.0f, 0.0f);
	float ndfSample = 0.0f;
	int ndfColorTest = 0;
	float energySum = 0.0f;
	vec3 shadings[5] = { vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f) };
	float energySums[5] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	float energies[5] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	int lookups[5] = { -1, -1, -1, -1, -1 };
	int tile_sample_count[5] = { 0, 0, 0, 0, 0 };
	float totalndfSamples = 0;
	float samplesPerPixel;
	float unit = 1.0f;// / float(maxSamplingRuns);
	
		// TODO: rename
		//vec2 voxelCoordF = samplePosition.xy;

		//vec3 diffuseColor = 0.25f * vec3(1.0f, 1.0f, 1.0f);
		vec3 diffuseColor = 1.0f * vec3(1.0f, 1.0f, 1.0f);

		// NOTE: for this model to be physically based the BRDF would have to be perfectly specular
		//float specularExponent = 128.0f;
		float specularCorrection = 1.0f;
		vec3 specularColor = specularCorrection * vec3(1.0f, 1.0f, 1.0f);

		
		vec3 lightViewSpace = normalize(viewSpaceLightDir);


		const float lightEnergyCorrection = 1.0f;

		float energyCollected = 0;

		//vec3 sourceNormal = 2.0f * vec3(float(histogramX) * histogramScaleX - 0.5f, float(histogramY) * histogramScaleY - 0.5f, 0.0f);
		//sourceNormal.z = sqrt(1.0f - sourceNormal.x*sourceNormal.x - sourceNormal.y*sourceNormal.y);

		vec3 sourceNormal;
		sourceNormal.xy = vec2(newRay.x, newRay.y)*2.0f - 1.0f;
		//sourceNormal.z = 1.0f - sqrt(sourceNormal.x*sourceNormal.x + sourceNormal.y*sourceNormal.y);
		sourceNormal.z = sqrt(1.0f - sourceNormal.x*sourceNormal.x - sourceNormal.y*sourceNormal.y);

		// project to actual view space since the nearest view samples does not necessarily equal the current view direction
		//vec3 viewSpaceNormal = normalize(viewCorrection * sourceNormal);
		vec3 viewSpaceNormal = sourceNormal;


		vec3 neighbourShading = vec3(0.0f, 0.0f, 0.0f);
		float neighbourdNdfSample = 0.0f;

		// add colors
#if 1
		const float threshold = 0.25f;//0.35f;
		//vec2 transfer = worldSpaceNormal.xy;
		vec2 transfer = viewSpaceNormal.xy;

		if (renderMode == 1)
		{
			const vec3 leftColor = vec3(.5, 0, 0);// vec3(0.35f, 0.65f, 0.8f);
			const vec3 rightColor = vec3(0f, 0.5f, 0f);
			const vec3 bottomColor = vec3(0.5f, 0.5f, 0.5f);
			const vec3 topColor = vec3(0.0f, 0.0f, 0.5f);

			mat3 lightRotationZ = rotationMatrix(vec3(0.0f, 0.0f, 1.0f), lightViewSpace.x + PI);
			mat3 lightRotationY = rotationMatrix(vec3(0.0f, 1.0f, 0.0f), lightViewSpace.y);

			transfer = (lightRotationZ * vec3(transfer.xy, 1.0f)).xy;
			transfer = (lightRotationY * vec3(transfer.xy, 1.0f)).xy;

			diffuseColor = 0.5f * leftColor * (1.0f - transfer.x) + 0.5f * rightColor * transfer.x +
				0.5f * bottomColor * (1.0f - transfer.y) + 0.5f * topColor * transfer.y;
			specularColor = diffuseColor;

			const float energyCorrection = 1.0f;
			diffuseColor *= energyCorrection;
			specularColor *= energyCorrection;
		}
		else if (renderMode == 2)
		{

			//mat3 lightRotationZ = rotationMatrix(vec3(0.0f, -1.0f, 0.0f), viewSpaceLightDir.x);
			//mat3 lightRotationY = rotationMatrix(vec3(1.0f, 0.0f, 0.0f), viewSpaceLightDir.y);

			//vec3 transformedNormal = viewSpaceNormal;
			//transformedNormal = (lightRotationZ * transformedNormal.xyz).xyz;
			//transformedNormal = (lightRotationY * transformedNormal.xyz).xyz;

			//const float seamCorrection = 0.125f;
			//transformedNormal.xy -= vec2(transformedNormal.xy * seamCorrection);
			//vec2 lookupCoord = 1.0f - (transformedNormal.xy * 0.5f + 0.5f);

			//vec3 lookupColor = texture(normalTransferSampler, lookupCoord).xyz;

			//diffuseColor = lookupColor.xyz;
			//specularColor = diffuseColor;

			{
				vec3 transformedNormal;
				vec3 L = viewSpaceLightDir;
				//L.z *= -1.0f;

				vec3 a = cross(viewSpaceNormal, L);
				a = normalize(a);

				float theta = acos(dot(viewSpaceNormal, L) / (length(viewSpaceNormal)*length(L)));

				mat3 R = rotationMatrix(a, theta);

				transformedNormal = R*viewSpaceNormal;

				const float seamCorrection = 0.125f;
				transformedNormal.xy -= vec2(transformedNormal.xy * seamCorrection);

				transfer.xy = transformedNormal.xy * 0.5f + 0.5f;

				vec3 lookupColor = texture(normalTransferSampler, vec2(transfer.x, transfer.y)).xyz;

				diffuseColor = lookupColor;
				specularColor = diffuseColor;
			}
		}

		//float lerp = viewSpaceNormal.y * 0.5f + 0.5f;
		//float saturation = 1.25f;
		//diffuseColor = (1.0f - lerp) * saturation * vec3(0.0f, 1.0f, 0.0f) + lerp * saturation * vec3(1.0f, 0.0f, 0.0f);
#endif

#if 1
		viewView = vec3(0.0f, 0.0f, 1.0f);
		//lightViewSpace = normalize(viewView + vec3(0.25f, 1.25f, 1.0f));
#endif

		vec3 centralShading;
		// >> granda <<
		if (renderMode <= 0)
		{
			centralShading = blinnPhong(viewSpaceNormal, -1*vec3(-lightViewSpace.x,-lightViewSpace.y,lightViewSpace.z), viewView, diffuseColor, specularColor, specularExp);
		}
		else
		{
			centralShading = (diffuseColor + specularColor) * 0.5f;
		}
	

	return centralShading;
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

	//if ((spatialCoordinate.x < blc_s.x) || (spatialCoordinate.x >= trc_s.x) || (spatialCoordinate.y < blc_s.y) || (spatialCoordinate.y >= trc_s.y))
	//	return;

	int lookup;
	int lookups[5] = { -1, -1, -1, -1, -1 };
	int tile_sample_count[5] = { 0, 0, 0, 0, 0 };



	//vec2 rayOffset = sPos;
	//rayOffset.xy *= 1.0f / viewportWidth;

	//calculate rayoffset
	

	// FIXME: race condition
	//const unsigned int downLookup = (spatialCoordinate.y / 2) * VolumeResolutionX + (spatialCoordinate.x / 2)) * HistogramHeight * HistogramWidth;

	ivec2 multisampleSpatialCoordinate = ivec2(spatialCoordinate.xy);
	vec2 newRay = imageLoad(tex, multisampleSpatialCoordinate).xy;

	bool miss = false;
	if (newRay.x == 0.0f && newRay.y == 0.0f)
		miss = true;// return;

	//in tiled raycasting mode, we don't need to quantize ray, we need to comput the color given the ray and store the color of the pixel in the cache
	vec3 color = computeColor(newRay);


	{
		//for the sample in the given pixel, map the sample to different pixels in floor and ciel lods

		//sample floor and ceil
		{
			lookup = calculate_lookup_pixel(floor_w, floor_h, tile_sample_count[0], true, vec2(spatialCoordinate), floor(lod), floor(lod), vec2(0, 0), vec2(0, 0));
			if (lookup >= 0)
			{
				sample_count[lookup / (HistogramHeight*HistogramWidth)] += 1.0f;

				if (!miss)
				{
					ndfVoxelData.histograms[lookup + 1] += color.x;
					ndfVoxelData.histograms[lookup + 2] += color.y;
					ndfVoxelData.histograms[lookup + 3] += color.z;
				}
				else
				{
					ndfVoxelData.histograms[lookup]++;   //store number of misses
				}
			}


			//lookup = calculate_lookup_pixel(ceil_w, tile_sample_count[4], false, vec2(spatialCoordinate), floor(lod), floor(lod+1), vec2(0, 0), vec2(0, 0));

			//if (lookup >= 0)
			//{
			//	ndfVoxelData.histograms[lookup] += 1;
			//	ndfVoxelData.histograms[lookup + 1] += color.x;
			//	ndfVoxelData.histograms[lookup + 2] += color.y;
			//	ndfVoxelData.histograms[lookup + 3] += color.z;
			//}
		}

	}
}