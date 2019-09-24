#version 430
#pragma optimize (off)

//#define OUTPUT_HISTOGRAM
//#define OUTPUT_EBRDF
#define NDF

in vec2 texCoords;
in vec3 rayBegin;
in vec3 frustumTarget;
in vec3 viewSpacePosition;
in vec3 worldSpacePosition;

out vec4 outColor;

uniform int renderMode;
uniform int zoomMode;
uniform float zoomScale;
uniform vec2 zoomWindow;
uniform int ssaoEnabled;
uniform int ssaoDownsampleCount;

uniform int samplingRunIndex;
uniform int maxSamplingRuns;
uniform int multiSamplingRate;

uniform float quantizationVarianceScale;
uniform float ndfIntensityCorrection;

uniform vec3 camPosi;
uniform vec2 camAngles;
uniform vec3 camDirection;
uniform vec3 viewSpaceCamDirection;
uniform vec3 viewSpaceLightDir;

uniform ivec2 viewDiscretizations;
uniform ivec2 histogramDiscretizations;
uniform ivec3 spatialDiscretizations;
uniform ivec2 viewportSize;

uniform sampler2D normalTransferSampler;

//sampling texture, for debugging purposes
layout(binding = 0, rg8) uniform image2D tex;
//end sampling texture

//layout(binding = 0, rgba16ui) uniform uimage2D tileTex;
layout(binding = 1, rgba32f) uniform image2D floorLevel;
layout(binding = 2, rgba32f) uniform image2D ceilLevel;

uniform float lod;

uniform int ceil_w;
uniform int floor_w;
uniform int tile_w;
uniform int tile_h;
uniform int phys_tex_dim;

uniform ivec3 lod_blc;
uniform ivec3 lod_trc;

uniform ivec3 ceil_blc;
uniform ivec3 floor_blc;

uniform ivec2 visible_region_blc;
uniform ivec2 visible_region_trc;

uniform vec2 blc_s, blc_l, trc_s;

uniform vec2 sampling_trc_s, sampling_blc_s, sampling_blc_l;

int HistogramWidth = histogramDiscretizations.x;
int HistogramHeight = histogramDiscretizations.y;
int ViewWidth = viewDiscretizations.x;
int ViewHeight = viewDiscretizations.y;
int VolumeResolutionX = spatialDiscretizations.x;
int VolumeResolutionY = spatialDiscretizations.y;

const float PI = 3.141592f;
const float toRadiants = PI / 180.0f;

float f_du, f_dv,c_du,c_dv;

layout(std430, binding = 0) buffer NdfVoxelData 
{
	coherent readonly float histograms[];
} ndfVoxelData;

layout(std430, binding = 1) buffer NdfVoxelDataHighRes 
{
	coherent readonly float histograms[];
} ndfVoxelDataHighRes;

mat3 rotationMatrix(vec3 axis, float angle) 
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0f - c;
    
    return mat3(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c);
}
float rand(vec2 co){
	return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}
vec3 blinnPhong(vec3 normal, vec3 light, vec3 view, vec3 diffuseColor, vec3 specularColor, float specularExponent) 
{
	vec3 halfVector = normalize(light + view);

	float diffuseIntensity = 1.0f * max(0.0f, -dot(normal, -light));

	//float diffuseIntensity = max(0.0f, abs(dot(normal, light)));
	//float diffuseIntensity = max(0.0f, dot(normal, -light));
	//float specularIntensity = max(0.0f, pow(dot(normal, halfVector), specularExponent));

	float nDotHalf = abs(dot(normal, halfVector));
	float specularIntensity = 1.0f * max(0.0f, pow(nDotHalf, specularExponent));
	//specularIntensity *= 0.0f;
	//diffuseIntensity *= 0.0f;

	const float ambientIntensity = 0.0f;//0.25f;
	const vec3 ambientColor = vec3(1.0f, 0.0f, 0.0f);

	//return specularIntensity * specularColor;
	//return specularIntensity * vec3(0.0f, 0.5f, 0.0f) + min(1.0f, (1.0f - specularIntensity)) * vec3(0.5f, 0.0f, 0.0f);
	//return specularIntensity * specularColor + ambientIntensity * ambientColor;
	return diffuseIntensity * diffuseColor + specularIntensity * specularColor + ambientIntensity * ambientColor;
	//return vec3(1.0f, 1.0f, 1.0f) - 0.25f * (diffuseIntensity * diffuseColor - specularIntensity * specularColor + ambientIntensity * ambientColor);
}
int calculate_lookup_pixel(int lod_w, inout int tile_sample_count, bool levelFlag, vec2 Coordinate, float F_lod, float E_lod, vec2 offset, int bin_indx)
{
	//mohamed's lookup
	int num_tiles_in_h = lod_w / tile_w;
	ivec2 bin_2d_indx = ivec2(bin_indx%HistogramWidth, bin_indx / HistogramWidth);

	vec2 PixelCoordIn_F_Lod = vec2(Coordinate - sampling_blc_s + sampling_blc_l);// +vec2(bin_2d_indx.x / float(HistogramWidth), bin_2d_indx.y / float(HistogramHeight));

	//now we get pixel coordinate in exact lod
	ivec2 PixelCoordIn_E_Lod = ivec2(PixelCoordIn_F_Lod.x*pow(2, F_lod - E_lod), PixelCoordIn_F_Lod.y*pow(2, F_lod - E_lod));

	//end newer


	//add offset to image in PixelCoordIN_E_Lod
	uvec2 spatialCoordinate = uvec2(PixelCoordIn_E_Lod + offset + ivec2(bin_2d_indx.x / (HistogramWidth / 2), bin_2d_indx.y / (HistogramWidth / 2)));


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


			//if ((levelFlag) && (samplingRunIndex % 4 == 0))
			//{
			//	tileCoords_InPhysTex.w = 1;   //prevent other threads from writing to it
			//	tileCoords_InPhysTex.z++;
			//	imageStore(floorLevel, tileCoords_InPageTex, tileCoords_InPhysTex);
			//}
			//else if (!levelFlag)
			//{
			//	tileCoords_InPhysTex.w = 1;   //prevent other threads from writing to it
			//	tileCoords_InPhysTex.z++;
			//	imageStore(ceilLevel, tileCoords_InPageTex, tileCoords_InPhysTex);
			//}
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
int calculate_lookup(int lod_w, inout int tile_sample_count, bool levelFlag, vec2 Coordinate, float F_lod, float E_lod, vec2 offset, inout float du, inout float dv, inout ivec2 pixel_in_exact)
{
	//mohamed's lookup
	int num_tiles_in_h = lod_w / tile_w;

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
	ivec2 PixelCoordIn_E_Lod = ivec2(PixelCoordIn_F_Lod.x*pow(2,F_lod - E_lod), PixelCoordIn_F_Lod.y*pow(2,F_lod - E_lod));
	pixel_in_exact = PixelCoordIn_E_Lod;
	//end newer


	//add offset to image in PixelCoordIN_E_Lod
	uvec2 spatialCoordinate = uvec2(PixelCoordIn_E_Lod + offset);

	//the offset may get us a pixel outside the iamge
	if ((spatialCoordinate.x >= 0) && (spatialCoordinate.x<lod_w) && (spatialCoordinate.y>=0) && (spatialCoordinate.y<lod_w))
	{
		//get tile of the pixel
		ivec2 tileindx2D = ivec2(spatialCoordinate.x / tile_w, spatialCoordinate.y / tile_w);
		int tile_indx = tileindx2D.x*num_tiles_in_h + tileindx2D.y;
		vec2 withinTileOffset = vec2(spatialCoordinate.x%tile_w,spatialCoordinate.y%tile_w);

		//get tile coordiantes in page texture

		ivec2  tileCoords_InPageTex = ivec2(tile_indx% num_tiles_in_h, tile_indx / num_tiles_in_h);

		//read physical texture coordinates from page texture
		vec4 tileCoords_InPhysTex;

		if (levelFlag)
			tileCoords_InPhysTex = imageLoad(floorLevel, tileCoords_InPageTex);
		else
			tileCoords_InPhysTex = imageLoad(ceilLevel, tileCoords_InPageTex);

		if (tileCoords_InPhysTex.w == 1)
		{
			tileCoords_InPhysTex.w = 0;   //allow compute shader to update sample count
			if (levelFlag)
				imageStore(floorLevel, tileCoords_InPageTex, tileCoords_InPhysTex);
			else
				imageStore(ceilLevel, tileCoords_InPageTex, tileCoords_InPhysTex);
		}

		//debug
		if ((tileCoords_InPhysTex.x < 0) || (tileCoords_InPhysTex.y < 0))
		{
			return -2;
		}
		//end debug

		//debug
		if (tileCoords_InPhysTex.z==0)
		{
			if (tileCoords_InPhysTex.z < 0)
				return -5;
		}
		//end debug

		tile_sample_count = int(tileCoords_InPhysTex.z);
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
void main()
{


	vec3 samplePosition = rayBegin;

	const vec3 offset = vec3(0.4f, 0.0f, 0.0f);
	//samplePosition = mod(samplePosition + offset, vec3(1.0f, 1.0f, 1.0f));

	// TODO: add depth map and parallax mapping? Better rotation...
	// TODO: normal of tangent frame using depth map
	vec3 viewWorld = normalize(camPosi - worldSpacePosition);
	vec3 viewView = normalize(camPosi - viewSpacePosition);

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
	float unit = 1.0f / float(maxSamplingRuns);
	{
		// TODO: rename
		vec2 voxelCoordF = samplePosition.xy;

		vec3 viewDir = viewWorld * 0.5f + 0.5f;

		if (viewDir.x < 0.0f || viewDir.x > 1.0f || viewDir.y < 0.0f || viewDir.y > 1.0f)
		{
			outColor = vec4(0.5f, 0.0f, 0.0f, 1.0f);
			return;
		}

		ivec2 voxelCoord = ivec2(int(voxelCoordF.x * float(VolumeResolutionX)), int(voxelCoordF.y * float(VolumeResolutionY)));


		//debug
		//vec2 newRay = imageLoad(tex, voxelCoord).xy;
		//outColor = vec4(newRay.xy, 0.0f, 1.0f);
		//return;
		//end debug

		//const float angleRange = 360.0f;//2.0f;//20.0f; //360.0f
		//vec2 camAngleScaled = vec2((camAngles.x / angleRange), (camAngles.y / angleRange));
		int viewCoordinateX = 0;//int(camAngleScaled.x * float(ViewWidth-1));
		int viewCoordinateY = 0;//int(camAngleScaled.y * float(ViewHeight-1));

		vec2 voxelFract;
		int index;
		if (zoomMode <= 0)
		{
			//patrick's index
			index = (voxelCoord.y * VolumeResolutionX + voxelCoord.x) * ViewHeight * ViewWidth * HistogramHeight * HistogramWidth + (viewCoordinateY * ViewWidth + viewCoordinateX) * HistogramHeight * HistogramWidth;
			//end patrick's index

			voxelFract = vec2(fract(voxelCoordF.x * float(VolumeResolutionX)), fract(voxelCoordF.y * float(VolumeResolutionY)));
		}
		else
		{
			//index = ((voxelCoord.y + VolumeResolutionY) * (VolumeResolutionX * multiSamplingRate) + (voxelCoord.x + VolumeResolutionX)) * HistogramHeight * HistogramWidth;

			const float zoomFactor = zoomScale;
			vec2 zoomWindowScale = vec2(VolumeResolutionX, VolumeResolutionY) / zoomFactor;
			vec2 zoomCoord = vec2(voxelCoord.x, voxelCoord.y) / zoomFactor;
			ivec2 zoomOffset = ivec2(zoomWindow.x * VolumeResolutionX * multiSamplingRate, zoomWindow.y * VolumeResolutionY * multiSamplingRate);

			//index = (((zoomCoord.y - (zoomWindowScale / 2)) + zoomOffset.y) * (VolumeResolutionX * multiSamplingRate) + (zoomCoord.x - (zoomWindowScale / 2) + zoomOffset.x)) * HistogramHeight * HistogramWidth;
			index = (((int(zoomCoord.y) - (int(zoomWindowScale) / 2)) + zoomOffset.y) * (VolumeResolutionX * multiSamplingRate) + (int(zoomCoord.x) - (int(zoomWindowScale) / 2) + zoomOffset.x)) * HistogramHeight * HistogramWidth;

			//index = (voxelCoord.y * 2 * (VolumeResolutionX * multiSamplingRate) + voxelCoord.x * 2) * HistogramHeight * HistogramWidth;

			voxelFract = vec2(fract(voxelCoordF.x * zoomWindowScale.x), fract(voxelCoordF.y * zoomWindowScale.y));
		}

		//vec3 diffuseColor = 0.25f * vec3(1.0f, 1.0f, 1.0f);
		vec3 diffuseColor = 1.0f * vec3(1.0f, 1.0f, 1.0f);

		// NOTE: for this model to be physically based the BRDF would have to be perfectly specular
		float specularExponent = 128.0f;
		float specularCorrection = 5.0f;
		vec3 specularColor = specularCorrection * vec3(1.0f, 1.0f, 1.0f);

		// world space transformation
		vec3 front = normalize(viewWorld);
		vec3 up = normalize(vec3(0.0f, 1.0f, 0.0f));
		vec3 right = normalize(cross(front, up));

		mat3x3 worldSpaceMatrix = mat3x3(right, up, front);

		// NOTE: light dir should be the other way around
		vec3 lightWorldSpace = normalize(worldSpaceMatrix * viewSpaceLightDir);
		vec3 lightViewSpace = normalize(viewSpaceLightDir);

		const float histogramScaleX = 1.0f / float(HistogramWidth - 1);
		const float histogramScaleY = 1.0f / float(HistogramHeight - 1);

		//patrick's code
		// the fraction of total energy over energy collected
		//const float energyCollected = (float(maxSamplingRuns) / float(samplingRunIndex));
		//end patrick's code


		// FIXME: remove test - splat BRDF and sample only current light dir
		//const float lightEnergyCorrection = 14.0f;
		//int histogramX = HistogramWidth / 2; {
		//	int histogramY = HistogramHeight - 3; {
		const float lightEnergyCorrection = 1.0f;

		/////////////////
		//mohamed's index
		//if a pixel is outside the visible region, it need not be rendered.
		if ((voxelCoord.x < blc_s.x) || (voxelCoord.x >= trc_s.x) || (voxelCoord.y < blc_s.y) || (voxelCoord.y >= trc_s.y))
			return;


		float energyCollected = 0;
		if (lod - int(lod) == 0)  //exact lod
		{
			int num_tiles_in_h = floor_w / tile_w;

			//newest



			//ivec2 blc_tile_indx = ivec2(int(lod_blc.x) / num_tiles_in_h, int(lod_blc.x) % num_tiles_in_h);

			//ivec2 tile_offset_of_current_pixel = ivec2((voxelCoord.x - visible_region_blc.x) / tile_w, (voxelCoord.y - visible_region_blc.y) / tile_h);
			//int   tile_indx = (blc_tile_indx.x + tile_offset_of_current_pixel.x)*num_tiles_in_h + (blc_tile_indx.y + tile_offset_of_current_pixel.y);
			//vec2 withinTileOffset = vec2((voxelCoord.x - visible_region_blc.x) % tile_w, (voxelCoord.y - visible_region_blc.y) % tile_h);


			//newwest 2
			//map pixel to lod space
			ivec2 coordinate = ivec2(voxelCoord - blc_s + blc_l);
			ivec2 tileindx_2d = ivec2(coordinate.x / tile_w, coordinate.y / tile_h);
			int tile_indx = tileindx_2d.x*num_tiles_in_h + tileindx_2d.y;
			vec2 withinTileOffset = vec2(coordinate.x%tile_w, coordinate.y%tile_h);
			//end newest

			//get tile coordiantes in page texture
			ivec2  tileCoords_InPageTex = ivec2(tile_indx% num_tiles_in_h, tile_indx / num_tiles_in_h);

			//read physical texture coordinates from page texture
			vec4 tileCoords_InPhysTex = imageLoad(floorLevel, tileCoords_InPageTex);
			energyCollected = (float(maxSamplingRuns) / float(tileCoords_InPhysTex.z));
			energies[0] = energyCollected;

			if (tileCoords_InPhysTex.w == 1)
			{
				tileCoords_InPhysTex.w = 0;   //allow compute shader to update sample count
				imageStore(floorLevel, tileCoords_InPageTex, tileCoords_InPhysTex);
			}

			tileCoords_InPhysTex.x *= tile_w;
			tileCoords_InPhysTex.y *= tile_h;

			//location in ndf tree is the physical texture location + within tile offset
			ivec2 Pixelcoord_InPhysTex = ivec2(tileCoords_InPhysTex.x + withinTileOffset.x, tileCoords_InPhysTex.y + withinTileOffset.y);

			//index = int((Pixelcoord_InPhysTex.y * VolumeResolutionX + Pixelcoord_InPhysTex.x) * HistogramHeight * HistogramWidth);
			index = int((Pixelcoord_InPhysTex.y * phys_tex_dim + Pixelcoord_InPhysTex.x) * HistogramHeight * HistogramWidth);
			lookups[0] = index;
			tile_sample_count[0] = int(tileCoords_InPhysTex.z);

			//debug
			if (voxelCoord.x == 511)
			{
				int fkdjdklfs = 8;
			}
			//end debug
		}
		else
		{
			//get five lookups
			//four for floor lod

			//debug
			//for (int j = 1; j < 64; j++)
			//{
			//	lookups[4] = calculate_lookup_pixel(ceil_w, tile_sample_count[4], false, vec2(voxelCoord), lod, ceil(lod), vec2(0, 0), j);
			//	lookups[0] = calculate_lookup_pixel(floor_w, tile_sample_count[0], true, vec2(voxelCoord), lod, int(lod), vec2(0, 0), j);

			//	if ((lookups[0] < 0) || (lookups[4] < 0))
			//	{
			//		outColor = vec4(0, 0, 1, 0.5);
			//		return;
			//	}
			//}
			//end debug
			//int i1, i2, i3, i4, i5;
		
			//for (int i = 0; i < 256; i++)
			//{
			//	for (int j = 0; j < 256; j++)
			//	{
			//		if (j == 128)
			//		{
			//			j = 128;
			//		}
			//if ((voxelCoord.x >= 0) && (voxelCoord.x<256) && (voxelCoord.y >= 0) && (voxelCoord.y<256))
			//{
			//	int i = voxelCoord.x;
			//	int j = voxelCoord.y;
			//	i1 = calculate_lookup_pixel(floor_w, tile_sample_count[0], true, vec2(i, j), ceil(lod), int(lod), vec2(0, 0), 1);
			//	i2 = calculate_lookup_pixel(ceil_w, tile_sample_count[4], false, vec2(i, j), ceil(lod), ceil(lod), vec2(0, 0), 1);

			//	if ((i1 < 0) || (i2 < 0))
			//	{
			//		outColor = vec4(0, 0, 1, 0.5);
			//		return;
			//	}

			//	i1 = calculate_lookup_pixel(floor_w, tile_sample_count[0], true, vec2(i, j), ceil(lod), int(lod), vec2(0, 0), 7);
			//	i2 = calculate_lookup_pixel(ceil_w, tile_sample_count[4], false, vec2(i, j), ceil(lod), ceil(lod), vec2(0, 0), 7);
			//	if ((i1 < 0) || (i2 < 0))
			//	{
			//		outColor = vec4(0, 0, 1, 0.5);
			//		return;
			//	}

			//	i1 = calculate_lookup_pixel(floor_w, tile_sample_count[0], true, vec2(i, j), ceil(lod), int(lod), vec2(0, 0), 56);
			//	i2 = calculate_lookup_pixel(ceil_w, tile_sample_count[4], false, vec2(i, j), ceil(lod), ceil(lod), vec2(0, 0), 56);
			//	if ((i1 < 0) || (i2 < 0))
			//	{
			//		outColor = vec4(0, 0, 1, 0.5);
			//		return;
			//	}

			//	i1 = calculate_lookup_pixel(floor_w, tile_sample_count[0], true, vec2(i, j), ceil(lod), int(lod), vec2(0, 0), 63);
			//	i2 = calculate_lookup_pixel(ceil_w, tile_sample_count[4], false, vec2(i, j), ceil(lod), ceil(lod), vec2(0, 0), 63);
			//	if ((i1 < 0) || (i2 < 0))
			//	{
			//		outColor = vec4(0, 0, 1, 0.5);
			//		return;
			//	}
			//}
				//}
			//}
			int i1, i2, i3, i4, i5, i, j;
			if ((voxelCoord.x >= sampling_blc_s.x) && (voxelCoord.y >= sampling_blc_s.y) && (voxelCoord.x < sampling_trc_s.x) && (voxelCoord.y < sampling_trc_s.y))
			{


				i = voxelCoord.x;
				j = voxelCoord.y;

				/*i1 = calculate_lookup_pixel(floor_w, tile_sample_count[0], true, vec2(i, j), ceil(lod), int(lod), vec2(0, 0), 0);
				i2 = calculate_lookup_pixel(floor_w, tile_sample_count[1], true, vec2(i, j), ceil(lod), int(lod), vec2(0, 0), 7);
				i3 = calculate_lookup_pixel(floor_w, tile_sample_count[2], true, vec2(i, j), ceil(lod), int(lod), vec2(0, 0), 56);
				i4 = calculate_lookup_pixel(floor_w, tile_sample_count[3], true, vec2(i, j), ceil(lod), int(lod), vec2(0, 0), 63);
				i5 = calculate_lookup_pixel(ceil_w, tile_sample_count[4], false, vec2(i, j), ceil(lod), ceil(lod), vec2(0, 0), 0);*/

				for (int k = 0; k < 64; k++)
				{
					i1 = calculate_lookup_pixel(floor_w, tile_sample_count[0], true, vec2(i, j), ceil(lod), int(lod), vec2(0, 0), k);
					i5 = calculate_lookup_pixel( ceil_w, tile_sample_count[4], false, vec2(i, j), ceil(lod), ceil(lod), vec2(0, 0), k);
					//if (i1 < 0 || i2 < 0 || i3 < 0 || i4 < 0)
					//{
					//	outColor = vec4(1, 0, 0, 1);
					//	return;
					//}
				}

				
				
				//{
				//	outColor = vec4(0, 1, 0, 1);
				//	return;
				//}

			}

			ivec2 pixel_in_ceil,garbage;

			lookups[4] = calculate_lookup(ceil_w, tile_sample_count[4], false, vec2(voxelCoord), lod, ceil(lod), vec2(0, 0), c_du, c_dv,pixel_in_ceil);

			lookups[0] = calculate_lookup(floor_w, tile_sample_count[0], true, vec2(voxelCoord), lod, int(lod), vec2(0, 0), f_du, f_dv,garbage);
			lookups[1] = calculate_lookup(floor_w, tile_sample_count[1], true, vec2(voxelCoord), lod, int(lod), vec2(1, 0), f_du, f_dv,garbage);
			lookups[2] = calculate_lookup(floor_w, tile_sample_count[2], true, vec2(voxelCoord), lod, int(lod), vec2(0, 1), f_du, f_dv,garbage);
			lookups[3] = calculate_lookup(floor_w, tile_sample_count[3], true, vec2(voxelCoord), lod, int(lod), vec2(1, 1), f_du, f_dv,garbage);
			


			for (int i = 0; i < 5; i++)
			{
				if (lookups[i] < 0)
				{
					outColor = vec4(0, 0, 1, 0.5);
					return;
				}

			}

			//compute energy collected
			/*float count = 0;
			for (int i = 0; i < 5; i++)
			{
			if (tile_sample_count[i] != 0)
			{
			energyCollected += float(maxSamplingRuns) / float(tile_sample_count[i]);
			count++;
			}
			}

			energyCollected = energyCollected / count;*/

			//float nrg_floor, nrg_ceil;

			//nrg_floor = (float(maxSamplingRuns) / float(tile_sample_count[0]) +
			//	         float(maxSamplingRuns) / float(tile_sample_count[1]) +
			//	         float(maxSamplingRuns) / float(tile_sample_count[2]) +
			// 	         float(maxSamplingRuns) / float(tile_sample_count[3])) / 4.0;

			//nrg_floor = f_du*f_dv*float(maxSamplingRuns) / float(tile_sample_count[3])          +
			//	        f_du*(1 - f_dv)*float(maxSamplingRuns) / float(tile_sample_count[2])      +
			//	        (1 - f_du)*f_dv*float(maxSamplingRuns) / float(tile_sample_count[1])      +
			//	        (1 - f_du)*(1 - f_dv)*float(maxSamplingRuns) / float(tile_sample_count[0]);

			//nrg_floor = float(maxSamplingRuns) / float(tile_sample_count[0]);

			//nrg_floor = (float(maxSamplingRuns) / float(tile_sample_count[0]) +
			//	float(maxSamplingRuns) / float(tile_sample_count[1]) +
			//	float(maxSamplingRuns) / float(tile_sample_count[2]) +
			//	float(maxSamplingRuns) / float(tile_sample_count[3])) / 4.0;

			//nrg_ceil = float(maxSamplingRuns) / float(tile_sample_count[4]);

			//float w1, w2;
			//w1 = lod - int(lod);
			//w2 = 1.0 - w1;

			//energyCollected = w2*nrg_floor + w1*nrg_ceil;

			//or

			//energies[0] = float(maxSamplingRuns) / float(tile_sample_count[0]);
			//energies[1] = float(maxSamplingRuns) / float(tile_sample_count[1]);
			//energies[2] = float(maxSamplingRuns) / float(tile_sample_count[2]);
			//energies[3] = float(maxSamplingRuns) / float(tile_sample_count[3]);
			//energies[4] = float(maxSamplingRuns) / float(tile_sample_count[4]);

			//debug
			if ((tile_sample_count[0] == 0) || (tile_sample_count[4] == 0) || (tile_sample_count[1] == 0) || (tile_sample_count[2] == 0) || (tile_sample_count[3] == 0))
			{
				//outColor = vec4(0, 0, 1, 0.5);
				//return;
			}
			//debug

			//energyCollected =(float(maxSamplingRuns) / float(tile_sample_count[4]));
		}

		//end mohamed's index
		/////////////////////

		for (int i = 0; i < 5; i++)
		{
			float ndfsamplesum = 0;
			for (int histogramY = 0; histogramY < HistogramHeight; ++histogramY)
			{
				for (int histogramX = 0; histogramX < HistogramWidth; ++histogramX)
				{
					int binIndex = histogramY * HistogramWidth + histogramX;

					//vec3 sourceNormal = 2.0f * vec3(float(histogramX) * histogramScaleX - 0.5f, float(histogramY) * histogramScaleY - 0.5f, 0.0f);
					//sourceNormal.z = sqrt(1.0f - sourceNormal.x*sourceNormal.x - sourceNormal.y*sourceNormal.y);

					vec3 sourceNormal;
					sourceNormal.xy = 2.0f * vec2(float(histogramX) * histogramScaleX - 0.5f, float(histogramY) * histogramScaleY - 0.5f);
					sourceNormal.z = sqrt(1.0f - sourceNormal.x*sourceNormal.x - sourceNormal.y*sourceNormal.y);

					// project to actual view space since the nearest view samples does not necessarily equal the current view direction
					//vec3 viewSpaceNormal = normalize(viewCorrection * sourceNormal);
					vec3 viewSpaceNormal = sourceNormal;

					// project view space normal to world space
					//vec3 worldSpaceNormal = normalize(worldSpaceMatrix * viewSpaceNormal);
					// alternatively:
					//vec3 worldSpaceNormal = normalize(viewSpaceNormal.x * right + viewSpaceNormal.y * up + viewSpaceNormal.z * front);

					// ignore last value since it contains the depth
					if (binIndex < HistogramWidth * HistogramHeight - 1)
					{
						if (zoomMode <= 0)
						{
							//old
							//ndfSample = ndfVoxelData.histograms[index + binIndex];

							ndfSample = ndfVoxelData.histograms[lookups[i] + binIndex];
							ndfsamplesum += ndfSample;
							totalndfSamples += ndfSample;
						}
						else
						{
							const int histogramIndexCentral = index;
							const int histogramIndexHorizontal = index + HistogramWidth * HistogramHeight;
							const int histogramIndexVertical = index + VolumeResolutionX * multiSamplingRate * HistogramWidth * HistogramHeight;
							const int histogramIndexHorizontalVertical = index + (VolumeResolutionX * multiSamplingRate + 1) * HistogramWidth * HistogramHeight;

							ndfSample = ndfVoxelDataHighRes.histograms[histogramIndexCentral + binIndex] * (1.0f - voxelFract.x) * (1.0f - voxelFract.y);
							ndfSample += ndfVoxelDataHighRes.histograms[histogramIndexHorizontal + binIndex] * (voxelFract.x) * (1.0f - voxelFract.y);
							ndfSample += ndfVoxelDataHighRes.histograms[histogramIndexVertical + binIndex] * (1.0f - voxelFract.x) * (voxelFract.y);
							ndfSample += ndfVoxelDataHighRes.histograms[histogramIndexHorizontalVertical + binIndex] * (voxelFract.x) * (voxelFract.y);
						}
					}

					// correct energy while downsampling using an unfinished state
					ndfSample *= ndfIntensityCorrection;

					vec3 neighbourShading = vec3(0.0f, 0.0f, 0.0f);
					float neighbourdNdfSample = 0.0f;

					// add colors
#if 1
					const float threshold = 0.25f;//0.35f;
					//vec2 transfer = worldSpaceNormal.xy;
					vec2 transfer = viewSpaceNormal.xy;

					if (renderMode == 1)
					{
						const vec3 leftColor = vec3(0.35f, 0.65f, 0.8f);
						const vec3 rightColor = vec3(0.7f, 0.95f, 0.1f);
						const vec3 bottomColor = vec3(0.5f, 0.5f, 0.5f);
						const vec3 topColor = vec3(0.35f, 0.65f, 0.8f);

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
						mat3 lightRotationZ = rotationMatrix(vec3(0.0f, -1.0f, 0.0f), viewSpaceLightDir.x);
						mat3 lightRotationY = rotationMatrix(vec3(1.0f, 0.0f, 0.0f), viewSpaceLightDir.y);

						vec3 transformedNormal = viewSpaceNormal;
						transformedNormal = (lightRotationZ * transformedNormal.xyz).xyz;
						transformedNormal = (lightRotationY * transformedNormal.xyz).xyz;

						const float seamCorrection = 0.125f;
						transformedNormal.xy -= vec2(transformedNormal.xy * seamCorrection);
						vec2 lookupCoord = 1.0f - (transformedNormal.xy * 0.5f + 0.5f);

						vec3 lookupColor = texture(normalTransferSampler, lookupCoord).xyz;

						diffuseColor = lookupColor.xyz;
						specularColor = diffuseColor;
					}

					//float lerp = viewSpaceNormal.y * 0.5f + 0.5f;
					//float saturation = 1.25f;
					//diffuseColor = (1.0f - lerp) * saturation * vec3(0.0f, 1.0f, 0.0f) + lerp * saturation * vec3(1.0f, 0.0f, 0.0f);
#endif

#if 1
					viewView = vec3(0.0f, 0.0f, 1.0f);
					//lightViewSpace = normalize(viewView + vec3(0.25f, 1.25f, 1.0f));

					viewWorld = normalize(worldSpaceMatrix * viewView);
					//lightWorldSpace = -normalize(viewWorld + vec3(0.75f, 0.5f, 0.25f));
#endif

					vec3 centralShading;
					// >> granda <<
					if (renderMode <= 0)
					{
						centralShading = blinnPhong(viewSpaceNormal, -lightViewSpace, viewView, diffuseColor, specularColor, specularExponent);
						//centralShading /= dot(viewSpaceNormal, viewView);
					}
					else
					{
						centralShading = (diffuseColor + specularColor) * 0.5f;
					}

					shadings[i] += lightEnergyCorrection * ndfSample *centralShading;
					energySums[i] += ndfSample;
				}
			}
			//debug
			samplesPerPixel = float(ndfsamplesum) / unit;
			if (samplesPerPixel > 0)
				energies[i] = float(maxSamplingRuns) / samplesPerPixel;
			else
				energies[i] = 0;



			//end debug

			// correct energy of missing rays
			shadings[i] *= energies[i];// energyCollected;
			energySums[i] *= energies[i];// energyCollected;
		}
	}

	//shading /= energySum;
	if (lod - int(lod) == 0)
	{
		outColor = vec4(shadings[0].xyz, 1.0f);
		//if (samplesPerPixel < 100)
		//	outColor = vec4(1, 0, 0, 0);
		//else if (samplesPerPixel<1024)
		//	outColor = vec4(0, 1, 0, 1);
		//else if (samplesPerPixel>800)
		//	outColor = vec4(0, 0, 1, 1);
	}
	else
	{
		//do bilinear interpolation on floor pixels
		vec3 floor_shading = f_du      *f_dv      *shadings[3] +
			                 f_du      *(1 - f_dv)*shadings[2] +
			                 (1 - f_du)*f_dv      *shadings[1] +
		                     (1 - f_du)*(1 - f_dv)*shadings[0];

		//debug
		float count = 0;
		floor_shading = vec3(0, 0, 0);
		for (int i = 0; i < 4; i++)
		{
			if (energies[i]>0)
			{
				count += energies[i];
				floor_shading += energies[i] * shadings[i];
			}
		}
		floor_shading *= 1.0f / count;
		//end debug

		//do linear interpolation between floor and ceil
		vec3 ceil_shading = shadings[4];

		float w1, w2;
		w1 = lod - int(lod);
		w2 = 1.0 - w1;


		vec3 final_shading = w2*floor_shading + w1 *ceil_shading;

		outColor = vec4(final_shading.xyz, 1.0f);

		//debug
		//count number of samples in each pixel
		//if (totalndfSamples / (1.0 / maxSamplingRuns) < 800)
		//	outColor = vec4(1, 0, 0, 1);
		//else if (totalndfSamples / (1.0 / maxSamplingRuns) >= 1000)
		//	outColor = vec4(0, 1, 0, 1);
		//end debug

		//debug
		//if (energies[0] == 0 && energies[1] == 0 && energies[2] == 0 && energies[3] == 0)
		//	outColor = vec4(1, 1, 1, 1);
		//else if (energies[0] == 0)
		//	outColor = vec4(1, 0, 0, 1);
		////else if (energies[1] == 0)
		//	//outColor = vec4(0, 1, 0, 1);
		////else if (energies[2] == 0)
		//	//outColor = vec4(0, 0, 1, 1);
		////else if (energies[3] == 0)
		//	//outColor = vec4(0, 0, 0, 1);
		//else if (energies[4] == 0)
		//	outColor = vec4(1, 1, 0, 1);

		//debug
		//if (energies[0] == 0 && energies[1] == 0 && energies[2] == 0 && energies[3] == 0)
		//{
		//	outColor = vec4(1, 1, 1, 1);
		//}
		//else if (energies[0] == 0)
		//{
		//outColor = vec4(1, 0, 0, 1);
		//}
		//else if (energies[1] == 0 && lookups[1] >= 0)
		//{
		//	outColor = vec4(0, 1, 0, 1);
		//}
		//else if (energies[2] == 0 && lookups[2] >= 0)
		//{
		//	outColor = vec4(0, 0, 1, 1);
		//}
		//else if (energies[3] == 0 && lookups[3] >= 0)
		//{
		//	outColor = vec4(0, 1, 1, 1);
		//}
		//end debug


		//end debug
	}

	// boost contrast
#if 1
	vec3 lowPass;
	{
		const float energyCorrection = 1.0f;
		const float constrast = 0.75f;
		const float brightness = 1.05f;
		lowPass = brightness.xxx * pow(outColor.xyz * energyCorrection.xxx, constrast.xxx);
	}
	vec3 highPass;
	{
		if(renderMode <= 0) 
		{
			const float energyCorrection = 1.0f;//2.125f;
			const float constrast = 2.0f;//2.125f;
			const float brightness = 1.0f;
			highPass = brightness.xxx * pow(outColor.xyz * energyCorrection.xxx, constrast.xxx);
			lowPass *= 0.0f;
		} 
		else if(renderMode == 2) 
		{
			//const float energyCorrection = 2.9f;
			//const float constrast = 4.0f;

			//const float energyCorrection = 1.75f;
			//const float constrast = 2.25f;

			// Video
			//const float energyCorrection = 1.55f;
			//const float constrast = 1.5f;

			// saw tooth sphere
			//const float energyCorrection = 1.8f;
			//const float constrast = 2.75f;

			// lit spheres
			const float energyCorrection = 1.8f;
			const float constrast = 3.5f;

			const float brightness = 1.0f;// * pow(energySum * 0.975f, 8.0f);
			highPass = brightness.xxx * pow(outColor.xyz * energyCorrection.xxx, constrast.xxx);
		}
		else
		{ 
			//const float energyCorrection = 1.45f;
			//const float constrast = 2.0f;
			//const float brightness = 0.95f;

			const float energyCorrection = 1.55f;
			const float constrast = 2.5f;
			const float brightness = 0.95f;

			highPass = brightness.xxx * pow(outColor.xyz * energyCorrection.xxx, constrast.xxx);
		}
	}
	const float brightness = 1.0f;
	//outColor.xyz = brightness * (lowPass * 0.5f + highPass * 0.5f);
	outColor.xyz = brightness * (highPass);

#endif // boost contrast

	// apply background color
	const vec3 backgroundColor = 1.0f * vec3(1.0f, 1.0f, 1.0f);
	//float opacity = energySum;

	// anythig lower than threshold will turn into background
	const float lowerEnergyThreshold = 0.0f;
	// range should be 1.0 - turn down range to give things with lower energy more opacity.
	const float energyRange = 1.0f;
	//const float energyRange = 0.5f;

	//float opacity = smoothstep(lowerEnergyThreshold, lowerEnergyThreshold + energyRange, energySum);
	
	// FIXME: white background is not consistent! gets darker if less smaples hit.
	float opacity = energySum;
	//outColor.xyz = mix(backgroundColor, outColor.xyz, opacity);

	//debug
	//float factor = 1000000000;
	//outColor = imageLoad(tileTex, ivec2(gl_FragCoord.x,gl_FragCoord.y));
	//outColor.x *= factor;
	//outColor.y *= factor;
	//outColor.z *= factor;
	//end debug
}