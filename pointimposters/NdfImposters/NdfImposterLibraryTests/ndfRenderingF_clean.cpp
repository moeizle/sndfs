#version 430
//#pragma optimize (off)

//#define OUTPUT_HISTOGRAM
//#define OUTPUT_EBRDF
#define NDF

//#define MAX_NUM_OF_BINS 64*64

in vec2 texCoords;
in vec3 rayBegin;
in vec3 frustumTarget;
in vec3 viewSpacePosition;
in vec3 worldSpacePosition;

out vec4 outColor;
uniform int binningMode;

uniform int renderMode;
uniform int zoomMode;
uniform float zoomScale;
uniform vec2 zoomWindow;
uniform int ssaoEnabled;
uniform int ssaoDownsampleCount;
uniform int visualizeTiles;
uniform int showEmptyTiles;

uniform int samplingRunIndex;
uniform int maxSamplingRuns;
uniform int multiSamplingRate;

uniform float quantizationVarianceScale;
uniform float ndfIntensityCorrection;

uniform float binLimit;

uniform int colorMapSize;

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
layout(binding = 3, rgba32f) uniform image2D tex;
//end sampling texture

//layout(binding = 0, rgba16ui) uniform uimage2D tileTex;
layout(binding = 1, rgba32f) uniform image2D floorLevel;
layout(binding = 2, rgba32f) uniform image2D ceilLevel;
layout(binding = 0, rgba32f) uniform image2D ndfTex;


uniform float lod;

uniform int plainRayCasting;
uniform int cachedRayCasting;
uniform int probeNDFsMode;
uniform int ndfOverlayMode;

vec4 backgroundColor = vec4(0,0, 0, 1);

uniform int win_w;
uniform int win_h;

uniform int ceil_w;
uniform int floor_w;
uniform int ceil_h;
uniform int floor_h;
uniform int tile_w;
uniform int tile_h;
uniform int phys_tex_dim;

uniform int progressiveSamplesCount;

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

uint simMeasureFactor = 10000u;
uint minVal, maxVal;
float minValf, maxValf;
const int colorSize = 5;
//vec3 colors[colorSize] = { vec3(160.f / 256.f, 32.f / 256.f, 240.f / 256.f), vec3(1, 0, 0), vec3(1, 1, 0), vec3(0, 1, 1),vec3(0,0,1) };
vec3 colors[colorSize] = { vec3(1,0,1), vec3(0, 0, 1), vec3(0, 1, 0), vec3(1, 1, 0), vec3(1, 0, 0) };

const float PI = 3.141592f;
const float toRadiants = PI / 180.0f;
//const int max_num_of_bins = 4096;

//bindings
uniform int ndfVoxelDataBining;


float f_du, f_dv, c_du, c_dv;

layout(std430) buffer NdfVoxelData
{
	coherent readonly float histograms[];
} ndfVoxelData;

//layout(std430) buffer NdfVoxelDataHighRes
//{
//	coherent readonly float histograms[];
//} ndfVoxelDataHighRes;



layout(std430) buffer progressive_raycasting_data
{
	float screen_data[]; // This is the important name (in the shader).
};

layout(std430) buffer preIntegratedBins
{
	float binColor[]; // This is the important name (in the shader).
};

layout(std430) buffer sampleCount
{
	float sample_count[]; // This is the important name (in the shader).
};

layout(std430) buffer binAreas
{
	double bArea[]; // This is the important name (in the shader).
};

//layout(std430, binding = 6) buffer regionColor
//{
//	float region_color[]; // This is the important name (in the shader).
//};
layout(std430) buffer avgNDF
{
	float avg_NDF[]; // This is the important name (in the shader).
};
layout(std430) buffer similarityLimits
{
	uint sim_limits[]; // This is the important name (in the shader).
};
layout(std430) buffer colorMap
{
	float color_map[]; // This is the important name (in the shader).
};
layout(std430) buffer similarityLimitsF
{
	float sim_limitsf[]; // This is the important name (in the shader).
};
layout(std430) buffer ndfOverlay
{
	float ndf_overlay[]; // This is the important name (in the shader).
};


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


		tileCoords_InPhysTex.x *= tile_w;
		tileCoords_InPhysTex.y *= tile_h;

		//location in ndf tree is the physical texture location + within tile offset
		//ivec2 Pixelcoord_InPhysTex = ivec2(tileCoords_InPhysTex.x + withinTileOffset.x, tileCoords_InPhysTex.y + withinTileOffset.y);
		int Pixelcoord_InPhysTex = int(tileCoords_InPhysTex.x + (withinTileOffset.y*tile_w + withinTileOffset.x));

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

		//to visualize tiles
		if (((visualizeTiles == 0 && !levelFlag) || (visualizeTiles == 1 && levelFlag) || (visualizeTiles == 2)) &&
			(withinTileOffset.x == 0 || withinTileOffset.y == 0 || withinTileOffset.x == tile_w - 1 || withinTileOffset.y == tile_h - 1))
		{
			//that is we draw a blue pixel for the tile borders
			return -9;
		}


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
float  get_sample(int binIndex, int lookup)
{
	float h = float(binIndex) / 2.0f;

	//read the value at the new histogram central, it should contain two bins
	float combined_samples = ndfVoxelData.histograms[lookup + int(floor(h))];

	vec2 components = unpackHalf2x16(floatBitsToUint(combined_samples));

	if (fract(h) > 0.49f)
	{
		//return b2
		return components.y;
	}
	else
	{
		//return b1
		return components.x;
	}
}
#if 0
void similarityBasedShading(float t, inout vec4 shadingColor,bool invert)
{
	//similarity Measure
	//vec3 red = vec3(1, 0, 0);
	//vec3 green = vec3(0, 1, 0);
	//vec3 blue = vec3(0, 0, 1);

	//instead of dealing with a whole spectrum, t in [0,1], we know that t is in [minVal,maxVal], so our spectrum should only be over that region.
	//float minValf = float(minVal);// / float(simMeasureFactor);
	//float maxValf = float(maxVal);// / float(simMeasureFactor);

	float T = (t - minValf) / (maxValf - minValf);

	if (invert)
		T = 1.0f - T;

	//vec3 similarityColor = c1 + p*(c2 - c1);
	int cindex = int(T*(colorMapSize-1));
	vec3 similarityColor = vec3(color_map[3*cindex], color_map[3*cindex+1], color_map[3*cindex+2]);
	//similarityColor = blue + t*(red - blue);
	//if (t<.08)
	shadingColor *= vec4(similarityColor,1);
	//else
	//	shadingColor *= vec4(red, 1);
}
void binsToBeCompared(inout float val1, inout float val2, int i, int lookup)
{
	double binArea;
	float totalNumberOfSamples = sample_count[lookup / (HistogramHeight*HistogramWidth)];

	val1 = avg_NDF[i];
	//val1 = min(avg_NDF[i], 1.0f);

	//ndf
#if 1
	//if ((bArea[i] != 0.0) && (bArea[i] == bArea[i]))
	//	binArea = bArea[i];
	//else
	//	binArea = 0.0;

	//if (binArea != 0.0)
	val2 = ndfVoxelData.histograms[lookup + i] / totalNumberOfSamples;// / float(binArea);
		//ndf2 = min((ndfVoxelData.histograms[lookup + i] / totalNumberOfSamples) / float(binArea), 1.0f);
	//else
	//	val2 = 0.0f;
#endif
	//cdf
#if 0
		val2 = ndfVoxelData.histograms[lookup + i] / totalNumberOfSamples;
#endif
	
}
float L1norm(int lookup)
{
	float t, d, f, w;
	float ndf1, ndf2;
	float l1norm = 0.0f;
	float diff;
	
	double binArea;

	for (int i = 0; i < HistogramHeight*HistogramWidth; i++)
	{
		binsToBeCompared(ndf1, ndf2, i, lookup);
		diff = ndf1 - ndf2;
		l1norm += abs(diff);
	}

	
	//l1norm /= float(HistogramWidth*HistogramHeight);
	return l1norm;
}
float L2norm(int lookup)
{
	//get l2norm between avg ndf and my ndf
	float l2Norm = 0.0f;
	float t;
	float ndf2,ndf1;
	//float totalNumberOfSamples = sample_count[lookup / (HistogramHeight*HistogramWidth)];
	double binArea;

	for (int i = 0; i < HistogramWidth*HistogramHeight; i++)
	{
		binsToBeCompared(ndf1, ndf2, i, lookup);

		l2Norm += (ndf2 - ndf1)*(ndf2 - ndf1);
	}

	l2Norm = sqrt(l2Norm);

	t = l2Norm;// / (HistogramHeight*HistogramWidth);
	return t;
}
float KolmogorovSmirnov(int lookup)
{
	//get l2norm between avg ndf and my ndf
	float MD = 0.0f;
	float t;
	float ndf2,ndf1;
	//float totalNumberOfSamples = sample_count[lookup / (HistogramHeight*HistogramWidth)];
	double binArea;
	float accumNDF1, accumNDF2;
	float diff;

	accumNDF1 = accumNDF2 = 0.0f;
	float maxDiff = 0.0f;

	for (int i = 0; i < HistogramWidth*HistogramHeight; i++)
	{
		binsToBeCompared(ndf1, ndf2, i, lookup);

		accumNDF1 += ndf2;
		accumNDF2 += ndf1;

		diff= abs(accumNDF1-accumNDF2);

		if (diff>maxDiff)
			maxDiff = diff;
	}

	//l2Norm = sqrt(l2Norm);
	float n = HistogramHeight*HistogramWidth;
	float nthTriangularNumber = ((n*n) + n) / 2.0f;

	t = maxDiff;// / n;// nthTriangularNumber;
	return t;
}
float HistogramIntersection(int lookup)
{
	//get l2norm between avg ndf and my ndf
	float intersection = 0.0f;
	float sumAvg_NDF = 0.0f;
	float t;
	float ndf2,ndf1;
	//float totalNumberOfSamples = sample_count[lookup / (HistogramHeight*HistogramWidth)];
	double binArea;
	float accumNDF1, accumNDF2;

	accumNDF2 = accumNDF1 = 0.0f;

	for (int i = 0; i < HistogramWidth*HistogramHeight; i++)
	{
		binsToBeCompared(ndf1, ndf2, i, lookup);
		//accumNDF1 += avg_NDF[i];
		//accumNDF2 += ndf2;
		intersection += min(ndf2, ndf1);           
		//sumAvg_NDF += ndf1;
	}

	//l2Norm = sqrt(l2Norm);

	t = intersection;// / sumAvg_NDF);            //ndf2 or avg_NDF[i] should have a maximum value of 1 per bin, so we divide by number of bins to normalize 
	return t;
}
float klDivergence(int lookup)
{
	//get l2norm between avg ndf and my ndf
	float kl = 0.0f;
	float sumAvg_NDF = 0.0f;
	float t;
	float ndf2, ndf1;
	//float totalNumberOfSamples = sample_count[lookup / (HistogramHeight*HistogramWidth)];
	double binArea;
	float accumNDF1, accumNDF2;
	float val;

	accumNDF2 = accumNDF1 = 0.0f;

	for (int i = 0; i < HistogramWidth*HistogramHeight; i++)
	{
		binsToBeCompared(ndf1, ndf2, i, lookup);
		//accumNDF1 += avg_NDF[i];
		//accumNDF2 += ndf2;
		if (ndf1 != 0.0f && ndf2 != 0.0f)
		{
			kl += ndf1*log(ndf1 / ndf2);
		}
			
		//sumAvg_NDF += ndf1;
	}

	//l2Norm = sqrt(l2Norm);

	t = kl;// / sumAvg_NDF);            //ndf2 or avg_NDF[i] should have a maximum value of 1 per bin, so we divide by number of bins to normalize 
	return t;
}
float jeffreyDivergence(int lookup)
{
	//get l2norm between avg ndf and my ndf
	float jl = 0.0f;
	float sumAvg_NDF = 0.0f;
	float t;
	float ndf2, ndf1;
	//float totalNumberOfSamples = sample_count[lookup / (HistogramHeight*HistogramWidth)];
	double binArea;
	float accumNDF1, accumNDF2;
	float m;
	float c1, c2;

	accumNDF2 = accumNDF1 = 0.0f;

	for (int i = 0; i < HistogramWidth*HistogramHeight; i++)
	{
		binsToBeCompared(ndf1, ndf2, i, lookup);
		//accumNDF1 += avg_NDF[i];
		//accumNDF2 += ndf2;

		m = (ndf1 + ndf2) / 2.0f;
		if (m != 0.0f)
		{
			jl +=ndf1*log(ndf1/m)+ndf2*log(ndf2/m);
		}
		

		//sumAvg_NDF += ndf1;
	}

	//l2Norm = sqrt(l2Norm);

	t = jl;// / sumAvg_NDF);            //ndf2 or avg_NDF[i] should have a maximum value of 1 per bin, so we divide by number of bins to normalize 
	return t;
}
float xSquaredStatistics(int lookup)
{
	//get l2norm between avg ndf and my ndf
	float x2 = 0.0f;
	float sumAvg_NDF = 0.0f;
	float t;
	float ndf2, ndf1;
	//float totalNumberOfSamples = sample_count[lookup / (HistogramHeight*HistogramWidth)];
	double binArea;
	float accumNDF1, accumNDF2;
	float m;

	accumNDF2 = accumNDF1 = 0.0f;

	for (int i = 0; i < HistogramWidth*HistogramHeight; i++)
	{
		binsToBeCompared(ndf1, ndf2, i, lookup);
		//accumNDF1 += avg_NDF[i];
		//accumNDF2 += ndf2;
		m = (ndf1 + ndf2) / 2.0f;
		if (m != 0.0f)
		{
			x2 += ((ndf1 - m) *(ndf1 - m)) / m;
		}
		//sumAvg_NDF += ndf1;
	}

	//l2Norm = sqrt(l2Norm);

	t = x2;// / sumAvg_NDF);            //ndf2 or avg_NDF[i] should have a maximum value of 1 per bin, so we divide by number of bins to normalize 
	return t;
}
float earthMoversDistance(int lookup)
{
#if 0
	{
		float work, weightSum, EMD;
		float dij, fij;
		float Wi, prevWi, Uj, prevUj;

		float accumMyNDF[64];
		float accumAvgNDF[64];

		float accum1, accum2;

		accum1 = accum2 = 0.0f;

		for (int i = 0; i < HistogramHeight*HistogramWidth; i++)
		{
			accum1 += avg_NDF[i];
			accumAvgNDF[i] = accum1;

			accum2 += myNDF[k][i];
			accumMyNDF[i] = accum2;
		}

		for (int i = 0; i < HistogramHeight*HistogramWidth; i++)
		{
			for (int j = 0; j < HistogramWidth*HistogramHeight; j++)
			{
				dij = abs(i - j) / float(HistogramHeight*HistogramWidth - 1);

				prevWi = prevUj = 0.0f;

				Wi = accumMyNDF[i];
				if (i>0)
					prevWi = accumMyNDF[i - 1];

				Uj = accumAvgNDF[j];
				if (j > 0)
					prevUj = accumAvgNDF[j - 1];


				fij = abs(max(prevWi, prevUj) - min(Wi, Uj));

				work += fij*dij;
				weightSum += fij;
			}
		}

		EMD = work / weightSum;

		return EMD;
	}
#endif
	{
		float nextEMD, curEMD;
		float EMD = 0.0f;
		float ndf2,ndf1;
		//float totalNumberOfSamples = sample_count[lookup / (HistogramHeight*HistogramWidth)];
		double binArea;
		curEMD = 0.0f;

		for (int i = 0; i < HistogramWidth*HistogramHeight; i++)
		{
			binsToBeCompared(ndf1, ndf2, i, lookup);

			nextEMD = (ndf2 + curEMD) - ndf1;
			
			EMD += abs(curEMD);

			curEMD = nextEMD;
		}
		//float n = HistogramHeight*HistogramWidth;
		//float nthTriangularNumber = ((n*n) + n) / 2.0f;

		//accumNDF1 can take a maximum value of nthtriangularnumber and so can accumndf2, so their diffrence is a maximum of 2*nthtriangularnumber

		//EMD = EMD / (2.0f* nthTriangularNumber);
		return EMD;// / (n + (n*(n + 1.0f) / 2.0f));
	}
}
float maxMinMetric(int j)
{
	return 0;
}

#endif
void renderFromNDFs()
{
	vec3 samplePosition = rayBegin;

	//outColor = vec4(1, 1, 1, 1);
	//return;

	//debug
	//samplePosition.x = 1.0f - samplePosition.x;
	//samplePosition.y = 1.0f - samplePosition.y;
	//end debug

	const vec3 offset = vec3(0.4f, 0.0f, 0.0f);
	//samplePosition = mod(samplePosition + offset, vec3(1.0f, 1.0f, 1.0f));

	// TODO: add depth map and parallax mapping? Better rotation...
	// TODO: normal of tangent frame using depth map
	vec3 viewWorld;// = normalize(camPosi - worldSpacePosition);
	vec3 viewView;// = normalize(camPosi - viewSpacePosition);

	//outColor = vec4(0.5f, 0.0f, 0.0f, 1.0f);
	//return;

	//for probendfs mode
	//read max value and min value that 't' took in previous iteration
	//minVal = sim_limits[0];
	//maxVal = sim_limits[1];

	vec3 shading = vec3(0.0f, 0.0f, 0.0f);
	float ndfSample = 0.0f;
	int ndfColorTest = 0;
	float energySum = 0.0f;
	vec3 shadings[5] = { vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 0.0f, 0.0f) };
	float energySums[5] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	float energies[5] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
	int lookups[5] = { -1, -1, -1, -1, -1 };
	int tile_sample_count[5] = { 0, 0, 0, 0, 0 };
	float totalHits = 0;
	float samplesPerPixel;
	float unit = 1.0f;// / float(maxSamplingRuns);
	
	float totalSampleCount;
	float totalHitCount;
	bool empty_pixel = true;
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

		////debug
		//vec2 newRay = imageLoad(tex, voxelCoord).xy;
		//outColor = vec4(newRay, 0.f, 1.0f);
		//return;
		//end debug

		//if (voxelCoord.x < 100 || voxelCoord.y < 100)
		//{
		//	outColor = vec4(1.0,0,0,0);
		//	return;
		//}


		//const float angleRange = 360.0f;//2.0f;//20.0f; //360.0f
		//vec2 camAngleScaled = vec2((camAngles.x / angleRange), (camAngles.y / angleRange));
		int viewCoordinateX = 0;//int(camAngleScaled.x * float(ViewWidth-1));
		int viewCoordinateY = 0;//int(camAngleScaled.y * float(ViewHeight-1));

		vec2 voxelFract;
		int index;

		voxelFract = vec2(fract(voxelCoordF.x * float(VolumeResolutionX)), fract(voxelCoordF.y * float(VolumeResolutionY)));


		//vec3 diffuseColor = 0.25f * vec3(1.0f, 1.0f, 1.0f);
		vec3 diffuseColor = 1.0f * vec3(1.0f, 1.0f, 1.0f);

		// NOTE: for this model to be physically based the BRDF would have to be perfectly specular
		float specularExponent = 128.0f;
		float specularCorrection = 5.0f;
		vec3 specularColor = specularCorrection * vec3(1.0f, 0.0f, 0.0f);

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



		// FIXME: remove test - splat BRDF and sample only current light dir
		const float lightEnergyCorrection = 1.0f;


		//if (voxelCoord.x < 100 || voxelCoord.y < 100)
		//{
		//	outColor = vec4(1, 0, 0, 0);
		//	return;
		//}


		/////////////////
		//mohamed's index
		//if a pixel is outside the visible region, it need not be rendered.
		if ((voxelCoord.x < blc_s.x) || (voxelCoord.x >= trc_s.x) || (voxelCoord.y < blc_s.y) || (voxelCoord.y >= trc_s.y))
		{
			outColor = backgroundColor;
			return;
		}


		float energyCollected = 0;

		//calculcate lookups
		ivec2 pixel_in_ceil, garbage;

		if (visualizeTiles == 0 || visualizeTiles == 2)
		{
			lookups[4] = calculate_lookup(ceil_w, ceil_h, tile_sample_count[4], false, vec2(voxelCoord), lod, floor(lod + 1), vec2(0, 0), c_du, c_dv, pixel_in_ceil);
			if (lookups[4] == -9 && (visualizeTiles == 0 || visualizeTiles == 2))
			{
				outColor = vec4(0, 1, 0, 0);
				return;
			}
		}


		lookups[0] = calculate_lookup(floor_w, floor_h, tile_sample_count[0], true, vec2(voxelCoord), lod, floor(lod), vec2(0, 0), f_du, f_dv, garbage);
		lookups[1] = calculate_lookup(floor_w, floor_h, tile_sample_count[1], true, vec2(voxelCoord), lod, floor(lod), vec2(1, 0), f_du, f_dv, garbage);
		lookups[2] = calculate_lookup(floor_w, floor_h, tile_sample_count[2], true, vec2(voxelCoord), lod, floor(lod), vec2(0, 1), f_du, f_dv, garbage);
		lookups[3] = calculate_lookup(floor_w, floor_h, tile_sample_count[3], true, vec2(voxelCoord), lod, floor(lod), vec2(1, 1), f_du, f_dv, garbage);

		for (int i = 0; i < 4; i++)
		{

			if (lookups[i] == -9 && (visualizeTiles == 1 || visualizeTiles == 2))
			{
				outColor = vec4(0, 0, 1, 0);
				return;
			}
		}

		//outColor = vec4(0,1,1,0);
		//return;


		//to display tiles that are not cached.

		if (showEmptyTiles == 1)
		{
			for (int i = 0; i < 5; i++)
			{

				if (lookups[i] == -2)
				{
					outColor = vec4(1, 0, 0, 0);
					return;
				}
			}
		}


		double binArea, binCDF, binNDF;
		float totalNumberOfSamples;
		vec3 binNTF;
		vec3 regionC;
		vec3 red = vec3(1, 0, 0);
		vec3 blue = vec3(0, 0, 1);
		vec3 white = vec3(1, 1, 1);
		vec3 yellow = vec3(1, 1, 0);
		vec3 similarityColor;

		const int colorSize = 3;
		//vec3 colors[colorSize] = { vec3(0, 0, 0), vec3(160.f/256.f,32.f/256.f,240.f/256.f), vec3(1, 0, 0), vec3(1,1,0), vec3(1, 1, 1) };
		//vec3 colors[colorSize] = { vec3(1, 0, 0), vec3(1, 1, 0), vec3(1, 1, 1) };


		
		////total sample count
		totalSampleCount = 0.0f;
		totalHitCount = 0.0f;

		for (int i = 0; i < 4; i++)
		{
			float ndfsamplesum = 0.0f;
			totalHits = 0;
			float similarityMeasure = 0.0;

			if (lookups[i] >= 0)
			{
				totalNumberOfSamples = sample_count[lookups[i] / (HistogramHeight*HistogramWidth)];
				totalSampleCount += totalNumberOfSamples;
				if (probeNDFsMode >0)
				{
					if (sim_limitsf[2 + (win_w*voxelCoord.y + voxelCoord.x)] == -2.0f)              //regionC is initialized to 1,1,1 in case it is picked by user, it will be 1,0,0, it is enough to check if y or z==0
					{
						outColor = vec4(1, 0, 1, 1);
						return;
					}
				}

				//count misses per pixel
				for (int histogramY = 0; histogramY < HistogramHeight; histogramY++)
				{
					for (int histogramX = 0; histogramX < HistogramWidth; histogramX++)
					{
						int binIndex = histogramY * HistogramWidth + histogramX;
						ndfSample = ndfVoxelData.histograms[lookups[i] + binIndex];
						ndfsamplesum += ndfSample;
					}
				}
				float misses = totalNumberOfSamples - ndfsamplesum;
				totalHitCount += ndfsamplesum;

				//outColor = (1.0/1024.0f) *vec4(totalNumberOfSamples, totalNumberOfSamples, totalNumberOfSamples,1);
				//outColor = (1.0 / 1024.0f) *vec4(misses, misses, misses, 1);
				//return;

				//count hits per bin
				for (int histogramY = 0; histogramY < HistogramHeight; histogramY++)
				{
					for (int histogramX = 0; histogramX < HistogramWidth; histogramX++)
					{

						int binIndex = histogramY * HistogramWidth + histogramX;
						ndfSample = ndfVoxelData.histograms[lookups[i] + binIndex];

						vec3 centralShadingPreComputed = vec3(binColor[binIndex * 3], binColor[binIndex * 3 + 1], binColor[binIndex * 3 + 2]);


						if ((bArea[binIndex] != 0.0) && (bArea[binIndex] == bArea[binIndex]))
							binArea = bArea[binIndex];
						else
							binArea = 0.0;

#if 0
						double mx, mn;
						mx = 0;
						mn = 10;
						for (int i = 0; i<HistogramWidth*HistogramHeight; i++)
						{
							if (bArea[i]>mx)
								mx = bArea[i];

							if (bArea[i]<mn)
								mn = bArea[i];
						}
						if (ndfSample > 0 && binIndex <= HistogramWidth*HistogramHeight)
						{
							double t = (bArea[binIndex] - mn) / (mx - mn);
							dvec3 c1 = dvec3(0.3, 0.3, 0.3);// vec3(250.0 / 255, 128.0 / 255, 114.0 / 255);
							dvec3 c2 = dvec3(1, 1, 1);// vec3(0, 191.0 / 255, 1);
							dvec3 c = c1 + t*(c2 - c1);
							shadings[i] = vec3(c);// vec3(64.0f* bArea[binIndex], 64.0f* bArea[binIndex], 64.0f* bArea[binIndex]);


							bool nl = false;

							{
								int NeighbourLookup = calculate_lookup(floor_w, floor_h, tile_sample_count[0], true, vec2(voxelCoord - 1.0f), lod, floor(lod), vec2(0, 0), f_du, f_dv, garbage);
								if (NeighbourLookup >= 0)
								{
									for (int k = 0; k < HistogramWidth*HistogramHeight; k++)
									{
										if (ndfVoxelData.histograms[lookups[i] + k]>0)
										{
											for (int h = 0; h<HistogramHeight*HistogramHeight && h != k; h++)
											{

												if (ndfVoxelData.histograms[NeighbourLookup + h] > 0)
												{
													shadings[i] = vec3(0, 0, 0);
													nl = true;
													break;
												}
											}
										}
										if (nl)
											break;

									}
								}
							}

							if (!nl)
							{
								int NeighbourLookup = calculate_lookup(floor_w, floor_h, tile_sample_count[0], true, vec2(voxelCoord + 1.0f), lod, floor(lod), vec2(0, 0), f_du, f_dv, garbage);
								if (NeighbourLookup >= 0)
								{
									for (int k = 0; k < HistogramWidth*HistogramHeight; k++)
									{
										if (ndfVoxelData.histograms[lookups[i] + k]>0)
										{
											for (int h = 0; h<HistogramHeight*HistogramHeight && h != k; h++)
											{

												if (ndfVoxelData.histograms[NeighbourLookup + h] > 0)
												{
													shadings[i] = vec3(0, 0, 0);
													nl = true;
													break;
												}
											}
										}
										if (nl)
											break;

									}
								}

							}

							if (!nl)
							{
								int NeighbourLookup = calculate_lookup(floor_w, floor_h, tile_sample_count[0], true, vec2(voxelCoord + vec2(1.0f, -1.0f)), lod, floor(lod), vec2(0, 0), f_du, f_dv, garbage);
								if (NeighbourLookup >= 0)
								{
									for (int k = 0; k < HistogramWidth*HistogramHeight; k++)
									{
										if (ndfVoxelData.histograms[lookups[i] + k]>0)
										{
											for (int h = 0; h<HistogramHeight*HistogramHeight && h != k; h++)
											{

												if (ndfVoxelData.histograms[NeighbourLookup + h] > 0)
												{
													shadings[i] = vec3(0, 0, 0);
													nl = true;
													break;
												}
											}
										}
										if (nl)
											break;

									}
								}

							}

							if (!nl)
							{
								int NeighbourLookup = calculate_lookup(floor_w, floor_h, tile_sample_count[0], true, vec2(voxelCoord + vec2(-1.0f, 1.0f)), lod, floor(lod), vec2(0, 0), f_du, f_dv, garbage);
								if (NeighbourLookup >= 0)
								{
									for (int k = 0; k < HistogramWidth*HistogramHeight; k++)
									{
										if (ndfVoxelData.histograms[lookups[i] + k]>0)
										{
											for (int h = 0; h<HistogramHeight*HistogramHeight && h != k; h++)
											{

												if (ndfVoxelData.histograms[NeighbourLookup + h] > 0)
												{
													shadings[i] = vec3(0, 0, 0);
													nl = true;
													break;
												}
											}
										}
										if (nl)
											break;

									}
								}

							}

							if (!nl)
							{
								int NeighbourLookup = calculate_lookup(floor_w, floor_h, tile_sample_count[0], true, vec2(voxelCoord + vec2(0, 1.0f)), lod, floor(lod), vec2(0, 0), f_du, f_dv, garbage);
								if (NeighbourLookup >= 0)
								{
									for (int k = 0; k < HistogramWidth*HistogramHeight; k++)
									{
										if (ndfVoxelData.histograms[lookups[i] + k]>0)
										{
											for (int h = 0; h<HistogramHeight*HistogramHeight && h != k; h++)
											{

												if (ndfVoxelData.histograms[NeighbourLookup + h] > 0)
												{
													shadings[i] = vec3(0, 0, 0);
													nl = true;
													break;
												}
											}
										}
										if (nl)
											break;

									}
								}

							}
							if (!nl)
							{
								int NeighbourLookup = calculate_lookup(floor_w, floor_h, tile_sample_count[0], true, vec2(voxelCoord + vec2(0, -1.0f)), lod, floor(lod), vec2(0, 0), f_du, f_dv, garbage);
								if (NeighbourLookup >= 0)
								{
									for (int k = 0; k < HistogramWidth*HistogramHeight; k++)
									{
										if (ndfVoxelData.histograms[lookups[i] + k]>0)
										{
											for (int h = 0; h<HistogramHeight*HistogramHeight && h != k; h++)
											{

												if (ndfVoxelData.histograms[NeighbourLookup + h] > 0)
												{
													shadings[i] = vec3(0, 0, 0);
													nl = true;
													break;
												}
											}
										}
										if (nl)
											break;

									}
								}

							}
							if (!nl)
							{
								int NeighbourLookup = calculate_lookup(floor_w, floor_h, tile_sample_count[0], true, vec2(voxelCoord + vec2(-1.0f, 0)), lod, floor(lod), vec2(0, 0), f_du, f_dv, garbage);
								if (NeighbourLookup >= 0)
								{
									for (int k = 0; k < HistogramWidth*HistogramHeight; k++)
									{
										if (ndfVoxelData.histograms[lookups[i] + k]>0)
										{
											for (int h = 0; h<HistogramHeight*HistogramHeight && h != k; h++)
											{

												if (ndfVoxelData.histograms[NeighbourLookup + h] > 0)
												{
													shadings[i] = vec3(0, 0, 0);
													nl = true;
													break;
												}
											}
										}
										if (nl)
											break;

									}
								}

							}
							if (!nl)
							{
								int NeighbourLookup = calculate_lookup(floor_w, floor_h, tile_sample_count[0], true, vec2(voxelCoord + vec2(1.0f, 0)), lod, floor(lod), vec2(0, 0), f_du, f_dv, garbage);
								if (NeighbourLookup >= 0)
								{
									for (int k = 0; k < HistogramWidth*HistogramHeight; k++)
									{
										if (ndfVoxelData.histograms[lookups[i] + k]>0)
										{
											for (int h = 0; h<HistogramHeight*HistogramHeight && h != k; h++)
											{

												if (ndfVoxelData.histograms[NeighbourLookup + h] > 0)
												{
													shadings[i] = vec3(0, 0, 0);
													nl = true;
													break;
												}
											}
										}
										if (nl)
											break;

									}
								}

							}

						}
#endif


						//if (ndfsamplesum > totalNumberOfSamples)
						//{
						//	float diff = ndfsamplesum - totalNumberOfSamples;
						//	if (diff > .000001)
						//	{	
						//		outColor = vec4(0, 1, 0, 1);
						//		return;
						//	}
						//}

						if (binArea != 0.0f)
						{
							binCDF = double(ndfSample / (totalNumberOfSamples));
							binNDF = binCDF / binArea;// / area;
							binNTF = centralShadingPreComputed;// *area;
							shadings[i] += float(binNDF)* binNTF;// centralShading;// / maxSamplingRuns;
							energySums[i] += ndfSample;
						}
					}
				}
			}
		}
	}


	//shading /= energySum;
	//if (lod - int(lod) == 0)
	//{
	//	outColor = vec4(shadings[0].xyz, 1.0f);
	//	//if (outColor.x == 0 && outColor.y == 0 && outColor.z == 0)
	//	//	outColor.xyz = vec3(1, 1, 1);
	//}
	//else
	{
		vec4 colors[5];
		for (int i = 0; i < 4; i++)
		{
			//if (energies[i] == 0)
			//{
			//	colors[i] = vec4(0, 0, 0, 1);
			//}
			//else
			{
				colors[i] = vec4(shadings[i].xyz, 1.0);
			}
		}
		//do bilinear interpolation on floor pixels
		vec4 floor_shading = f_du         *f_dv         *colors[3] +
			f_du         *(1.0f - f_dv)*colors[1] +
			(1.0f - f_du)*f_dv         *colors[2] +
			(1.0f - f_du)*(1.0f - f_dv)*colors[0];

		//debug
		//float count = 0;
		//floor_shading = vec4(0, 0, 0,1);
		//for (int i = 0; i < 4; i++)
		//{
		//	if (energies[i]>0)
		//	{
		//		count += energies[i];
		//		floor_shading += vec4(energies[i] * shadings[i],0);
		//	}
		//}
		//floor_shading *= 1.0f / count;
		//floor_shading = vec4(floor_shading.xyz, 1.0);
		//end debug

		//do linear interpolation between floor and ceil
		vec4 ceil_shading = (colors[0] + colors[1] + colors[2] + colors[3]) / 4.0f;

		


		//old
		//float w1, w2;
		//w1 = lod - int(lod);
		//w2 = 1.0 - w1;
		//outColor = w2*floor_shading + w1 *ceil_shading;
		//end old

		//if (length(ceil_shading.xyz) == 0)
		//{
		//	outColor = backgroundColor;
		//	return;
		//}

		outColor = mix(floor_shading, ceil_shading, fract(lod));



		// apply background color
		if (totalSampleCount == 0)
		{
			outColor = backgroundColor;
			return;
		}
		//outColor.xyz *= vec3(141 / 255.f, 211 / 255.f, 199 / 255.f);
		outColor.xyz = mix(backgroundColor.xyz, outColor.xyz, totalHitCount/totalSampleCount);
		

//NDF overlay mode
#if 1
		if (ndfOverlayMode > 0 )
		{
			vec3 floorColors[4];
			for (int i = 0; i<4; i++)
			{
#if 1
				floorColors[i].x = ndf_overlay[3*(lookups[i] / (HistogramHeight*HistogramWidth))];
				floorColors[i].y = ndf_overlay[3*(lookups[i] / (HistogramHeight*HistogramWidth))+1];
				floorColors[i].z = ndf_overlay[3*(lookups[i] / (HistogramHeight*HistogramWidth))+2];
#else
				int filterDim = 8;
				int l;
				float s = 0.0f;
				vec2 voxelCoordF = rayBegin.xy;
				ivec2 voxelCoord = ivec2(int(voxelCoordF.x * float(VolumeResolutionX)), int(voxelCoordF.y * float(VolumeResolutionY)));
				ivec2 garbage;
				for (int j = -filterDim/2; j < filterDim/2; j++)
				{
					for (int k = -filterDim / 2; k < filterDim / 2; k++)
					{
						l=calculate_lookup(floor_w, floor_h, tile_sample_count[0], true, vec2(voxelCoord), lod, floor(lod), vec2(j, k), f_du, f_dv, garbage);
						if (l >= 0)
						{
							if (ndf_overlay[3 * (l / (HistogramHeight*HistogramWidth))] == 1 && 
								ndf_overlay[3 * (l / (HistogramHeight*HistogramWidth))+1] == 1 && 
								ndf_overlay[3 * (l / (HistogramHeight*HistogramWidth))+2] ==1 )
							{
							}
							else
							{

								floorColors[i].x += ndf_overlay[3 * (l / (HistogramHeight*HistogramWidth))];
								floorColors[i].y += ndf_overlay[3 * (l / (HistogramHeight*HistogramWidth)) + 1];
								floorColors[i].z += ndf_overlay[3 * (l / (HistogramHeight*HistogramWidth)) + 2];
								s++;
							}
						}
					}
				}

				floorColors[i] = floorColors[i] / s;
#endif
			}

			vec3 floorColor = f_du         *f_dv         *floorColors[3] +
				              f_du         *(1.0f - f_dv)*floorColors[1] +
				              (1.0f - f_du)*f_dv         *floorColors[2] +
				              (1.0f - f_du)*(1.0f - f_dv)*floorColors[0];

			vec3 ceilColor = (floorColors[0] + floorColors[1] + floorColors[2] + floorColors[3]) / 4.0f;

			vec3 finalColor = mix(floorColor, ceilColor, fract(lod));
			outColor.xyz = mix(outColor.xyz,floorColor,0.5);
			//outColor = vec4(finalColor, 1.0f);
		}
#endif
#if 0
		if (probeNDFsMode > 0 && totalHitCount>0)
		{
			float simMeasure[4];

			for (int i = 0; i < 4; i++)
			{

				if (probeNDFsMode == 2)
				{
					simMeasure[i] = L1norm(lookups[i]);

				}
				else if (probeNDFsMode == 3)
				{
					simMeasure[i] = L2norm(lookups[i]);
				}
				else if (probeNDFsMode == 4)
				{
					simMeasure[i] = HistogramIntersection(lookups[i]);
				}
				else if (probeNDFsMode == 5)
				{
					simMeasure[i] = xSquaredStatistics(lookups[i]);
				}

				if (simMeasure[i] < 0)
				{
					outColor = vec4(1, 0, 0, 1);
					return;
				}
				//if (simMeasure[i] > 1)
				//{
				//	outColor = vec4(0, 1, 0, 1);
				//	return;
				//}
			}

			float floorSimMeasure = f_du         *f_dv         *simMeasure[3] +
				f_du         *(1.0f - f_dv)*simMeasure[1] +
				(1.0f - f_du)*f_dv         *simMeasure[2] +
				(1.0f - f_du)*(1.0f - f_dv)*simMeasure[0];

			float ceilSimMeasure = (simMeasure[0] + simMeasure[1] + simMeasure[2] + simMeasure[3]) / 4.0f;

			float similarityMeasure = mix(floorSimMeasure, ceilSimMeasure, fract(lod));
#if 0
			minVal = min(atomicMin(sim_limits[0], uint(similarityMeasure * simMeasureFactor)), uint(similarityMeasure * simMeasureFactor));
			maxVal = max(atomicMax(sim_limits[1], uint(similarityMeasure * simMeasureFactor)), uint(similarityMeasure * simMeasureFactor));
#endif
#if 1
			minValf = min(sim_limitsf[0],similarityMeasure);
			maxValf = max(sim_limitsf[1],similarityMeasure);

			ivec2 voxelCoord = ivec2(int(samplePosition.x * float(VolumeResolutionX)), int(samplePosition.y * float(VolumeResolutionY)));
			int pixel_ind = voxelCoord.y*win_w + voxelCoord.x;

			//update similarity measure to be used in calculating simlimitsf for the next iteration
			sim_limitsf[2+ pixel_ind] = similarityMeasure;

#endif
			if (probeNDFsMode == 4)
				similarityBasedShading(similarityMeasure, outColor,true);
			else if (probeNDFsMode!=1)
				similarityBasedShading(similarityMeasure, outColor, false);

			
		}
#endif

	}
}

void renderFromTexture()
{
	//vec2 newRay = imageLoad(tex, voxelCoord).xy;
	//outColor = vec4(newRay.xy, 0.0f, 1.0f);
	//return;

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
	float unit = 1.0f;// / float(maxSamplingRuns);
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

		//vec3 bar = texture(normalTransferSampler,vec2(10,10)).xyz;
		//outColor = vec4(bar, 1);

		//return;
		//end debug


		//const float angleRange = 360.0f;//2.0f;//20.0f; //360.0f
		//vec2 camAngleScaled = vec2((camAngles.x / angleRange), (camAngles.y / angleRange));
		int viewCoordinateX = 0;//int(camAngleScaled.x * float(ViewWidth-1));
		int viewCoordinateY = 0;//int(camAngleScaled.y * float(ViewHeight-1));

		vec2 voxelFract;
		int index;

		voxelFract = vec2(fract(voxelCoordF.x * float(VolumeResolutionX)), fract(voxelCoordF.y * float(VolumeResolutionY)));


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

		const float histogramScaleX = 1.0f / float(HistogramWidth  - 1);
		const float histogramScaleY = 1.0f / float(HistogramHeight - 1);



		// FIXME: remove test - splat BRDF and sample only current light dir
		//const float lightEnergyCorrection = 14.0f;
		//int histogramX = HistogramWidth*2 / 2; {
		//	int histogramY = HistogramHeight - 3; {
		const float lightEnergyCorrection = 1.0f;

		vec3 sourceNormal;

		sourceNormal.xy =2.0f * imageLoad(tex, voxelCoord).xy - vec2(1.0f, 1.0f); 
		//sourceNormal.z = 1.0f - sqrt(sourceNormal.x*sourceNormal.x + sourceNormal.y*sourceNormal.y);
		sourceNormal.z = sqrt(1.0f - sourceNormal.x*sourceNormal.x - sourceNormal.y*sourceNormal.y);


		//debug
		//outColor.xy = imageLoad(tex, voxelCoord).xy;
		//return;
		//end debug

		// project to actual view space since the nearest view samples does not necessarily equal the current view direction
		//vec3 viewSpaceNormal = normalize(viewCorrection * sourceNormal);
		vec3 viewSpaceNormal = sourceNormal;

		viewView = vec3(0.0f, 0.0f, 1.0f);
		//lightViewSpace = normalize(viewView + vec3(0.25f, 1.25f, 1.0f));

		viewWorld = normalize(worldSpaceMatrix * viewView);
		//lightWorldSpace = -normalize(viewWorld + vec3(0.75f, 0.5f, 0.25f));


		vec3 centralShading = vec3(0, 0, 0);

		if (viewSpaceNormal.x == 0 && viewSpaceNormal.y == 0)
		{
			outColor = vec4(0, 0, 0, 1);
		}
		else
		{
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


			if (renderMode <= 0)
			{
				centralShading = blinnPhong(viewSpaceNormal, -lightViewSpace, viewView, .8*diffuseColor, 0 * specularColor, specularExponent);
				//centralShading = vec3(dot(-lightViewSpace, viewSpaceNormal));
			}
			else
			{
				centralShading = (diffuseColor + specularColor) * 0.5f;
			}

			//The color in 'centralShading' is only from the current sample, to enable progressive sampling, we'll add the color to the progressive sampling data per pixel,
			//and take the average color


			if (progressiveSamplesCount < maxSamplingRuns)
			{
				screen_data[3 * (voxelCoord.y*win_w + voxelCoord.x)] += centralShading.x;
				screen_data[3 * (voxelCoord.y*win_w + voxelCoord.x) + 1] += centralShading.y;
				screen_data[3 * (voxelCoord.y*win_w + voxelCoord.x) + 2] += centralShading.z;
			}


			int count = min(progressiveSamplesCount + 1, maxSamplingRuns);

			vec3 weighted_color = vec3(screen_data[3 * (voxelCoord.y*win_w + voxelCoord.x)] / count,
				screen_data[3 * (voxelCoord.y*win_w + voxelCoord.x) + 1] / count,
				screen_data[3 * (voxelCoord.y*win_w + voxelCoord.x) + 2] / count);  //+1 so we don't divide by 0


			//if (samplingRunIndex == 0)
			//{
			//	outColor = vec4(1, 0, 0, 1);
			//	return;
			//}
			//else if (samplingRunIndex == 1)
			//{
			//	outColor = vec4(0, 1, 0, 1);
			//	return;
			//}
			//else if (samplingRunIndex == 2)
			//{
			//	outColor = vec4(0, 0, 1, 1);
			//	return;
			//}

			outColor = vec4(weighted_color, 1); //vec4(centralShading.x,centralShading.x,centralShading.x, 1);
			
			//	outColor.xyz = vec3(1, 1, 1);
		}
	}
}

void renderCachedRaycasting()
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
	float unit = 1.0f;// / float(maxSamplingRuns);
	float totalMisses = 0;
	float totalsamples = 0;
	{
		// TODO: rename
		vec2 voxelCoordF = samplePosition.xy;

		vec3 viewDir = viewWorld * 0.5f + 0.5f;

		//if (viewDir.x < 0.0f || viewDir.x > 1.0f || viewDir.y < 0.0f || viewDir.y > 1.0f)
		//{
		//	outColor = vec4(0.5f, 0.0f, 0.0f, 1.0f);
		//	return;
		//}

		ivec2 voxelCoord = ivec2(int(voxelCoordF.x * float(VolumeResolutionX)), int(voxelCoordF.y * float(VolumeResolutionY)));

		////debug
		//vec2 newRay = imageLoad(tex, voxelCoord).xy;
		//outColor = vec4(newRay, 0.f, 1.0f);
		//return;
		//end debug

		//if (voxelCoord.x < 100 || voxelCoord.y < 100)
		//{
		//	outColor = vec4(1.0,0,0,0);
		//	return;
		//}


		//const float angleRange = 360.0f;//2.0f;//20.0f; //360.0f
		//vec2 camAngleScaled = vec2((camAngles.x / angleRange), (camAngles.y / angleRange));
		int viewCoordinateX = 0;//int(camAngleScaled.x * float(ViewWidth-1));
		int viewCoordinateY = 0;//int(camAngleScaled.y * float(ViewHeight-1));

		vec2 voxelFract;
		int index;

		voxelFract = vec2(fract(voxelCoordF.x * float(VolumeResolutionX)), fract(voxelCoordF.y * float(VolumeResolutionY)));


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

		const float histogramScaleX = 1.0f / float(HistogramWidth  - 1);
		const float histogramScaleY = 1.0f / float(HistogramHeight - 1);



		// FIXME: remove test - splat BRDF and sample only current light dir
		const float lightEnergyCorrection = 1.0f;


		//if (voxelCoord.x < 100 || voxelCoord.y < 100)
		//{
		//	outColor = vec4(1, 0, 0, 0);
		//	return;
		//}


		/////////////////
		//mohamed's index
		//if a pixel is outside the visible region, it need not be rendered.
		//if ((voxelCoord.x < blc_s.x) || (voxelCoord.x >= trc_s.x) || (voxelCoord.y < blc_s.y) || (voxelCoord.y >= trc_s.y))
		//	return;


		float energyCollected = 0;

		//calculcate lookups
		ivec2 pixel_in_ceil, garbage;

		lookups[4] = calculate_lookup(ceil_w, ceil_h, tile_sample_count[4], false, vec2(voxelCoord), lod, floor(lod + 1), vec2(0, 0), c_du, c_dv, pixel_in_ceil);
		if (lookups[4] == -9 && (visualizeTiles == 0 || visualizeTiles == 2))
		{
			outColor = vec4(0, 1, 0, 0);
			return;
		}

		lookups[0] = calculate_lookup(floor_w, floor_h, tile_sample_count[0], true, vec2(voxelCoord), lod, floor(lod), vec2(0, 0), f_du, f_dv, garbage);
		lookups[1] = calculate_lookup(floor_w, floor_h, tile_sample_count[1], true, vec2(voxelCoord), lod, floor(lod), vec2(1, 0), f_du, f_dv, garbage);
		lookups[2] = calculate_lookup(floor_w, floor_h, tile_sample_count[2], true, vec2(voxelCoord), lod, floor(lod), vec2(0, 1), f_du, f_dv, garbage);
		lookups[3] = calculate_lookup(floor_w, floor_h, tile_sample_count[3], true, vec2(voxelCoord), lod, floor(lod), vec2(1, 1), f_du, f_dv, garbage);

		for (int i = 0; i < 4; i++)
		{

			if (lookups[i] == -9 && (visualizeTiles == 1 || visualizeTiles == 2))
			{
				outColor = vec4(0, 0, 1, 0);
				return;
			}
		}

		//outColor = vec4(0,1,1,0);
		//return;

		//debug
		//for (int i = 0; i < 5; i++)
		//{

		//	if (lookups[i] == -2)
		//	{
		//		outColor = vec4(1, 0, 0, 0);
		//		return;
		//	}
		//}
		//debug

		


		for (int i = 0; i < 4; i++)
		{
			float ndfsamplesum = 0;
			if (lookups[i] >= 0)
			{
				float count = sample_count[lookups[i] / (HistogramHeight*HistogramWidth)];
				totalsamples += count;
				totalMisses += ndfVoxelData.histograms[lookups[i]];

				shadings[i] = vec3(ndfVoxelData.histograms[lookups[i] + 1], ndfVoxelData.histograms[lookups[i] + 2], ndfVoxelData.histograms[lookups[i] + 3]);
				shadings[i] *= 1.0f / count;
				
				//if (count< 1.0f)
				//{
				//	outColor = vec4(1,0,0, 1.0);
				//	return;
				//}
			}

		}
	}


	//shading /= energySum;
	//if (lod - floor(lod) == 0)
	//{
	//	outColor = vec4(shadings[0].xyz, 1.0f);
	//}
	//else
	{
		vec4 colors[5];
		for (int i = 0; i < 4; i++)
		{
			//if (energies[i] == 0)
			//{
			//	colors[i] = vec4(0, 0, 0, 1);
			//}
			//else
			{
				colors[i] = vec4(shadings[i].xyz, 1.0);
			}
		}
		//do bilinear interpolation on floor pixels
		vec4 floor_shading = f_du         *f_dv         *colors[3] +
			f_du         *(1.0f - f_dv)*colors[1] +
			(1.0f - f_du)*f_dv         *colors[2] +
			(1.0f - f_du)*(1.0f - f_dv)*colors[0];

		//debug
		//float count = 0;
		//floor_shading = vec4(0, 0, 0,1);
		//for (int i = 0; i < 4; i++)
		//{
		//	if (energies[i]>0)
		//	{
		//		count += energies[i];
		//		floor_shading += vec4(energies[i] * shadings[i],0);
		//	}
		//}
		//floor_shading *= 1.0f / count;
		//floor_shading = vec4(floor_shading.xyz, 1.0);
		//end debug

		//do linear interpolation between floor and ceil
		vec4 ceil_shading = (colors[0] + colors[1] + colors[2] + colors[3]) / 4.0f;

		outColor = mix(floor_shading, ceil_shading, fract(lod));

		//vec4 backgroundColor = vec4(1, 1, 1, 1);
		if (totalsamples == 0)
		{
			outColor = backgroundColor;
			return;
		}
		outColor.xyz = mix(backgroundColor.xyz, outColor.xyz, (totalsamples-totalMisses)/totalsamples);
		//if (outColor.x == 0 && outColor.y == 0 && outColor.z == 0)
	}


}

void main()
{
	if (bool(plainRayCasting))
	{
		renderFromTexture();
	}
	else
	{
		if (bool(cachedRayCasting))
		{
			renderCachedRaycasting();
		}
		else
		{
			renderFromNDFs();

		}
	}

	
}