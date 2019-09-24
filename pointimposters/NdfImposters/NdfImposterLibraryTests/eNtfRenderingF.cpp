#version 430

//#define OUTPUT_HISTOGRAM
//#define OUTPUT_EBRDF
#define NDF

in vec2 texCoords;
in vec3 rayBegin;
in vec3 frustumTarget;
in vec3 viewSpacePosition;
in vec3 worldSpacePosition;

out vec4 outColor;

/*uniform int renderMode;
uniform int zoomMode;
uniform float zoomScale;
uniform vec2 zoomWindow;
uniform int ssaoEnabled;
uniform int ssaoDownsampleCount;

uniform int samplingRunIndex;
uniform int maxSamplingRuns;
uniform int multiSamplingRate;*/

// limit the amount of gaussians per pixel so it doesnt crash if the offset map has errors.
//uniform int maxGaussians;

uniform float quantizationVarianceScale;

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

int HistogramWidth = histogramDiscretizations.x;
int HistogramHeight = histogramDiscretizations.y;
int ViewWidth = viewDiscretizations.x;
int ViewHeight = viewDiscretizations.y;
int VolumeResolutionX = spatialDiscretizations.x;
int VolumeResolutionY = spatialDiscretizations.y;

const float PI = 3.141592f;
const float toRadiants = PI / 180.0f;

layout(std430, binding = 0) buffer OffsetMap {
	coherent readonly uint gaussianOffsets[];
} offsetMap;

layout(std430, binding = 1) buffer Gaussians {
	coherent readonly uint quantizedGaussians[];
} gaussians;

float gaussKernel(float x, float variance, float mean) {
	const float M_PI = 3.141593f;

	float diff_squared = (x - mean) * (x - mean);
	float exponent = -(diff_squared / (2.0f * variance));
	float numerator = exp(exponent);
	float denominator = sqrt(variance) * sqrt(2.0f * M_PI);

	return numerator / denominator;
}

void main() {
	vec3 samplePosition = rayBegin;

	const vec3 offset = vec3(0.4f, 0.0f, 0.0f);
	//samplePosition = mod(samplePosition + offset, vec3(1.0f, 1.0f, 1.0f));

	// TODO: add depth map and parallax mapping? Better rotation...
	// TODO: normal of tangent frame using depth map
	vec3 viewWorld = normalize(camPosi - worldSpacePosition);
	vec3 viewView = normalize(camPosi - viewSpacePosition);

	vec3 shading = vec3(0.0f, 0.0f, 0.0f);
	float ndfSample = 0.0f;
	int ndfColorTest = 0;
	float energySum = 0.0f;

	// TODO: rename
	vec2 voxelCoordF = samplePosition.xy;

	vec3 viewDir = viewWorld * 0.5f + 0.5f;

	if(viewDir.x < 0.0f || viewDir.x > 1.0f || viewDir.y < 0.0f || viewDir.y > 1.0f) {
		outColor = vec4(0.5f, 0.0f, 0.0f, 1.0f);
		return;
	}

	ivec2 voxelCoord = ivec2(
		int(voxelCoordF.x * float(VolumeResolutionX)),
		int(voxelCoordF.y * float(VolumeResolutionY)));

	int index = (voxelCoord.y * VolumeResolutionX + voxelCoord.x);

	// FIXME: correct?
	//vec2 light = 1.0f - (viewSpaceLightDir.xy + 0.5f);
	vec2 light = 1.0f - (viewSpaceLightDir.xy * 0.5f + 0.5f);

	unsigned int gaussianOffset = offsetMap.gaussianOffsets[index];
	unsigned int gaussianNextOffset = offsetMap.gaussianOffsets[index + 1];

	/*if(gaussianNextOffset > gaussianOffset) {
		outColor = vec4(0.75f, 0.0f, 0.0f, 1.0f);
		return;
	}*/

	// has to be bounded
	//gaussianNextOffset = min(gaussianNextOffset, gaussianOffset + maxGaussians);
	// has to be larger than previous offset
	//gaussianNextOffset = max(gaussianNextOffset, gaussianOffset);

	// more aggressive quantization
	float totalLikelihood = 0.0f;
	// TODO: use 1 / max variance as scale
	// NOTE: variance will be too small if not scaled to an appropriate range
	float lambertian = 0.0f;
	float radiance = 0.0f;
	for(unsigned int gaussianI = gaussianOffset; gaussianI < gaussianNextOffset; ++gaussianI) {
		// dequantize
		vec3 means;
		vec2 variances;

	unsigned int packedj = gaussians.quantizedGaussians[gaussianI];

		// assuming big endian
		means.x = float(packedj & 0x0000003f) / 64.0f;
		means.y = float((packedj & 0x00000fc0) >> 6) / 64.0f;
		means.z = float((packedj & 0x000ff000) >> 12) / 256.0f;

		// FIXME: quantizationVarianceScale should probably half as high
		//const float varianceScale = 0.5f;
		const float varianceScale = 0.2f;
		variances.x = varianceScale * quantizationVarianceScale * float((packedj & 0x03f00000) >> 20) / 64.0f;
		variances.y = varianceScale * quantizationVarianceScale * float((packedj & 0xfc000000) >> 26) / 64.0f;

		if(means.x > 0.0f || means.y > 0.0f || variances.x > 0.0f || variances.y > 0.0f) {
			float likelihood = 
				gaussKernel(light.x, variances.x, means.x) *
				gaussKernel(light.y, variances.y, means.y);

			totalLikelihood += likelihood;
			radiance += likelihood * means.z;
		} else {
			lambertian = means.z;
		}
	}

	if(totalLikelihood > 0.0f) {
		radiance /= totalLikelihood; 
	}
	radiance += lambertian;

	outColor.xyz = radiance.xxx;
	energySum = totalLikelihood;

	// boost contrast
#if 1
	vec3 highPass;
	{
		// NOTE: changing the Gaussian variance is another way to adjust the contrast!

		// time dependent: video
		const float energyCorrection = 4.25f;
		const float constrast = 2.5f;
		const float brightness = 1.5f;

		//const float energyCorrection = 4.0f;
		//const float constrast = 2.0f;
		//const float brightness = 1.0f;

		highPass = brightness.xxx * pow(outColor.xyz * energyCorrection.xxx, constrast.xxx);
	}
	outColor.xyz = highPass;
#endif // boost contrast

#if 0
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
	outColor.xyz = mix(backgroundColor, outColor.xyz, opacity);
#endif
}