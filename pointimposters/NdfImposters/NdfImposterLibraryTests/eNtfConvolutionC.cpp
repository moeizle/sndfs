#version 440
//#extension GL_NV_bindless_texture : require
//#extension GL_NV_gpu_shader5 : require

layout(std430, binding = 0) buffer NdfVoxelData {
	coherent readonly float histograms[];
	//coherent float histograms[];
} ndfVoxelData;

layout(std430, binding = 1) buffer OutputENtf {
	coherent float histograms[];
} outputENtf;

vec3 blinnPhong(vec3 normal, vec3 light, vec3 view, vec3 diffuseColor, vec3 specularColor, float specularExponent) {
	vec3 halfVector = normalize(light + view);

	float diffuseIntensity =  max(0.0f, dot(normal, -light));

	float nDotHalf = abs(dot(normal, halfVector));
	float specularIntensity = max(0.0f, pow(nDotHalf, specularExponent));

	return diffuseIntensity * diffuseColor + specularIntensity * specularColor;
}

/**
Evaluate brdf.
*/
vec3 shading(vec3 sampleNormal, float ndfSample, vec3 viewSpaceLightDir, vec3 viewDirection) {
	// FIXME: use the transfer function during NDF rendering
	// NOTE: careful not to overbrighten! The approximation algorithm can NOT handle extremely bright spots
	// Rather do some correctiongs during rendering

	// working settings:
	//float specularExponent = 2.0f;
	//float specularCorrection = 0.75f;
	//float energyCorrection = 0.5f;

	float specularExponent = 32.0f;
	float specularCorrection = 2.0f;
	float energyCorrection = 0.5f;

	vec3 specularColor = specularCorrection * vec3(1.0f, 1.0f, 1.0f);
	vec3 diffuseColor = vec3(1.0f, 1.0f, 1.0f);

	vec3 lightViewSpace = -normalize(viewSpaceLightDir);

	return energyCorrection * ndfSample * blinnPhong(sampleNormal, lightViewSpace, viewDirection, diffuseColor, specularColor, specularExponent);
}

const int HistogramWidth = 8;
const int HistogramHeight = 8;

// has to be recompiled each time the size changes - consider the local size division
layout (local_size_x = HistogramWidth, local_size_y = HistogramHeight, local_size_z = 1) in;

void main() {
	uvec2 spatial = gl_WorkGroupID.xy;
	uvec2 light = gl_LocalInvocationID.xy;
	unsigned int offset =  (spatial.y * gl_NumWorkGroups.x + spatial.x) * HistogramHeight * HistogramWidth;

	const vec3 viewDirectionViewSpace = vec3(0.0f, 0.0f, -1.0f);
	const vec2 sampleStep = vec2(1.0f / float(HistogramWidth-1), 1.0f / float(HistogramHeight-1));

	// FIXME: Convolve horizontal first then vertical! much fewer samples required!
	// FIXME: ndf reading is more efficient but has to write to output light - race condition

	vec2 transformedLightCoord = 2.0f * vec2(float(light.x) * sampleStep.x - 0.5f, float(light.y) * sampleStep.y - 0.5f);

	vec3 sampleLight = vec3(
		transformedLightCoord.x,
		transformedLightCoord.y,
		sqrt(1.0f - transformedLightCoord.x*transformedLightCoord.x - transformedLightCoord.y*transformedLightCoord.y));

	float radiance = 0.0f;
	for(unsigned int y = 0; y < HistogramHeight; ++y) {
		for(unsigned int x = 0; x < HistogramWidth; ++x) {
			vec3 sampleNormal;
			sampleNormal.xy = 2.0f * vec2(float(x) * sampleStep.x - 0.5f, float(y) * sampleStep.y - 0.5f);
			sampleNormal.z = sqrt(1.0f - sampleNormal.x*sampleNormal.x - sampleNormal.y*sampleNormal.y);

			float ndf = ndfVoxelData.histograms[offset + y * HistogramWidth + x];

			radiance += shading(sampleNormal, ndf, sampleLight, viewDirectionViewSpace).x;
		}
	}

	outputENtf.histograms[offset + light.y * HistogramWidth + light.x] += radiance;
}