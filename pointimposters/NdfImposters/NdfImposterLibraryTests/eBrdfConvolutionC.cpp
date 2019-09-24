#version 430
//#extension GL_NV_bindless_texture : require
//#extension GL_NV_gpu_shader5 : require

uniform unsigned int localDivision;
uniform uvec2 spatialOffset;
uniform uvec2 invocationOffset;
uniform uvec2 invocationSize;
uniform uvec2 currentView;

uniform ivec2 viewDiscretizations;
uniform ivec2 histogramDiscretizations;
uniform ivec3 spatialDiscretizations;
uniform ivec2 viewportSize;

// statically sized buffers are faster...
//const int HistogramWidth = 9;
//const int HistogramHeight = 9;
//const int ViewWidth = 1;
//const int ViewHeight = 1;
//const int VolumeResolution = 64;
//const int VolumeResolutionX = VolumeResolution;
//const int VolumeResolutionY = VolumeResolution;
//const int VolumeResolutionZ = 1;

int HistogramWidth = histogramDiscretizations.x;
int HistogramHeight = histogramDiscretizations.y;
int ViewWidth = viewDiscretizations.x;
int ViewHeight = viewDiscretizations.y;
int VolumeResolutionX = spatialDiscretizations.x;
int VolumeResolutionY = spatialDiscretizations.y;
int VolumeResolutionZ = spatialDiscretizations.z;

layout(std430, binding = 0) buffer NdfVoxelData {
	//coherent readonly float histograms[];
	coherent float histograms[];
} ndfVoxelData;

layout(std430, binding = 1) buffer OutNdfVoxelData {
	//coherent writeonly float histograms[];
	coherent float histograms[];
} outNdfVoxelData;

/**
Evaluate brdf.
*/
vec3 shading(vec3 sampleNormal, float ndfSample, vec3 viewSpaceLightDir, vec3 viewDirection) {
	vec3 color = vec3(0.0f, 0.0f, 0.0f);

	//vec3 diffuse = 0.75 * vec3(1.0f, 1.0f, 1.0f) * dot(sampleNormal, viewSpaceLightDir);
	vec3 diffuse = 0.75f * vec3(1.0f, 1.0f, 1.0f) * dot(sampleNormal, viewSpaceLightDir);

	// phong shading
	//const float specularExponent = 32.0f;
	//const float specularBrightnessCorrection = 1.5f;

	//const float specularExponent = 1024.0f;
	//const float specularBrightnessCorrection = 50.0f;
	
	//const float specularExponent = 96.0f;
	//const float specularBrightnessCorrection = 3.0f;
		
	const float specularExponent = 128.0f;
	const float specularBrightnessCorrection = 3.5f;
	
	const vec3 specularColor = vec3(1.0f, 1.0f, 1.0f) * specularBrightnessCorrection;
	vec3 halfVector = normalize(-viewDirection + viewSpaceLightDir);
	vec3 specular = specularColor * pow(max(0.0f, dot(sampleNormal, halfVector)), specularExponent);

	color = vec3(ndfSample * (specular + diffuse) * 0.5f).xyz;
	//color = vec3(ndfSample * (specular)).xyz;

	return color;
}

// has to be recompiled each time the size changes - consider the local size division
layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
//layout (local_size_x = 9, local_size_y = 9, local_size_z = 1) in;
//layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
//layout (local_size_x = 17, local_size_y = 17, local_size_z = 1) in;
//layout (local_size_x = 25, local_size_y = 25, local_size_z = 1) in;
//layout (local_size_x = 32, local_size_y = 32, local_size_z = 1) in;
//layout (local_size_x = 64, local_size_y = 64, local_size_z = 1) in;

float convolve(uvec2 reference, unsigned int offset) {
	const vec3 viewDirectionViewSpace = vec3(0.0f, 0.0f, -1.0f);
	const vec2 sampleStep = vec2(1.0f / float(HistogramWidth-1), 1.0f / float(HistogramHeight-1));

	// convolve NDF and BRDF -> integrating over normals with fixed light
	vec2 transformedLightCoord = vec2(float(reference.x) * sampleStep.x, float(reference.y) * sampleStep.y) - 0.5f;
	
	// FIXME: is this correct?
	// clamp to unit circle if out of range
	//if(length(transformedLightCoord) > 0.5f) {
	//	transformedLightCoord = normalize(transformedLightCoord) * 0.5f;
	//}

	vec3 sampleLight = vec3(
		transformedLightCoord.x,
		transformedLightCoord.y,
		sqrt(1.0f - transformedLightCoord.x*transformedLightCoord.x - transformedLightCoord.y*transformedLightCoord.y));
	
	vec3 eBRDF = vec3(0.0f, 0.0f, 0.0f);
	for(unsigned int y = invocationOffset.y; y < invocationOffset.y + invocationSize.y; ++y) {
		for(unsigned int x = invocationOffset.x; x < invocationOffset.x + invocationSize.x; ++x) {
			vec2 sampleCoords = vec2(float(x) * sampleStep.x, float(y) * sampleStep.y) - 0.5f;

			vec3 sampleNormal = vec3(
				sampleCoords.x,
				sampleCoords.y,
				sqrt(1.0f - sampleCoords.x*sampleCoords.x - sampleCoords.y*sampleCoords.y));

			float ndf = ndfVoxelData.histograms[offset + y * HistogramWidth + x];;

			eBRDF += shading(sampleNormal, ndf, sampleLight, viewDirectionViewSpace).xyz;
		}
	}

	return eBRDF.x;
}

void main() {
	uvec2 offsetVolume = uvec2(gl_WorkGroupID.x, gl_WorkGroupID.y) + spatialOffset.xy;
	uvec2 volume = offsetVolume / localDivision;
	uvec2 localGroup = offsetVolume % localDivision;
	uvec2 view = currentView;

	unsigned int offset = 
				  volume.y * VolumeResolutionX * ViewHeight * ViewWidth * HistogramHeight * HistogramWidth
				+ volume.x * ViewHeight * ViewWidth * HistogramHeight * HistogramWidth;

	unsigned int viewOffset = offset
				+ view.y * ViewWidth * HistogramHeight * HistogramWidth
				+ view.x * HistogramHeight * HistogramWidth;

	const uvec2 localHistogramDomainSize = uvec2(HistogramWidth, HistogramHeight) / localDivision;

	uvec2 outgoingRadianceCoordinate = gl_LocalInvocationID.xy + uvec2(localHistogramDomainSize.x * localGroup.x, localHistogramDomainSize.y * localGroup.y);
	//unsigned int outgoingRadianceIndex = offset + outgoingRadianceCoordinate.y * HistogramWidth + outgoingRadianceCoordinate.x;
	unsigned int outgoingRadianceIndex = viewOffset + outgoingRadianceCoordinate.y * HistogramWidth + outgoingRadianceCoordinate.x;

	// FIXME: Convolve horizontal first then vertical! much fewer samples required!

	// NOTE: assumes a clear output ssbo
	outNdfVoxelData.histograms[outgoingRadianceIndex] += convolve(outgoingRadianceCoordinate, viewOffset);
}