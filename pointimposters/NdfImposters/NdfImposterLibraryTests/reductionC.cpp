#version 430

uniform unsigned int localDivision;
uniform uvec2 spatialOffset;
uniform uvec2 invocationOffset;
uniform uvec2 invocationSize;
uniform uvec2 currentView;

uniform ivec2 viewDiscretizations;
uniform ivec2 histogramDiscretizations;
uniform ivec3 spatialDiscretizations;
uniform ivec2 viewportSize;

int HistogramWidth = histogramDiscretizations.x;
int HistogramHeight = histogramDiscretizations.y;
int ViewWidth = viewDiscretizations.x;
int ViewHeight = viewDiscretizations.y;
int VolumeResolutionX = spatialDiscretizations.x;
int VolumeResolutionY = spatialDiscretizations.y;
int VolumeResolutionZ = spatialDiscretizations.z;

uniform sampler2D inputSamples;
uniform int offset;
uniform ivec2 sampleResolution;

layout(std430, binding = 0) buffer NdfVoxelData {
	coherent float histograms[];
} ndfVoxelData;

// has to be recompiled each time the size changes - consider the local size division
layout (local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main() {
	const int sampleCount = sampleResolution.x*sampleResolution.y;
	const float unit = 1.0f / float(sampleCount);
	const float HistogramWidthF = float(HistogramWidth-1);
	const float HistogramHeightF = float(HistogramHeight-1);

	for(int sampleY = 0; sampleY < sampleResolution.y; ++sampleY) {
		for(int sampleX = 0; sampleX < sampleResolution.x; ++sampleX) {
			vec2 normal = texelFetch(inputSamples, ivec2(sampleX, sampleY), 0).xy;

			ivec2 histogramBin = ivec2(int(normal.y * HistogramWidthF), int(normal.x * HistogramHeightF));
			int histogramIndex = histogramBin.y * HistogramWidth + histogramBin.x;

			ndfVoxelData.histograms[offset + histogramIndex] += unit;
		}
	}
}