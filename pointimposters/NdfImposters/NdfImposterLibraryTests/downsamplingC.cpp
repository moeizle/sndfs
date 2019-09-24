#version 430

uniform ivec2 viewDiscretizations;
uniform ivec2 histogramDiscretizations;
uniform ivec2 spatialDiscretizations;

uniform int maxSamplingRuns;
uniform int multiSamplingRate;

int HistogramWidth = histogramDiscretizations.x;
int HistogramHeight = histogramDiscretizations.y;
int ViewWidth = viewDiscretizations.x;
int ViewHeight = viewDiscretizations.y;
int VolumeResolutionX = spatialDiscretizations.x;
int VolumeResolutionY = spatialDiscretizations.y;

layout(std430, binding = 0) buffer NdfVoxelData {
	coherent float histograms[];
} ndfVoxelData;

layout(std430, binding = 1) buffer NdfVoxelDataDownsampled {
	coherent float histograms[];
} ndfVoxelDataDownsampled;

// has to be recompiled each time the size changes - consider the local size division
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() {
	uvec2 spatialCoordinate = uvec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);
	
	const unsigned int downLookup = (spatialCoordinate.y * VolumeResolutionX + spatialCoordinate.x) * HistogramHeight * HistogramWidth;

	const unsigned int lookup = (spatialCoordinate.y * 2 * VolumeResolutionX + spatialCoordinate.x * 2) * HistogramHeight * HistogramWidth;
	const unsigned int lookupHor = (spatialCoordinate.y * 2 * VolumeResolutionX + spatialCoordinate.x * 2 + 1) * HistogramHeight * HistogramWidth;
	const unsigned int lookupVer = ((spatialCoordinate.y * 2 + 1) * VolumeResolutionX + spatialCoordinate.x * 2) * HistogramHeight * HistogramWidth;
	const unsigned int lookupHorVer = ((spatialCoordinate.y * 2 + 1) * VolumeResolutionX + spatialCoordinate.x * 2 + 1) * HistogramHeight * HistogramWidth;

	for(int y = 0; y < HistogramHeight; ++y) {
		for(int x = 0; x < HistogramWidth; ++x) {
			int binIndex = y * HistogramWidth + x;

			ndfVoxelDataDownsampled.histograms[downLookup + binIndex] += ndfVoxelData.histograms[lookup + binIndex] * 0.25f;
			ndfVoxelDataDownsampled.histograms[downLookup + binIndex] += ndfVoxelData.histograms[lookupHor + binIndex] * 0.25f;
			ndfVoxelDataDownsampled.histograms[downLookup + binIndex] += ndfVoxelData.histograms[lookupVer + binIndex] * 0.25f;
			ndfVoxelDataDownsampled.histograms[downLookup + binIndex] += ndfVoxelData.histograms[lookupHorVer + binIndex] * 0.25f;
		}
	}
}