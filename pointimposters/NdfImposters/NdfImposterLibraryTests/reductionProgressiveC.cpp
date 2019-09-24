#version 430

//#define SINGLE_PRECISION
//#define HIGH_RES

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

layout(binding = 0, rg8) uniform image2D tex;

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

// has to be recompiled each time the size changes - consider the local size division
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main() 
{
	// NOTE: constant has to be equal to multisamling rate on host
	const int multiSampling = multiSamplingRate;
	//const float unit = 1.0f / float(maxSamplingRuns * multiSampling);
#ifdef SINGLE_PRECISION
	const float unit = 1.0f / float(maxSamplingRuns * multiSampling * multiSampling);
#else
	const double unit = 1.0 / double(maxSamplingRuns * multiSampling * multiSampling);
#endif // SINGLE_PRECISION

#ifdef SINGLE_PRECISION
	//const float unitHighRes = 1.0f / float(maxSamplingRuns * multiSampling * multiSampling * multiSampling * multiSampling);
	//const float unitHighRes = 1.0f / float(maxSamplingRuns * multiSampling * multiSampling);
	const float unitHighRes = 1.0f / float(maxSamplingRuns);
#else
	//const double unitHighRes = 1.0 / double(maxSamplingRuns * multiSampling * multiSampling * multiSampling * multiSampling);
	//const double unitHighRes = 1.0 / double(maxSamplingRuns * multiSampling * multiSampling);
	const double unitHighRes = 1.0 / double(maxSamplingRuns);
#endif // SINGLE_PRECISION

	uvec2 spatialCoordinate = uvec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);
	
	const unsigned int lookup = (spatialCoordinate.y * VolumeResolutionX + spatialCoordinate.x) * HistogramHeight * HistogramWidth;
	// FIXME: race condition
	//const unsigned int downLookup = (spatialCoordinate.y / 2) * VolumeResolutionX + (spatialCoordinate.x / 2)) * HistogramHeight * HistogramWidth;

	for(int y = 0; y < multiSampling; ++y) {
		for(int x = 0; x < multiSampling; ++x) {
			ivec2 multisampleSpatialCoordinate = ivec2(spatialCoordinate.xy) * multiSampling + ivec2(x, y);
			vec2 newRay = imageLoad(tex, multisampleSpatialCoordinate).xy;
	
#if 0
			splatReconstructionKernel(newRay, lookup, unit);
#else
			vec2 quantizedRay = vec2(newRay.x * float(HistogramWidth-1), newRay.y * float(HistogramHeight-1));

			// NOTE: the total amount of energy will only be available if the max. number of samples is reached.
			// before that the renderer has to upscale the ndf samples according to the total energy.
			const int histogramIndexCentral = 
				int(quantizedRay.y) * HistogramWidth +
				int(quantizedRay.x);
			const int histogramIndexHorizontal = 
				int(quantizedRay.y) * HistogramWidth +
				int(quantizedRay.x) + 1;
			const int histogramIndexVertical = 
				(int(quantizedRay.y) + 1) * HistogramWidth +
				int(quantizedRay.x);
			const int histogramIndexHorizontalVertical = 
				(int(quantizedRay.y) + 1) * HistogramWidth +
				int(quantizedRay.x) + 1;

			// TODO: use epanechnikov kernel instead of piecewise linear (supposed to be better than gaussian wrt MSE)

#ifdef SINGLE_PRECISION
			vec2 bilinearWeights = vec2(fract(quantizedRay.x), fract(quantizedRay.y));
#else
			dvec2 bilinearWeights = dvec2(double(fract(quantizedRay.x)), double(fract(quantizedRay.y)));
#endif // SINGLE_PRECISION

			// NOTE: the total amount of energy will only be available if the max. number of samples is reached.
			// before that the renderer has to upscale the ndf samples according to the total energy.
			if(histogramIndexCentral > 0 && histogramIndexCentral < HistogramHeight * HistogramWidth) {
#ifdef SINGLE_PRECISION
				ndfVoxelData.histograms[lookup + histogramIndexCentral] += (1.0f - bilinearWeights.x) * (1.0f - bilinearWeights.y) * unit;
#else
				ndfVoxelData.histograms[lookup + histogramIndexCentral] += float((1.0 - bilinearWeights.x) * (1.0 - bilinearWeights.y) * unit);
#endif // SINGLE_PRECISION;
			}
			if(histogramIndexHorizontal > 0 && histogramIndexHorizontal < HistogramHeight * HistogramWidth) {
#ifdef SINGLE_PRECISION
				ndfVoxelData.histograms[lookup + histogramIndexHorizontal] += (bilinearWeights.x) * (1.0f - bilinearWeights.y) * unit;
#else
				ndfVoxelData.histograms[lookup + histogramIndexHorizontal] += float((bilinearWeights.x) * (1.0 - bilinearWeights.y) * unit);
#endif // SINGLE_PRECISION
			}
			if(histogramIndexVertical > 0 && histogramIndexVertical < HistogramHeight * HistogramWidth) {
#ifdef SINGLE_PRECISION
				ndfVoxelData.histograms[lookup + histogramIndexVertical] += (1.0f - bilinearWeights.x) * (bilinearWeights.y) * unit;
#else
				ndfVoxelData.histograms[lookup + histogramIndexVertical] += float((1.0 - bilinearWeights.x) * (bilinearWeights.y) * unit);
#endif // SINGLE_PRECISION
			}
			if(histogramIndexHorizontalVertical > 0 && histogramIndexHorizontalVertical < HistogramHeight * HistogramWidth) {
#ifdef SINGLE_PRECISION
				ndfVoxelData.histograms[lookup + histogramIndexHorizontalVertical] += (bilinearWeights.x) * (bilinearWeights.y) * unit;
#else
				ndfVoxelData.histograms[lookup + histogramIndexHorizontalVertical] += float((bilinearWeights.x) * (bilinearWeights.y) * unit);
#endif // SINGLE_PRECISION
			}

#ifdef HIGH_RES
			unsigned int lookupHigRes = ((multisampleSpatialCoordinate.y * VolumeResolutionX * multiSampling) + multisampleSpatialCoordinate.x)
			//unsigned int lookupHigRes = ((multisampleSpatialCoordinate.y * VolumeResolutionX) + multisampleSpatialCoordinate.x)
				* HistogramHeight * HistogramWidth;

			// NOTE: the total amount of energy will only be available if the max. number of samples is reached.
			// before that the renderer has to upscale the ndf samples according to the total energy.
			if(histogramIndexCentral > 0 && histogramIndexCentral < HistogramHeight * HistogramWidth) {
#ifdef SINGLE_PRECISION
				ndfVoxelDataHighRes.histograms[lookupHigRes + histogramIndexCentral] += (1.0f - bilinearWeights.x) * (1.0f - bilinearWeights.y) * unitHighRes;
#else
				ndfVoxelDataHighRes.histograms[lookupHigRes + histogramIndexCentral] += float((1.0 - bilinearWeights.x) * (1.0 - bilinearWeights.y) * unitHighRes);
#endif // SINGLE_PRECISION;
			}
			if(histogramIndexHorizontal > 0 && histogramIndexHorizontal < HistogramHeight * HistogramWidth) {
#ifdef SINGLE_PRECISION
				ndfVoxelDataHighRes.histograms[lookupHigRes + histogramIndexHorizontal] += (bilinearWeights.x) * (1.0f - bilinearWeights.y) * unitHighRes;
#else
				ndfVoxelDataHighRes.histograms[lookupHigRes + histogramIndexHorizontal] += float((bilinearWeights.x) * (1.0 - bilinearWeights.y) * unitHighRes);
#endif // SINGLE_PRECISION
			}
			if(histogramIndexVertical > 0 && histogramIndexVertical < HistogramHeight * HistogramWidth) {
#ifdef SINGLE_PRECISION
				ndfVoxelDataHighRes.histograms[lookupHigRes + histogramIndexVertical] += (1.0f - bilinearWeights.x) * (bilinearWeights.y) * unitHighRes;
#else
				ndfVoxelDataHighRes.histograms[lookupHigRes + histogramIndexVertical] += float((1.0 - bilinearWeights.x) * (bilinearWeights.y) * unitHighRes);
#endif // SINGLE_PRECISION
			}
			if(histogramIndexHorizontalVertical > 0 && histogramIndexHorizontalVertical < HistogramHeight * HistogramWidth) {
#ifdef SINGLE_PRECISION
				ndfVoxelDataHighRes.histograms[lookupHigRes + histogramIndexHorizontalVertical] += (bilinearWeights.x) * (bilinearWeights.y) * unitHighRes;
#else
				ndfVoxelDataHighRes.histograms[lookupHigRes + histogramIndexHorizontalVertical] += float((bilinearWeights.x) * (bilinearWeights.y) * unitHighRes);
#endif // SINGLE_PRECISION
			}
#endif
#endif
		}
	}

	// TODO: on the fly eBRDF splatting - even sparse approximation?

}