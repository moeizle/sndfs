#include "NdfSampler.h"
#include <cmath>

using namespace NdfImposters;

// instatiate for static linking
template class NdfSampler<float>;

template <typename BinType>
void NdfSampler<BinType>::histogramBinning(NdfTreeLevel<BinType> &ndfTreeLevel, glm::ivec2 spatial, glm::ivec2 view, int samplingRate, const std::vector<glm::vec2> &normalSamples, float depth) {
	assert(spatial.x < ndfTreeLevel.GetSpatialResolution().x);
	assert(spatial.x >= 0);
	assert(spatial.y < ndfTreeLevel.GetSpatialResolution().y);
	assert(spatial.y >= 0);
	
	assert(view.x < ndfTreeLevel.GetViewDirectionResolution().x);
	assert(view.x >= 0);
	assert(view.y < ndfTreeLevel.GetViewDirectionResolution().y);
	assert(view.y >= 0);

	assert(ndfTreeLevel.GetHistogramResolution().x > 0 && ndfTreeLevel.GetHistogramResolution().y > 0);

	assert((samplingRate*samplingRate) > 0);

	const auto sampleCount = normalSamples.size();
	if(sampleCount <= 0) {
		return;
	}

	assert(sampleCount <= (samplingRate*samplingRate));
	//const auto unit = static_cast<float>(sampleCount) / static_cast<float>(samplingRate * samplingRate * samplingRate * samplingRate);
	const auto unit = 1.0f / static_cast<float>(samplingRate * samplingRate);
	
	auto histogramRef = ndfTreeLevel.GetHistogram(spatial, view);

#ifdef _DEBUG
	// clear histogram
	for(auto it = histogramRef.NdfDataPointer; 
		it != histogramRef.NdfDataPointer + histogramRef.HistogramResolution.x * histogramRef.HistogramResolution.y;
		++it) {
		*it = 0.0f;
	}
#endif

	const glm::vec2 histogramScale = {
		static_cast<float>((histogramRef.HistogramResolution.x - 1)),
		static_cast<float>((histogramRef.HistogramResolution.y - 1))
	};

	// NOTE: no range checks in release mode
	for(auto &&sample : normalSamples) {
		assert(sample.x <= 1.0f);
		assert(sample.x >= 0.0f);
		assert(sample.y <= 1.0f);
		assert(sample.y >= 0.0f);

		glm::vec2 sampleCoordinate = {
			static_cast<float>(sample.x) * histogramScale.x,
			static_cast<float>(sample.y) * histogramScale.y
		};

		glm::ivec2 quantizedSample = {
			static_cast<int>(sampleCoordinate.x),
			static_cast<int>(sampleCoordinate.y)
		};

		// filter out invalid samples
		// TODO: filter all invalid ones, not just the clear color
		// 0, 0 is not necessarilly invalid - consider 1x1 histogram
		if(quantizedSample.x > 0 || quantizedSample.y > 0) {

			// TODO: modular binning functions
#if 0
			// nearest neighbour
			auto binIndex = quantizedSample.y * histogramRef.HistogramResolution.x + quantizedSample.x;

			histogramRef.NdfDataPointer[binIndex] += unit;
#endif

#if 0
			// nearest neighbour
			auto binIndex = quantizedSample.y * histogramRef.HistogramResolution.x + quantizedSample.x;

			// nearest neighbour
			glm::vec2 weight = { std::fmodf(sampleCoordinate.x, 1.0f), std::fmodf(sampleCoordinate.y, 1.0f) };
			if(weight.x >= 0.5f) {
				binIndex += 1;
			}
			if(weight.y >= 0.5f) {
				binIndex += histogramRef.HistogramResolution.x;
			}

			histogramRef.NdfDataPointer[binIndex] += unit;
#endif

#if 1
			// bilinear
			auto binIndex = quantizedSample.y * histogramRef.HistogramResolution.x + quantizedSample.x;
			auto binIndexHor = quantizedSample.y * histogramRef.HistogramResolution.x + (quantizedSample.x + 1);
			auto binIndexVer = (quantizedSample.y + 1) * histogramRef.HistogramResolution.x + quantizedSample.x;
			auto binIndexHorVer = (quantizedSample.y + 1) * histogramRef.HistogramResolution.x + (quantizedSample.x + 1);

			glm::vec2 weight = { std::fmodf(sampleCoordinate.x, 1.0f), std::fmodf(sampleCoordinate.y, 1.0f) };

			histogramRef.NdfDataPointer[binIndex] += unit * (1.0f - weight.x) * (1.0f - weight.y);
			histogramRef.NdfDataPointer[binIndexHor] += unit * (weight.x) * (1.0f - weight.y);
			histogramRef.NdfDataPointer[binIndexVer] += unit * (1.0f - weight.x) * (weight.y);
			histogramRef.NdfDataPointer[binIndexHorVer] += unit * (weight.x) * (weight.y);
#endif

			// TODO: gaussian
		}
	}
	
	// NOTE: this has to be considered during normalization
	histogramRef.NdfDataPointer[(histogramRef.HistogramResolution.x * histogramRef.HistogramResolution.y) - 1] = depth;
}