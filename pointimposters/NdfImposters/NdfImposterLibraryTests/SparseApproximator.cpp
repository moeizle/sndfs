#include "SparseApproximator.h"

using namespace SparseApproximation;

template class SparseApproximator<float>;

template <typename DataType>
ApproximationResult SparseApproximator<DataType>::Approximate(glm::ivec2 histogramResolution, glm::ivec2 spatialResolution, int maxClusters, std::vector<float> data) {
	SparseApproximation::ApproximationResult result(spatialResolution);
	
	#pragma omp parallel for
	for(int histogramIndex = 0; histogramIndex < spatialResolution.x * spatialResolution.y; ++histogramIndex) {
		auto dataOffset = size_t(histogramIndex * histogramResolution.x * histogramResolution.y);
		auto ssboBegin = data.begin() + dataOffset;
		auto ssboEnd = ssboBegin + histogramResolution.x * histogramResolution.y;

		Helpers::Approximation::ExpectationMaximization<3> approximation(ssboBegin, ssboEnd, { histogramResolution.x, histogramResolution.y, 1 }, maxClusters);

		#pragma omp critical
		{
			std::move(approximation.Gaussians.begin(), approximation.Gaussians.end(), std::back_inserter(result.gaussians[histogramIndex]));

			// Adds the lambertian as a special case of a gaussian.
			// The shader has to check whether the gaussian is a proper gaussian or a lambertian
			// This reduces the memory consumption since only lambertian > 0 are stored.
			if(approximation.Lambertian > 0.0f) {
				Helpers::Approximation::GaussianBase<float, 3> lambertianGaussian;
				lambertianGaussian.variances[0] = 0.0f;
				lambertianGaussian.variances[1] = 0.0f;
				lambertianGaussian.variances[2] = 0.0f;

				lambertianGaussian.means[0] = 0.0f;
				lambertianGaussian.means[1] = 0.0f;
				lambertianGaussian.means[2] = approximation.Lambertian;

				result.gaussians[histogramIndex].emplace_back(lambertianGaussian);
			}
		}
	}

	return result;
}

template <typename DataType>
GpuGmm SparseApproximator<DataType>::CreateGpuGmm(ApproximationResult approximation) {
	GpuGmm result;

	// count gaussians
	result.quantizationVarianceScale = 0.0f;
	auto gaussianCount = std::accumulate(approximation.gaussians.begin(), approximation.gaussians.end(), 0,
		[&] (size_t current, decltype(*approximation.gaussians.begin()) &pixelGaussians) {
			for(const auto &gaussian : pixelGaussians) {
				result.quantizationVarianceScale = std::max(gaussian.variances[0], result.quantizationVarianceScale);
			}
			return current + pixelGaussians.size();
		});

	result.gaussians.reserve(gaussianCount);
	result.offsets.reserve(approximation.gaussians.size());
	uint32_t currentOffset = 0;
	for(auto pixelGmm : approximation.gaussians) {
		result.offsets.emplace_back(currentOffset);
		currentOffset += pixelGmm.size();

		//std::move(pixelGmm.begin(), pixelGmm.end(), std::back_inserter(result.gaussians));

		for(auto &gaussian : pixelGmm) {
			result.gaussians.emplace_back(gaussian, result.quantizationVarianceScale);
		}
	}

	return result;
}