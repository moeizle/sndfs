#ifndef SPARSE_APPROXIMATOR_H_
#define SPARSE_APPROXIMATOR_H_

#include "ExpectationMaximization.h"

// undefine both min and max to remove possible interference of win with stdlib
#undef max
#undef min
#include <algorithm>

namespace SparseApproximation {

class ApproximationResult {
public:
	ApproximationResult() : dimension(0, 0) {}
	ApproximationResult(glm::ivec2 spatialDimension) : dimension(spatialDimension) {
		gaussians.resize(spatialDimension.x * spatialDimension.y);
	}
	glm::ivec2 dimension;
	std::vector<std::vector<Helpers::Approximation::GaussianBase<float, 3>>> gaussians;
};

class QuantizedGaussian {
public:
	QuantizedGaussian() : data(0) {}
	QuantizedGaussian(Helpers::Approximation::GaussianBase<float, 3> gaussian, float quantizationVarianceScale) {
		const auto quantizationScale6 = 63.0f;
		const auto quantizationScale8 = 255.0f;

		assert(gaussian.means[0] >= 0.0f && gaussian.means[0] <= 1.0f);
		assert(gaussian.means[1] >= 0.0f && gaussian.means[1] <= 1.0f);
		assert(gaussian.means[2] >= 0.0f && gaussian.means[2] <= 1.0f);

		assert(gaussian.variances[0] >= 0.0f && gaussian.variances[0] <= 1.0f);
		assert(gaussian.variances[1] >= 0.0f && gaussian.variances[1] <= 1.0f);
		assert(gaussian.variances[2] >= 0.0f && gaussian.variances[2] <= 1.0f);
			
		auto meanX = std::max(static_cast<unsigned char>(0), std::min(static_cast<unsigned char>(63), static_cast<unsigned char>(gaussian.means[0] * quantizationScale6)));
		auto meanY = std::max(static_cast<unsigned char>(0), std::min(static_cast<unsigned char>(63), static_cast<unsigned char>(gaussian.means[1] * quantizationScale6)));
		auto meanZ = std::max(static_cast<unsigned char>(0), std::min(static_cast<unsigned char>(255), static_cast<unsigned char>(gaussian.means[2] * quantizationScale8)));

		const float varianceScale = 1.0f / quantizationVarianceScale;
		auto varianceX = std::max(static_cast<unsigned char>(0), std::min(static_cast<unsigned char>(63), static_cast<unsigned char>(varianceScale * gaussian.variances[0] * quantizationScale6)));
		auto varianceY = std::max(static_cast<unsigned char>(0), std::min(static_cast<unsigned char>(63), static_cast<unsigned char>(varianceScale * gaussian.variances[1] * quantizationScale6)));

		data = 0;
		data |= uint32_t(meanX);
		data |= uint32_t(meanY) << 6;
		data |= uint32_t(meanZ) << 12;
		data |= uint32_t(varianceX) << 20;
		data |= uint32_t(varianceY) << 26;
	}

	uint32_t data;
};

class GpuGmm {
public:
	std::vector<uint32_t> offsets;
	std::vector<QuantizedGaussian> gaussians;
	float quantizationVarianceScale;
};

template <typename BinType = float>
class SparseApproximator {
public:
	ApproximationResult Approximate(glm::ivec2 histogramResolution, glm::ivec2 spatialResolution, int maxClusters, std::vector<float> data);
	GpuGmm CreateGpuGmm(ApproximationResult approximation);

private:
};

} // namespace SparseApproximation

#endif // SPARSE_APPROXIMATOR_H_