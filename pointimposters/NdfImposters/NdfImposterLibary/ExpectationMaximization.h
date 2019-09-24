#ifndef EXPECTATION_MAXIMIZATION_H_
#define EXPECTATION_MAXIMIZATION_H_

#if 1
#include "NdfTree.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>
#include <numeric>
#include <random>
#include <array>

#include <glm\vec2.hpp>

//#define USE_AMP_
#ifdef USE_AMP_
#include <amp.h>
#endif // USE_AMP_

namespace Helpers {
namespace Approximation {
/*
Axis algined multi variate gaussians - No covariances.
*/
template <typename DataType, int Dimensions> class GaussianBase {
public:
	GaussianBase() {
		std::fill(means.begin(), means.end(), 0.5f);
		std::fill(variances.begin(), variances.end(), 0.2f);
	}

	GaussianBase(std::array<DataType, Dimensions> means, std::array<DataType, Dimensions> variances) {
		this->means = means;
		this->variances = variances;
	}
	
	// FIXME: add covariances for arbitrarily aligned gaussians
	std::array<DataType, Dimensions> means;
	std::array<DataType, Dimensions> variances;

	template<int DimensionsUsed>
	DataType GetLikelihood(std::array<DataType, DimensionsUsed> values) {
		// TODO: enable if or static assert - not fully supported by VS 2013
		//static_assert(DimensionsUsed <= Dimensions);

		DataType out = 1.0f;
		for(auto i = 0; i < DimensionsUsed; ++i) {
			out *= gaussKernel(values[i], variances[i], means[i]);
		}

		return out;
	}
	
private:
	inline DataType gaussKernel(DataType x) {
		DataType numerator = std::expf(-(x*x) / 0.4f);
		static const DataType denominator = std::sqrtf(0.2f) / std::sqrtf(2.0f * M_PI);

		return numerator / denominator;
	}

	inline DataType gaussKernel(DataType x, DataType variance, DataType mean) {
		DataType diff_squared = (x - mean) * (x - mean);
		DataType exponent = -(diff_squared / (2.0f * variance));
		DataType numerator = std::expf(exponent);
		DataType denominator = std::sqrtf(variance) * std::sqrtf(2.0f * M_PI);

		return numerator / denominator;
	}
};

using DataType = float;

/*
Sparse approximation of an eBRDF using GMM/EM.

Assumes 1D range domain.
*/
template<int DimensionsUsed>
class ExpectationMaximization {
public:
	ExpectationMaximization::ExpectationMaximization(std::vector<float>::const_iterator dataBegin, std::vector<float>::const_iterator dataEnd, std::array<int, DimensionsUsed> dimensionSizes, const int initialClusterCount, const int iterations = 5) {
		sourceData.clear();
		sourceData.reserve(std::distance(dataBegin, dataEnd));
		std::copy(dataBegin, dataEnd, std::back_inserter(sourceData));

		//assert(sourceData.size() == std::accumulate(dimensionSizes.begin(), dimensionSizes.end(), 0));

		// find min value that represents the lambertia
		Lambertian = *std::min_element(sourceData.begin(), sourceData.end());
		// subtract lambertian
		std::transform(sourceData.begin(), sourceData.end(), sourceData.begin(),
			[=] (decltype(*sourceData.begin()) histogramBin) {
				return histogramBin - Lambertian;
		});

		// generate random uniformly distributed intial gaussians
		Gaussians.reserve(initialClusterCount);
		{
			static const auto initialVariance = 0.2f;
			std::random_device randomSeed;
			std::default_random_engine generator(randomSeed());
			std::uniform_real_distribution<DataType> distribution(0.0f, 1.0f);
			std::generate_n(std::back_inserter(Gaussians), initialClusterCount, [&] () {
				std::array<DataType, DimensionsUsed> means;
				std::generate_n(means.begin(), DimensionsUsed - 1, [&] () {
					return distribution(generator);
				});
				// NOTE: assumes 1D range domain which should always be greater than zero
				means[DimensionsUsed - 1] = distribution(generator) * 0.5f + 0.5f;

				std::array<DataType, DimensionsUsed> variances;
				std::generate_n(variances.begin(), DimensionsUsed, [] () {
					return initialVariance;
				});

				return GaussianBase<DataType, DimensionsUsed>(means, variances);
			});
		}

		likelihood = std::vector<std::vector<std::vector<LikelihoodType>>>(dimensionSizes[1], std::vector<std::vector<LikelihoodType>>(dimensionSizes[0], std::vector<LikelihoodType>(Gaussians.size(), 0.0f)));

#ifdef USE_AMP_
		// TODO: NYI: not actually used yet
		std::vector<LikelihoodType> likelihoodContiguously(std::accumulate(dimensionSizes.begin(), dimensionSizes.end(), 0), 0.0f);
		concurrency::array_view<LikelihoodType, DimensionsUsed> likelihoodView(likelihoodContiguously);
#endif // USE_AMP_

		expectation = std::vector<LikelihoodType>(Gaussians.size(), 0.0f);

		//static const auto emIterations = 50;
		const auto emIterations = iterations;
		const auto rejectionThreshold = 0.00005f;
		// TODO: recursive
		for(auto i = 0; i < emIterations; ++i) {
			this->EmStep(dimensionSizes);
			
			// reject too gaussians with small range value
			// NOTE: assumes 1D range domain
			Gaussians.erase(std::remove_if(Gaussians.begin(), Gaussians.end(),
				[=] (decltype(*Gaussians.begin())& gaussian) {
					return gaussian.means[DimensionsUsed-1] <= rejectionThreshold;
				}), Gaussians.end());

			// TODO: stop if close enough


			if(Gaussians.size() <= 0) {
				break;
			}
		}

#if 0
		// splat gaussians and lambertian
		const std::array<DataType, 2> scale = {
			static_cast<DataType>(1.0f) / static_cast<DataType>(xSize),
			static_cast<DataType>(1.0f) / static_cast<DataType>(ySize) };

		auto &histogramRef = ndfRef.HistogramData2D;
		for(auto yData = 0; yData < histogramRef.size(); ++yData) {
			for(auto xData = 0; xData < histogramRef[0].size(); ++xData) {
				auto gaussianI = 0;
				histogramRef[yData][xData] = 0.0f;
				auto totalLikelihood = 0.0f;
				for(auto gaussian : Gaussians) {
					std::array<DataType, 2> coordinate = { xData * scale[0], yData * scale[1] };	

					auto likelihood = gaussian.GetLikelihood(coordinate);

					// TODO: splatted gaussians seem to have hard edges - might be an error during splatting
					histogramRef[yData][xData] += likelihood * gaussian.means[2];
					totalLikelihood += likelihood;

					++gaussianI;
				}

				if(totalLikelihood > 0.0f) {
					histogramRef[yData][xData] /= totalLikelihood;
				}
				histogramRef[yData][xData] += Lambertian;
			}
		}
#endif
	}

	void EmStep(std::array<int, DimensionsUsed> dimensionSizes) {
		// expectation - calculate expectation of data points to gaussians and likelihood using baysian posterior
		const std::array<DataType, DimensionsUsed - 1> scale = { 
			static_cast<DataType>(1.0f) / static_cast<DataType>(dimensionSizes[1]),
			static_cast<DataType>(1.0f) / static_cast<DataType>(dimensionSizes[0])};

		/*template<class T> class flatten { public: using type = T; };
		template<typename T> class flatten<T[]> { public: using type = T; };
		template<typename T, std::size_t N> class flatten<T[N]> { public: using type = T; };

		template<typename T[], size_t N, size_t M>
		auto flattenAuto(T (&a)[M][N]) -> T (&)[M*N] { return flattenAuto<T, N, M>(a, N, M); }

		template<typename T, size_t N, size_t M>
		auto flattenAuto(T (&a)[M][N]) -> T (&)[M*N] { return reinterpret_cast<T (&)[M*N]>(a); }

		int[10] test;
		auto test2 = flatten(test);
		auto test3 = flattenAuto(test);

		int[10][20][30] test4;
		auto test5 = flattenAuto(test4);*/

		// NYI: for generic amount of dimensions
		assert(DimensionsUsed == 3);
		for(auto yData = 0; yData < dimensionSizes[1]; ++yData) {
			for(auto xData = 0; xData < dimensionSizes[0]; ++xData) {
				auto expectationSum = 0.0f;
				auto gaussianI = 0;
				// TODO: profile if by value or by ref is faster
				for(auto gaussian : Gaussians) {
					std::array<DataType, DimensionsUsed> coordinate = { xData * scale[0], yData * scale[1], sourceData[yData * dimensionSizes[0] + xData] };	

					auto fitness = gaussian.GetLikelihood(coordinate);

					expectation[gaussianI] = fitness;
					expectationSum += fitness;

					++gaussianI;
				}

				std::transform(expectation.begin(), expectation.end(), likelihood[yData][xData].begin(),
					[expectationSum] (decltype(*expectation.begin()) value) {
						return value / expectationSum;
					});
			}
		}	
	
		// maximization - move the gaussians to maximize expected values using current knowledge			
		auto gaussianI = 0;
		for(auto &gaussian : Gaussians) {
			// update means and variances
			std::array<LikelihoodType, 3> weightedMean = { 0.0f, 0.0f, 0.0f };
			std::array<LikelihoodType, 3> weightedVariance = { 0.0f, 0.0f, 0.0f };
			auto totalWeight = 0.0f;
			for(auto yData = 0; yData < dimensionSizes[1]; ++yData) {
				for(auto xData = 0; xData < dimensionSizes[0]; ++xData) {
					std::array<DataType, 3> coordinate = { xData * scale[0], yData * scale[1], sourceData[yData * dimensionSizes[0] + xData] };

					auto weight = likelihood[yData][xData][gaussianI];
					totalWeight += weight;

					weightedMean[0] += weight * coordinate[0];
					weightedMean[1] += weight * coordinate[1];
					weightedMean[2] += weight * coordinate[2];

					weightedVariance[0] += weight * ((coordinate[0] - gaussian.means[0]) * (coordinate[0] - gaussian.means[0]));
					weightedVariance[1] += weight * ((coordinate[1] - gaussian.means[1]) * (coordinate[1] - gaussian.means[1]));
					weightedVariance[2] += weight * ((coordinate[2] - gaussian.means[2]) * (coordinate[2] - gaussian.means[2]));
				}
			}

			gaussian.variances[0] = weightedVariance[0] / totalWeight;
			gaussian.variances[1] = weightedVariance[1] / totalWeight;
			gaussian.variances[2] = weightedVariance[2] / totalWeight;

			gaussian.means[0] = weightedMean[0] / totalWeight;
			gaussian.means[1] = weightedMean[1] / totalWeight;
			gaussian.means[2] = weightedMean[2] / totalWeight;
			
			// TODO: calculate total variation to terminate earlier


			++gaussianI;
		}
	}

	std::vector<GaussianBase<DataType, DimensionsUsed>> Gaussians;
	float Lambertian;

private:
	/*union {
		decltype(Ndf::NdfSlice<Ndf::histogramResolution, Ndf::histogramResolution>::HistogramData2D) histogramCopy;
		decltype(Ndf::NdfSlice<Ndf::histogramResolution, Ndf::histogramResolution>::HistogramData1D) histogramCopy1D;
	};*/

	/*class {
	public:
		std::vector<float>::iterator dataBegin;
		std::vector<float>::iterator dataEnd;

		auto begin() -> decltype(dataBegin) {
			return sourceDataBegin;
		}

		auto end() -> decltype(dataEnd) {
			return dataEnd;
		}
	} sourceData;*/
	std::vector<float> sourceData;

	using LikelihoodType = float;
	std::vector<LikelihoodType> expectation;
	std::vector<std::vector<std::vector<LikelihoodType>>> likelihood;
};
} // namespace Approximation
} // namespace Helpers
#endif

#if 0
#include "NdfTree.h"

#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>
#include <numeric>
#include <random>

#include <glm\vec2.hpp>

namespace Helpers {
namespace Approximation {
/*
Axis algined multi variate gaussians - No covariances.
*/
template <typename DataType, int Dimensions> class GaussianBase {
public:
	GaussianBase() {
		std::fill(means.begin(), means.end(), 0.5f);
		std::fill(variances.begin(), variances.end(), 0.2f);
	}

	GaussianBase(std::array<DataType, Dimensions> means, std::array<DataType, Dimensions> variances) {
		this->means = means;
		this->variances = variances;
	}
	
	// FIXME: add covariances for arbitrarily aligned gaussians
	std::array<DataType, Dimensions> means;
	std::array<DataType, Dimensions> variances;

	template<int DimensionsUsed>
	DataType GetLikelihood(std::array<DataType, DimensionsUsed> values) {
		// TODO: enable if or static assert - not fully supported by VS 2013
		//static_assert(DimensionsUsed <= Dimensions);

		DataType out = 1.0f;
		for(auto i = 0; i < DimensionsUsed; ++i) {
			out *= gaussKernel(values[i], variances[i], means[i]);
		}

		return out;
	}
	
private:
	inline DataType gaussKernel(DataType x) {
		DataType numerator = std::expf(-(x*x) / 0.4f);
		static const DataType denominator = std::sqrtf(0.2f) / std::sqrtf(2.0f * M_PI);

		return numerator / denominator;
	}

	inline DataType gaussKernel(DataType x, DataType variance, DataType mean) {
		DataType diff_squared = (x - mean) * (x - mean);
		DataType exponent = -(diff_squared / (2.0f * variance));
		DataType numerator = std::expf(exponent);
		DataType denominator = std::sqrtf(variance) * std::sqrtf(2.0f * M_PI);

		return numerator / denominator;
	}
};

using DataType = float;
using TrivariateGaussian = GaussianBase<DataType, 3>;

/*
Sparse approximation of an eBRDF using GMM/EM.
*/
class ExpectationMaximization {
public:
	ExpectationMaximization::ExpectationMaximization(NdfImposters::NormalDistributionFunction& viewDependentNdf, const int initialClusterCount, glm::ivec2 view) {
		auto &ndfRef = viewDependentNdf.ViewDependentHistograms[view.y][view.x];
		histogramCopy1D = ndfRef.HistogramData1D;
		
		// find min value that represents the lambertia
		Lambertian = *std::min_element(histogramCopy1D.cbegin(), histogramCopy1D.cend());
		for(auto &value : histogramCopy1D) {
			value -= Lambertian;
		}

		// generate random intial gaussians
		Gaussians.reserve(initialClusterCount);
		static const auto initialVariance = 0.2f;
		std::random_device randomSeed;
		std::default_random_engine generator(randomSeed());
		std::uniform_real_distribution<DataType> distribution(0.0f, 1.0f);
		std::generate_n(std::back_inserter(Gaussians), initialClusterCount, [&] () {
			return TrivariateGaussian(
				{{ distribution(generator), distribution(generator), distribution(generator) * 0.5f + 0.5f }},
				{{ initialVariance, initialVariance, initialVariance }});
		});

		const auto xSize = histogramCopy.size();
		const auto ySize = histogramCopy[0].size();
		likelihood = std::vector<std::vector<std::vector<LikelihoodType>>>(ySize, std::vector<std::vector<LikelihoodType>>(xSize, std::vector<LikelihoodType>(Gaussians.size(), 0.0f)));
		expectation = std::vector<LikelihoodType>(Gaussians.size(), 0.0f);

		//static const auto emIterations = 40;
		static const auto emIterations = 50;
		static const auto rejectionThreshold = 0.00005f;
		for(auto i = 0; i < emIterations; ++i) {
			this->EmStep();
			
			// reject too small gaussians
			Gaussians.erase(std::remove_if(Gaussians.begin(), Gaussians.end(),
				[] (decltype(*Gaussians.begin())& gaussian) {
					return gaussian.means[2] <= rejectionThreshold;
				}), Gaussians.end());

			// TODO: stop if close enough

			if(Gaussians.size() <= 0) {
				break;
			}
		}

#if 0
		// splat gaussians and lambertian
		const std::array<DataType, 2> scale = {
			static_cast<DataType>(1.0f) / static_cast<DataType>(xSize),
			static_cast<DataType>(1.0f) / static_cast<DataType>(ySize) };

		auto &histogramRef = ndfRef.HistogramData2D;
		for(auto yData = 0; yData < histogramRef.size(); ++yData) {
			for(auto xData = 0; xData < histogramRef[0].size(); ++xData) {
				auto gaussianI = 0;
				histogramRef[yData][xData] = 0.0f;
				auto totalLikelihood = 0.0f;
				for(auto gaussian : Gaussians) {
					std::array<DataType, 2> coordinate = { xData * scale[0], yData * scale[1] };	

					auto likelihood = gaussian.GetLikelihood(coordinate);

					// TODO: splatted gaussians seem to have hard edges - might be an error during splatting
					histogramRef[yData][xData] += likelihood * gaussian.means[2];
					totalLikelihood += likelihood;

					++gaussianI;
				}

				if(totalLikelihood > 0.0f) {
					histogramRef[yData][xData] /= totalLikelihood;
				}
				histogramRef[yData][xData] += Lambertian;
			}
		}
#endif
	}
	
	void EmStep() {
		const auto xSize = histogramCopy.size();
		const auto ySize = histogramCopy[0].size();
		
		// expectation - calculate expectation of data points to gaussians and likelihood using baysian posterior
		const std::array<DataType, 2> scale = { static_cast<DataType>(1.0f) / static_cast<DataType>(ySize), static_cast<DataType>(1.0f) / static_cast<DataType>(xSize) };
		for(auto yData = 0; yData < histogramCopy.size(); ++yData) {
			for(auto xData = 0; xData < histogramCopy[0].size(); ++xData) {
				auto expectationSum = 0.0f;
				auto gaussianI = 0;
				// TODO: profile if by value or by ref is faster
				for(auto gaussian : Gaussians) {	
					std::array<DataType, 3> coordinate = { xData * scale[0], yData * scale[1], histogramCopy[yData][xData] };	

					auto fitness = gaussian.GetLikelihood(coordinate);

					expectation[gaussianI] = fitness;
					expectationSum += fitness;

					++gaussianI;
				}

				std::transform(expectation.begin(), expectation.end(), likelihood[yData][xData].begin(),
					[expectationSum] (decltype(*expectation.begin()) value) {
						return value / expectationSum;
					});
			}
		}	
	
		// maximization - move the gaussians to maximize expected values using current knowledge			
		auto gaussianI = 0;
		for(auto &gaussian : Gaussians) {
			// update means and variances
			std::array<LikelihoodType, 3> weightedMean = { 0.0f, 0.0f, 0.0f };
			std::array<LikelihoodType, 3> weightedVariance = { 0.0f, 0.0f, 0.0f };
			auto totalWeight = 0.0f;
			for(auto yData = 0; yData < histogramCopy.size(); ++yData) {
				for(auto xData = 0; xData < histogramCopy[0].size(); ++xData) {
					std::array<DataType, 3> coordinate = { xData * scale[0], yData * scale[1], histogramCopy[yData][xData] };

					auto weight = likelihood[yData][xData][gaussianI];
					totalWeight += weight;

					weightedMean[0] += weight * coordinate[0];
					weightedMean[1] += weight * coordinate[1];
					weightedMean[2] += weight * coordinate[2];

					weightedVariance[0] += weight * ((coordinate[0] - gaussian.means[0]) * (coordinate[0] - gaussian.means[0]));
					weightedVariance[1] += weight * ((coordinate[1] - gaussian.means[1]) * (coordinate[1] - gaussian.means[1]));
					weightedVariance[2] += weight * ((coordinate[2] - gaussian.means[2]) * (coordinate[2] - gaussian.means[2]));
				}
			}

			gaussian.variances[0] = weightedVariance[0] / totalWeight;
			gaussian.variances[1] = weightedVariance[1] / totalWeight;
			gaussian.variances[2] = weightedVariance[2] / totalWeight;

			gaussian.means[0] = weightedMean[0] / totalWeight;
			gaussian.means[1] = weightedMean[1] / totalWeight;
			gaussian.means[2] = weightedMean[2] / totalWeight;
			
			// TODO: calculate total variation to terminate earlier


			++gaussianI;
		}
	}

	std::vector<TrivariateGaussian> Gaussians;
	float Lambertian;

private:
	union {
		decltype(Ndf::NdfSlice<Ndf::histogramResolution, Ndf::histogramResolution>::HistogramData2D) histogramCopy;
		decltype(Ndf::NdfSlice<Ndf::histogramResolution, Ndf::histogramResolution>::HistogramData1D) histogramCopy1D;
	};

	using LikelihoodType = float;
	std::vector<LikelihoodType> expectation;
	std::vector<std::vector<std::vector<LikelihoodType>>> likelihood;
};
} // namespace Approximation
} // namespace Helpers
#endif

#endif // EXPECTATION_MAXIMIZATION_H_