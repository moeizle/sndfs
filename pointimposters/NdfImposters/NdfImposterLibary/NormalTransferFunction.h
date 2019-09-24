#ifndef NORMAL_TRANSFER_FUNCTION_H_
#define NORMAL_TRANSFER_FUNCTION_H_

#include <vector>
#include <type_traits>

#include <GlHelpers.h>

#include <glm/vec3.hpp>
#include <glm/vec2.hpp>

// undefine both min and max to remove possible interference of win with stdlib
#undef max
#undef min
#include <algorithm>

#define _USE_MATH_DEFINES
#include <math.h>

namespace NdfImposters {

template<typename DataType = float>
auto LinearKernel(DataType x) -> DataType {
	static const auto lowerBound = DataType(0.0);
	static const auto upperBound = DataType(1.0);

	return std::max(lowerBound, upperBound - std::abs(x));
};

template<typename DataType = float>
auto QuadraticKernel(DataType x) -> DataType { 
	static const auto lowerBound = DataType(0.0);
	static const auto upperBound = DataType(1.0);

	return std::max(lowerBound, upperBound - (x * x));
};

template<typename DataType = float>
inline auto GaussianKernel(DataType x) -> DataType {
	static const DataType mean = 0.0;
	static const DataType variance = 0.4;
	// FIXME: should integrate to 1, not be 1 at f(0)
	static const DataType scale = 1.0f;//std::sqrt(variance) * std::sqrt(2.0f * M_PI); // = 1.0 / f(0)

	DataType diff_squared = (x - mean) * (x - mean);
	DataType exponent = -(diff_squared / (2.0f * variance));
	DataType numerator = std::exp(exponent);
	DataType denominator = std::sqrt(variance) * std::sqrt(2.0f * M_PI);

	return scale * (numerator / denominator);
};

// pattern matching: more than one coordinate -> evaluate first element from set and continue recursively
template<typename KernelFunctorType, typename FirstCoordinateType, typename ...CoordinatesType>
auto MultivariateKernel(KernelFunctorType functor, FirstCoordinateType firstCoordinate, CoordinatesType ...coordinates) 
-> std::result_of_t<KernelFunctorType(FirstCoordinateType)> {
	return functor(firstCoordinate) * MultivariateKernel(std::forward<KernelFunctorType>(functor), coordinates...);
}

// pattern matching: only one left -> use 1D kernel
template<typename KernelFunctorType, typename CoordinateType>
auto MultivariateKernel(KernelFunctorType functor, CoordinateType coordinate)
-> std::result_of_t<KernelFunctorType(CoordinateType)> {
	return functor(coordinate);
}

template<typename DataType = float>
class NormalTransferFunction {
public:
	NormalTransferFunction(glm::ivec2 resolution);

	// Updates the transfer function texture from the current data.
	void UpdateTexture();

	void FromPicture(std::string pngFilePath);

	// splats a given kernel into the texture data.
	//template<typename KernelFunctor>
	//void Splat(glm::vec2 splatCoordinate, glm::vec3 splatColor, KernelFunctor kernelFunctor);
	void Splat(glm::vec2 splatCoordinate, glm::vec3 splatColor, bool left, bool clear);
	void FetchColor(glm::vec2 splatCoordinate, glm::vec3& color);

	Helpers::Gl::TextureHandle2D transferFunctionTexture;

private:
	std::vector<std::array<GLubyte, 4>> transferFunctionTextureData;
	const glm::ivec2 functionResolution;
};

} // namespace NdfImposters

#endif // NORMAL_TRANSFER_FUNCTION_H_