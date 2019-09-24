#include "NormalTransferFunction.h"

//#include "GlhSingle.h"
#include "glad/glad.h"

#include "FileIO.h"

// NOTE: on windows min and max are defined as macros
// undefine both to remove possible interference with stdlib
#undef max
#undef min
#include <algorithm>

using namespace NdfImposters;

// instatiate for static linking
template class NormalTransferFunction<float>;
//void NormalTransferFunction<float>::Splat<NdfImposters::LinearKernel<float>>(glm::vec2, glm::vec3, NdfImposters::LinearKernel<float>);

template<typename DataType>
NormalTransferFunction<DataType>::NormalTransferFunction(glm::ivec2 resolution) : functionResolution(resolution) {
	transferFunctionTextureData.resize(resolution.x * resolution.y, {0, 0, 0, 255});
}

template<typename DataType>
void NormalTransferFunction<DataType>::FromPicture(std::string pngFilePath) {
	auto pngData = Helpers::IO::ReadPngFile(pngFilePath);

	if((pngData.width_ * pngData.height_)!= transferFunctionTextureData.size()) {
		std::cout << "Error: NTF image does not fit size of NTF texture!" << std::endl;
		return;
	}

	std::copy(pngData.imageData.begin(), pngData.imageData.end(), transferFunctionTextureData.data());
}

template<typename DataType>
void NormalTransferFunction<DataType>::UpdateTexture() {
#if 0 // kernel test
	auto linearKernelLambda = [] (float x) { return std::max(0.0f, 1.0f - std::abs(x)); };
	auto quadraticKernelLambda = [] (float x) { 
		static const auto lowerBound = 0.0f;
		static const auto upperBound = 1.0f;
		return std::max(lowerBound, upperBound - (x * x));
	};

	auto testLinear = MultivariateKernel(linearKernelLambda, 0.25f);
	auto testQuadratic = MultivariateKernel(quadraticKernelLambda, 0.25f);
	auto testQuadratic2 = MultivariateKernel(QuadraticKernel<DataType>, 0.25f);

	auto testLinear2D = MultivariateKernel(linearKernelLambda, 0.25f, 0.75f, 0.6f, 0.2f);
	auto testLinear2D2 = MultivariateKernel(linearKernelLambda, 0.25f, 0.75f, 0.6f, 0.2f);
	
	auto testQuadratic4D = MultivariateKernel(QuadraticKernel<DataType>, 0.25f, 0.75f, 0.6f, 0.2f);
	auto testLinear4D = MultivariateKernel(LinearKernel<DataType>, 0.25f, 0.75f, 0.6f, 0.2f);
	auto testGaussian4D = MultivariateKernel(GaussianKernel<DataType>, 0.25f, 0.75f, 0.6f, 0.2f);

	auto testGaussian2D2 = MultivariateKernel(GaussianKernel<DataType>, 0.0f, 0.0f);
	auto testGaussian1D = MultivariateKernel(GaussianKernel<DataType>, 0.0f);

	auto debug = 0;
#endif // kernel tests

	if(!transferFunctionTexture.Texture) {
		Helpers::Gl::CreateTexture(transferFunctionTexture, functionResolution.x, functionResolution.y, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, transferFunctionTextureData.data());
	} else {
		glBindTexture(GL_TEXTURE_2D, transferFunctionTexture.Texture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, transferFunctionTexture.Width, transferFunctionTexture.Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, transferFunctionTextureData.data());
		glGenerateMipmap(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
}

template<typename DataType>
//template<typename KernelFunctor>
//void NormalTransferFunction<DataType>::Splat(glm::vec2 splatCoordinate, glm::vec3 color, KernelFunctor kernelFunctor) {
void NormalTransferFunction<DataType>::Splat(glm::vec2 splatCoordinate, glm::vec3 splatColor, bool left, bool clear) {
	if(!transferFunctionTexture.Texture) {
		Helpers::Gl::CreateTexture(transferFunctionTexture, functionResolution.x, functionResolution.y, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE, transferFunctionTextureData.data());
	}

	auto lightMapX = static_cast<int>(splatCoordinate.x * transferFunctionTexture.Width);
	auto lightMapY = static_cast<int>(splatCoordinate.y * transferFunctionTexture.Height);

	glBindTexture(GL_TEXTURE_2D, transferFunctionTexture.Texture);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, transferFunctionTextureData.data());
	
	std::array<float, 4> color = {1.0f, 0, 0, 1.0f};
	if(left) {
		color[0] = 0.0f;
		color[1] = 1.0f;
		color[2] = 0.0f;
		color[3] = 1.0f;
	}

	const auto intensity = 0.5f;
	//lightMapData.fill(color);
	const auto footprint = 5;
	for(auto y = std::max(0, lightMapY - footprint); y < std::min(transferFunctionTexture.Height, lightMapY + footprint); ++y) {
		for(auto x = std::max(0, lightMapX - footprint); x < std::min(transferFunctionTexture.Width, lightMapX + footprint); ++x) {
			auto index = y * transferFunctionTexture.Width + x;

			float distanceX = std::abs(static_cast<float>(x - lightMapX)) / static_cast<float>(footprint + 1);
			float distanceY = std::abs(static_cast<float>(y - lightMapY)) / static_cast<float>(footprint + 1);
			float weight = intensity * std::max(0.0f, 1.0f - distanceX) * std::max(0.0f, 1.0f - distanceY);

			std::array<float, 4> weightedColor =  { color[0] * weight, color[1] * weight, color[2] * weight, color[3] * weight};
			std::array<GLubyte, 4> colorUbyte = { static_cast<GLubyte>(weightedColor[0] * 255.0f), static_cast<GLubyte>(weightedColor[1] * 255.0f),
				static_cast<GLubyte>(weightedColor[2] * 255.0f), static_cast<GLubyte>(weightedColor[3] * 255.0f)};

			if(!clear) {
				transferFunctionTextureData[index] = {
					static_cast<GLubyte>(std::max(0, std::min(255, transferFunctionTextureData.data()[index][0] + colorUbyte[0]))),
					static_cast<GLubyte>(std::max(0, std::min(255, transferFunctionTextureData.data()[index][1] + colorUbyte[1]))),
					static_cast<GLubyte>(std::max(0, std::min(255, transferFunctionTextureData.data()[index][2] + colorUbyte[2]))),
					static_cast<GLubyte>(std::max(0, std::min(255, transferFunctionTextureData.data()[index][3] + colorUbyte[3]))) };
			} else {
				auto clearColor = static_cast<GLubyte>(weight * 255.0f);

				transferFunctionTextureData[index] = {
					static_cast<GLubyte>(std::max(0, std::min(255, transferFunctionTextureData.data()[index][0] - clearColor))),
					static_cast<GLubyte>(std::max(0, std::min(255, transferFunctionTextureData.data()[index][1] - clearColor))),
					static_cast<GLubyte>(std::max(0, std::min(255, transferFunctionTextureData.data()[index][2] - clearColor))),
					255 };
			}
		}
	}

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, transferFunctionTexture.Width, transferFunctionTexture.Height, 0, GL_RGBA, GL_UNSIGNED_BYTE, transferFunctionTextureData.data());
	glGenerateMipmap(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, 0);
}

template<typename DataType>
void NormalTransferFunction<DataType>::FetchColor(glm::vec2 splatCoordinate, glm::vec3& color)
{
	//fetch color at splat coordinate
	auto lightMapX = static_cast<int>(splatCoordinate.x * transferFunctionTexture.Width);
	auto lightMapY = static_cast<int>(splatCoordinate.y * transferFunctionTexture.Height);

	glBindTexture(GL_TEXTURE_2D, transferFunctionTexture.Texture);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, transferFunctionTextureData.data());


	auto index = lightMapY * transferFunctionTexture.Width + lightMapX;
	color = glm::vec3(transferFunctionTextureData.data()[index][0], transferFunctionTextureData.data()[index][1], transferFunctionTextureData.data()[index][2]);
	glBindTexture(GL_TEXTURE_2D, 0);
}