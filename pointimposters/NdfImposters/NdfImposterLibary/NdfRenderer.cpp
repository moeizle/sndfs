#include "NdfRenderer.h"
#include "NdfTree.h"

//#include "GlhSingle.h"
#include "glad/glad.h"
#include <array>
#include <string>
#include <iostream>

using namespace NdfImposters;

template class NdfRenderer<float>;

template <typename BinType>
void NdfRenderer<BinType>::Render(NdfTree<BinType> &ndfTree) {
#if 0
	assert(glBindBufferBase);
	assert(glUseProgram);
	assert(glBindFramebuffer);

	GLuint ssboBindingPointIndex = 0;
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, ndfTree->hierarchy[Ndf::MipLevelIndex]->ssbo);

#ifdef FROM_FILE
	ssboBindingPointIndex = 1;
	const int ssaoMipLevelIndex = Ndf::MipLevelIndex + Ndf::SsaoDownsampleCount;
	assert(ssaoMipLevelIndex < Ndf::MipLevelCount);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, ndfTree->hierarchy[ssaoMipLevelIndex]->ssbo);
#endif
	
	ssboBindingPointIndex = 2;
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, ndfTree->hierarchy[Ndf::MipLevelIndex]->ssboOffsetMap);
	
	ssboBindingPointIndex = 3;
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, ndfTree->hierarchy[Ndf::MipLevelIndex]->ssboGaussians);

	auto &tree = ndfTree;
	auto &l0 = tree->hierarchy[0];

	auto programHandle = ndf4DPreviewSingleShaderProgram->GetGlShaderProgramHandle();
	glUseProgram(programHandle);

	auto viewportScale = glm::vec2(windowWidth / l0->Width, windowHeight / l0->Height);

	glUniform1i(glGetUniformLocation(programHandle, "endRaySampler"), 0);
	glUniform1i(glGetUniformLocation(programHandle, "lightMapSampler"), 1);

	glUniform1i(glGetUniformLocation(programHandle, "renderMode"), RenderMode);
	glUniform1i(glGetUniformLocation(programHandle, "ssaoEnabled"), SsaoEnabled);
	glUniform1i(glGetUniformLocation(programHandle, "ssaoDownsampleCount"), Ndf::SsaoDownsampleCount);
	

	glUniformMatrix4fv(glGetUniformLocation(programHandle, "MVP"), 1, false, &modelViewProjectionMatrix[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(programHandle, "MV"), 1, false, &modelViewMatrix[0][0]);

	glUniform3fv(glGetUniformLocation(programHandle, "camPosi"), 1, &camPosi[0]);

	auto camDirection = glm::normalize(camTarget - camPosi);
	glUniform3fv(glGetUniformLocation(programHandle, "camDirection"), 1, &camDirection[0]);

	auto viewSpaceCamDirection = viewMatrix * glm::vec4(camDirection, 0.0f);
	viewSpaceCamDirection = glm::normalize(viewSpaceCamDirection);
	glUniform3fv(glGetUniformLocation(programHandle, "viewSpaceCamDirection"), 1, &viewSpaceCamDirection[0]);

	auto viewSpaceLightDir = viewMatrix * glm::vec4(LightDir, 0.0f);
	viewSpaceLightDir = glm::normalize(viewSpaceLightDir);
	viewSpaceLightDir.y *= -1.0f;

	glUniform3fv(glGetUniformLocation(programHandle, "viewSpaceLightDir"), 1, &viewSpaceLightDir[0]);

	glUniform2i(glGetUniformLocation(programHandle, "viewDiscretizations"), Ndf::viewSampleResolution, Ndf::viewSampleResolution);
	glUniform2i(glGetUniformLocation(programHandle, "histogramDiscretizations"), Ndf::histogramResolution, Ndf::histogramResolution);

	int vx = Ndf::volumeResolutionX / pow(2, Ndf::MipLevelIndex);
	int vy = Ndf::volumeResolutionX / pow(2, Ndf::MipLevelIndex);
	glUniform3i(glGetUniformLocation(programHandle, "spatialDiscretizations"), vx, vx, Ndf::volumeResolutionZ);
	glUniform2i(glGetUniformLocation(programHandle, "viewportSize"), windowWidth, windowHeight);

	glUniform1f(glGetUniformLocation(programHandle, "quantizationVarianceScale"), l0->quantizationVarianceScale);


	// FIXME: quad handle does not work for some reason
	//QuadHandle.Render();
	//BoxHandle.Render();

	static const auto instances = 1;//20;
	static const auto spacing = 0.75f * glm::vec3(0.175f, 0.115f, 0.15f);
	static const auto scale = 1.0f / static_cast<float>(instances);
	static const auto halfInstance = instances / 2;

	const auto viewProjectionMatrix = projectionMatrix * viewMatrix;
	auto modelMatrix = glm::mat4x4();
	modelMatrix[0][0] = modelMatrix[1][1] = modelMatrix[2][2] = scale;
	for(auto z = 0; z < instances; ++z) {
		modelMatrix[3][2] = static_cast<float>(z - instances) * spacing.z;
		for(auto y = 0; y < instances; ++y) {
			modelMatrix[3][1] = static_cast<float>(y - halfInstance) * spacing.y;
			for(auto x = 0; x < instances; ++x) {
				modelMatrix[3][0] = static_cast<float>(x - halfInstance) * spacing.x;

				auto modelViewProjectionMatrix = viewProjectionMatrix * modelMatrix;

				glUniformMatrix4fv(glGetUniformLocation(programHandle, "MVP"), 1, false, &modelViewProjectionMatrix[0][0]);
				glUniformMatrix4fv(glGetUniformLocation(programHandle, "M"), 1, false, &modelMatrix[0][0]);

				BoxHandle.Render();
			}
		}
	}
	
	//glBindTexture(GL_TEXTURE_2D, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glUseProgram(0);
#endif
}

template <typename BinType>
void NdfRenderer<BinType>::BindSsbo(NdfTree<BinType> &ndfTree, const Helpers::Gl::ShaderProgram &shaderProgram) {
	assert(glShaderStorageBlockBinding);
	assert(glGetProgramResourceIndex);

	// TODO: bind base
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, ssbo.GlSsbo);

	GLuint ssboBindingPointIndex = 0;
	auto blockIndex = glGetProgramResourceIndex(shaderProgram.GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "NdfVoxelData");
	glShaderStorageBlockBinding(shaderProgram.GetGlShaderProgramHandle(), blockIndex, ssboBindingPointIndex);
	auto glErr = glGetError();
    if (glErr != GL_NO_ERROR) {
		std::cout << "glError: " << __FILE__ << " " << __LINE__ << std::endl;
    }
}