#include "ProgressiveParticleSampler.h"

#include <glm/gtc/matrix_transform.hpp>

#include "GlHelpers.h"

using namespace NdfImposters;

// instatiate for static linking
template class NdfProgressiveParticleSampler<float>;

template <typename DataType>
void NdfProgressiveParticleSampler<DataType>::ComputeFromParticles(std::vector<glm::vec3> particleCenters, float particleRadius,
		Helpers::Gl::ShaderProgram &normalSamplingProgram, int samplingRate,
		bool gpuReduction, Helpers::Gl::ShaderProgram &reductionShaderProgram, bool computeBinning) {
	
	// binning not necessary - particles are all expected to be visible
	// add an offset in the view plane but it should not be entierly random - the whole footprint should be covered
	// create render target for ray casting solutions
	static const auto outputResolution = glm::ivec2(1024, 1024);
	static const auto superSampling = 1;
	static const auto renderTargetResolution = glm::ivec2(outputResolution.x * superSampling, outputResolution.y * superSampling);
	// TODO: check if within device specific boundaries
	auto renderTarget = Helpers::Gl::CreateRenderTarget(renderTargetResolution.x, renderTargetResolution.y, GL_RG8, GL_RG, GL_UNSIGNED_BYTE);

	const auto samplingProgramHandle = normalSamplingProgram.GetGlShaderProgramHandle();

	// check gl functions
	assert(glUniform1f);
	assert(glUniform2f);
	assert(glUniform3f);
	assert(glUniform4f);
	assert(glUniform1fv);
	assert(glUniform2fv);
	assert(glUniform3fv);
	assert(glUniform4fv);
	assert(glUniform1i);
	assert(glUniform2i);
	assert(glUniform3i);
	assert(glUniform4i);
	assert(glUniformMatrix4fv);
	assert(glGetUniformLocation);
	assert(glFrontFace);
	assert(glUseProgram);
	assert(glBindFramebuffer);
	assert(glClearColor);
	assert(glClear);
	assert(glDisable);
	assert(glDispatchCompute);
		
	// sample particles from all positions and directions
	assert(ndfTreeRef.GetLevels().size() > 0);
	auto &ndfTreeLevel = ndfTreeRef.GetLevels().front();

	const auto particleScale = particleRadius;

	// convert particles to quads to account for particles covering multiple tiles
	decltype(particleCenters) quadVertices;

	for(auto &&particle : particleCenters) {
		quadVertices.emplace_back(particle.x-particleScale, particle.y-particleScale, particle.z);
		quadVertices.emplace_back(particle.x+particleScale, particle.y-particleScale, particle.z);
		quadVertices.emplace_back(particle.x+particleScale, particle.y+particleScale, particle.z);
		quadVertices.emplace_back(particle.x-particleScale, particle.y+particleScale, particle.z);
	}
		
	// upload particle quads
	Helpers::Gl::BufferHandle particlesGlBuffer;
    Helpers::Gl::MakeBuffer(quadVertices, particlesGlBuffer);

	//GLuint reductionSsbo = 0;

	auto histogramBinResolution = ndfTreeLevel.GetHistogramResolution();

	if(gpuReduction) {
		// setup reduction buffers
		//auto ssboSize = 
		//glGenBuffers(1, &reductionSsbo);
		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, reductionSsbo);
		//glBufferData(GL_SHADER_STORAGE_BUFFER, ssboSize, nullptr, GL_DYNAMIC_COPY);

		// make sure the output ssbo already exists
		const GLuint ssboBindingPointIndex = 0;
		ndfTreeLevel.UploadData(ssboBindingPointIndex);	

		// bind ssbo
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, ndfTreeLevel.GetShaderStorageBufferOject());

		auto reductionHandle = reductionShaderProgram.GetGlShaderProgramHandle();
		glUseProgram(reductionHandle);

		glUniform1i(glGetUniformLocation(reductionHandle, "offset"), 0);
		glUniform2i(glGetUniformLocation(reductionHandle, "histogramDiscretizations"), histogramBinResolution.x, histogramBinResolution.y);
		glUniform2i(glGetUniformLocation(reductionHandle, "sampleResolution"), renderTarget.Width, renderTarget.Height);

		glUseProgram(0);
	}
		
	// calculate global projection matrix
	auto leftOriginal = -0.5f;
	auto rightOriginal = 0.5f;
	auto bottomOriginal = -0.5f;
	auto topOriginal = 0.5f;

	glm::mat4x4 modelMatrix;
	static const auto modelScale = 1.0f;
	modelMatrix[0][0] = modelScale;
	modelMatrix[1][1] = modelScale;
	modelMatrix[2][2] = modelScale;

	// TODO: set as parameters
	static const auto camUp = glm::vec3(0.0f, 1.0f, 0.0f);
	static const auto nearPlane = 0.001f;
	static const auto farPlane = 10.0f;

	glBindFramebuffer(GL_FRAMEBUFFER, renderTarget.FrameBufferObject);
	// FIXME: viewport causes tears for some reason if sampling rate < 1024
	glViewport(0, 0, renderTarget.Width, renderTarget.Height);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	//glEnable(GL_POINT_SPRITE);
	//glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_DEPTH_TEST);
	glFrontFace(GL_CCW);

	glUseProgram(samplingProgramHandle);

	glUniform1f(glGetUniformLocation(samplingProgramHandle, "far"), farPlane);
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "near"), nearPlane);
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "particleSize"), particleScale);
	// FIXME: implement not known here
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "viewportWidth"), 512);

	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "Model"), 1, GL_FALSE, &modelMatrix[0][0]);

	static const auto radius = 1.0f;
	const auto camTarget = glm::vec3(0.0f, 0.0f, 0.0f);

	auto viewDirection = ndfTreeLevel.GetViewDirection({0, 0});
	auto camPosi = -radius * viewDirection;

	auto viewMatrix = glm::lookAt(camPosi, camTarget, camUp);
	auto modeViewMatrix = modelMatrix * viewMatrix;

	//auto right = glm::normalize(glm::cross(normalize(camTarget-camPosi), camUp));
	//glUniform3fv(glGetUniformLocation(samplingProgramHandle, "right"), 1, &right[0]);
	//glUniform3fv(glGetUniformLocation(samplingProgramHandle, "up"), 1, &camUp[0]);

	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "ModelView"), 1, GL_FALSE, &modeViewMatrix[0][0]);

	glUniform3fv(glGetUniformLocation(samplingProgramHandle, "ViewPosition"), 1, &camPosi[0]);
	glUniform2i(glGetUniformLocation(samplingProgramHandle, "viewSlice"), 0, 0);

	// read back samples from render target
	// TODO: use bytes
	//std::vector<glm::vec2> normalSamples(renderTarget.Width * renderTarget.Height, glm::vec2(0.0f, 0.0f));

	glBindTexture(GL_TEXTURE_2D, renderTarget.RenderTexture);

	std::cout << "Sampling" << std::endl;

	auto spatial = glm::ivec2(0, 0);
						
	auto &particleHandleRef = particlesGlBuffer;

	auto projectionMatrix = glm::ortho(leftOriginal, rightOriginal, bottomOriginal, topOriginal, nearPlane, farPlane);
	auto viewProjectionMatrix = projectionMatrix * viewMatrix;

	// set uniforms
	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "ViewProjection"), 1, GL_FALSE, &viewProjectionMatrix[0][0]);

	// clear render target
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// render samples
	particleHandleRef.Render(GL_QUADS);
		
	std::cout << "Sampling finished" << std::endl;


	glBindTexture(GL_TEXTURE_2D, 0);

	Helpers::Gl::DeleteRenderTarget(renderTarget);

	glUseProgram(0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//glDisable(GL_POINT_SPRITE);
	//glDisable(GL_PROGRAM_POINT_SIZE);
	glDisable(GL_DEPTH_TEST);
}