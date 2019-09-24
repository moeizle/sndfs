#include "NdfParticleSampler.h"

#include "GlHelpers.h"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/mat4x4.hpp>

#include <iostream>

using namespace NdfImposters;

//#define FLOATING_POINT_SAMPLES
//#define FULL_SAMPLES
#define PARTICLE_BINNING

// number of bins per spatial cell - important for large numbers of particles where having too many bins introduces too many duplicate particles
// FIXME: there might be an error if division > 1
static const auto binningDivision = 1;
// number of subdomains computed separately - important for large numbers of particles where storing them in vram is not an option
static const auto binningSubdomainCount = 1;

// instatiate for static linking
template class NdfParticleSampler<float>;

template <typename BinType>
void NdfParticleSampler<BinType>::ComputeFromParticles(std::vector<glm::vec3> particleCenters, float particleRadius,
	Helpers::Gl::ShaderProgram &normalSamplingProgram, int samplingRate, bool gpuReduction, Helpers::Gl::ShaderProgram &reductionShaderProgram, bool computeBinning) {
	auto histogramBinResolution = ndfTreeRef.GetHistogramResolution();
	auto viewDirectionResolution = ndfTreeRef.GetViewDirectionResolution();
	auto spatialResolution = ndfTreeRef.GetSpatialResolution();

	// check resolutions
	{
		if(ndfTreeRef.GetLevelCount() <= 0) {
			std::cerr << "Ndf tree does not have any levels" << std::endl;
			return;
		}

		if(histogramBinResolution.x <= 0 || histogramBinResolution.y <= 0) {
			std::cerr << "Histogram resolution within invalid range: " << histogramBinResolution.x << ", " << histogramBinResolution.y << std::endl;
		}

		if(viewDirectionResolution.x <= 0 || viewDirectionResolution.y <= 0) {
			std::cerr << "View direction resolution within invalid range: " << viewDirectionResolution.x << ", " << viewDirectionResolution.y << std::endl;
		}

		if(spatialResolution.x <= 0 || spatialResolution.y <= 0) {
			std::cerr << "Spatial resolution within invalid range: " << spatialResolution.x << ", " << spatialResolution.y << std::endl;
		}
	}

	{
		// create render target
#ifdef FLOATING_POINT_SAMPLES
#ifdef FULL_SAMPLES
		auto renderTarget = Helpers::Gl::CreateRenderTarget(samplingRate, samplingRate, GL_RGBA32F, GL_RGBA, GL_FLOAT);
#else
		auto renderTarget = Helpers::Gl::CreateRenderTarget(samplingRate, samplingRate, GL_RG32F, GL_RG, GL_FLOAT);
#endif
#else
#ifdef FULL_SAMPLES
		auto renderTarget = Helpers::Gl::CreateRenderTarget(samplingRate, samplingRate, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
#else
		auto renderTarget = Helpers::Gl::CreateRenderTarget(samplingRate, samplingRate, GL_RG8, GL_RG, GL_UNSIGNED_BYTE);
#endif

		assert(renderTarget.Width == samplingRate);
		assert(renderTarget.Height == samplingRate);

		if(histogramBinResolution.x > 256 || histogramBinResolution.y > 256) {
			std::cout << "8 bit quantization of normal samples too low for histogram resoltuion" << std::endl;
		}
#endif

		// check gl functions in debug mode
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

#ifdef PARTICLE_BINNING
		
#else
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
		Helpers::Gl::CreateGlMeshFromBuffers(quadVertices, particlesGlBuffer);
#endif

		//GLuint reductionSsbo = 0;

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
		auto tileScaleHorizontal = (rightOriginal - leftOriginal) / static_cast<float>(ndfTreeLevel.GetSpatialResolution().x);
		auto tileScaleVertical = (topOriginal - bottomOriginal) / static_cast<float>(ndfTreeLevel.GetSpatialResolution().y);

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

		auto samplingProgramHandle = normalSamplingProgram.GetGlShaderProgramHandle();
		assert(glUseProgram);
		glUseProgram(samplingProgramHandle);

		glUniform1f(glGetUniformLocation(samplingProgramHandle, "far"), farPlane);
		glUniform1f(glGetUniformLocation(samplingProgramHandle, "near"), nearPlane);
		glUniform1f(glGetUniformLocation(samplingProgramHandle, "particleSize"), particleScale);
		// FIXME: implement not known here
		glUniform1f(glGetUniformLocation(samplingProgramHandle, "viewportWidth"), 512);

		glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "Model"), 1, GL_FALSE, &modelMatrix[0][0]);

		// read back samples from render target
#ifdef FLOATING_POINT_SAMPLES
#ifdef FULL_SAMPLES
		std::vector<glm::vec2> normalSamples(renderTarget.Width * renderTarget.Height, glm::vec2(0.0f, 0.0f));
		std::vector<glm::vec4> normalSamples4D(renderTarget.Width * renderTarget.Height, glm::vec4(0.0f, 0.0f, 0.0f, 0.0f));
#else
		std::vector<glm::vec2> normalSamples(renderTarget.Width * renderTarget.Height, glm::vec2(0.0f, 0.0f));
#endif
#else
#ifdef FULL_SAMPLES
		std::vector<glm::u8vec4> normalSamplesU8(renderTarget.Width * renderTarget.Height, glm::u8vec4(0, 0, 0, 255));
#else
		std::vector<glm::u8vec2> normalSamplesU8(renderTarget.Width * renderTarget.Height, glm::u8vec2(0, 0));
		std::vector<glm::vec2> normalSamples(renderTarget.Width * renderTarget.Height, glm::vec2(0.0f, 0.0f));
#endif
#endif

		static const auto radius = 1.0f;
		const auto camTarget = glm::vec3(0.0f, 0.0f, 0.0f);
		for(auto viewY = 0; viewY < ndfTreeLevel.GetViewDirectionResolution().y; ++viewY) {
			for(auto viewX = 0; viewX < ndfTreeLevel.GetViewDirectionResolution().x; ++viewX) {
				std::cout << "view " << viewX << ", " << viewY << std::endl;

				auto view = glm::ivec2(viewX, viewY);

				auto viewDirection = ndfTreeLevel.GetViewDirection(view);
				auto camPosi = -radius * viewDirection;

				auto viewMatrix = glm::lookAt(camPosi, camTarget, camUp);
				auto modeViewMatrix = modelMatrix * viewMatrix;

				//auto right = glm::normalize(glm::cross(normalize(camTarget-camPosi), camUp));
				//glUniform3fv(glGetUniformLocation(samplingProgramHandle, "right"), 1, &right[0]);
				//glUniform3fv(glGetUniformLocation(samplingProgramHandle, "up"), 1, &camUp[0]);

				glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "ModelView"), 1, GL_FALSE, &modeViewMatrix[0][0]);

				glUniform3fv(glGetUniformLocation(samplingProgramHandle, "ViewPosition"), 1, &camPosi[0]);
				glUniform2i(glGetUniformLocation(samplingProgramHandle, "viewSlice"), viewX, viewY);

				// read back samples from render target
				// TODO: use bytes
				//std::vector<glm::vec2> normalSamples(renderTarget.Width * renderTarget.Height, glm::vec2(0.0f, 0.0f));

				glBindTexture(GL_TEXTURE_2D, renderTarget.RenderTexture);

#ifdef PARTICLE_BINNING
				
				const auto binningResolution = ndfTreeLevel.GetSpatialResolution() / binningDivision;
				const auto binningSubdomainResolution = binningResolution / binningSubdomainCount;

				for(int subdomainI = 0; subdomainI < binningSubdomainCount; ++subdomainI) {
				// use less bins than pixels
				const auto binningOffset = binningSubdomainResolution * subdomainI;

				//std::vector<Helpers::Gl::BufferHandle> particlesGlBuffer(binningResolution.x * binningResolution.y);

				if(computeBinning)
				{
					particlesGlBuffer.clear();
					particlesGlBuffer.resize(binningResolution.x * binningResolution.y);

					std::cout << "Particle binning" << std::endl;
					// convert particles to quads to account for particles covering multiple tiles
					std::vector<decltype(particleCenters)> quadVertices(binningResolution.x * binningResolution.y);

					assert(binningResolution.x > 1);
					assert(binningResolution.y > 1);
					auto inverseSpatialScale = glm::vec2(
						static_cast<float>(ndfTreeLevel.GetSpatialResolution().x - 1) / static_cast<float>(binningDivision),
						static_cast<float>(ndfTreeLevel.GetSpatialResolution().y - 1) / static_cast<float>(binningDivision));
					auto spatialScale = glm::vec2(1.0f / inverseSpatialScale.x, 1.0f / inverseSpatialScale.y);

					// NOTE: this approximation has to always overestimate.
					// compute how large a particle is in sizes of bins and use accordingly many neighbouring bins that could also be covered by the particle 
					//const auto neighbourBinning = static_cast<int>(std::ceilf(std::min(spatialScale.x, spatialScale.y) / (particleRadius * static_cast<float>(binningDivision))));
					// FIXME: use correct one:	const auto neighbourBinning = static_cast<int>(std::ceilf(particleRadius / std::min(spatialScale.x, spatialScale.y)));
					const auto neighbourBinning = static_cast<int>(std::ceilf(particleRadius / std::min(spatialScale.x, spatialScale.y)));
					//const auto neighbourBinning = 1;
					//const auto neighbourBinning = 0;

					std::cout << "\tNeighbouring bins: " << neighbourBinning << std::endl;
					std::cout << "\tNumber of bins: " << binningResolution.x << ", " << binningResolution.y << std::endl;
					
					const auto particleMessageInterval = static_cast<long long>(particleCenters.size() / 10);

					// single threaded is faster especially with multiprocessing
					//for(auto &particle : particleCenters) {
					//#pragma omp parallel for
					for(auto particleI = 0; particleI < particleCenters.size(); ++particleI) {
						if(particleI % particleMessageInterval == 0) {
							std::cout << "Binning particles " << static_cast<long long>(particleI) << " to " << std::min(static_cast<long long>(particleI) + particleMessageInterval, static_cast<long long>(particleCenters.size())) <<  std::endl;
						}

						auto &particle = particleCenters[particleI];
						auto viewSpaceParticle = modeViewMatrix * glm::vec4(particle, 0.0f);
						//auto viewSpaceParticle = glm::vec4(particle, 0.0f);

						assert(viewSpaceParticle.x >= -0.5f && viewSpaceParticle.x <= 0.5f);
						assert(viewSpaceParticle.y >= -0.5f && viewSpaceParticle.y <= 0.5f);
						assert(viewSpaceParticle.z >= -0.5f && viewSpaceParticle.z <= 0.5f);

						auto scaledParticle = glm::vec2((viewSpaceParticle.x + 0.5f) * inverseSpatialScale.x, (viewSpaceParticle.y + 0.5f) * inverseSpatialScale.y);

						// find spatial bin in which the particle center is
						auto binCoordinate = glm::ivec2(static_cast<int>(scaledParticle.x), static_cast<int>(scaledParticle.y));// + binningResolution / 2;
						assert(binCoordinate.x >= 0 && binCoordinate.x < binningResolution.x);
						assert(binCoordinate.y >= 0 && binCoordinate.y < binningResolution.y);
						//auto binIndex = binCoordinate.y * ndfTreeLevel.GetSpatialResolution().x + binCoordinate.x;

#if 1

						// TODO: calculate actually needed neighbours
						// calculate relative position within bin and see if it is close enough to the edge to reach into further bins

						auto horizontalNeighbours = neighbourBinning;
						auto verticalNeighbours = neighbourBinning;

						for(auto binY = std::max(0, binCoordinate.y - verticalNeighbours); binY < std::min(binCoordinate.y + 1 + verticalNeighbours, binningResolution.y - 1); ++binY) {
							for(auto binX = std::max(0, binCoordinate.x - horizontalNeighbours); binX < std::min(binCoordinate.x + 1 + horizontalNeighbours, binningResolution.x - 1); ++binX) {
								assert(binX >= 0 && binX < binningResolution.x);
								assert(binY >= 0 && binY < binningResolution.y);

								// TODO: this could be directly written into an ssbo
								// add to bin and all neighbouring bins within the particle radius
								auto binIndex = binY * binningResolution.x + binX;
								assert(binIndex >= 0 && binIndex < binningResolution.x * binningResolution.y);

								//#pragma omp critical
								{
									quadVertices[binIndex].emplace_back(particle.x-particleScale, particle.y-particleScale, particle.z);
									quadVertices[binIndex].emplace_back(particle.x+particleScale, particle.y-particleScale, particle.z);
									quadVertices[binIndex].emplace_back(particle.x+particleScale, particle.y+particleScale, particle.z);
									quadVertices[binIndex].emplace_back(particle.x-particleScale, particle.y+particleScale, particle.z);
								}
							}
						}
#else
						// FIXME: error if no neighbouring bins are used - everything above half stays black
						// FIXME: remove test
						auto binIndex = binCoordinate.y * binningResolution.x + binCoordinate.x;

						quadVertices[binIndex].emplace_back(particle.x-particleScale, particle.y-particleScale, particle.z);
						quadVertices[binIndex].emplace_back(particle.x+particleScale, particle.y-particleScale, particle.z);
						quadVertices[binIndex].emplace_back(particle.x+particleScale, particle.y+particleScale, particle.z);
						quadVertices[binIndex].emplace_back(particle.x-particleScale, particle.y+particleScale, particle.z);
#endif
					}
					std::cout << "Particle binning finished" << std::endl;

					std::cout << "Uploading particles" << std::endl;
					// upload particle quads
					const auto uploadMessageInterval = static_cast<long long>(particlesGlBuffer.size() / 10);
					decltype(quadVertices.begin()->size()) maxParticles = 0;
					auto averageParticles = 0.0f;
					auto totalMemory = size_t(0);
					for(auto binIndex = 0; binIndex < particlesGlBuffer.size(); ++binIndex) {
						if(binIndex % uploadMessageInterval == 0) {
							std::cout << "Uploading bin " << static_cast<long long>(binIndex) << " to " << std::min(static_cast<long long>(binIndex) + uploadMessageInterval, static_cast<long long>(particlesGlBuffer.size())) <<  std::endl;
						}

						// there have to be four vertices per quad
						assert(quadVertices[binIndex].size() % 4 == 0);

						maxParticles = std::max(maxParticles, quadVertices[binIndex].size());
						averageParticles += static_cast<float>(quadVertices[binIndex].size());

                        Helpers::Gl::MakeBuffer(quadVertices[binIndex], particlesGlBuffer[binIndex]);

						totalMemory += quadVertices[binIndex].size() * sizeof(*quadVertices.begin()->begin());
					}
					averageParticles /= static_cast<float>(binningResolution.x * binningResolution.y);

					std::cout << "Maximum number of particles in a bin " << maxParticles << ", average " << averageParticles << std::endl;
					std::cout << "Total particle memory " << totalMemory / (1024 * 1024) << " mb" << std::endl;

					auto glErr = glGetError();
					if (glErr != GL_NO_ERROR) {
						std::cout << "glError: " << __FILE__ << " " << __LINE__ << std::endl;
					}

					glFinish();
					std::cout << "Uploading particles finished" << std::endl;
				}
#else

#endif

				std::cout << "Sampling" << std::endl;
				for(auto spatialY = 0; spatialY < ndfTreeLevel.GetSpatialResolution().y; ++spatialY) {
					static const auto rowMessageInterval = 8LL;
					if(spatialY % rowMessageInterval == 0) {
						std::cout << "Rows " << static_cast<long long>(spatialY) << " to " << std::min(static_cast<long long>(spatialY) + rowMessageInterval, static_cast<long long>(ndfTreeLevel.GetSpatialResolution().y)) <<  std::endl;
					}

					size_t particlesRenderPerRow = 0;
					for(auto spatialX = 0; spatialX < ndfTreeLevel.GetSpatialResolution().x; ++spatialX) {
						auto spatial = glm::ivec2(spatialX, spatialY);
						
#ifdef PARTICLE_BINNING
						auto spatialBinning = spatial / binningDivision;
						auto spatialBinningIndex = spatialBinning.y * binningResolution.x + spatialBinning.x;
						auto &particleHandleRef = particlesGlBuffer[spatialBinningIndex];
#else
						auto &particleHandleRef = particlesGlBuffer;
#endif
						if(particleHandleRef.VertexCount_ > 0) {
							particlesRenderPerRow += particleHandleRef.VertexCount_;

							// calculate tile of global projection matrix
							auto xSubdomainF = static_cast<float>(spatialX);
							auto ySubdomainF = static_cast<float>(spatialY);

							auto leftTiled = leftOriginal + (tileScaleHorizontal) * xSubdomainF;
							auto rightTiled = leftTiled + tileScaleHorizontal;
							auto bottomTiled = bottomOriginal + (tileScaleVertical) * ySubdomainF;
							auto topTiled = bottomTiled + tileScaleVertical;
							auto projectionMatrix = glm::ortho(leftTiled, rightTiled, bottomTiled, topTiled, nearPlane, farPlane);
							auto viewProjectionMatrix = projectionMatrix * viewMatrix;

							// set uniforms
							glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "ViewProjection"), 1, GL_FALSE, &viewProjectionMatrix[0][0]);

							// clear render target
							glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

							// render samples
							particleHandleRef.Render(GL_QUADS);

							if(!gpuReduction) {
								// TODO: this should be the bottleneck - use a binning pass directly on the gpu
#ifdef FLOATING_POINT_SAMPLES
#ifdef FULL_SAMPLES
								glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, normalSamples4D.data());

								float depth = std::numeric_limits<float>::max();

								// calculate depth for ray package
#if 1
								// min non zero depth
								for(auto sampleI = 0; sampleI < normalSamples4D.size(); ++sampleI) {
									normalSamples[sampleI] = glm::vec2(normalSamples4D[sampleI].x, normalSamples4D[sampleI].y);
									if(normalSamples4D[sampleI].z > 0.0f) {
										depth = std::min(normalSamples4D[sampleI].z, depth);
									}
								}
#else
								for(auto sampleI = 0; sampleI < normalSamples4D.size(); ++sampleI) {
									normalSamples[sampleI] = glm::vec2(normalSamples4D[sampleI].x, normalSamples4D[sampleI].y);
								}

								// median excluding zero
								// remove all zero depth samples
								normalSamples4D.erase(std::remove_if(normalSamples4D.begin(), normalSamples4D.end(),
									[] (decltype(*normalSamples4D.begin()) &sample) {
										return sample.z <= 0.0f;
									}
								), normalSamples4D.end());
								
								// sort by depth
								std::sort(normalSamples4D.begin(), normalSamples4D.end(),
									[] (decltype(*normalSamples4D.begin()) &left, decltype(*normalSamples4D.begin()) &right) {
										return left.z  < right.z;
									}
								);
								
								// median = pick value in the middle
								depth = normalSamples4D[normalSamples4D.size() / 2].z;
#endif
								if(depth == std::numeric_limits<float>::max()) {
									depth = 0.0f;
								}
#else
								glGetTexImage(GL_TEXTURE_2D, 0, GL_RG, GL_FLOAT, normalSamples.data());
#endif
#else

#ifdef FULL_SAMPLES
								glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, normalSamplesU8.data());
#else
								glGetTexImage(GL_TEXTURE_2D, 0, GL_RG, GL_UNSIGNED_BYTE, normalSamplesU8.data());
								const float depth = 0.0f;
#endif

								static const auto dequantizationScale = 1.0f / (std::pow(2.0f, 8.0f)-1);
								// data size probably too small for multithreading - especially when using multiprocessing as well
								//#pragma omp parallel for
								for(auto sampleI = 0; sampleI < renderTarget.Width * renderTarget.Height; ++sampleI) {
									normalSamples[sampleI] = glm::vec2(
										static_cast<float>(normalSamplesU8[sampleI].x) * dequantizationScale,
										static_cast<float>(normalSamplesU8[sampleI].y) * dequantizationScale);
								}
#endif

								histogramBinning(ndfTreeLevel, spatial, view, samplingRate, normalSamples, depth);
							} else {
								// start reduction shader
								auto reductionHandle = reductionShaderProgram.GetGlShaderProgramHandle();
							
								glUseProgram(reductionHandle);

								int offset = ndfTreeLevel.GetDataIndex(spatial, view, {0, 0});
								glUniform1i(glGetUniformLocation(reductionHandle, "offset"), offset);
								glDispatchCompute(1, 1, 1);

								glUseProgram(samplingProgramHandle);
							}
						}
					}

					if(spatialY % rowMessageInterval == 0) {
						std::cout << static_cast<int>(static_cast<float>(particlesRenderPerRow) / static_cast<float>(ndfTreeLevel.GetSpatialResolution().x) * 0.25f) << " average particles per bin out of " << particleCenters.size()<< std::endl;
					}
				}
				}
			}
		}
		
		std::cout << "Sampling finished" << std::endl;


		glBindTexture(GL_TEXTURE_2D, 0);

		Helpers::Gl::DeleteRenderTarget(renderTarget);

		glUseProgram(0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		//glDisable(GL_POINT_SPRITE);
		//glDisable(GL_PROGRAM_POINT_SIZE);
		glDisable(GL_DEPTH_TEST);

		if(gpuReduction) {
			ndfTreeLevel.DownloadData();
		}

		ndfTreeRef.Downsample();
	}
}