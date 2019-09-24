#ifndef NDF_PARTICLE_SAMPLER_H_
#define NDF_PARTICLE_SAMPLER_H_

#include "NdfSampler.h"

#include "ShaderUtility.h"

#include <glm\vec3.hpp>

#include <vector>

namespace Helpers {
	namespace Gl {
		class BufferHandle;
	}
}

namespace NdfImposters {

template <typename BinType = float>
class NdfParticleSampler : public NdfSampler<BinType> {
public:
	NdfParticleSampler(NdfTree<BinType> &ndfTree) : NdfSampler(ndfTree) {}

	/*
	Important: InitializeStorage has to be called first.

	Builds the Ndf Tree by sampling the surface normal vectors of a given set of particles.
	*/
	// TODO: add callback to set uniforms.
	// TODO: move to generator project?
	void ComputeFromParticles(std::vector<glm::vec3> particleCenters, float particleRadius,
		Helpers::Gl::ShaderProgram &normalSamplingProgram, int samplingRate,
		bool gpuReduction, Helpers::Gl::ShaderProgram &reductionShaderProgram, bool computeBinning);

private:
	std::vector<Helpers::Gl::BufferHandle> particlesGlBuffer;
};

} // namespace NdfImposters

#endif // NDF_PARTICLE_SAMPLER_H_