#ifndef HELPERS_MEGA_MOL_PARTICLE_FILE_H_
#define HELPERS_MEGA_MOL_PARTICLE_FILE_H_

#include <vector>
#include <string>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/mat2x3.hpp>

#include "../NdfImposterLibraryTests/octree.h"

namespace Helpers {
namespace IO {

	// ensure types exist
#ifndef uint8_t
	typedef unsigned __int8 uint8_t;
#endif
#ifndef uint16_t
	typedef unsigned __int16 uint16_t;
#endif
#ifndef uint32_t
	typedef unsigned __int32 uint32_t;
#endif
#ifndef uint64_t
	typedef unsigned __int64 uint64_t;
#endif

	using ParticleType = glm::vec4;
	std::vector<ParticleType> ParticlesFromMmpld(std::string filePath, uint64_t maxParticles,
		float &outGlobalRadius, glm::mat2x3 areaOfInterest, uint32_t maxFrameCount, glm::mat2x3 &outBoundingBox, bool rescale = true, glm::mat2x3 *ownBBox = nullptr);

	void readParticleData(std::string filePath, uint64_t maxParticles, float &outGlobalRadius, glm::mat2x3 areaOfInterest, uint32_t maxFrameCount, glm::mat2x3 &outBoundingBox, octree& tree, long long in_node_max_memory_consumption, bool rescale = true, glm::mat2x3 *ownBBox = nullptr);
	void computeBoundingBox(std::string filePath, uint64_t maxParticles, float &outGlobalRadius, glm::mat2x3 areaOfInterest, uint32_t maxFrameCount, glm::mat2x3 &outBoundingBox);
	void placeDataInTree(std::string filePath, uint64_t maxParticles, float &outGlobalRadius, glm::mat2x3 areaOfInterest, uint32_t maxFrameCount, glm::mat2x3 &outBoundingBox, octree& tree, long long in_node_max_memory_consumption, bool rescale = true, glm::mat2x3 *ownBBox = nullptr);
}
}

#endif // HELPERS_MEGA_MOL_PARTICLE_FILE_H_