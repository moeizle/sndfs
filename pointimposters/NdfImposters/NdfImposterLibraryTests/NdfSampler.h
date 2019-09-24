#ifndef NDF_SAMPLER_H_
#define NDF_SAMPLER_H_

#include "NdfTree.h"

namespace NdfImposters {

/*
Fills an Ndf Tree by sampling.
*/
template <typename BinType = float>
class NdfSampler {
public:
	NdfSampler(NdfTree<BinType> &ndfTree) : ndfTreeRef(ndfTree) {}

protected:
	// function to fill a histogram using a set of samples
	void histogramBinning(NdfTreeLevel<BinType> &ndfTreeLevel, glm::ivec2 spatial, glm::ivec2 view, int samplingRate, const std::vector<glm::vec2> &normalSamples, float depth);
	
	// TODO: function that iterates over the spatial locations and executes a lambda
	// that samples and the surface
	//void SampleSpatial(lambda someLambda);

	NdfTree<BinType> &ndfTreeRef;
};

} // namespace NdfImposters

#endif // NDF_SAMPLER_H_