#ifndef NDF_RENDERER_H_
#define NDF_RENDERER_H_

#include "NdfTree.h"

#include "ShaderUtility.h"

namespace NdfImposters {

template <typename BinType = float>
class NdfRenderer {
public:
	

	void Render(NdfTree<BinType> &ndfTree);
	void BindSsbo(NdfTree<BinType> &ndfTree, const Helpers::Gl::ShaderProgram &shaderProgram);

private:
	
};

} // NdfImposters

#endif // NDF_RENDERER_H_