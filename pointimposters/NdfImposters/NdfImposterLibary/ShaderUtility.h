#ifndef HELPERS_SHADER_UTILITY_H_
#define HELPERS_SHADER_UTILITY_H_

//#include "GlhSingle.h"
#include "glad/glad.h"

#include <memory>
#include <string>
#include <map>

namespace Helpers {
namespace Gl {

class Shader {
public:
	Shader();
	~Shader();

	// FIXME: should not be a member function
	void FromSource(std::string source, GLuint shaderType);
	// FIXME: should not be a member function
	void FromFile(std::string filePath, GLuint shaderType, std::string preprendString = "");

	GLuint GetGlShaderHandle() const;

private:
	GLuint glShaderHandle;
};

class ShaderProgram {
public:
	ShaderProgram();
	~ShaderProgram();

	void Initialize();
	void AttachVertexShader(const std::shared_ptr<Shader> &shader);
	void AttachFragmentShader(const std::shared_ptr<Shader> &shader);
	void AttachGeometryShader(const std::shared_ptr<Shader> &shader);
	void AttachComputeShader(const std::shared_ptr<Shader> &shader);
	void LinkProgram();
	void setSsboBindingIndex(unsigned int name, unsigned int& index);
	unsigned int getSsboBiningIndex(unsigned int name);
	std::map<unsigned int, unsigned int> getSsboBindingIndecies();
	void setSsboBindingIndecies(std::map<unsigned int, unsigned int> m);

	GLuint GetGlShaderProgramHandle() const;

private:
	std::shared_ptr<Shader> vertexShader;
	std::shared_ptr<Shader> fragmentShader;
	std::shared_ptr<Shader> geometryShader;
	std::shared_ptr<Shader> computeShader;

	std::map<unsigned int, unsigned int> ssboBindingIndecies;

	GLuint glShaderProgramHandle;
};

} // namespace Gl
} // namespace Helpers

#endif // HELPERS_SHADER_UTILITY_H_
