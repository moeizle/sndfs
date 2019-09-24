#include "ShaderUtility.h"
#include "FileIO.h"

#include <array>
#include <memory>

#include <iostream>
#include <fstream>
#include <sstream>
#include <assert.h>
#include <array>
#include <assert.h>

using namespace Helpers::Gl;

Shader::Shader() {
	glShaderHandle = 0;
}

Shader::~Shader() {
	if(glShaderHandle) {
		glDeleteShader(glShaderHandle);
		glShaderHandle = 0;
	}
}

void Shader::FromSource(std::string source, GLuint shaderType) {
	assert(glCreateShader);
	assert(glShaderSource);
	assert(glCompileShader);
	assert(glGetShaderInfoLog);

	const GLchar* code = source.c_str();

	glShaderHandle = glCreateShader(shaderType);
	
	glShaderSource(glShaderHandle, 1, &code, 0);
	glCompileShader(glShaderHandle);

	static const int BUFFER_SIZE = 1024;
	std::array<char, BUFFER_SIZE> infoBuffer;
	std::fill(infoBuffer.begin(), infoBuffer.end(), 0);

	GLsizei length = 0;

	// get the log
	glGetShaderInfoLog(glShaderHandle, static_cast<GLsizei>(infoBuffer.size()), &length, infoBuffer.data());
	if (length > 1)	{
		std::cout << "Shader info log: " << infoBuffer.data() << std::endl;

		assert(length <= 1);
	}
}

void Shader::FromFile(std::string filePath, GLenum shaderType, std::string preprendString) {
	// TODO: delete previous if needed
	if(glShaderHandle) {

	}

	{
		Helpers::IO::FileGuard<std::ifstream> fileGuard(filePath, std::ifstream::in);
		if(fileGuard.file.is_open()) {
			std::stringstream buffer;
			buffer << preprendString;
			buffer << fileGuard.file.rdbuf();

			FromSource(buffer.str(), shaderType);
		} else {
			std::cout << "Failed to load shader from file: " << filePath << std::endl;

			assert(fileGuard.file.is_open());
		}
	}
}

GLuint Shader::GetGlShaderHandle() const {
	return this->glShaderHandle;
}


ShaderProgram::ShaderProgram() {
	glShaderProgramHandle = 0;
}

ShaderProgram::~ShaderProgram() {

}
void ShaderProgram::Initialize() {
	// TODO: delete previous program if needed

	assert(glCreateProgram);
	glShaderProgramHandle = glCreateProgram();
}

void ShaderProgram::AttachVertexShader(const std::shared_ptr<Shader> &shader) {
	assert(glAttachShader);
	assert(glShaderProgramHandle);

	vertexShader = shader;

	// TODO: detach previous shader if needed

	glAttachShader(glShaderProgramHandle, shader->GetGlShaderHandle());
}

void ShaderProgram::AttachFragmentShader(const std::shared_ptr<Shader> &shader) {
	assert(glAttachShader);
	assert(glShaderProgramHandle);

	fragmentShader = shader;

	// TODO: detach previous shader if needed

	
	glAttachShader(glShaderProgramHandle, shader->GetGlShaderHandle());
}

void ShaderProgram::AttachGeometryShader(const std::shared_ptr<Shader> &shader) {
	assert(glAttachShader);
	assert(glShaderProgramHandle);

	geometryShader = shader;

	// TODO: detach previous shader if needed

	
	glAttachShader(glShaderProgramHandle, shader->GetGlShaderHandle());
}


void ShaderProgram::AttachComputeShader(const std::shared_ptr<Shader> &shader) {
	assert(glAttachShader);
	assert(glShaderProgramHandle);

	computeShader = shader;

	// TODO: detach previous shader if needed

	
	glAttachShader(glShaderProgramHandle, shader->GetGlShaderHandle());
}

void ShaderProgram::LinkProgram() {
	assert(glBindAttribLocation);
	assert(glLinkProgram);
	assert(glGetProgramInfoLog);
	assert(glValidateProgram);
	assert(glGetProgramiv);
	assert(glAttachShader);

	glBindAttribLocation(glShaderProgramHandle, 0, "inPosition");

	glLinkProgram(glShaderProgramHandle);

	const unsigned int BUFFER_SIZE = 512;
	std::array<char, BUFFER_SIZE> infoBuffer;
	std::fill(infoBuffer.begin(), infoBuffer.end(), 0);

	GLsizei length = 0;
	glGetProgramInfoLog(glShaderProgramHandle, BUFFER_SIZE, &length, infoBuffer.data()); 
	if (length > 1)	{
		std::cout << "Shader program log: " << infoBuffer.data() << std::endl;
	}

	glValidateProgram(glShaderProgramHandle); 

	// Find out if the shader program validated correctly
	GLint status;
	glGetProgramiv(glShaderProgramHandle, GL_VALIDATE_STATUS, &status);
	if (status == GL_FALSE)	{
		std::cout << "Shader program invalid" << std::endl;

		length = 0;
		glGetProgramInfoLog(glShaderProgramHandle, BUFFER_SIZE, &length, infoBuffer.data()); 
		if (length > 1)	{
			std::cout << "Shader program log: " << infoBuffer.data() << std::endl;
		}

		assert(status);
	} else {
		std::cout << "Shader program successfully validated" << std::endl;
		
	}
}

void ShaderProgram::setSsboBindingIndex(unsigned int name, unsigned int& index)
{
	ssboBindingIndecies[name] = index;
	index++;
}

unsigned int ShaderProgram::getSsboBiningIndex(unsigned int name)
{
	return ssboBindingIndecies.find(name)->second;
}

GLuint ShaderProgram::GetGlShaderProgramHandle() const {
	return this->glShaderProgramHandle;
}

std::map<unsigned int, unsigned int> ShaderProgram::getSsboBindingIndecies()
{
	return ssboBindingIndecies;
}

void ShaderProgram::setSsboBindingIndecies(std::map<unsigned int, unsigned int> m)
{
	ssboBindingIndecies = m;
}

