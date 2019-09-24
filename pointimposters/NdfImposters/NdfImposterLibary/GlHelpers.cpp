#include "GlHelpers.h"

using namespace Helpers::Gl;

glm::vec3 Helpers::Gl::SphericalCoordinatesToWorld(glm::vec2 sphericalCoordinate) {
	return { std::sinf(sphericalCoordinate.x) * std::cosf(sphericalCoordinate.y), std::sinf(sphericalCoordinate.x) * std::sinf(sphericalCoordinate.y), std::cosf(sphericalCoordinate.x) };
}

// FIXME: ewww void pointer
void Helpers::Gl::CreateTexture(TextureHandle2D &textrueHandle, int width, int height, GLint internalFormat, GLenum format, GLenum type, void *data) {
	glGenTextures(1, &textrueHandle.Texture);
	glBindTexture(GL_TEXTURE_2D, textrueHandle.Texture);
	glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, data);
	glGenerateMipmap(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, 0);

	textrueHandle.Width = width;
	textrueHandle.Height = height;
}

/**
 * internalFormat, format and type are the same as in glTextImage2D.
 */
RenderTargetHandle Helpers::Gl::CreateRenderTarget(int width, int height, GLint internalFormat, GLenum format, GLenum type) 
{
	assert(glGenFramebuffers);

	GLuint glFramebufferObject = 0;
	GLuint glDepthMap = 0;
	GLuint glRenderTexture = 0;

	// initialize render target
	glGenFramebuffers(1, &glFramebufferObject);
	glGenRenderbuffers(1, &glDepthMap);
	glGenTextures(1, &glRenderTexture);
	
	glBindTexture(GL_TEXTURE_2D, glRenderTexture);
	glTexImage2D(GL_TEXTURE_2D,	0, internalFormat, width, height,0, format,	type, nullptr);
	glGenerateMipmap(GL_TEXTURE_2D);

	glBindFramebuffer(GL_FRAMEBUFFER, glFramebufferObject);

	glBindRenderbuffer(GL_RENDERBUFFER, glDepthMap);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, glDepthMap);

	// attach depth buffer
	glBindRenderbuffer(GL_RENDERBUFFER, glDepthMap);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, glDepthMap);


	GLenum textureIndex = GL_TEXTURE0;
	GLenum attachementIndex = GL_COLOR_ATTACHMENT0;

	glActiveTexture(textureIndex);
	glBindTexture(GL_TEXTURE_2D, glRenderTexture);

	glBindFramebuffer(GL_FRAMEBUFFER, glFramebufferObject);
	glFramebufferTexture2D(GL_FRAMEBUFFER, attachementIndex, GL_TEXTURE_2D, glRenderTexture, 0);

	glBindTexture(GL_TEXTURE_2D, 0);
	
	// update draw buffers
	std::array<GLuint, 1> drawBufferAttachements = {{GL_COLOR_ATTACHMENT0}};

	glDrawBuffers(1, drawBufferAttachements.data());

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	return RenderTargetHandle(glFramebufferObject, glDepthMap, glRenderTexture, width, height);
}



void Helpers::Gl::DeleteRenderTarget(GLuint &glFramebufferObject, GLuint &glDepthMap, GLuint &glRenderTexture) {
	glDeleteFramebuffers(1, &glFramebufferObject);
	glFramebufferObject = 0;

	glDeleteRenderbuffers(1, &glDepthMap);
	glDepthMap = 0;

	glDeleteTextures(1, &glRenderTexture);
	glRenderTexture = 0;
}

void Helpers::Gl::DeleteRenderTarget(RenderTargetHandle &handle) {
	Helpers::Gl::DeleteRenderTarget(handle.FrameBufferObject, handle.DepthMap, handle.RenderTexture);
}

std::vector<glm::vec3> Helpers::Gl::CreateBoxVertexPositions(float boxExtent, glm::vec3 center) {
	return {
		//front
		{-boxExtent, -boxExtent, boxExtent},
		{-boxExtent, boxExtent, boxExtent},
		{boxExtent, boxExtent, boxExtent},
		{boxExtent, boxExtent, boxExtent},
		{boxExtent, -boxExtent, boxExtent},
		{-boxExtent, -boxExtent, boxExtent},

		//back
		{boxExtent, boxExtent, -boxExtent},
		{-boxExtent, boxExtent, -boxExtent},
		{-boxExtent, -boxExtent, -boxExtent},
		{-boxExtent, -boxExtent, -boxExtent},
		{boxExtent, -boxExtent, -boxExtent},
		{boxExtent, boxExtent, -boxExtent},

		//top
		{boxExtent, boxExtent, -boxExtent},
		{boxExtent, boxExtent, boxExtent},
		{-boxExtent, boxExtent, boxExtent},
		{-boxExtent, boxExtent, boxExtent},
		{-boxExtent, boxExtent, -boxExtent},
		{boxExtent, boxExtent, -boxExtent},

		//bottom
		{boxExtent, -boxExtent, boxExtent},
		{boxExtent, -boxExtent, -boxExtent},
		{-boxExtent, -boxExtent, -boxExtent},
		{-boxExtent, -boxExtent, -boxExtent},
		{-boxExtent, -boxExtent, boxExtent},
		{boxExtent, -boxExtent, boxExtent},

		//right
		{boxExtent, -boxExtent, -boxExtent},
		{boxExtent, -boxExtent, boxExtent},
		{boxExtent, boxExtent, boxExtent},
		{boxExtent, boxExtent, boxExtent},
		{boxExtent, boxExtent, -boxExtent},
		{boxExtent, -boxExtent, -boxExtent},

		//left
		{-boxExtent, boxExtent, boxExtent},
		{-boxExtent, -boxExtent, boxExtent},
		{-boxExtent, -boxExtent, -boxExtent},
		{-boxExtent, -boxExtent, -boxExtent},
		{-boxExtent, boxExtent, -boxExtent},
		{-boxExtent, boxExtent, boxExtent}
	};
}