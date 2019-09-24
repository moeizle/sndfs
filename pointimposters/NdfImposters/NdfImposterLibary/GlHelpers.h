#ifndef GL_HELPERS_H_
#define GL_HELPERS_H_

//#include "GlhSingle.h"
#include "glad/glad.h"

#include <assert.h>
#include <vector>
#include <array>
#include <string>
#include <iostream>

#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/vec2.hpp>

#include "../NdfImposterLibraryTests/octree.h"

namespace Helpers {
namespace Gl {
	void writeBitmap(std::string filePath);

	glm::vec3 SphericalCoordinatesToWorld(glm::vec2 sphericalCoordinate);

	class TextureHandle2D {
	public:
		TextureHandle2D() : Texture(0), Width(0), Height(0) {}
		~TextureHandle2D() {
			glDeleteTextures(1, &Texture);
		}

		GLuint Texture;
		int Width;
		int Height;
	};

	void CreateTexture(TextureHandle2D &textrueHandle, int width, int height, GLint internalFormat, GLenum format, GLenum type, void *data);

	class RenderTargetHandle {
	public:
		RenderTargetHandle() : FrameBufferObject(0), DepthMap(0), RenderTexture(0) {}
		RenderTargetHandle(GLuint fbo, GLuint depthMap, GLuint renderTexture, int width, int height) :
			FrameBufferObject(fbo), DepthMap(depthMap), RenderTexture(renderTexture), Width(width), Height(height) {}

		GLuint FrameBufferObject;
		GLuint DepthMap;
		GLuint RenderTexture;
		int Width;
		int Height;
	};

	RenderTargetHandle CreateRenderTarget(int width, int height, GLint internalFormat, GLenum format, GLenum type);

	void DeleteRenderTarget(GLuint &glFramebufferObject, GLuint &glDepthMap, GLuint &glRenderTexture);
	void DeleteRenderTarget(RenderTargetHandle &handle);

	class BufferHandle {
	//protected:
	//	static std::unordered_map<GLuint, size_t> hVBO;
	//	static std::unordered_map<GLuint, size_t> hVAO;
	public:
		BufferHandle() : Vbo_(0), Vao_(0), VertexCount_(0) {}
		BufferHandle(GLuint vbo, GLuint vao, int vertexCount) : Vbo_(vbo), Vao_(vao), VertexCount_(vertexCount) {}
		~BufferHandle() 
		{
		//	auto it = hVBO.find(Vbo_);
		//	if (it != hVBO.end()) {
		//		it->second--;
		//		if (it->second==0) glDelete
		//	}
            if (Vao_ != 0)
				glDeleteVertexArrays(1, &Vao_);
            if (Vbo_ != 0)
				glDeleteBuffers(1, &Vbo_);
		}
		void clear(void) {
			glDeleteVertexArrays(1, &Vao_);
			glDeleteBuffers(1, &Vbo_);
		}
		BufferHandle& operator=(const BufferHandle& other) {
			Vbo_ = other.Vbo_;
			Vao_ = other.Vao_;
			VertexCount_ = other.VertexCount_;
			//if (Vbo_ != 0) hVBO[Vbo_]++;
			return *this;
		}
		void Render(GLenum primitiveType = GL_TRIANGLES) 
		{
			glBindVertexArray(Vao_);
			glDrawArrays(primitiveType, 0, VertexCount_);
			glBindVertexArray(0);
		}

		void Render_Selective(GLenum primitiveType, int* inds,int count) {
			glBindVertexArray(Vao_);
			glDrawElements(primitiveType, count, GL_UNSIGNED_INT, inds);
			//glDrawArrays(primitiveType, 0, VertexCount_);
			glBindVertexArray(0);
		}

		void Delete() {
			glDeleteVertexArrays(1, &Vao_);
			glDeleteBuffers(1, &Vbo_);

			Vao_ = 0;
			Vbo_ = 0;
			VertexCount_ = 0;
		}
		//void set_vbo(GLuint id) {
		//	hVBO[id]++;
		//	Vbo_ = id;
		//}
	//protected:
		GLuint Vbo_;
		GLuint Vao_;
		std::size_t VertexCount_;
	};

#if 0
	template <typename DataType>
	struct BufferType {
		std::string Label_;
		std::vector<DataType> Data_;
	};

	template <typename DataType>
	MeshHandle CreateGlMeshFromBuffers(std::vector<BufferType<DataType>> buffers);
#else

    namespace detail {
        template <class T> struct scalar_traits {};
        template <> struct scalar_traits<float> { static const GLenum value = GL_FLOAT; };
        template <> struct scalar_traits<double> { static const GLenum value = GL_DOUBLE; };
    }

    template <typename T>
    void MakeBuffer(std::vector<T>& vertexPositions, BufferHandle &handle) 
	{
        typedef typename T::value_type scalar_type;

        static_assert(sizeof(T) % sizeof(scalar_type) == 0,
            "MakeBuffer can only be used with structures consisting of integral multiples of their value_type");
        static_assert(sizeof(T) >= sizeof(scalar_type),
            "MakeBuffer can only be used with structures consisting of at least one of their value_type");

        handle.Vao_ = 0;
        handle.Vbo_ = 0;
        handle.VertexCount_ = vertexPositions.size();

        if (handle.VertexCount_ <= 0) {
            return;
        }

        auto positionsSize = vertexPositions.size() * sizeof(T);

        GLuint vao;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);

        GLuint vbo;
        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, positionsSize, static_cast<GLvoid*>(vertexPositions.data()), GL_DYNAMIC_DRAW);

        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, sizeof(T) / sizeof(scalar_type), detail::scalar_traits<scalar_type>::value, GL_FALSE, 0, static_cast<GLvoid*>(0));

        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        handle.Vao_ = vao;
        handle.Vbo_ = vbo;
        handle.VertexCount_ = vertexPositions.size();
    }

	template <typename T>
	void MakeBuffer(std::vector<T>& vertexPositions, octree& ot,std::vector<std::pair<int, BufferHandle>>& handles)
	{
		//for the sake of testing, we'll brick the datat into the leaves of the octree
		std::vector<int> leaves;
		ot.get_leaves(leaves);
		
		
		handles.clear();
		
		//GLuint vaos[30];// = new GLuint[leaves.size()];
		//GLuint vbos[30];// = new GLuint[leaves.size()];

		//glGenVertexArrays(leaves.size(), &vaos[0]);
		//glGenBuffers(leaves.size(), &vbos[0]);

		
		typedef typename T::value_type scalar_type;

		static_assert(sizeof(T) % sizeof(scalar_type) == 0,
			"MakeBuffer can only be used with structures consisting of integral multiples of their value_type");
		static_assert(sizeof(T) >= sizeof(scalar_type),
			"MakeBuffer can only be used with structures consisting of at least one of their value_type");


		for (int i = 0; i < leaves.size(); i++)
		{
			//get the vertices that correspond to this leaf
			std::vector<T> verticesInLeaf;
			for (int j = 0; j < ot.nodes[leaves[i]].indicies.size(); j++)
			{
				verticesInLeaf.push_back(vertexPositions[ot.nodes[leaves[i]].indicies[j]]);
			}

			handles.push_back(std::make_pair(leaves[i], BufferHandle()));

			//handles[i].first = leaves[i];
			handles[i].second.Vao_ = 0;
			handles[i].second.Vbo_ = 0;
			handles[i].second.VertexCount_ = verticesInLeaf.size();

			if (handles[i].second.VertexCount_ > 0) 
			{
				auto positionsSize = verticesInLeaf.size() * sizeof(T);

				//create vbo to contain the vertices of the leaf
				GLuint vbo=0;
				glGenBuffers(1, &vbo);
				glBindBuffer(GL_ARRAY_BUFFER, vbo);
				glBufferData(GL_ARRAY_BUFFER, positionsSize, static_cast<GLvoid*>(verticesInLeaf.data()), GL_DYNAMIC_DRAW);
				glBindBuffer(GL_ARRAY_BUFFER, 0);

				//create a vao containing the vbo
				GLuint vao=0;
				glGenVertexArrays(1, &vao);
				glBindVertexArray(vao);

				//bind vbo to vao
				glBindBuffer(GL_ARRAY_BUFFER, vbo);
				glVertexAttribPointer(0, sizeof(T) / sizeof(scalar_type), detail::scalar_traits<scalar_type>::value, GL_FALSE, 0, static_cast<GLvoid*>(0));

				glEnableVertexAttribArray(0);
				
				handles[i].second.Vao_ = vao;
				handles[i].second.Vbo_ = vbo;
				handles[i].second.VertexCount_ = verticesInLeaf.size();
				glBindBuffer(GL_ARRAY_BUFFER, 0);
				glBindVertexArray(0);
				
			}
		}
	}

	template <typename T>
	void MakeBuffer(std::vector<T>& vertexPositions, octree::node& n, BufferHandle& handle)
	{
		typedef typename T::value_type scalar_type;

		static_assert(sizeof(T) % sizeof(scalar_type) == 0,
			"MakeBuffer can only be used with structures consisting of integral multiples of their value_type");
		static_assert(sizeof(T) >= sizeof(scalar_type),
			"MakeBuffer can only be used with structures consisting of at least one of their value_type");



		//get the vertices that correspond to this leaf
		std::vector<T> verticesInLeaf;
		for (int j = 0; j < n.indicies.size(); j++)
		{
			verticesInLeaf.push_back(vertexPositions[n.indicies[j]]);
		}

		//handles[i].first = leaves[i];
		handle.Vao_ = 0;
		handle.Vbo_ = 0;
		handle.VertexCount_ = verticesInLeaf.size();

		if (handle.VertexCount_ > 0)
		{
			auto positionsSize = verticesInLeaf.size() * sizeof(T);

			//create vbo to contain the vertices of the leaf
			GLuint vbo = 0;
			glGenBuffers(1, &vbo);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(GL_ARRAY_BUFFER, positionsSize, static_cast<GLvoid*>(verticesInLeaf.data()), GL_DYNAMIC_DRAW);

			//create a vao containing the vbo
			GLuint vao = 0;
			glGenVertexArrays(1, &vao);
			glBindVertexArray(vao);

			//bind vbo to vao
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glVertexAttribPointer(0, sizeof(T) / sizeof(scalar_type), detail::scalar_traits<scalar_type>::value, GL_FALSE, 0, static_cast<GLvoid*>(0));

			glEnableVertexAttribArray(0);


			glBindVertexArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			handle.Vao_ = vao;
			handle.Vbo_ = vbo;
			handle.VertexCount_ = verticesInLeaf.size();
		}

	}

	template <typename T>
	void MakeBuffer(__int64 max_node_memory_consumption, BufferHandle& handle, T vert)
	{
		typedef typename T::value_type scalar_type;

		static_assert(sizeof(T) % sizeof(scalar_type) == 0,
			"MakeBuffer can only be used with structures consisting of integral multiples of their value_type");
		static_assert(sizeof(T) >= sizeof(scalar_type),
			"MakeBuffer can only be used with structures consisting of at least one of their value_type");



		//get the vertices that correspond to this leaf


		//handles[i].first = leaves[i];
		handle.Vao_ = 0;
		handle.Vbo_ = 0;
		handle.VertexCount_ = 0;


			auto positionsSize = max_node_memory_consumption;

			//create vbo to contain the vertices of the leaf
			GLuint vbo = 0;
			glGenBuffers(1, &vbo);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(GL_ARRAY_BUFFER, positionsSize, NULL, GL_DYNAMIC_DRAW);

			//create a vao containing the vbo
			GLuint vao = 0;
			glGenVertexArrays(1, &vao);
			glBindVertexArray(vao);

			//bind vbo to vao
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glVertexAttribPointer(0, sizeof(T) / sizeof(scalar_type), detail::scalar_traits<scalar_type>::value, GL_FALSE, 0, static_cast<GLvoid*>(0));

			glEnableVertexAttribArray(0);


			glBindVertexArray(0);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			handle.Vao_ = vao;
			handle.Vbo_ = vbo;
			handle.VertexCount_ = 0;
		

	}

#endif

	std::vector<glm::vec3> CreateBoxVertexPositions(float boxExtent, glm::vec3 center);

} // namespace Gl
} // namespace Helpers
#endif // GL_HELPERS_H_