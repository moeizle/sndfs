#ifndef GLH_SINGLE_H_
#define GLH_SINGLE_H_

#include <glh/glh_extensions.h>

#include <array>
#include <string>
#include <iostream>

namespace Helpers {

namespace {
	static std::array< std::string, 17 > const extensions_ = {
		"GL_VERSION_1_2"
		, "GL_VERSION_1_3"
		, "GL_VERSION_1_5"
		, "GL_VERSION_2_0"
		, "GL_VERSION_3_0"
		, "GL_VERSION_3_1"
		, "GL_VERSION_4_3"
		, "GL_KHR_debug"
		, "GL_ARB_compute_shader"
		, "GL_ARB_framebuffer_object"
		, "GL_ARB_program_interface_query"
		, "GL_ARB_shader_image_load_store"
		, "GL_ARB_texture_storage"
		, "GL_ARB_shader_storage_buffer_object"
		//, "GL_EXT_direct_state_access"
		, "GL_ARB_sampler_objects"
		, "GL_ARB_vertex_array_object"
		, "GL_ARB_uniform_buffer_object"
		//, "GL_ARB_bindless_texture"
		//, "GL_NV_bindless_texture"
		//, "GL_NV_gpu_shader5"
		//, "GL_ARB_vertex_attrib_64bit"
		};
} // namespace

// NOTE: intialize glut before calling this function!
inline bool InitializeExtensions() {
	for(auto &&extension : extensions_) {
	    if(::glh_init_extensions(extension.data()) != GL_TRUE) {
			std::cerr << "Failed to initialize extension: " << extension.data() << std::endl;

			return false;
		}
	}

	return true;
} 

} // namespace Helpers


#endif // GLH_SINGLE_H_
