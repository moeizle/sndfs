#version 430


uniform mat4 Projection;
uniform mat4 ModelView;

out int gl_PrimitiveID;

layout(triangles) in;
layout(triangle_strip, max_vertices = 3) out;

void main() {


	mat4 M = Projection*ModelView;
	{
		vec3 objectSpacePosition =  gl_in[0].gl_Position.xyz;
		gl_Position = vec4(objectSpacePosition.xyz, 1.0f);
		gl_Position = vec4(M * gl_Position);
		EmitVertex();
	}
	{
		vec3 objectSpacePosition = gl_in[1].gl_Position.xyz;
		gl_Position = vec4(objectSpacePosition.xyz, 1.0f);
		gl_Position = vec4(M * gl_Position);
		EmitVertex();
	}
	{
		vec3 objectSpacePosition =  gl_in[2].gl_Position.xyz;
		gl_Position = vec4(objectSpacePosition.xyz, 1.0f);
		gl_Position = vec4(M * gl_Position);
		EmitVertex();
	}


	gl_PrimitiveID = gl_PrimitiveIDIn;
	EndPrimitive();
}