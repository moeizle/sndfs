#version 430

uniform mat4 ViewProjection;
uniform mat4 Projection;
uniform mat4 ModelView;
uniform mat4 View;
uniform mat4 Model;
uniform mat4 ViewAlignmentMatrix;
uniform mat3 RotationMatrix;

uniform vec3 ViewPosition;
uniform vec3 modelOffset;

uniform float far;
uniform float near;
uniform float particleScale;
uniform float tileW;
uniform float tileH;

uniform float viewportWidth;

uniform int samplingRunIndex;
uniform int maxSamplingRuns;

uniform int pointCloudRendering;


uniform vec3 right;
uniform vec3 up;

layout(lines_adjacency) in;
layout(triangle_strip, max_vertices = 4) out;

void main() {
	

	mat4 M = Projection*ModelView;
	{
		vec3 objectSpacePosition = RotationMatrix * gl_in[0].gl_Position.xyz;
		gl_Position = vec4(objectSpacePosition.xyz , 1.0f);
		gl_Position = vec4(M * gl_Position);
		EmitVertex();
	}
	{
		vec3 objectSpacePosition = RotationMatrix * gl_in[1].gl_Position.xyz;
		gl_Position = vec4(objectSpacePosition.xyz, 1.0f);
		gl_Position = vec4(M * gl_Position);
		EmitVertex();
	}
	{
		vec3 objectSpacePosition = RotationMatrix * gl_in[2].gl_Position.xyz;
		gl_Position = vec4(objectSpacePosition.xyz, 1.0f);
		gl_Position = vec4(M * gl_Position);
		EmitVertex();
	}
	{
		vec3 objectSpacePosition = RotationMatrix * gl_in[3].gl_Position.xyz;
		gl_Position = vec4(objectSpacePosition.xyz, 1.0f);
		gl_Position = vec4(M * gl_Position);
		EmitVertex();
	}


	gl_PrimitiveID = gl_PrimitiveIDIn;
	EndPrimitive();
}