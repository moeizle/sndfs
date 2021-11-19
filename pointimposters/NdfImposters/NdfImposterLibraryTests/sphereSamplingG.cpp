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

in vec2 rayOffset[1];

out vec2 texCoord;

layout(points) in;
layout(triangle_strip, max_vertices = 4) out;

void main() {
	vec3 objectSpacePosition = RotationMatrix * gl_in[0].gl_Position.xyz;
	vec2 offset = rayOffset[0].xy;
	float radius = gl_in[0].gl_Position.w*particleScale;
	mat4 M = Projection*ModelView;
	{
		gl_Position = vec4(objectSpacePosition.xyz - radius*(right + up), 1.0f);
		gl_Position = vec4(M * gl_Position)+2.0*vec4(offset,0,0);
		texCoord = vec2(0.0f, 0.0f);
		EmitVertex();
		/*
		gl_Position = vec4(objectSpacePosition, 1.0f);
		gl_Position.xyz -= right * radius;
		gl_Position.xyz -= up * radius;
		texCoord = vec2(0.0f, 0.0f);
		
		vec4 viewSpacePosition = ModelView * gl_Position;
		viewSpacePosition.xy += offset.xy;
		gl_Position = vec4(Projection * viewSpacePosition);
		EmitVertex();
		*/
	}

	{
		gl_Position = vec4(objectSpacePosition.xyz + radius*(right - up), 1.0f);
		gl_Position = vec4(M * gl_Position) + 2.0*vec4(offset, 0, 0);
		texCoord = vec2(1.0f, 0.0f);
		EmitVertex();
		/*
		gl_Position = vec4(objectSpacePosition, 1.0f);
		gl_Position.xyz += right * radius;
		gl_Position.xyz -= up * radius;
		texCoord = vec2(1.0, 0.0f);

		vec4 viewSpacePosition = ModelView * gl_Position;
		viewSpacePosition.xy += offset.xy;
		gl_Position = vec4(Projection * viewSpacePosition);
		EmitVertex();
		*/
	}
   
	{
		gl_Position = vec4(objectSpacePosition.xyz + radius*(up - right), 1.0f);
		gl_Position = vec4(M * gl_Position) + 2.0*vec4(offset, 0, 0);
		texCoord = vec2(0.0f, 1.0f);
		EmitVertex();
		/*
		gl_Position = vec4(objectSpacePosition, 1.0f);
		gl_Position.xyz -= right * radius;
		gl_Position.xyz += up * radius;
		texCoord = vec2(0.0f, 1.0);

		vec4 viewSpacePosition = ModelView * gl_Position;
		viewSpacePosition.xy += offset.xy;
		gl_Position = vec4(Projection * viewSpacePosition);
		EmitVertex();
		*/
	}

	{
		gl_Position = vec4(objectSpacePosition.xyz + radius*(right + up), 1.0f);
		gl_Position = vec4(M * gl_Position) + 2.0*vec4(offset, 0, 0);
		texCoord = vec2(1.0f, 1.0f);
		EmitVertex();
		/*
		gl_Position = vec4(objectSpacePosition, 1.0f);
		gl_Position.xyz += right * radius;
		gl_Position.xyz += up * radius;
		texCoord = vec2(1.0, 1.0);

		vec4 viewSpacePosition = ModelView * gl_Position;
		viewSpacePosition.xy += offset.xy;
		gl_Position = vec4(Projection * viewSpacePosition);
		EmitVertex();
		*/
	}
	gl_PrimitiveID = gl_PrimitiveIDIn;
	EndPrimitive();
}  