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
uniform float viewportHeight;

uniform int highestSampleCount;
uniform int maxSamplingRuns;
uniform int samplescount;
uniform int sampleIndex;

uniform vec2 sPos;


uniform vec3 right;
uniform vec3 up;
uniform vec2 samplePos;

uniform float objPerPixel;

in vec4 inPosition;




void main()
{
	gl_Position = vec4(inPosition.xyz, inPosition.w);
}