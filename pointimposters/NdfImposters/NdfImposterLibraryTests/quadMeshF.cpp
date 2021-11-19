#version 430

// NOTE: always use Quad since point sprites do not work with sub-pixel rays
#define QUAD
//#define DEPTH

#ifdef QUAD

#endif

layout(location = 0) out vec4 outFragColor;


uniform float far;
uniform float near;
uniform float particleScale;
uniform float tileW;
uniform float tileH;

uniform int samplingRunIndex;
uniform int maxSamplingRuns;

uniform int plainRayCasting;
uniform int pointCloudRendering;

uniform int buildingHOM;



void main()
{
	outFragColor = vec4(0.0, gl_FragCoord.z, 0.0, 1.0);
}