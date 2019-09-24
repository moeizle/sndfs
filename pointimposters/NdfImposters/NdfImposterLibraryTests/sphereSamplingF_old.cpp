#version 420

// NOTE: always use Quad since point sprites do not work with sub-pixel rays
#define QUAD
//#define DEPTH

#ifdef QUAD
in vec2 texCoord;
#endif


layout(location = 0) out vec4 outFragColor0;
layout(location = 1) out vec4 outFragColor1;
layout(location = 2) out vec4 outFragColor2;
layout(location = 3) out vec4 outFragColor3;
layout(location = 4) out vec4 outFragColor4;
layout(location = 5) out vec4 outFragColor5;
layout(location = 6) out vec4 outFragColor6;
layout(location = 7) out vec4 outFragColor7;

uniform float far;
uniform float near;
uniform float particleSize;
uniform float tileW;
uniform float tileH;

uniform int samplingRunIndex;
uniform int maxSamplingRuns;

uniform int ttor;
uniform vec3 tc_0;
uniform vec3 tc_1;
uniform vec3 tc_2;
uniform vec3 tc_3;
uniform vec3 tc_4;
uniform vec3 tc_5;
uniform vec3 tc_6;
uniform vec3 tc_7;

void main() 
{
#ifdef QUAD
	vec2 coord = texCoord.xy;
#else
	vec2 coord = gl_PointCoord.xy;
#endif

	vec2 transformedCoord = coord.xy * 2.0f - 1.0f;

	float depth = length(transformedCoord);
	if(depth > 1.0f)
	{
#ifdef DEPTH
		outFragColor = vec4(0.0f, 0.0f, float(gl_FragCoord.z), 1.0f);
#endif

		discard;
	}

#if 0 // filter out everything but top and bottom
	const float threshold = 0.85f;
	if(abs(transformedCoord.y) < threshold) 
	{
		outFragColor = vec4(0.0f, 0.0f, float(gl_FragCoord.z), 1.0f);
		discard;
	}
#endif

//#ifndef QUAD
	float relativeDepth = depth;
	relativeDepth /= sqrt(2.0f);
	relativeDepth *= relativeDepth;
	relativeDepth /= (far - near);

	//const float magicConstantIDontRememberTheMeaningOf = 0.0005f;// 0.001f;
	const float magicConstantIDontRememberTheMeaningOf = 0.001f;
	float outDepth = float(gl_FragCoord.z) + relativeDepth * magicConstantIDontRememberTheMeaningOf;
	//float outDepth = float(gl_FragCoord.z) + relativeDepth * 0.001f;
	gl_FragDepth = outDepth;
//#endif

	//vec3 normal = normalize(vec3(coord.x, coord.y, sqrt(1.0f - coord.x*coord.x - coord.y*coord.y)));

	vec2 transformedNormal = vec2(coord.x, 1.0f - coord.y);

	//float hw = tileW / 2.0;
	//float hh = tileH / 2.0;
	//
	//if (ttor > 7)
	//{
	//	if ((gl_FragCoord.x >= tc_7.x - hw) && (gl_FragCoord.x < tc_7.x + hw) && (gl_FragCoord.y >= tc_7.y - hh) && (gl_FragCoord.y < tc_7.y + hh))
	//		outFragColor7 = vec4(transformedNormal.xy, 0.0f, 1.0f);
	//}
	//if (ttor > 6)
	//{
	//	if ((gl_FragCoord.x >= tc_6.x - hw) && (gl_FragCoord.x < tc_6.x + hw) && (gl_FragCoord.y >= tc_6.y - hh) && (gl_FragCoord.y < tc_6.y + hh))
	//		outFragColor6 = vec4(transformedNormal.xy, 0.0f, 1.0f);
	//}
	//if (ttor > 5)
	//{
	//	if ((gl_FragCoord.x >= tc_5.x - hw) && (gl_FragCoord.x < tc_5.x + hw) && (gl_FragCoord.y >= tc_5.y - hh) && (gl_FragCoord.y < tc_5.y + hh))
	//		outFragColor5 = vec4(transformedNormal.xy, 0.0f, 1.0f);
	//}
	//if (ttor > 4)
	//{
	//	if ((gl_FragCoord.x >= tc_4.x - hw) && (gl_FragCoord.x < tc_4.x + hw) && (gl_FragCoord.y >= tc_4.y - hh) && (gl_FragCoord.y < tc_4.y + hh))
	//		outFragColor4 = vec4(transformedNormal.xy, 0.0f, 1.0f);
	//}
	//if (ttor > 3)
	//{
	//	if ((gl_FragCoord.x >= tc_3.x - hw) && (gl_FragCoord.x < tc_3.x + hw) && (gl_FragCoord.y >= tc_3.y - hh) && (gl_FragCoord.y < tc_3.y + hh))
	//		outFragColor3 = vec4(transformedNormal.xy, 0.0f, 1.0f);
	//}

	//if (ttor > 2)
	//{
	//	if ((gl_FragCoord.x >= tc_2.x - hw) && (gl_FragCoord.x < tc_2.x + hw) && (gl_FragCoord.y >= tc_2.y - hh) && (gl_FragCoord.y < tc_2.y + hh))
	//		outFragColor2 = vec4(transformedNormal.xy, 0.0f, 1.0f);
	//}

	//if (ttor > 1)
	//{
	//	if ((gl_FragCoord.x >= tc_1.x - hw) && (gl_FragCoord.x < tc_1.x + hw) && (gl_FragCoord.y >= tc_1.y - hh) && (gl_FragCoord.y < tc_1.y + hh))
	//		outFragColor1 = vec4(transformedNormal.xy, 0.0f, 1.0f);
	//}
	//
	//if (ttor > 0)
	//{
	//	if ((gl_FragCoord.x >= tc_0.x - hw) && (gl_FragCoord.x < tc_0.x + hw) && (gl_FragCoord.y >= tc_0.y - hh) && (gl_FragCoord.y < tc_0.y + hh))
	//	{
			outFragColor0 = vec4(transformedNormal.xy, 0.0f, 1.0f);
	//	}
	//}
	
	//else
	//	discard;

#ifdef DEPTH
	outFragColor.z = outDetph;
#endif
}