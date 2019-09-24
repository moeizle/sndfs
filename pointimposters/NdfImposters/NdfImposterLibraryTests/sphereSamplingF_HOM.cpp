#version 430

// NOTE: always use Quad since point sprites do not work with sub-pixel rays
#define QUAD
//#define DEPTH

#ifdef QUAD
in vec2 texCoord;
#endif

in int gl_PrimitiveID;
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

	//if ( gl_FragCoord.x < 500 || gl_FragCoord.y<100)
	//{
	//	outFragColor = vec4(1, 0, 0, 0);
	//	return;
	//}

	vec2 coord = vec2(texCoord.x, texCoord.y);

	vec2 transformedCoord = coord.xy * 2.0f - 1.0f;

	float l2 = dot(transformedCoord, transformedCoord); // length(transformedCoord);
	if (pointCloudRendering == 0 && l2 > 1.0f)
	{
		discard;
	}

	//#ifndef QUAD
	/*float relativeDepth = l;
	relativeDepth /= sqrt(2.0f);
	relativeDepth *= relativeDepth;
	relativeDepth /= (far - near);*/

	float fragmentDepthOffset = 0;// sqrt(1.0 - l*l);

	//const float magicConstantIDontRememberTheMeaningOf = 0.0005f;// 0.001f;
	const float magicConstantIDontRememberTheMeaningOf = 0.001f;
	float outDepth = float(gl_FragCoord.z) + fragmentDepthOffset;// *magicConstantIDontRememberTheMeaningOf;
	//float outDepth = float(gl_FragCoord.z) + relativeDepth * 0.001f;
	//gl_FragDepth = outDepth;
	//#endif

	//vec3 normal = normalize(vec3(coord.x, coord.y, sqrt(1.0f - coord.x*coord.x - coord.y*coord.y)));

	//vec2 transformedNormal = vec2(coord.x, 1.0f - coord.y);


	outFragColor = vec4(outDepth,0,0,1.0f);

	//if ((gl_FragCoord.x < 100) && (gl_FragCoord.y < 100))
	//{
	//	outFragColor = vec4(1, 0, 0, 1);
	//}
}