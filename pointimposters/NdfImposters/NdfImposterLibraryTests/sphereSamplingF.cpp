#version 430

// NOTE: always use Quad since point sprites do not work with sub-pixel rays
#define QUAD
//#define DEPTH

#ifdef QUAD
in vec2 texCoord;
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

	//if ( gl_FragCoord.x < 500 || gl_FragCoord.y<100)
	//{
	//	outFragColor = vec4(1, 0, 0, 0);
	//	return;
	//}

	vec2 coord = vec2(texCoord.x,texCoord.y);

	vec2 transformedCoord = coord.xy * 2.0f - 1.0f;

	float l2 = dot(transformedCoord, transformedCoord); // length(transformedCoord);
	if (pointCloudRendering==0 && l2 > 1.0f)
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


	//send the normals for descirtization step
	if (pointCloudRendering == 0)
	{
		outFragColor = vec4(coord.x, coord.y, 0.0f, 1.0f);

		//debug
		//if (gl_PrimitiveID>250000)
		//	outFragColor = vec4(1,0,0,1.0);
		//else
			//outFragColor = vec4(gl_PrimitiveID, 0, 0, 1.0);
		//end debug
#if 0
		int x, y, z, w;
		int Index = gl_PrimitiveID+1;
		int D1, D2, D3, D4;
		D1 = D2 = D3 = D4 = 1000;

		x = Index % D1;
		y = (Index / D1)%D2;
		z = Index / (D1*D2);
		//z = ((Index - y * D1 - x) / (D1 * D2)) % D3;
		//w = ((Index - z * D2 * D1 - y * D1 - x) / (D1 * D2 * D3)) % D4;
		outFragColor = vec4(x / float(D1), y / float(D2), z/float(D3), 1.0);
#endif
	}
	else
	{
		outFragColor = vec4(1, 1, 1, 1);
	}

	//if ((gl_FragCoord.x < 100) && (gl_FragCoord.y < 100))
	//{
	//	outFragColor = vec4(1, 0, 0, 1);
	//}
}