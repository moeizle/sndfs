#version 430

in vec2 texCoords;

out vec4 outColor;

uniform sampler2D normalTransferSampler;
//uniform sampler2D chromeTexture;
uniform sampler2D ndfExplorerTexture;

uniform vec3 lightViewSpace;
uniform int renderMode;
uniform int activej;
uniform float specularExp;

const float PI = 3.141592f;
const float toRadiants = PI / 180.0f;

uniform int ndfOverlayMode;
uniform int colorMapSize;

layout(binding = 0, rgba32f) uniform image2D tex;

layout(std430) buffer colorMap
{
	float color_map[]; // This is the important name (in the shader).
};

mat3 rotationMatrix(vec3 axis, float angle) 
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0f - c;
    
    return mat3(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c);
}

vec3 blinnPhong(vec3 normal, vec3 light, vec3 view, vec3 diffuseColor, vec3 specularColor, float specularExponent) 
{
	vec3 halfVector = normalize(light + view);

	float diffuseIntensity = max(0.0f, -dot(normal, -light));
	float diffuseWeight = 1.0f;

	float nDotHalf = abs(dot(normal, halfVector));
	float specularIntensity = max(0.0f, pow(nDotHalf, specularExponent));
	float specularWeight = 1.0f;

	float ambientWeight = 0.0f;
	const float ambientIntensity = 0.0f;//0.25f;
	const vec3 ambientColor = vec3(1.0f, 0.0f, 0.0f);
	return(diffuseWeight* diffuseIntensity * diffuseColor +specularWeight* specularIntensity * specularColor + ambientIntensity * ambientColor*ambientWeight)/(diffuseWeight + specularWeight + ambientWeight);
}

void main() 
{
	//lightViewSpace.y = -lightViewSpace.y;
	vec3 viewSpaceLightDir = lightViewSpace;
	//viewSpaceLightDir.x = -viewSpaceLightDir.x;
	//viewSpaceLightDir.y = -viewSpaceLightDir.y;
	viewSpaceLightDir.z *= -1.0f;

	vec3 normal = vec3(texCoords.xy, 0.0f) * 2.0f - 1.0f;
	//vec3 normal = vec3(texCoords.xy, 0.0f) - 0.5f;
	normal.z = sqrt(1.0f - normal.x*normal.x - normal.y*normal.y);

	//viewSpaceLightDir = -1.0f*viewSpaceLightDir;
	if (ndfOverlayMode > 0)
	{
		vec2 quantizedRay = vec2(texCoords.x * 8.0f, texCoords.y * 8.0f);
#if 1 //color map
		int k = int(quantizedRay.y) * 8 + int(quantizedRay.x);
		float factor = colorMapSize / 64.0f;
		vec3 kColor = vec3(color_map[3 * int(k*factor)], color_map[3 * int(k*factor) + 1], color_map[3 * int(k*factor) + 2]);
		outColor = vec4(kColor, 1.0f);
#else
		const vec3 leftColor = vec3(.5, 0, 0);// vec3(0.35f, 0.65f, 0.8f);
		const vec3 rightColor = vec3(0f, 0.5f, 0f);
		const vec3 bottomColor = vec3(0.5f, 0.5f, 0.5f);
		const vec3 topColor = vec3(0.0f, 0.0f, 0.5f);

		outColor = vec4(0.5f * leftColor * (1.0f - texCoords.x) + 0.5f * rightColor * texCoords.x +
			0.5f * bottomColor * (1.0f - texCoords.y) + 0.5f * topColor * texCoords.y, 1.0f);
#endif
	}
	else
	{
		if (renderMode == 0)
		{
			vec3 shading = blinnPhong(normal, viewSpaceLightDir, vec3(0.0f, 0.0f, 1.0f), vec3(1.0f, 1.0f, 1.0f), vec3(1.0f, 1.0f, 1.0f), specularExp);
			outColor = vec4(shading.xyz, 1.0f);
		}
		else
		{
			// render analytic transfer function with current rotation
			vec2 transfer = normal.xy;
			//outColor = vec4(normal.xyz, 1.0f) * 0.5f + 0.5f;
			if (renderMode == 1)
			{
#if 0 //old
				vec3 diffuseColor;
				vec3 specularColor;

				const vec3 leftColor = vec3(.5, 0, 0);// vec3(0.35f, 0.65f, 0.8f);
				const vec3 rightColor = vec3(0f, 0.5f, 0f);
				const vec3 bottomColor = vec3(0.5f, 0.5f, 0.5f);
				const vec3 topColor = vec3(0.0f, 0.0f, 0.5f);

				mat3 lightRotationZ = rotationMatrix(vec3(0.0f, 0.0f, 1.0f), viewSpaceLightDir.x);
				mat3 lightRotationY = rotationMatrix(vec3(0.0f, 1.0f, 0.0f), viewSpaceLightDir.y);

				transfer = (lightRotationZ * vec3(transfer.xy, 1.0f)).xy;
				transfer = (lightRotationY * vec3(transfer.xy, 1.0f)).xy;

				diffuseColor = 0.5f * leftColor * (1.0f - transfer.x) + 0.5f * rightColor * transfer.x +
					0.5f * bottomColor * (1.0f - transfer.y) + 0.5f * topColor * transfer.y;
				specularColor = diffuseColor;

				outColor = vec4(diffuseColor.xyz, 1.0f);
#else
				//outColor = vec4(texture(chromeTexture, texCoords.xy).xyz, 1.0f);
				//outColor = vec4(texture(ndfExplorerTexture, texCoords.xy).xyz, 1.0f);
				outColor=vec4(imageLoad(tex, ivec2(512*texCoords.xy)).xyz,1.0f);
#endif
			}
			else if (renderMode == 2)
			{
				vec3 transformedNormal;
				/*mat3 lightRotationZ = rotationMatrix(vec3(0.0f, -1.0f, 0.0f), viewSpaceLightDir.x);
				mat3 lightRotationY = rotationMatrix(vec3(1.0f, 0.0f, 0.0f), viewSpaceLightDir.y);

				vec3 transformedNormal = normal;
				transformedNormal = (lightRotationZ * transformedNormal.xyz).xyz;
				transformedNormal = (lightRotationY * transformedNormal.xyz).xyz;
				const float seamCorrection = 0.125f;
				transformedNormal.xy -= vec2(transformedNormal.xy * seamCorrection);

				transfer.xy = transformedNormal.xy * 0.5f + 0.5f;

				vec3 lookupColor = texture(normalTransferSampler, vec2(transfer.x, 1.0f - transfer.y)).xyz;

				outColor = vec4(lookupColor.xyz, 1.0f);
				*/


				{
					//normal = normalize(normal);
					//viewSpaceLightDir = normalize(viewSpaceLightDir);
					/*vec3 a = cross(normal, viewSpaceLightDir);
					a = normalize(a);

					float theta = acos(dot(normal,viewSpaceLightDir)/(length(normal)*length(viewSpaceLightDir)));

					mat3 R = rotationMatrix(a, theta);

					transformedNormal = R*normal;

					const float seamCorrection = 0.125f;
					transformedNormal.xy -= vec2(transformedNormal.xy * seamCorrection);

					transfer.xy = transformedNormal.xy * 0.5f + 0.5f;

					vec3 lookupColor = texture(normalTransferSampler, vec2(transfer.x, transfer.y)).xyz;*/

					mat3 lightRotationZ = rotationMatrix(vec3(0.0f, -1.0f, 0.0f), viewSpaceLightDir.x);
					mat3 lightRotationY = rotationMatrix(vec3(1.0f, 0.0f, 0.0f), viewSpaceLightDir.y);

					transformedNormal = normal;
					transformedNormal = (lightRotationZ * transformedNormal.xyz).xyz;
					transformedNormal = (lightRotationY * transformedNormal.xyz).xyz;

					const float seamCorrection = 0.125f;
					transformedNormal.xy -= vec2(transformedNormal.xy * seamCorrection);
					vec2 lookupCoord = 1.0f - (transformedNormal.xy * 0.5f + 0.5f);

					vec3 lookupColor = texture(normalTransferSampler, lookupCoord).xyz;

					//diffuseColor = lookupColor.xyz;
					//specularColor = diffuseColor;

					outColor = vec4(lookupColor.xyz, 1.0f);

				}


			}
		}
	}
	/*const float energyCorrection = 1.2f;
	const float constrast = 2.0f;
	const float brightness = 1.0f;
	outColor.xyz = brightness.xxx * pow(outColor.xyz * energyCorrection.xxx, constrast.xxx);*/
	
	const vec3 backgroundColor = 0.0f * vec3(1.0f, 1.0f, 1.0f);
	const vec3 highlightColor = vec3(0.0f, 1.0f, 0.0f);
	if(activej > 0) 
	{
		// anti aliasing around the edge
		const float smoothness = 0.06f;
		float border = max(0.0f, normal.x*normal.x + normal.y*normal.y - (1.0f - smoothness));
		if(border > smoothness) {
			discard;
		}
		float opacity = smoothstep(smoothness, 0.0f, border);

		outColor.xyz = mix(highlightColor, outColor.xyz, opacity);
	} 
	else 
	{
		// anti aliasing around the edge
		const float smoothness = 0.06f;
		float border = max(0.0f, normal.x*normal.x + normal.y*normal.y - (1.0f - smoothness));
		if(border > smoothness) 
		{
			discard;
		}
		float opacity = smoothstep(smoothness, 0.0f, border);

		outColor.xyz = mix(backgroundColor, outColor.xyz, opacity);
	}


}