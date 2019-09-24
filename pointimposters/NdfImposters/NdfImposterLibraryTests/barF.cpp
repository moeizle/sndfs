#version 430

in vec2 texCoords;

out vec4 outColor;

uniform sampler2D barSampler;




void main()
{
	vec3 lookupColor = texture(barSampler, texCoords).xyz;

	outColor = vec4(lookupColor.xyz, 1.0f);

	//outColor = vec4(1,0,0, 1.0f);
}