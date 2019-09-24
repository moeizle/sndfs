#version 430

in vec2 texCoords;

out vec4 outColor;

//layout(binding = 1, rgba32f) uniform image2D depthMap;




void main()
{
	outColor = vec4(gl_FragCoord.z, 0, 0, 1.0f);
}