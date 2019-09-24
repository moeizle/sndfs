#version 430

in vec2 texCoords;

out vec4 outColor;

layout(binding = 1, rgba32f) uniform image2D homTexture;


layout(depth_any) out float gl_FragDepth;

void main()
{
	vec4 lookupColor = imageLoad(homTexture, ivec2(gl_FragCoord.x,gl_FragCoord.y));

	
	gl_FragDepth = lookupColor.x;
	outColor = vec4(gl_FragDepth,0,0,1.0f);
}