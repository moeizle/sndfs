#version 430


in vec4 inPosition;

void main()
{
	gl_Position = vec4(inPosition.xyz, inPosition.w);
}