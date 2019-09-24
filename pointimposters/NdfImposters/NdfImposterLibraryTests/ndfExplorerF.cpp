#version 430

uniform vec3 color;
in int gl_PrimitiveID;
layout(location = 0) out vec4 outFragColor;


void main()
{
	outFragColor = vec4(color.x, color.y, color.z, 1.0);
}