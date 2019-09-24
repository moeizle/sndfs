#version 420

in vec3 Position;

out vec2 myTexCoord;

void main()
{
	// We compute the vertex position as the fixed function does.
	gl_Position = vec4(Position.xyz, 1.0f);
	// We fill our varying variable with the texture
	//coordinate related to the texture unit 0 (gl_MultiTexCoord0 refers to the TU0
	//interpolator).
	myTexCoord = (Position.xy + 1.0f) * 0.5f;
}