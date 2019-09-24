#version 420

in vec3 Position;

out vec2 texCoords;

void main() {
	gl_Position = vec4(Position.xyz, 1.0f);

	vec2 val = (Position.xy + 1.0f) * 0.5f;
	val.y = 1.0f - val.y;
	texCoords = val;
}