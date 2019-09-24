#version 420

uniform mat4 MVP;
uniform mat4 MV;
uniform mat4 M;
uniform mat4 ViewProjection;
uniform mat4 Projection;

uniform float aspectRatio;

uniform vec3 right;
uniform vec3 up;

in vec3 Position;

out vec2 texCoords;
out vec3 rayBegin;
out vec3 frustumTarget;
out vec3 viewSpacePosition;
out vec3 worldSpacePosition;

void main() {
	// TODO: does not account for persepctive projection
	//vec3 viewAlignedPosition = Position.x * right + Position.y * up;
	vec3 viewAlignedPosition = Position.xyz;

	viewSpacePosition = (MV * vec4(viewAlignedPosition.xyz, 1.0f)).xyz;
	worldSpacePosition = (M * vec4(viewAlignedPosition.xyz, 1.0f)).xyz;
	
	//gl_Position = MVP * vec4(viewAlignedPosition.xyz, 1.0f);
	gl_Position = vec4(Position.xyz, 1.0f);

	// NOTE: used for NDF output
	//gl_Position = vec4(ViewProjection * vec4(viewAlignedPosition.xyz, 1.0f));
	//gl_Position = vec4(Projection * vec4(viewAlignedPosition.xyz, 1.0f));

	texCoords = (Position.xy + 1.0f) * 0.5f;
	rayBegin = (Position.xyz + 1.0f) * 0.5f;
	rayBegin.z = 0.0f;//1.0f - rayBegin.z;

	float invAspectRatio = 1.0f / aspectRatio;
	rayBegin.y = invAspectRatio - rayBegin.y * invAspectRatio;
	rayBegin.x = 1.0f - rayBegin.x;
}