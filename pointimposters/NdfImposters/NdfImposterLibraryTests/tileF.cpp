#version 420

in vec2 myTexCoord;
uniform sampler2D myTexture;
out vec4 outColor;

void main()
{
	//Use myTexCoord by any way, for example, to access a texture.
	outColor = texture2D(myTexture, myTexCoord);
}