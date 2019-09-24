#version 420

in vec2 Position;

out vec2 texCoords;

void main() 
{
	
	gl_Position = vec4(Position.xy, 0.0f ,1.0f);
	gl_Position.x /=(.5f* 1280.0f);
	//gl_Position.x = gl_Position.x;
	gl_Position.y /= (.5f*720.0f);
	

	gl_Position.x -= 1f;
	gl_Position.y -= 1f;
	gl_Position.y *= -1.0f;

	texCoords = vec2(1.0f-Position.x/1280.0f,Position.y/720.0f); //(gl_Position.xy + 1.0f) * 0.5f;
}