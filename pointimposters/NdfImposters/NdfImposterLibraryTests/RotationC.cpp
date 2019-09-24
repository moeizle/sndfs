#version 430

#extension GL_ARB_arrays_of_arrays : enable

uniform mat3 R;


layout(std430, binding = 0) buffer verts
{
	 float Positions[];
};


layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()
{
	int x = int(gl_GlobalInvocationID.x);

	vec3 i, f;
	i = vec3(Positions[x * 3], Positions[x * 3 + 1], Positions[x * 3 + 2]);
	f = R*i;

	Positions[x * 3] = f.x;
	Positions[x * 3 + 1] = f.y;
	Positions[x * 3 + 2] = f.z;
	
}