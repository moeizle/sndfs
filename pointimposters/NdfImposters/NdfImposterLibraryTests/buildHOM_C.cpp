#version 430

layout(binding = 0, rgba32f) uniform volatile image2D floorLevel;
layout(binding = 1, rgba32f) uniform image2D ceilLevel;


// has to be recompiled each time the size changes - consider the local size division
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()
{
	ivec2 ceilLocation = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);

	ivec2 floorLocation = 2*ceilLocation;
	float ceilVal = imageLoad(floorLevel, floorLocation).x;

#if 1  //max of children
	floorLocation = 2 * ceilLocation + ivec2(1, 0);
	ceilVal = max(ceilVal,imageLoad(floorLevel, floorLocation).x);
	
	floorLocation = 2 * ceilLocation + ivec2(0, 1);
	ceilVal = max(ceilVal, imageLoad(floorLevel, floorLocation).x);

	floorLocation = 2 * ceilLocation + ivec2(1, 1);
	ceilVal = max(ceilVal, imageLoad(floorLevel, floorLocation).x);
#endif

#if 0  //average of children
	floorLocation = 2 * ceilLocation + ivec2(1, 0);
	ceilVal += imageLoad(floorLevel, floorLocation).x;

	floorLocation = 2 * ceilLocation + ivec2(0, 1);
	ceilVal +=imageLoad(floorLevel, floorLocation).x;

	floorLocation = 2 * ceilLocation + ivec2(1, 1);
	ceilVal +=imageLoad(floorLevel, floorLocation).x;

	ceilVal *= 0.25f;
#endif

	imageStore(ceilLevel, ceilLocation, vec4(ceilVal, 0, 0, 1.0f));
}