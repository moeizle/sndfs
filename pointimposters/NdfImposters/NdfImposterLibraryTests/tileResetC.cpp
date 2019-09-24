#version 430

uniform ivec2 histogramDiscretizations;

int HistogramWidth = histogramDiscretizations.x;
int HistogramHeight = histogramDiscretizations.y;

uniform int phys_tex_dim;

uniform int firstTile;

uniform int tile_w;

layout(std430) buffer NdfVoxelData
{
	coherent float histograms[];
} ndfVoxelData;

layout(std430) buffer sampleCount
{
	float sample_count[]; // This is the important name (in the shader).
};

//layout(std430) buffer circularSampleCount
//{
//	float circular_sample_count[]; // This is the important name (in the shader).
//};

//layout(std430, binding = 3) buffer regionColor
//{
//	float region_color[]; // This is the important name (in the shader).
//};

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()
{
	//old
	//int r = int(gl_GlobalInvocationID.x);
	//int c = int(gl_GlobalInvocationID.y);
	//int k = int(gl_GlobalInvocationID.z);

	//int ind = ((phys_y + r)* phys_tex_dim +(phys_x + c)) *HistogramWidth*HistogramHeight + k;
	//end old

	//int r = int(gl_GlobalInvocationID.x);
	//int c = int(gl_GlobalInvocationID.y);
	//int k = int(gl_GlobalInvocationID.z);

	//int ind = ((phys_y + gl_GlobalInvocationID.x)* phys_tex_dim + phys_x + gl_GlobalInvocationID.y) *HistogramWidth*HistogramHeight + gl_GlobalInvocationID.z;


	//ndfVoxelData.histograms[((phys_y + gl_GlobalInvocationID.x)* phys_tex_dim + phys_x + gl_GlobalInvocationID.y) *HistogramWidth*HistogramHeight + gl_GlobalInvocationID.z] = 0;
	//sample_count[((phys_y + gl_GlobalInvocationID.x)* phys_tex_dim + phys_x + gl_GlobalInvocationID.y)] = 0;

	int tilePixelCoord = firstTile;

	tilePixelCoord += int(gl_GlobalInvocationID.y);
	tilePixelCoord *= tile_w*tile_w;

	ndfVoxelData.histograms[(tilePixelCoord + gl_GlobalInvocationID.x)*HistogramWidth*HistogramHeight + gl_GlobalInvocationID.z] = 0;
	sample_count[(tilePixelCoord + gl_GlobalInvocationID.x)] = 0;
	//circular_sample_count[(tilePixelCoord + gl_GlobalInvocationID.x)] = 0;

	//region_color[3 * (tilePixelCoord + gl_GlobalInvocationID.x)  ] = 1;
	//region_color[3 * (tilePixelCoord + gl_GlobalInvocationID.x)+1] = 1;
	//region_color[3 * (tilePixelCoord + gl_GlobalInvocationID.x)+2] = 1;

}