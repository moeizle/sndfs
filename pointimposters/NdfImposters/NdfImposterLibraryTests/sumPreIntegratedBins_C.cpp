#version 430

uniform ivec2 histogramDiscretizations;

int HistogramWidth = histogramDiscretizations.x;
int HistogramHeight = histogramDiscretizations.y;

uniform int binDiscretizations;

layout(std430) buffer superPreIntegratedBins
{
	float bins[];
};

layout(std430) buffer preIntegratedBins
{
	float binColor[]; // This is the important name (in the shader).
};

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()
{
	//for (int i = 0; i < histogramResolution.x*histogramResolution.y; i++)
	//{
	int i = int(gl_GlobalInvocationID.x);
	int k = int(gl_GlobalInvocationID.y);
	
	binColor[i * 3 + k] = 0;


	for (int j = 0; j < binDiscretizations*binDiscretizations; j++)
	{
		binColor[i * 3 + k] += bins[(i*binDiscretizations*binDiscretizations * 3) + j * 3 + k];
	}
	//}
}