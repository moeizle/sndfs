#version 430

in vec2 texCoords;

uniform ivec2 histogramDiscretizations;
uniform int binningMode;

out vec4 outColor;

uniform sampler2D normalTransferSampler;
uniform int colorMapSize;


const float PI = 3.141592f;
const float toRadiants = PI / 180.0f;
int HistogramWidth = histogramDiscretizations.x;
int HistogramHeight = histogramDiscretizations.y;


layout(std430) buffer avgNDF
{
	float avg_NDF[]; // This is the important name (in the shader).
};

layout(std430) buffer colorMap
{
	float color_map[]; // This is the important name (in the shader).
};

layout(std430) buffer binAreas
{
	double bArea[]; // This is the important name (in the shader).
};

void main() 
{
	//discard;
	vec2 quantizedRay;
	int histogramIndexCentral, histogramIndexR, histogramIndexB, histogramIndexBR;
	dvec2 bilinearWeights;
	vec2 newRay = texCoords;
	bool miss;
	
	vec2 transformedCoord = texCoords.xy * 2.0f - 1.0f;
	float l2 = dot(transformedCoord, transformedCoord);
	if (l2 <= 10.0f)
	{

		if (binningMode == 0)
		{
			quantizedRay = vec2(texCoords.x * float(HistogramWidth), texCoords.y * float(HistogramHeight));

			// NOTE: the total amount of energy will only be available if the max. number of samples is reached.
			// before that the renderer has to upscale the ndf samples according to the total energy.8
			histogramIndexCentral = int(quantizedRay.y) * HistogramWidth + int(quantizedRay.x);
			miss = (histogramIndexCentral == 0);
			if (histogramIndexCentral < 0 || histogramIndexCentral >= HistogramHeight * HistogramWidth)
			{
				return;
			}
		}
		else if (binningMode == 1)
		{
			//spherical coordinates binning

			//get 3d normal
			vec3 sourceNormal;
			sourceNormal.xy = vec2(newRay.x, newRay.y)*2.0f - 1.0f;
			//sourceNormal.z = 1.0f - sqrt(sourceNormal.x*sourceNormal.x + sourceNormal.y*sourceNormal.y);
			sourceNormal.z = sqrt(1.0f - sourceNormal.x*sourceNormal.x - sourceNormal.y*sourceNormal.y);

			quantizedRay = vec2(newRay.x * float(HistogramWidth), newRay.y * float(HistogramHeight));

			// NOTE: the total amount of energy will only be available if the max. number of samples is reached.
			// before that the renderer has to upscale the ndf samples according to the total energy.
			histogramIndexCentral = int(quantizedRay.y) * HistogramWidth + int(quantizedRay.x);

			miss = (histogramIndexCentral == 0);

			if (histogramIndexCentral < 0 || histogramIndexCentral >= HistogramHeight * HistogramWidth)
			{
				return;
			}

			//miss = ((newRay.x == 0) && (newRay.y==0));

			if (!miss)
			{
				if (sourceNormal.z != sourceNormal.z)
					discard;


				float theta = atan(sourceNormal.y, sourceNormal.x);
				float fi = acos(sourceNormal.z);


				//range of atan: -pi to pi
				//range of acos: 0 to pi

				//push all thetas by pi/2 to make the range from 0 to pi
				theta += PI;

				theta = mod(theta, 2 * PI);
				fi = mod(fi, 2 * PI);


				float s1 = 2 * PI / HistogramHeight;
				float s2 = (PI / 2) / HistogramHeight;


				ivec2 binIndex = ivec2(theta / s1, fi / s2);
				binIndex = ivec2(min(binIndex.x, HistogramHeight - 1), min(binIndex.y, HistogramHeight - 1));
				histogramIndexCentral = binIndex.y  * HistogramWidth + binIndex.x;
			}

		}
		else if (binningMode == 2)
		{
			//longitude/latitude binning
			//get 3d normal
			vec3 sourceNormal;
			sourceNormal.xy = vec2(newRay.x, newRay.y)*2.0f - 1.0f;
			//sourceNormal.z = 1.0f - sqrt(sourceNormal.x*sourceNormal.x + sourceNormal.y*sourceNormal.y);
			sourceNormal.z = sqrt(1.0f - sourceNormal.x*sourceNormal.x - sourceNormal.y*sourceNormal.y);

			quantizedRay = vec2(newRay.x * float((HistogramWidth)), newRay.y * float(HistogramHeight));

			// NOTE: the total amount of energy will only be available if the max. number of samples is reached.
			// before that the renderer has to upscale the ndf samples according to the total energy.
			histogramIndexCentral = int(quantizedRay.y) * HistogramWidth + int(quantizedRay.x);

			miss = (histogramIndexCentral == 0);

			if (histogramIndexCentral < 0 || histogramIndexCentral >= HistogramHeight * HistogramWidth)
			{
				return;
			}

			//miss = ((newRay.x == 0) && (newRay.y==0));

			if (!miss)
			{
				if (sourceNormal.z != sourceNormal.z)
					discard;


				float X = sqrt(2.0f / (1.0 + sourceNormal.z))*sourceNormal.x;
				float Y = sqrt(2.0f / (1.0 + sourceNormal.z))*sourceNormal.y;

				//range of x and y: -sqrt(2) -> sqrt(2)

				X += sqrt(2.0f);
				Y += sqrt(2.0f);

				float s1 = (2.0f*sqrt(2.0f)) / HistogramHeight;    //X is in the range of -2 to 2
				float s2 = (2.0f*sqrt(2.0f)) / HistogramHeight;


				ivec2 binIndex = ivec2(X / s1, Y / s2);
				binIndex = ivec2(min(binIndex.x, HistogramHeight - 1), min(binIndex.y, HistogramHeight - 1));
				histogramIndexCentral = binIndex.y  * HistogramWidth + binIndex.x;
			}
		}
	}
	else
	{
		histogramIndexCentral = -1;
		miss = true;
	}

	//invert display of bin for clarity
	//ivec2 twodindx = ivec2(histogramIndexCentral%HistogramWidth, histogramIndexCentral / HistogramWidth);
	//twodindx.x = HistogramHeight - twodindx.x;
	//twodindx.y = HistogramHeight - twodindx.y;
	//histogramIndexCentral = twodindx.y*HistogramWidth + twodindx.x;

	//empty bin color
	float factor = float(histogramIndexCentral);// float(int(texCoords.y*HistogramWidth)*HistogramWidth) + float(int(texCoords.x*HistogramWidth));
	factor /= float(HistogramWidth*HistogramWidth+HistogramWidth);

	int colorIndex =int(factor*float(colorMapSize-1));
	vec4 emptyBinColor = vec4(color_map[colorIndex * 3], color_map[colorIndex * 3 + 1], color_map[colorIndex * 3 + 2], 1);

	if (bArea[histogramIndexCentral] == 0)
		discard;

	if (!miss)
 	{
		if (avg_NDF[histogramIndexCentral] == 0)
			outColor = emptyBinColor;
		else
		{
			//debug
			float maxVal = 0.0f;
			for (int i = 0; i < HistogramHeight*HistogramWidth; i++)
			{
				if (avg_NDF[i]>maxVal)
					maxVal = avg_NDF[i];
			}

			//end debug
			outColor =vec4(avg_NDF[histogramIndexCentral]/maxVal, avg_NDF[histogramIndexCentral]/maxVal, avg_NDF[histogramIndexCentral]/maxVal, 1);
		}
		
			
	}
//	else
//		discard;// outColor = emptyBinColor;
	
	//return;

	/*const float energyCorrection = 1.2f;
	const float constrast = 2.0f;
	const float brightness = 1.0f;
	outColor.xyz = brightness.xxx * pow(outColor.xyz * energyCorrection.xxx, constrast.xxx);*/
	
	//vec3 normal = vec3(texCoords.xy, 0.0f) * 2.0f - 1.0f;

	//const vec3 backgroundColor = 0.0f * vec3(1.0f, 1.0f, 1.0f);
	//const vec3 highlightColor = vec3(0.0f, 1.0f, 0.0f);
	//
	//// anti aliasing around the edge
	//const float smoothness = 0.06f;
	//float border = max(0.0f, normal.x*normal.x + normal.y*normal.y - (1.0f - smoothness));
	//if(border > smoothness) 
	//{
	//	discard;
	//}
	//float opacity = smoothstep(smoothness, 0.0f, border);

	//outColor.xyz = mix(backgroundColor, outColor.xyz, opacity);
	
}