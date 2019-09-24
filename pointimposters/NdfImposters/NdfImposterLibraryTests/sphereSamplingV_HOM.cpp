#version 430

uniform mat4 ViewProjection;
uniform mat4 Projection;
uniform mat4 ModelView;
uniform mat4 View;
uniform mat4 Model;
uniform mat4 ViewAlignmentMatrix;
uniform mat3 RotationMatrix;

uniform vec3 ViewPosition;
uniform vec3 modelOffset;

uniform float far;
uniform float near;
uniform float particleScale;
uniform float tileW;
uniform float tileH;

uniform float viewportWidth;
uniform float viewportHeight;

uniform int highestSampleCount;
uniform int maxSamplingRuns;
uniform int samplescount;
uniform int sampleIndex;

uniform vec2 sPos;


uniform vec3 right;
uniform vec3 up;
uniform vec2 samplePos;

uniform float objPerPixel;

in vec4 inPosition;

out vec2 rayOffset;


void main()
{
	//old 
	//float squareRootMaxSamplesF = floor(sqrt(float(maxSamplingRuns)));
	//int squareRootMaxSamples = int(squareRootMaxSamplesF);

	//// prime in residue class wraps around and introduces a pseudo random sampling pattern that reduces aliasing
	////const float prime = 3163.0f;
	//const float prime = 149.0f;
	//float sampleIndex = float(samplingRunIndex) * prime;

	//// offset by half dimension ensures that the first ray is in the center of the pixel
	//float horizontalSampleIndex = mod((sampleIndex + squareRootMaxSamplesF * 0.5f), squareRootMaxSamplesF);
	//float verticalSampleIndex = mod(((sampleIndex / squareRootMaxSamplesF) + squareRootMaxSamplesF * 0.5f), squareRootMaxSamplesF);

	//// NOTE: this dependes on total samples and resolution
	//const float smoothingFactor = 1.0f; // 1.0f = no smoothing, 0.0f = center ray only
	//const float rayOffsetStrength = smoothingFactor / (squareRootMaxSamplesF * viewportWidth);
	//rayOffset = rayOffsetStrength * vec2(horizontalSampleIndex, verticalSampleIndex);

	//end old
	//int val;
	//for (int i = 0; i < 128 * 128; i++)
	//{
	//	val = samples[i];
	//	if (val + 56>19999999)
	//		return;
	//}


	rayOffset = sPos;// *vec2(viewportWidth / viewportHeight, 1);
	rayOffset.xy /= vec2(viewportWidth, viewportHeight);
	//rayOffset.xy *= objPerPixel;// 1.0f / viewportWidth; 

	gl_Position = vec4(inPosition.xyz + modelOffset.xyz, inPosition.w);
}