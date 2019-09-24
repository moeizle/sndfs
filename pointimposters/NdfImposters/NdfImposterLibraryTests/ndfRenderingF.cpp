#version 430

//#define OUTPUT_HISTOGRAM
//#define OUTPUT_EBRDF
#define NDF

in vec2 texCoords;
in vec3 rayBegin;
in vec3 frustumTarget;
in vec3 viewSpacePosition;
in vec3 worldSpacePosition;

out vec4 outColor;

uniform int renderMode;
uniform int zoomMode;
uniform float zoomScale;
uniform vec2 zoomWindow;
uniform int ssaoEnabled;
uniform int ssaoDownsampleCount;

uniform int samplingRunIndex;
uniform int maxSamplingRuns;
uniform int multiSamplingRate;

uniform float quantizationVarianceScale;
uniform float ndfIntensityCorrection;

uniform vec3 camPosi;
uniform vec2 camAngles;
uniform vec3 camDirection;
uniform vec3 viewSpaceCamDirection;
uniform vec3 viewSpaceLightDir;

uniform ivec2 viewDiscretizations;
uniform ivec2 histogramDiscretizations;
uniform ivec3 spatialDiscretizations;
uniform ivec2 viewportSize;

uniform sampler2D normalTransferSampler;

layout(binding = 1, rgba16f) uniform image2D tileTex;
layout(binding = 2, rg8) uniform image2D pageTex;


uniform int floor_w;

int HistogramWidth = histogramDiscretizations.x;
int HistogramHeight = histogramDiscretizations.y;
int ViewWidth = viewDiscretizations.x;
int ViewHeight = viewDiscretizations.y;
int VolumeResolutionX = spatialDiscretizations.x;
int VolumeResolutionY = spatialDiscretizations.y;

const float PI = 3.141592f;
const float toRadiants = PI / 180.0f;

layout(std430, binding = 0) buffer NdfVoxelData {
	coherent readonly float histograms[];
} ndfVoxelData;

layout(std430, binding = 1) buffer NdfVoxelDataHighRes {
	coherent readonly float histograms[];
} ndfVoxelDataHighRes;

mat3 rotationMatrix(vec3 axis, float angle) 
{
    axis = normalize(axis);
    float s = sin(angle);
    float c = cos(angle);
    float oc = 1.0f - c;
    
    return mat3(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
                oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
                oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c);
}

vec3 blinnPhong(vec3 normal, vec3 light, vec3 view, vec3 diffuseColor, vec3 specularColor, float specularExponent) 
{
	vec3 halfVector = normalize(light + view);

	float diffuseIntensity = 1.0f * max(0.0f, -dot(normal, -light));

	//float diffuseIntensity = max(0.0f, abs(dot(normal, light)));
	//float diffuseIntensity = max(0.0f, dot(normal, -light));
	//float specularIntensity = max(0.0f, pow(dot(normal, halfVector), specularExponent));

	float nDotHalf = abs(dot(normal, halfVector));
	float specularIntensity = 1.0f * max(0.0f, pow(nDotHalf, specularExponent));
	//specularIntensity *= 0.0f;
	//diffuseIntensity *= 0.0f;

	const float ambientIntensity = 0.0f;//0.25f;
	const vec3 ambientColor = vec3(1.0f, 0.0f, 0.0f);

	//return specularIntensity * specularColor;
	//return specularIntensity * vec3(0.0f, 0.5f, 0.0f) + min(1.0f, (1.0f - specularIntensity)) * vec3(0.5f, 0.0f, 0.0f);
	//return specularIntensity * specularColor + ambientIntensity * ambientColor;
	return diffuseIntensity * diffuseColor + specularIntensity * specularColor + ambientIntensity * ambientColor;
	//return vec3(1.0f, 1.0f, 1.0f) - 0.25f * (diffuseIntensity * diffuseColor - specularIntensity * specularColor + ambientIntensity * ambientColor);
}

void main() 
{
	vec3 samplePosition = rayBegin;

	const vec3 offset = vec3(0.4f, 0.0f, 0.0f);
	//samplePosition = mod(samplePosition + offset, vec3(1.0f, 1.0f, 1.0f));

	// TODO: add depth map and parallax mapping? Better rotation...
	// TODO: normal of tangent frame using depth map
	vec3 viewWorld = normalize(camPosi - worldSpacePosition);
	vec3 viewView = normalize(camPosi - viewSpacePosition);

	//outColor = vec4(0.5f, 0.0f, 0.0f, 1.0f);
	//return;

	vec3 shading = vec3(0.0f, 0.0f, 0.0f);
	float ndfSample = 0.0f;
	int ndfColorTest = 0;
	float energySum = 0.0f;
	{
		// TODO: rename
		vec2 voxelCoordF = samplePosition.xy;

		vec3 viewDir = viewWorld * 0.5f + 0.5f;

		if(viewDir.x < 0.0f || viewDir.x > 1.0f || viewDir.y < 0.0f || viewDir.y > 1.0f) {
			outColor = vec4(0.5f, 0.0f, 0.0f, 1.0f);
			return;
		}

		ivec2 voxelCoord = ivec2(
			int(voxelCoordF.x * float(VolumeResolutionX)),
			int(voxelCoordF.y * float(VolumeResolutionY)));

		//const float angleRange = 360.0f;//2.0f;//20.0f; //360.0f
		//vec2 camAngleScaled = vec2((camAngles.x / angleRange), (camAngles.y / angleRange));
		int viewCoordinateX = 0;//int(camAngleScaled.x * float(ViewWidth-1));
		int viewCoordinateY = 0;//int(camAngleScaled.y * float(ViewHeight-1));

		vec2 voxelFract;
		int index;
		if(zoomMode <= 0) {
			index = (voxelCoord.y * VolumeResolutionX + voxelCoord.x) * ViewHeight * ViewWidth * HistogramHeight * HistogramWidth
				+ (viewCoordinateY * ViewWidth + viewCoordinateX) * HistogramHeight * HistogramWidth;

			voxelFract = vec2(fract(voxelCoordF.x * float(VolumeResolutionX)), fract(voxelCoordF.y * float(VolumeResolutionY)));
		} else {
			//index = ((voxelCoord.y + VolumeResolutionY) * (VolumeResolutionX * multiSamplingRate) + (voxelCoord.x + VolumeResolutionX)) * HistogramHeight * HistogramWidth;
		
			const float zoomFactor = zoomScale;
			vec2 zoomWindowScale = vec2(VolumeResolutionX, VolumeResolutionY) / zoomFactor;
			vec2 zoomCoord = vec2(voxelCoord.x, voxelCoord.y) / zoomFactor;
			ivec2 zoomOffset = ivec2(zoomWindow.x * VolumeResolutionX * multiSamplingRate, zoomWindow.y * VolumeResolutionY * multiSamplingRate);

			//index = (((zoomCoord.y - (zoomWindowScale / 2)) + zoomOffset.y) * (VolumeResolutionX * multiSamplingRate) + (zoomCoord.x - (zoomWindowScale / 2) + zoomOffset.x)) * HistogramHeight * HistogramWidth;
			index = (((int(zoomCoord.y) - (int(zoomWindowScale) / 2)) + zoomOffset.y) * (VolumeResolutionX * multiSamplingRate) + (int(zoomCoord.x) - (int(zoomWindowScale) / 2) + zoomOffset.x)) * HistogramHeight * HistogramWidth;

			//index = (voxelCoord.y * 2 * (VolumeResolutionX * multiSamplingRate) + voxelCoord.x * 2) * HistogramHeight * HistogramWidth;

			voxelFract = vec2(fract(voxelCoordF.x * zoomWindowScale.x), fract(voxelCoordF.y * zoomWindowScale.y));
		}

#if defined(OUTPUT_HISTOGRAM) || defined(OUTPUT_EBRDF)
		const float histogramScaling = 32.0f;
		voxelCoordF.xy /= histogramScaling;

		voxelCoord = ivec2(
			int(voxelCoordF.x * float(VolumeResolutionX)),
			int(voxelCoordF.y * float(VolumeResolutionY)));

		//voxelCoord = ivec2(0, 0);
		index = (voxelCoord.y * VolumeResolutionX + voxelCoord.x) * HistogramHeight * HistogramWidth;



		// histogram bin per pixel
		int histogramX = int(fract(voxelCoordF.x * 32.0f) * float(HistogramWidth));
		int histogramY = int(fract(voxelCoordF.y * 32.0f) * float(HistogramHeight));
		ndfSample = ndfVoxelData.histograms[index + histogramY * HistogramWidth + histogramX];
		
		/*if(histogramX > 16) {
			ndfSample = 10.0f;
		} else {
			ndfSample = 0.0f;
		}*/

		voxelCoordF.xy *= histogramScaling;

		//if(volumeCoordinateY == 128 && volumeCoordinateX == 230) {
		//	ndfColorTest = 1;
		//}
#else // OUTPUT_HISTOGRAM || OUTPUT_EBRDF
		//vec3 diffuseColor = 0.25f * vec3(1.0f, 1.0f, 1.0f);
		vec3 diffuseColor = 1.0f * vec3(1.0f, 1.0f, 1.0f);
		
		// NOTE: for this model to be physically based the BRDF would have to be perfectly specular
		float specularExponent = 128.0f;
		float specularCorrection = 5.0f;
		vec3 specularColor = specularCorrection * vec3(1.0f, 1.0f, 1.0f);

		// world space transformation
		vec3 front = normalize(viewWorld);
		vec3 up = normalize(vec3(0.0f, 1.0f, 0.0f));
		vec3 right = normalize(cross(front, up));

		mat3x3 worldSpaceMatrix = mat3x3(right, up, front);
#if 1
		// NOTE: light dir should be the other way around
		vec3 lightWorldSpace = normalize(worldSpaceMatrix * viewSpaceLightDir);
		vec3 lightViewSpace = normalize(viewSpaceLightDir);
#else
	// head mounted light
		vec3 lightWorldSpace = normalize(viewWorld);
		vec3 lightViewSpace = normalize(viewView);
#endif

		const float histogramScaleX = 1.0f / float(HistogramWidth-1);
		const float histogramScaleY = 1.0f / float(HistogramHeight-1);
		
		// the fraction of total energy over energy collected
		const float energyCollected = (float(maxSamplingRuns) / float(samplingRunIndex));

		// FIXME: remove test - splat BRDF and sample only current light dir
		//const float lightEnergyCorrection = 14.0f;
		//int histogramX = HistogramWidth / 2; {
		//	int histogramY = HistogramHeight - 3; {
		const float lightEnergyCorrection = 1.0f;
#ifdef NDF
		for(int histogramY = 0; histogramY < HistogramHeight; ++histogramY) {
			for(int histogramX = 0; histogramX < HistogramWidth; ++histogramX) {
				int binIndex = histogramY * HistogramWidth + histogramX;
				
				//vec3 sourceNormal = 2.0f * vec3(float(histogramX) * histogramScaleX - 0.5f, float(histogramY) * histogramScaleY - 0.5f, 0.0f);
				//sourceNormal.z = sqrt(1.0f - sourceNormal.x*sourceNormal.x - sourceNormal.y*sourceNormal.y);

				vec3 sourceNormal;
				sourceNormal.xy = 2.0f * vec2(float(histogramX) * histogramScaleX - 0.5f, float(histogramY) * histogramScaleY - 0.5f);
				sourceNormal.z = sqrt(1.0f - sourceNormal.x*sourceNormal.x - sourceNormal.y*sourceNormal.y);

				// project to actual view space since the nearest view samples does not necessarily equal the current view direction
				//vec3 viewSpaceNormal = normalize(viewCorrection * sourceNormal);
				vec3 viewSpaceNormal = sourceNormal;

				// project view space normal to world space
				vec3 worldSpaceNormal = normalize(worldSpaceMatrix * viewSpaceNormal);
				// alternatively:
				//vec3 worldSpaceNormal = normalize(viewSpaceNormal.x * right + viewSpaceNormal.y * up + viewSpaceNormal.z * front);

				//mohamed's index
				int tile_no = int(imageLoad(tileTex, ivec2(voxelCoord.xy)).x);
				vec2 within_tile = imageLoad(tileTex, ivec2(voxelCoord.xy)).yz;

				ivec2 page_tex_ind;
				page_tex_ind.x = int(mod(tile_no, floor_w));
				page_tex_ind.y = int(tile_no / floor_w);

				vec2 phys_xy = imageLoad(pageTex, page_tex_ind).xy;

				index = int(((phys_xy.y + within_tile.y) * VolumeResolutionX + (phys_xy.x + within_tile.x)) * HistogramHeight * HistogramWidth);
				//end mohamed's index

				// ignore last value since it contains the depth
				if(binIndex < HistogramWidth * HistogramHeight - 1) {
if(zoomMode <= 0) {
						ndfSample = ndfVoxelData.histograms[index + binIndex];
} else {
						const int histogramIndexCentral = index;
						const int histogramIndexHorizontal = index + HistogramWidth * HistogramHeight;
						const int histogramIndexVertical = index + VolumeResolutionX * multiSamplingRate * HistogramWidth * HistogramHeight;
						const int histogramIndexHorizontalVertical = index + (VolumeResolutionX * multiSamplingRate + 1) * HistogramWidth * HistogramHeight;

#if 0
						ndfSample = ndfVoxelDataHighRes.histograms[histogramIndexCentral + binIndex] * 0.25f;
						ndfSample += ndfVoxelDataHighRes.histograms[histogramIndexHorizontal + binIndex] * 0.25f;
						ndfSample += ndfVoxelDataHighRes.histograms[histogramIndexVertical + binIndex] * 0.25f;
						ndfSample += ndfVoxelDataHighRes.histograms[histogramIndexHorizontalVertical + binIndex] * 0.25f;
#else
						ndfSample = ndfVoxelDataHighRes.histograms[histogramIndexCentral + binIndex] * (1.0f - voxelFract.x) * (1.0f - voxelFract.y);
						ndfSample += ndfVoxelDataHighRes.histograms[histogramIndexHorizontal + binIndex] * (voxelFract.x) * (1.0f - voxelFract.y);
						ndfSample += ndfVoxelDataHighRes.histograms[histogramIndexVertical + binIndex] * (1.0f - voxelFract.x) * (voxelFract.y);
						ndfSample += ndfVoxelDataHighRes.histograms[histogramIndexHorizontalVertical + binIndex] * (voxelFract.x) * (voxelFract.y);
#endif
}
				}

				// correct energy while downsampling using an unfinished state
				ndfSample *= ndfIntensityCorrection;

				vec3 neighbourShading = vec3(0.0f, 0.0f, 0.0f);
				float neighbourdNdfSample = 0.0f;

				// add colors
#if 1
				const float threshold = 0.25f;//0.35f;
				//vec2 transfer = worldSpaceNormal.xy;
				vec2 transfer = viewSpaceNormal.xy;

				if(renderMode == 1) {
					const vec3 leftColor = vec3(0.35f, 0.65f, 0.8f);
					const vec3 rightColor = vec3(0.7f, 0.95f, 0.1f);
					const vec3 bottomColor = vec3(0.5f, 0.5f, 0.5f);
					const vec3 topColor = vec3(0.35f, 0.65f, 0.8f);

					mat3 lightRotationZ = rotationMatrix(vec3(0.0f, 0.0f, 1.0f), lightViewSpace.x+PI);
					mat3 lightRotationY = rotationMatrix(vec3(0.0f, 1.0f, 0.0f), lightViewSpace.y);

					transfer = (lightRotationZ * vec3(transfer.xy, 1.0f)).xy;
					transfer = (lightRotationY * vec3(transfer.xy, 1.0f)).xy;

					diffuseColor = 0.5f * leftColor * (1.0f - transfer.x) + 0.5f * rightColor * transfer.x +
						0.5f * bottomColor * (1.0f - transfer.y) + 0.5f * topColor * transfer.y;
					specularColor = diffuseColor;

					const float energyCorrection = 1.0f;
					diffuseColor *= energyCorrection;
					specularColor *= energyCorrection;
				} else if(renderMode == 2) {
					mat3 lightRotationZ = rotationMatrix(vec3(0.0f, -1.0f, 0.0f), viewSpaceLightDir.x);
					mat3 lightRotationY = rotationMatrix(vec3(1.0f, 0.0f, 0.0f), viewSpaceLightDir.y);

					vec3 transformedNormal = viewSpaceNormal;
					transformedNormal = (lightRotationZ * transformedNormal.xyz).xyz;
					transformedNormal = (lightRotationY * transformedNormal.xyz).xyz;

					const float seamCorrection = 0.125f;
					transformedNormal.xy -= vec2(transformedNormal.xy * seamCorrection);
					vec2 lookupCoord = 1.0f - (transformedNormal.xy * 0.5f + 0.5f);

					vec3 lookupColor = texture(normalTransferSampler, lookupCoord).xyz;

					diffuseColor = lookupColor.xyz;
					specularColor = diffuseColor;
				}

				//float lerp = viewSpaceNormal.y * 0.5f + 0.5f;
				//float saturation = 1.25f;
				//diffuseColor = (1.0f - lerp) * saturation * vec3(0.0f, 1.0f, 0.0f) + lerp * saturation * vec3(1.0f, 0.0f, 0.0f);
#endif

#if 1
				viewView = vec3(0.0f, 0.0f, 1.0f);
				//lightViewSpace = normalize(viewView + vec3(0.25f, 1.25f, 1.0f));

				viewWorld = normalize(worldSpaceMatrix * viewView);
				//lightWorldSpace = -normalize(viewWorld + vec3(0.75f, 0.5f, 0.25f));
#endif

				vec3 centralShading;
				// >> granda <<
				if(renderMode <= 0) {
					centralShading = blinnPhong(viewSpaceNormal, -lightViewSpace, viewView, diffuseColor, specularColor, specularExponent);
					//centralShading /= dot(viewSpaceNormal, viewView);
				} else { 
					centralShading = (diffuseColor + specularColor) * 0.5f;
				}

				shading += lightEnergyCorrection * ndfSample * centralShading;
				energySum += ndfSample;
			}
		}
#else
		// lookup pre integrated
		//vec3 sourceNormal = vec3(float(histogramX) * histogramScaleX - 0.5f, float(histogramY) * histogramScaleY - 0.5f, 0.0f);
		//sourceNormal.z = sqrt(1.0f - sourceNormal.x*sourceNormal.x - sourceNormal.y*sourceNormal.y);

		vec2 brdfLookup = (-lightViewSpace.xy) * 0.5f + 0.5f;
		vec2 brdfWeight = vec2(fract(brdfLookup.x * float(HistogramWidth-1)), fract(brdfLookup.y * float(HistogramHeight-1)));
		int brdfBinIndex = int(brdfLookup.y * float(HistogramHeight) * float(HistogramWidth)) + int(brdfLookup.x * float(HistogramWidth));

		float brdfSampleCentral = ndfVoxelData.histograms[index + brdfBinIndex];

		const float correction = 10.0f;
		float brdfValue;
#if 0
		float brdfSampleHor = ndfVoxelData.histograms[index + brdfBinIndex + 1];
		float brdfSampleVer = ndfVoxelData.histograms[index + brdfBinIndex + HistogramWidth];
		float brdfSampleHorVer = ndfVoxelData.histograms[index + brdfBinIndex + HistogramWidth + 1];

		brdfValue = (brdfSampleCentral * (1.0f - brdfWeight.x) * (1.0f - brdfWeight.y) +
			brdfSampleHor * (brdfWeight.x) * (1.0f - brdfWeight.y) +
			brdfSampleVer * (1.0f - brdfWeight.x) * (brdfWeight.y) +
			brdfSampleHorVer * (brdfWeight.x) * (brdfWeight.y));
#else
		brdfValue = brdfSampleCentral;

		vec3 normal = -lightViewSpace;
		brdfValue = (brdfSampleCentral * blinnPhong(normal, -lightViewSpace, viewView, diffuseColor, specularColor, specularExponent)).x;
#endif

		shading = pow(brdfValue * correction, 0.25f).xxx;
#endif // NDF

		// correct energy of missing rays
		shading *= energyCollected;
		energySum *= energyCollected;
#endif
	}

	//shading /= energySum;

	outColor = vec4(shading.xyz, 1.0f);

	// boost contrast
#if 1
#if defined(OUTPUT_HISTOGRAM) || defined(OUTPUT_EBRDF)
	//const float sampleScale = 1000.0f * float(maxSamplingRuns) / float(samplingRunIndex); //60.0f;
	const float sampleScale = 60.0f;
	outColor = sampleScale * vec4(ndfSample, ndfSample, ndfSample, 1.0f);

	if(ndfColorTest > 0) {
		outColor *= vec4(1.0f, 0.0f, 0.0f, 1.0f);
	}

	// NOTE: the energy correction depends on the amount of histogram bins
	const float energyCorrection = 1.5f;
	const float constrast = 0.25f;
	const float brightness = 1.0f;

	outColor.xyz = brightness.xxx * pow(outColor.xyz * energyCorrection.xxx, constrast.xxx);
#else
	vec3 lowPass;
	{
		const float energyCorrection = 1.0f;
		const float constrast = 0.75f;
		const float brightness = 1.05f;
		lowPass = brightness.xxx * pow(outColor.xyz * energyCorrection.xxx, constrast.xxx);
	}
	vec3 highPass;
	{
		if(renderMode <= 0) {
			const float energyCorrection = 1.0f;//2.125f;
			const float constrast = 2.0f;//2.125f;
			const float brightness = 1.0f;
			highPass = brightness.xxx * pow(outColor.xyz * energyCorrection.xxx, constrast.xxx);
			lowPass *= 0.0f;
		} else if(renderMode == 2) {
			//const float energyCorrection = 2.9f;
			//const float constrast = 4.0f;

			//const float energyCorrection = 1.75f;
			//const float constrast = 2.25f;

			// Video
			//const float energyCorrection = 1.55f;
			//const float constrast = 1.5f;

			// saw tooth sphere
			//const float energyCorrection = 1.8f;
			//const float constrast = 2.75f;

			// lit spheres
			const float energyCorrection = 1.8f;
			const float constrast = 3.5f;

			const float brightness = 1.0f;// * pow(energySum * 0.975f, 8.0f);
			highPass = brightness.xxx * pow(outColor.xyz * energyCorrection.xxx, constrast.xxx);
		}
		else { 
			//const float energyCorrection = 1.45f;
			//const float constrast = 2.0f;
			//const float brightness = 0.95f;

			const float energyCorrection = 1.55f;
			const float constrast = 2.5f;
			const float brightness = 0.95f;

			highPass = brightness.xxx * pow(outColor.xyz * energyCorrection.xxx, constrast.xxx);
		}
	}
	const float brightness = 1.0f;
	//outColor.xyz = brightness * (lowPass * 0.5f + highPass * 0.5f);
	outColor.xyz = brightness * (highPass);
#endif // OUTPUT_HISTOGRAM || OUTPUT_EBRDF 
#endif // boost contrast

	// apply background color
	const vec3 backgroundColor = 1.0f * vec3(1.0f, 1.0f, 1.0f);
	//float opacity = energySum;

	// anythig lower than threshold will turn into background
	const float lowerEnergyThreshold = 0.0f;
	// range should be 1.0 - turn down range to give things with lower energy more opacity.
	const float energyRange = 1.0f;
	//const float energyRange = 0.5f;

	//float opacity = smoothstep(lowerEnergyThreshold, lowerEnergyThreshold + energyRange, energySum);
	
	// FIXME: white background is not consistent! gets darker if less smaples hit.
	float opacity = energySum;
	//outColor.xyz = mix(backgroundColor, outColor.xyz, opacity);
}