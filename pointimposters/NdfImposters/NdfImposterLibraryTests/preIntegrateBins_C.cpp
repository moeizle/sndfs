#version 430


uniform ivec2 histogramDiscretizations;

int HistogramWidth = histogramDiscretizations.x;
int HistogramHeight = histogramDiscretizations.y;

uniform int binDiscretizations;
uniform int binningMode;
uniform int renderMode;

uniform int areaIndx;

uniform float specularExp;

uniform vec3 viewSpaceLightDir;

uniform sampler2D normalTransferSampler;
uniform sampler2D chromeTexture;

layout(binding = 0, rgba32f) uniform image2D tex;

layout(std430) buffer superPreIntegratedBins
{
	float bins[];
};

layout(std430) buffer simple_binAreas
{
	double Sbins[];
};

//layout(std430, binding = 3) buffer preIntegratedBins
//{
//	float binColor[]; // This is the important name (in the shader).
//};


layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

mat3 rotationMatrix(vec3 axis, float angle)
{
	axis = normalize(axis);
	float s = sin(angle);
	float c = cos(angle);
	float oc = 1.0f - c;

	return mat3(oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s,
		oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z - axis.x * s,
		oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c);
}

vec3 blinnPhong(vec3 normal, vec3 light, vec3 view, vec3 diffuseColor, vec3 specularColor, float specularExponent)
{

	light.x *= -1;
	light.y *= -1;
	

	vec3 halfVector = normalize(light + view);

	float diffuseIntensity = 1.0f * max(0.0f, -dot(normal, -light));
	float diffuseWeight = 1.0f;
	//float diffuseIntensity = max(0.0f, abs(dot(normal, light)));
	//float diffuseIntensity = max(0.0f, dot(normal, -light));
	//float specularIntensity = max(0.0f, pow(dot(normal, halfVector), specularExponent));

	float nDotHalf = abs(dot(normal, halfVector));
	float specularIntensity = 1.0f * max(0.0f, pow(nDotHalf, specularExponent));
	float specularWeight = 1.0f;
	//specularIntensity *= 0.0f;
	//diffuseIntensity *= 0.0f;

	const float ambientWeight = 0.0f;//0.25f;
	const vec3 ambientColor = vec3(1.0f, 0.0f, 0.0f);


	

	//return specularIntensity * specularColor;
	//return specularIntensity * vec3(0.0f, 0.5f, 0.0f) + min(1.0f, (1.0f - specularIntensity)) * vec3(0.5f, 0.0f, 0.0f);
	//return specularIntensity * specularColor + ambientIntensity * ambientColor;
	return (diffuseIntensity * diffuseColor * diffuseWeight + specularIntensity * specularColor * specularWeight + ambientWeight * ambientColor) / (diffuseWeight + specularWeight + ambientWeight);
	//return kd;
	//return vec3(1.0f, 1.0f, 1.0f) - 0.25f * (diffuseIntensity * diffuseColor - specularIntensity * specularColor + ambientIntensity * ambientColor);

	//toon shading
	//{
	//	float b, y, alpha, beta;
	//	b = 0.55;
	//	y = 0.3;
	//	alpha = 0.25;
	//	beta = 0.5;
	//	vec3 kblue = vec3(0, 0, b);
	//	vec3 kyellow = vec3(y, y, 0);
	//	vec3 kcool, kwarm;
	//	vec3 kd = vec3(1, 1, 1);

	//	kcool = kblue + alpha*kd;
	//	kwarm = kyellow + beta*kd;

	//	normalize(normal);
	//	normalize(light);

	//	vec3 I = ((1 + dot(normal, halfVector)) / 2.0f)*kcool + (1 - ((1 + dot(normal, halfVector)) / 2.0f))*kwarm;

	//	return I;
	//}

}

void main()
{

	float discretizationArea = 0;

	const float histogramScaleX = 1.0f / float(HistogramWidth);
	const float histogramScaleY = 1.0f / float(HistogramHeight);
	int histogramX = int(gl_GlobalInvocationID.x) % (HistogramWidth);
	int histogramY = int(gl_GlobalInvocationID.x / HistogramHeight);

	int j = int(gl_GlobalInvocationID.y) % (binDiscretizations);
	int k = int(gl_GlobalInvocationID.y) / (binDiscretizations);


	vec3 diffuseColor = vec3(1.0f, 1.0f, 1.0f);

	// NOTE: for this model to be physically based the BRDF would have to be perfectly specular
	float specularCorrection = 1.0f;
	vec3 specularColor = specularCorrection * vec3(1.0f, 1.0f, 1.0f);

	vec2 diskCenter = vec2(0.5, 0.5);
	vec2 pos,v;
	vec3 N,c;
	const float PI = 3.141592f;


	c = vec3(0, 0, 0);

	if (binningMode == 0)
	{
		//discretizationArea = 0.0;

		////initialize position big bin's bottom left corner
		//pos.x = histogramX*histogramScaleX;       //   ((histogramX + j / binDiscretizations)* histogramScaleX + 1.0f / (2.0f*binDiscretizations)*histogramScaleX) *2.0f - 1.0f;// -0.5f;
		//pos.y = histogramY*histogramScaleY;      // ((histogramY + k / binDiscretizations)* histogramScaleY + 1.0f / (2.0f*binDiscretizations)*histogramScaleY) *2.0f - 1.0f;// -0.5f;

		////add to that the sub-bin discretization
		//pos.x += j*(histogramScaleX / binDiscretizations);
		//pos.y += k*(histogramScaleY / binDiscretizations);

		////now we are at the bottom left corner of the smaller bins (bins within big bin
		////we would like to be in the middle of that bin, so we add half the bin dimension to each coordiante
		//pos.x += 0.5f*(histogramScaleX / binDiscretizations);
		//pos.y += 0.5f*(histogramScaleY / binDiscretizations);


		//N.x = pos.x*2.0f - 1.0f;
		//N.y = pos.y*2.0f - 1.0f;
		//N.z = sqrt(1.0f - N.x*N.x - N.y*N.y);
		//float l = length(N);
		//if (l <= 1.0f)
		//{
		//	//float length = std::sqrt((N.x * N.x) + (N.y * N.y));
		//	//N.z = sqrt(1.0f - length);
		//	discretizationArea = (histogramScaleX / binDiscretizations)*(histogramScaleY / binDiscretizations) / (PI*0.5f*0.5f);  //add normalized area, so total area is = 1
		//}

		//new
		

		ivec2 twoDIndx = ivec2(histogramX*binDiscretizations+j,histogramY*binDiscretizations+k);
		int oneDIndx = twoDIndx.y*HistogramWidth*binDiscretizations + twoDIndx.x;
		int indx = areaIndx + oneDIndx * 3;

		//int s = int(gl_GlobalInvocationID.x*gl_GlobalInvocationID.y);

		//int mydim = (HistogramWidth*HistogramHeight*binDiscretizations*binDiscretizations) / 2;
		//int v = histogramX*j;
		//int u = histogramY*k;

		//int indx = areaIndx + 3*(u*mydim + v);
		
		N.x = float(Sbins[indx + 1]) * 2.0 - 1.0;
		N.y = float(Sbins[indx + 2]) * 2.0 - 1.0;
		N.z = sqrt(1.0 - N.x*N.x - N.y*N.y);

		//float l = length(N);
		//if (l <= 1.0f)
		//{
		discretizationArea = float(Sbins[indx]);
		//}
		

	}
	else if (binningMode == 1)
	{
		float s1 = 2 * PI;
		float s2 = (PI / 2);
		vec2 ind = vec2(histogramX, histogramY);
		float interval = 1.0 / HistogramWidth;
		//get the position within bin, between (0,0) to (1,1)
		pos = ind*vec2(interval, interval);

		//add sub-bin psotion
		pos += vec2(interval, interval) * vec2(j / float(binDiscretizations), k / float(binDiscretizations));

		//move to middle of sub-bin
		pos += vec2(interval, interval) * vec2(0.5*(1.0 / float(binDiscretizations)), 0.5*(1.0 / float(binDiscretizations)));


		float theta = pos.x*s1;
		float fi = pos.y*s2;


		N = vec3(-cos(theta)*sin(fi),
			-sin(theta)*sin(fi),
			cos(fi));

		//if (length(N) <= 1)//calculate the area
		//{
		//get small bin area
		float ba_onSphere = sin(fi)*(1.0f / binDiscretizations)*interval*s1*(1.0f / binDiscretizations)*interval*s2;      //solid angle (dw)
		float B = fi;                                                                                       //angle between projection plane normal and bin normal, acos(dot(N,(0,0,1)))=acos(N.z)=fi
		float ba_projected = N.z*ba_onSphere;                                                               //ba_projected=cos(B)*ba_onsphere=cos(fi)*ba_onsphere=N.z*ba_onsphere
		discretizationArea = ba_projected / PI;                                                                            //divide by PI to total area is '1'
		//}
	}
	else if (binningMode == 2)
	{
		//float s1 = (2.0f*sqrt(2.0f)) / histogramResolution.y;
		//float s2 = (2.0f*sqrt(2.0f)) / histogramResolution.y;

		//float X = histogramX*s1 + 1.0f / (2.0f*binDiscretizations)*s1;
		//float Y = histogramY*s2 + 1.0f / (2.0f*binDiscretizations)*s2;

		//X -= sqrt(2.0f);
		//Y -= sqrt(2.0f);


		//N = vec3(sqrt(1.f - (X*X + Y*Y) / 4.0f)*X,
		//	sqrt(1.f - (X*X + Y*Y) / 4.0f)*Y,
		//	-1 * (-1.f + (X*X + Y*Y) / 2.0f));

		//N = vec3(X / (2 * sqrt(1 / (-X*X + Y*Y + 4))),
		//	Y / (2 * sqrt(1 / (-X*X + Y*Y + 4))),
		//	-X*X / 2 + Y*Y / 2 + 1);

		//float s1 = 4.0f / histogramResolution.y;
		//float s2 = (2*PI) / histogramResolution.y;

		//float R = histogramX*s1 + (1.0f / (2.0f*binDiscretizations))*s1;
		//float theta = histogramY*s2 + (1.0f / (2.0f*binDiscretizations))*s2;


		//R = R - 2;
		//theta -= PI;

		//float fi = 2 * acos(R / 2.0);


		//N = vec3(cos(theta)*sin(fi),
		//	sin(theta)*sin(fi),
		//	cos(fi));

		//newest
		{
			discretizationArea = 0.0;
			float s1 = (2.0f*sqrt(2.0f));
			float s2 = (2.0f*sqrt(2.0f));

			//get the position within bin, between (0,0) to (1,1)
			float interval = 1.0f / HistogramWidth;
			vec2 ind = vec2(histogramX, histogramY);
			pos = ind*vec2(interval, interval);

			//add sub-bin psotion
			pos += vec2(interval, interval) * vec2(j / float(binDiscretizations), k / float(binDiscretizations));

			vec2 s = pos;
			vec2 e = s + vec2(interval, interval) * vec2((1.0 / float(binDiscretizations)), (1.0 / float(binDiscretizations)));;

			//move to middle of sub-bin
			pos += vec2(interval, interval) * vec2(0.5*(1.0 / float(binDiscretizations)), 0.5*(1.0 / float(binDiscretizations)));


			float X = pos.x*s1;
			float Y = pos.y*s2;

			X -= sqrt(2.0f);
			Y -= sqrt(2.0f);

			if (length(vec2(X, Y)) <= sqrt(2.0f))//calculate the area
			{
				N = vec3(sqrt(1.f - (X*X + Y*Y) / 4.0f)*X,
					sqrt(1.f - (X*X + Y*Y) / 4.0f)*Y,
					-1 * (-1.f + (X*X + Y*Y) / 2.0f));

				//area is dXdY
				//get X and Y of s
				vec2 sXY = vec2(s.x*s1, s.y*s2);
				vec2 eXY = vec2(e.x*s1, e.y*s2);

				float dX = eXY.x - sXY.x;
				float dY = eXY.y - sXY.y;

				discretizationArea = dX*dY / (PI*sqrt(2.0f)*sqrt(2.0f));
			}
		}
	}

	vec2 transfer = vec2(N.x, N.y);

	if (renderMode <= 0)
	{
		vec3 lightViewSpace = vec3(viewSpaceLightDir.x, viewSpaceLightDir.y, viewSpaceLightDir.z);
		c = blinnPhong(N, -lightViewSpace, vec3(0.0f, 0.0f, 1.0f), diffuseColor, specularColor, specularExp);
	}
	else if (renderMode == 1)
	{
#if 0
		//const vec3 leftColor = vec3(.5, 0, 0);// vec3(0.35f, 0.65f, 0.8f);
		//const vec3 rightColor = vec3(0.7f, 0.95f, 0.1f);
		//const vec3 bottomColor = vec3(0.5f, 0.5f, 0.5f);
		//const vec3 topColor = vec3(0.35f, 0.65f, 0.8f);

		const vec3 leftColor = vec3(.5, 0, 0);// vec3(0.35f, 0.65f, 0.8f);
		const vec3 rightColor = vec3(0f, 0.5f, 0f);
		const vec3 bottomColor = vec3(0.5f, 0.5f, 0.5f);
		const vec3 topColor = vec3(0.0f, 0.0f, 0.5f);

		mat3 lightRotationZ = rotationMatrix(vec3(0.0f, 0.0f, 1.0f), viewSpaceLightDir.x + PI);
		mat3 lightRotationY = rotationMatrix(vec3(0.0f, 1.0f, 0.0f), viewSpaceLightDir.y);

		transfer = vec2((lightRotationZ * vec3(transfer.x, transfer.y, 1.0f)).x, (lightRotationZ * vec3(transfer.x, transfer.y, 1.0f)).y);
		transfer = vec2((lightRotationY * vec3(transfer.x, transfer.y, 1.0f)).x, (lightRotationY * vec3(transfer.x, transfer.y, 1.0f)).y);

		diffuseColor = 0.5f * leftColor * (1.0f - transfer.x) + 0.5f * rightColor * transfer.x + 0.5f * bottomColor * (1.0f - transfer.y) + 0.5f * topColor * transfer.y;
		//diffuseColor = texture(chromeTexture, transfer).xyz;
		specularColor = diffuseColor;

		c = (diffuseColor + specularColor) * 0.5f;
#else
		/*vec3 lightViewSpace = vec3(viewSpaceLightDir.x, viewSpaceLightDir.y, viewSpaceLightDir.z);
		c = blinnPhong(N, -lightViewSpace, vec3(0.0f, 0.0f, 1.0f), diffuseColor, specularColor, specularExp);
		vec2 twoDIndx = vec2(histogramX*binDiscretizations + j, histogramY*binDiscretizations + k);
		twoDIndx = twoDIndx / (HistogramWidth*binDiscretizations);*/
		vec2 lookupCoord = 0.5f*(N.xy + vec2(1, 1));
		//c = texture(chromeTexture,lookupCoord ).xyz;
		//vec3 lightViewSpace = vec3(viewSpaceLightDir.x, viewSpaceLightDir.y, viewSpaceLightDir.z);
		//c = blinnPhong(N, -lightViewSpace, vec3(0.0f, 0.0f, 1.0f), diffuseColor, specularColor, specularExp);
		c = imageLoad(tex, ivec2(512 * lookupCoord.xy)).xyz;
#endif

	}
	else if (renderMode == 2)
	{

		//mat3 lightRotationZ = rotationMatrix(vec3(0.0f, -1.0f, 0.0f), -viewSpaceLightDir.x);
		//mat3 lightRotationY = rotationMatrix(vec3(1.0f, 0.0f, 0.0f), -viewSpaceLightDir.y);

		//vec3 transformedNormal = N;
		//transformedNormal = (lightRotationZ * transformedNormal.xyz).xyz;
		//transformedNormal = (lightRotationY * transformedNormal.xyz).xyz;

		//const float seamCorrection = 0.125f;
		//transformedNormal.xy -= vec2(transformedNormal.xy * seamCorrection);
		//vec2 lookupCoord = 1.0f - (transformedNormal.xy * 0.5f + 0.5f);

		//vec3 lookupColor = texture(normalTransferSampler, lookupCoord).xyz;

		//diffuseColor = lookupColor.xyz;
		//specularColor = diffuseColor;


		//c = (diffuseColor + specularColor) * 0.5f;

		{
			vec3 transformedNormal;
			vec3 L = viewSpaceLightDir;
			//L.z *= -1.0f;

			/*vec3 a = cross(N, L);
			a = normalize(a);

			float theta = acos(dot(N, L) / (length(N)*length(L)));

			mat3 R = rotationMatrix(a, theta);

			transformedNormal = R*N;

			const float seamCorrection = 0.125f;
			transformedNormal.xy -= vec2(transformedNormal.xy * seamCorrection);

			transfer.xy = transformedNormal.xy * 0.5f + 0.5f;

			vec3 lookupColor = texture(normalTransferSampler, vec2(transfer.x, transfer.y)).xyz;

			diffuseColor = lookupColor;
			specularColor = diffuseColor;*/


			mat3 lightRotationZ = rotationMatrix(vec3(0.0f, -1.0f, 0.0f), viewSpaceLightDir.x);
			mat3 lightRotationY = rotationMatrix(vec3(1.0f, 0.0f, 0.0f), viewSpaceLightDir.y);

			transformedNormal = N;
			transformedNormal = (lightRotationZ * transformedNormal.xyz).xyz;
			transformedNormal = (lightRotationY * transformedNormal.xyz).xyz;

			const float seamCorrection = 0.125f;
			transformedNormal.xy -= vec2(transformedNormal.xy * seamCorrection);
			vec2 lookupCoord = 1.0f - (transformedNormal.xy * 0.5f + 0.5f);

			vec3 lookupColor = texture(normalTransferSampler, lookupCoord).xyz;

			diffuseColor = lookupColor.xyz;
			specularColor = diffuseColor;

			c = (diffuseColor + specularColor) * 0.5f;

		}
	}


	bins[(gl_GlobalInvocationID.x*binDiscretizations*binDiscretizations*3) + gl_GlobalInvocationID.y * 3 + 0] = c.x*discretizationArea;
	bins[(gl_GlobalInvocationID.x*binDiscretizations*binDiscretizations*3) + gl_GlobalInvocationID.y * 3 + 1] = c.y*discretizationArea;
	bins[(gl_GlobalInvocationID.x*binDiscretizations*binDiscretizations*3) + gl_GlobalInvocationID.y * 3 + 2] = c.z*discretizationArea;

}