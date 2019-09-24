#ifndef PROCEDURAL_H_
#define PROCEDURAL_H_

#include<vector>
#define _USE_MATH_DEFINES
#include <math.h>

#include<glm/vec3.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace Helpers {
namespace Procedural {

std::vector<glm::vec3> GenerateCheckerboardSphere() {
	std::vector<glm::vec3> outParticles;

	const int yCount = 1000;
	const int xCount = 1000;
	const auto xScale = 1.0f / static_cast<float>(xCount);
	const auto yScale = 1.0f / static_cast<float>(yCount);
	const auto radius = 0.5f;
	const auto particlesPerGrove = 4.0f;
	const auto checkerScale = 0.002f;

	for(int y = 0; y < yCount; ++y) {
		for(int x = 0; x < xCount; ++x) {
			auto xRad = static_cast<float>(x) * xScale * 0.5f + 0.75f;
			auto yRad = static_cast<float>(y) * yScale * 0.5f;

			auto xf = radius * std::sinf(xRad * 2.0f * M_PI) * std::cosf(yRad * 2.0f * M_PI);
			auto yf = radius * std::sinf(xRad * 2.0f * M_PI) * std::sinf(yRad * 2.0f * M_PI);
			auto zf = radius * std::cosf(xRad * 2.0f * M_PI);

			auto checker = 0.0f;
			auto checkerBoardX = std::fmodf(xRad, (xScale * particlesPerGrove * 2.0f));
			auto checkerBoardY = std::fmodf(yRad, (yScale * particlesPerGrove * 2.0f));
			if((checkerBoardX > (xScale * particlesPerGrove)) ^ (checkerBoardY > (yScale * particlesPerGrove))) {
				checker = checkerScale;
			}

			glm::vec3 normal = glm::vec3(xf, yf, zf);
			normal = glm::normalize(normal);
			auto offset = normal * checker;
			xf += offset.x;
			yf += offset.y;
			zf += offset.z;

			outParticles.emplace_back(glm::vec3(xf, yf, zf));// + glm::vec3(-0.5f, -0.5f, 0.0f));
		}
	}

	return outParticles;
}

std::vector<glm::vec3> GenerateSawToothSphere() {
	std::vector<glm::vec3> outParticles;

	const float section = 1.0f;
	//const int yCount = 5000;
	//const int xCount = 5000;
	//const auto particlesPerGrove = 8.0f;

	const int yCount = 5000;
	const int xCount = 5000;
	//const auto particlesPerGrove = 40.0f;
	const auto particlesPerGrove = 40.0f;

	const auto xScale = section / static_cast<float>(xCount);
	const auto yScale = section / static_cast<float>(yCount);
	const auto radius = 0.45f;
	const auto vGrooveScale = 2.0f;

	for(int y = 0; y < yCount; ++y) {
		for(int x = 0; x < xCount; ++x) {
			auto xRad = static_cast<float>(x) * xScale * 0.5f + 0.75f;
			auto yRad = static_cast<float>(y) * yScale * 0.5f;

			auto xf = radius * std::sinf(xRad * 2.0f * M_PI) * std::cosf(yRad * 2.0f * M_PI);
			auto yf = radius * std::sinf(xRad * 2.0f * M_PI) * std::sinf(yRad * 2.0f * M_PI);
			auto zf = radius * std::cosf(xRad * 2.0f * M_PI);

			auto vGroove = std::fmodf(yRad * 2.0f, (yScale * particlesPerGrove * 2.0f));

			glm::vec3 normal = glm::vec3(xf, yf, zf);
			normal = glm::normalize(normal);
			auto offset = normal * vGroove * vGrooveScale;
			xf += offset.x;
			yf += offset.y;
			zf += offset.z;

			outParticles.emplace_back(glm::vec3(xf, yf, zf));
		}
	}

	// fill gaps
	const auto yScaleGaps = section / (static_cast<float>(yCount) / particlesPerGrove);
	const auto fillScale = yScaleGaps * (vGrooveScale / particlesPerGrove);

	for(int y = 0; y < yCount / (particlesPerGrove * 2); ++y) {
		for(int x = 0; x < xCount; ++x) {
			auto xRad = static_cast<float>(x) * xScale * 0.5f + 0.75f;
			auto yRad = static_cast<float>(y) * yScaleGaps;

			auto xf = radius * std::sinf(xRad * 2.0f * M_PI) * std::cosf(yRad * 2.0f * M_PI);
			auto yf = radius * std::sinf(xRad * 2.0f * M_PI) * std::sinf(yRad * 2.0f * M_PI);
			auto zf = radius * std::cosf(xRad * 2.0f * M_PI);
		
			glm::vec3 normal = glm::vec3(xf, yf, zf);
			normal = glm::normalize(normal);
			for(int fillI = 0; fillI < particlesPerGrove * 2; ++fillI) {
				auto offset = normal * fillScale;
				xf += offset.x;
				yf += offset.y;
				zf += offset.z;

				outParticles.emplace_back(glm::vec3(xf, yf, zf));
			}
		}
	}

	return outParticles;
}


std::vector<glm::vec4> GenerateSawToothSphere(float particleRadius, int xCount, int yCount) {
	std::vector<glm::vec4> outParticles;

	const float section = 1.0f;
	//const int yCount = 5000;
	//const int xCount = 5000;
	//const auto particlesPerGrove = 8.0f;

	//const int yCount = 5000;
	//const int xCount = 5000;
	//const auto particlesPerGrove = 40.0f;
	const auto particlesPerGrove = 40.0f;

	const auto xScale = section / static_cast<float>(xCount);
	const auto yScale = section / static_cast<float>(yCount);
	const auto radius = 0.45f;
	const auto vGrooveScale = 2.0f;

	for (int y = 0; y < yCount; ++y) {
		for (int x = 0; x < xCount; ++x) {
			auto xRad = static_cast<float>(x)* xScale * 0.5f + 0.75f;
			auto yRad = static_cast<float>(y)* yScale * 0.5f;

			auto xf = radius * std::sinf(xRad * 2.0f * M_PI) * std::cosf(yRad * 2.0f * M_PI);
			auto yf = radius * std::sinf(xRad * 2.0f * M_PI) * std::sinf(yRad * 2.0f * M_PI);
			auto zf = radius * std::cosf(xRad * 2.0f * M_PI);

			auto vGroove = std::fmodf(yRad * 2.0f, (yScale * particlesPerGrove * 2.0f));

			glm::vec3 normal = glm::vec3(xf, yf, zf);
			normal = glm::normalize(normal);
			auto offset = normal * vGroove * vGrooveScale;
			xf += offset.x;
			yf += offset.y;
			zf += offset.z;

			outParticles.emplace_back(glm::vec4(xf, yf, zf,particleRadius));
		}
	}

	// fill gaps
	const auto yScaleGaps = section / (static_cast<float>(yCount) / particlesPerGrove);
	const auto fillScale = yScaleGaps * (vGrooveScale / particlesPerGrove);

	for (int y = 0; y < yCount / (particlesPerGrove * 2); ++y) {
		for (int x = 0; x < xCount; ++x) {
			auto xRad = static_cast<float>(x)* xScale * 0.5f + 0.75f;
			auto yRad = static_cast<float>(y)* yScaleGaps;

			auto xf = radius * std::sinf(xRad * 2.0f * M_PI) * std::cosf(yRad * 2.0f * M_PI);
			auto yf = radius * std::sinf(xRad * 2.0f * M_PI) * std::sinf(yRad * 2.0f * M_PI);
			auto zf = radius * std::cosf(xRad * 2.0f * M_PI);

			glm::vec3 normal = glm::vec3(xf, yf, zf);
			normal = glm::normalize(normal);
			for (int fillI = 0; fillI < particlesPerGrove * 2; ++fillI) {
				auto offset = normal * fillScale;
				xf += offset.x;
				yf += offset.y;
				zf += offset.z;

				outParticles.emplace_back(glm::vec4(xf, yf, zf,particleRadius));
			}
		}
	}

	return outParticles;
}

std::vector<glm::vec3> GenerateVGrooveSphere() {
	std::vector<glm::vec3> outParticles;

	const float section = 1.0f;
	const int yCount = 5000;
	const int xCount = 5000;
	const auto xScale = section / static_cast<float>(xCount);
	const auto yScale = section / static_cast<float>(yCount);
	const auto radius = 0.45f;
	const auto particlesPerGrove = 8.0f;
	//const auto particlesPerGrove = 40.0f;
	const auto vGrooveScale = 2.0f;
	const float repeat = yScale * particlesPerGrove * 2.0f;

	for(int y = 0; y < yCount; ++y) {
		for(int x = 0; x < xCount; ++x) {
			auto xRad = static_cast<float>(x) * xScale * 0.5f + 0.75f;
			auto yRad = static_cast<float>(y) * yScale * 0.5f;

			auto xf = radius * std::sinf(xRad * 2.0f * M_PI) * std::cosf(yRad * 2.0f * M_PI);
			auto yf = radius * std::sinf(xRad * 2.0f * M_PI) * std::sinf(yRad * 2.0f * M_PI);
			auto zf = radius * std::cosf(xRad * 2.0f * M_PI);

			auto vGroove = std::fmodf(yRad * 2.0f, repeat);
			if(vGroove > repeat * 0.5f) {
				vGroove = repeat - vGroove;
			}

			glm::vec3 normal = glm::vec3(xf, yf, zf);
			normal = glm::normalize(normal);
			auto offset = normal * vGroove * vGrooveScale;
			xf += offset.x;
			yf += offset.y;
			zf += offset.z;

			outParticles.emplace_back(glm::vec3(xf, yf, zf));
		}
	}

	return outParticles;
}

} // namespace Procedural
} // namespace Helpers
#endif // PROCEDURAL_H_