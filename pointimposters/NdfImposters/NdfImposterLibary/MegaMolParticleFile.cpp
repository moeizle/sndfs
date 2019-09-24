#include "MegaMolParticleFile.h"

#include <string>
#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <memory>
#include <assert.h>

#include <glm\mat2x3.hpp>

// FIXME: A POINTER!?!?!? get rid of it...
std::vector<Helpers::IO::ParticleType> Helpers::IO::ParticlesFromMmpld(std::string filePath, uint64_t maxParticles, float &outGlobalRadius, glm::mat2x3 areaOfInterest, uint32_t maxFrameCount, glm::mat2x3 &outBoundingBox, bool rescale, glm::mat2x3 *ownBBox) {
	//////////////////////////////////////////////////////////////////////////////////////////
	// copied from megamol source code: https://svn.vis.uni-stuttgart.de/trac/megamol/attachment/wiki/mmpld/mmpldinfo.cpp
	//////////////////////////////////////////////////////////////////////////////////////////
	std::vector<Helpers::IO::ParticleType> particlePositions;

	const bool list_framedata = true;

	// open file
	std::ifstream file(filePath, std::ifstream::in | std::ifstream::binary);
	if (!file.is_open()) {
		throw std::runtime_error("Unable to open file");
	}
	// request errors
	file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	// read and check header
	char magicid[6];
	file.read(magicid, 6);
	if (::memcmp(magicid, "MMPLD", 6) != 0) {
		throw std::runtime_error("File does not seem to be of MMPLD format");
	}
	// read version
	uint16_t version;
	file.read(reinterpret_cast<char*>(&version), 2);
	if (version == 100) {
		std::cout << "    mmpld - version 1.0" << std::endl;
	} else if (version == 101) {
		std::cout << "    mmpld - version 1.1" << std::endl;
    } else if (version == 102) {
        std::cout << "    mmpld - version 1.2" << std::endl;
    } else {
		std::cerr << "    mmpld - version " << (version / 100) << "." << (version % 100) << std::endl;
		//throw std::runtime_error("Unsupported mmpld version encountered");
	}
	// read number of frames
	uint32_t frame_count;
	file.read(reinterpret_cast<char*>(&frame_count), 4);
	std::cout << "Number of frames: " << frame_count << std::endl;
	// read bounding boxes

	glm::mat2x3 boundingBox;
	file.read(reinterpret_cast<char*>(&boundingBox[0]), sizeof(float) * 6);
	std::cout << "Bounding box: (" << boundingBox[0][0] << ", " << boundingBox[0][1] << ", " << boundingBox[0][2] << ", "
		<< boundingBox[1][0] << ", " << boundingBox[1][1] << ", " << boundingBox[1][2] << ")" << std::endl;

	glm::mat2x3 &clippingBox = outBoundingBox;
	file.read(reinterpret_cast<char*>(&clippingBox[0]), sizeof(float) * 6);
	std::cout << "Clipping box: (" << clippingBox[0][0] << ", " << clippingBox[0][1] << ", " << clippingBox[0][2] << ", "
		<< clippingBox[1][0] << ", " << clippingBox[1][1] << ", " << clippingBox[1][2] << ")" << std::endl;
	// abort if no data
	if (frame_count <= 0) {
		throw std::runtime_error("No data");
	}
	// frame index table
	std::vector<uint64_t> frame_table(frame_count + 1);
	file.read(reinterpret_cast<char*>(frame_table.data()), (frame_count + 1) * 8);
	if (static_cast<uint64_t>(file.tellg()) != frame_table[0]) {
		std::cerr << "WRN: dead data trailing head" << std::endl;
	}
	file.seekg(0, std::ios_base::end);
	if (static_cast<uint64_t>(file.tellg()) < frame_table[frame_count]) {
		std::cerr << "WRN: file truncated" << std::endl;
	}
	if (static_cast<uint64_t>(file.tellg()) > frame_table[frame_count]) {
		std::cerr << "WRN: dead data trailing body" << std::endl;
	}
	{
		std::string errmsg;
		for (unsigned int fi = 0; fi < maxFrameCount; fi++) {
			if (frame_table[fi + 1] <= frame_table[fi]) {
				errmsg += "Frame table corrupted at frame ";
				errmsg += fi;
				errmsg += "\n";
			}
		}
		if (!errmsg.empty()) {
			throw std::runtime_error(errmsg);
		}
	}
	// test frames
	uint32_t min_listCnt;
	uint32_t max_listCnt;
	uint64_t min_partCnt;
	uint64_t acc_partCnt(0);
	uint64_t max_partCnt;
	uint64_t frm_partCnt;
	uint64_t lst_partCnt;
	
	if (maxFrameCount > 0) {
		frame_count = std::min(maxFrameCount, frame_count);
	}

	for (uint32_t fi = frame_count-1; fi < frame_count; fi++) {
		if (list_framedata) std::cout << "Frame #" << fi;
		file.seekg(frame_table[fi]);
        if (version == 102) {
            float theTime;
            file.read(reinterpret_cast<char*>(&theTime), 4);
            if (list_framedata) std::cout << " Simulation time " << theTime;
        }
		// number of lists
		uint32_t lists_cnt;
		file.read(reinterpret_cast<char*>(&lists_cnt), 4);
		if (list_framedata) std::cout << " - " << lists_cnt << " list" << ((lists_cnt != 1) ? "s" : "") << std::endl;
		if (fi == 0) {
			min_listCnt = max_listCnt = lists_cnt;
		} else {
			if (min_listCnt > lists_cnt) min_listCnt = lists_cnt;
			if (max_listCnt < lists_cnt) max_listCnt = lists_cnt;
		}
		frm_partCnt = 0;
		auto particlePostionsInsertIterator = std::back_inserter(particlePositions);
		for (uint32_t li = 0; li < lists_cnt; li++) {
			// list data format info
			uint8_t vert_type, col_type;
			size_t vrt_size;
			size_t col_size;
			file.read(reinterpret_cast<char*>(&vert_type), 1);
			file.read(reinterpret_cast<char*>(&col_type), 1);
			if (list_framedata) std::cout << "    #" << li << ": ";
			switch (vert_type) {
			case 1: vrt_size = 12; if (list_framedata) std::cout << "VERTDATA_FLOAT_XYZ"; break;
			case 2: vrt_size = 16; if (list_framedata) std::cout << "VERTDATA_FLOAT_XYZR"; break;
			case 3: vrt_size = 6; if (list_framedata) std::cout << "VERTDATA_SHORT_XYZ"; break;
			case 0: // falls through
			default: vrt_size = 0; if (list_framedata) std::cout << "VERTDATA_NONE"; break;
			}
			if (list_framedata) std::cout << ", ";
			if (vert_type == 0) col_type = 0;
			switch (col_type) {
			case 1: col_size = 3; if (list_framedata) std::cout << "COLDATA_UINT8_RGB"; break;
			case 2: col_size = 4; if (list_framedata) std::cout << "COLDATA_UINT8_RGBA"; break;
			case 3: col_size = 4; if (list_framedata) std::cout << "COLDATA_FLOAT_I"; break;
			case 4: col_size = 12; if (list_framedata) std::cout << "COLDATA_FLOAT_RGB"; break;
			case 5: col_size = 16; if (list_framedata) std::cout << "COLDATA_FLOAT_RGBA"; break;
			case 0: // falls through
			default: col_size = 0; if (list_framedata) std::cout << "COLDATA_NONE"; break;
			}
			if (list_framedata) std::cout << std::endl;
			size_t stride = vrt_size + col_size;
			if (list_framedata) std::cout << "        " << stride << " byte" << ((stride != 1) ? "s" : "") << " per particle" << std::endl;
			float glob_rad(0.05f);
			if ((vert_type == 1) || (vert_type == 3)) {
				file.read(reinterpret_cast<char*>(&glob_rad), 4);
				if (list_framedata) std::cout << "        global radius: " << glob_rad << std::endl;
			}
			float col_range[2];
			if (col_type == 0) {
				uint8_t col[4];
				file.read(reinterpret_cast<char*>(col), 4);
				if (list_framedata) std::cout << "        global color: (" << col[0] << ", " << col[1] << ", " << col[2] << ", " << col[3] << ")" << std::endl;
			}
			else if (col_type == 3) {
				file.read(reinterpret_cast<char*>(col_range), 8);
				if (list_framedata) std::cout << "        intensity color range: [" << col_range[0] << ", " << col_range[1] << "]" << std::endl;
			}
			col_range[0] = 0.0f;
			col_range[1] = 1.0f;
			file.read(reinterpret_cast<char*>(&lst_partCnt), 8);
			if (list_framedata) std::cout << "        " << lst_partCnt << " particle" << ((lst_partCnt != 1) ? "s" : "") << std::endl;
			frm_partCnt += lst_partCnt;




			//////////////////////////////////////////////////////////////////////////////////////////
			// own code
			//
			// read particle data
			//assert(vrt_size == 12);
			
			if (vrt_size == 12) {
				outGlobalRadius = glob_rad;

				auto particleCount = std::min(maxParticles, lst_partCnt);

				std::unique_ptr<char[]> rawData = std::unique_ptr<char[]>(new char[stride]);
				if(particleCount > 0) 
				{
					file.read(reinterpret_cast<char*>(rawData.get()), stride);
					auto particle = *reinterpret_cast<ParticleType*>(rawData.get());

					//debug
					if (lists_cnt > 1)
					{
						if (li == 0)
							particle.w = 2.011f;
						else
							particle.w = 1.78f;
					}
					else
							particle.w = glob_rad;
					//end debug

					particlePostionsInsertIterator = particle;

					clippingBox[0] = glm::vec3(particle);
					clippingBox[1] = glm::vec3(particle);
				}

				for (uint64_t particleI = 1; particleI < particleCount; ++particleI) {
					file.read(reinterpret_cast<char*>(rawData.get()), stride);

					// ignore additional data
					auto particle = *reinterpret_cast<ParticleType*>(rawData.get());

					//debug

					if (lists_cnt > 1)
					{
						if (li == 0)
							particle.w = 2.011f;
						else
							particle.w = 1.78f;
					}
					else
						particle.w = glob_rad;
					//end debug

#if 0
					auto low = areaOfInterest[0];
					auto high = areaOfInterest[1];
					if (particle.x >= low.x && particle.x <= high.x &&
						particle.y >= low.y && particle.y <= high.y &&
						particle.z >= low.z && particle.z <= high.z) {
						particlePostionsInsertIterator = particle;
					}
#else
					//if(fi == frame_count - 1) {
						particlePostionsInsertIterator = particle;
					//}
#endif
				}
			}

			//////////////////////////////////////////////////////////////////////////////////////////



			// list data
			// skip (for now; in a later version, we could check for faulty data: particles leaving clipbox or intensity color outside the color range)
			//file.seekg(lst_partCnt * stride, std::ios_base::cur);
			//break;
		}


		// rescale
		if (rescale) 
		{
			if (ownBBox) 
			{
				clippingBox = *ownBBox;
			}
			else 
			{
				clippingBox[0] = glm::vec3(particlePositions.front());
				clippingBox[1] = glm::vec3(particlePositions.front());
				for (auto &particle : particlePositions) 
				{
					// calculate bounding box
					clippingBox[0] = { std::min(clippingBox[0].x, particle.x), std::min(clippingBox[0].y, particle.y), std::min(clippingBox[0].z, particle.z) };
					clippingBox[1] = { std::max(clippingBox[1].x, particle.x), std::max(clippingBox[1].y, particle.y), std::max(clippingBox[1].z, particle.z) };
				}
			}

			const glm::vec3 clippingBoxCenter = (clippingBox[1] + clippingBox[0]) * 0.5f;
			const glm::vec3 invClippingBoxSize = 1.0f / (clippingBox[1] - clippingBox[0]);
			const float invLongestEdge = std::min(std::min(invClippingBoxSize[0], invClippingBoxSize[1]), invClippingBoxSize[2]);

			for (auto &particle : particlePositions) 
			{
				particle -= glm::vec4(clippingBoxCenter,0);
				particle.x *= invLongestEdge;
				particle.y *= invLongestEdge;
				particle.z *= invLongestEdge;
			}
		}

		if (static_cast<uint64_t>(file.tellg()) != frame_table[fi + 1]) {
			std::cerr << "WRN: trailing data after frame " << fi << std::endl;
		}
		// collect info for particle summary
		if (fi == 0) {
			min_partCnt = max_partCnt = frm_partCnt;
		} else {
			if (min_partCnt > frm_partCnt) min_partCnt = frm_partCnt;
			if (max_partCnt < frm_partCnt) max_partCnt = frm_partCnt;
		}
		acc_partCnt += frm_partCnt;
	}
	acc_partCnt /= frame_count;
	// particle summary
	std::cout << "Data Summary" << std::endl;
	std::cout << "    " << frame_count << " time frame" << ((frame_count != 1) ? "s" : "") << std::endl;
	if (min_listCnt == max_listCnt) {
		std::cout << "    " << min_listCnt << " particle list" << ((min_listCnt != 1) ? "s" : "") << " per frame" << std::endl;
	} else {
		std::cout << "    " << min_listCnt << " .. " << max_listCnt << " particle lists per frame" << std::endl;
	}
	if (min_partCnt == max_partCnt) {
		std::cout << "    " << min_partCnt << " particle" << ((min_partCnt != 1) ? "s" : "") << " per frame" << std::endl;
	} else {
		std::cout << "    " << min_partCnt << " .. " << max_partCnt << " particles per frame" << std::endl;
		std::cout << "    " << acc_partCnt << " on average" << std::endl;
	}

	return particlePositions;
}
void Helpers::IO::readParticleData(std::string filePath, uint64_t maxParticles, float &outGlobalRadius, glm::mat2x3 areaOfInterest, uint32_t maxFrameCount, glm::mat2x3 &outBoundingBox, octree& tree, long long in_node_max_memory_consumption, bool rescale, glm::mat2x3 *ownBBox)
{
	//pass over data once to compute the unscaled bounding box
	computeBoundingBox(filePath, maxParticles, outGlobalRadius, areaOfInterest, maxFrameCount, outBoundingBox);

	//now that we ccomputed the unscaled bouding box, we start reading and placing the data in the tree
	placeDataInTree(filePath, maxParticles, outGlobalRadius, areaOfInterest, maxFrameCount, outBoundingBox, tree, in_node_max_memory_consumption, true, &outBoundingBox);
}
void Helpers::IO::computeBoundingBox(std::string filePath, uint64_t maxParticles, float &outGlobalRadius, glm::mat2x3 areaOfInterest, uint32_t maxFrameCount, glm::mat2x3 &outBoundingBox)
{
	//////////////////////////////////////////////////////////////////////////////////////////
	// copied from megamol source code: https://svn.vis.uni-stuttgart.de/trac/megamol/attachment/wiki/mmpld/mmpldinfo.cpp
	//////////////////////////////////////////////////////////////////////////////////////////

	const bool list_framedata = true;

	// open file
	std::ifstream file(filePath, std::ifstream::in | std::ifstream::binary);
	if (!file.is_open()) 
	{
		throw std::runtime_error("Unable to open file");
	}
	// request errors
	file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	// read and check header
	char magicid[6];
	file.read(magicid, 6);
	if (::memcmp(magicid, "MMPLD", 6) != 0) {
		throw std::runtime_error("File does not seem to be of MMPLD format");
	}
	// read version
	uint16_t version;
	file.read(reinterpret_cast<char*>(&version), 2);
	if (version == 100) {
		std::cout << "    mmpld - version 1.0" << std::endl;
	}
	else if (version == 101) {
		std::cout << "    mmpld - version 1.1" << std::endl;
	}
	else if (version == 102) {
		std::cout << "    mmpld - version 1.2" << std::endl;
	}
	else {
		std::cerr << "    mmpld - version " << (version / 100) << "." << (version % 100) << std::endl;
		//throw std::runtime_error("Unsupported mmpld version encountered");
	}
	// read number of frames
	uint32_t frame_count;
	file.read(reinterpret_cast<char*>(&frame_count), 4);
	std::cout << "Number of frames: " << frame_count << std::endl;
	// read bounding boxes

	glm::mat2x3 boundingBox;
	file.read(reinterpret_cast<char*>(&boundingBox[0]), sizeof(float)* 6);
	std::cout << "Bounding box: (" << boundingBox[0][0] << ", " << boundingBox[0][1] << ", " << boundingBox[0][2] << ", "
		<< boundingBox[1][0] << ", " << boundingBox[1][1] << ", " << boundingBox[1][2] << ")" << std::endl;

	glm::mat2x3 &clippingBox = outBoundingBox;
	file.read(reinterpret_cast<char*>(&clippingBox[0]), sizeof(float)* 6);
	std::cout << "Clipping box: (" << clippingBox[0][0] << ", " << clippingBox[0][1] << ", " << clippingBox[0][2] << ", "
		<< clippingBox[1][0] << ", " << clippingBox[1][1] << ", " << clippingBox[1][2] << ")" << std::endl;
	// abort if no data
	if (frame_count <= 0) {
		throw std::runtime_error("No data");
	}
	// frame index table
	std::vector<uint64_t> frame_table(frame_count + 1);
	file.read(reinterpret_cast<char*>(frame_table.data()), (frame_count + 1) * 8);
	if (static_cast<uint64_t>(file.tellg()) != frame_table[0]) {
		std::cerr << "WRN: dead data trailing head" << std::endl;
	}
	file.seekg(0, std::ios_base::end);
	if (static_cast<uint64_t>(file.tellg()) < frame_table[frame_count]) {
		std::cerr << "WRN: file truncated" << std::endl;
	}
	if (static_cast<uint64_t>(file.tellg()) > frame_table[frame_count]) {
		std::cerr << "WRN: dead data trailing body" << std::endl;
	}
	{
		std::string errmsg;
		for (unsigned int fi = 0; fi < maxFrameCount; fi++) {
			if (frame_table[fi + 1] <= frame_table[fi]) {
				errmsg += "Frame table corrupted at frame ";
				errmsg += fi;
				errmsg += "\n";
			}
		}
		if (!errmsg.empty()) {
			throw std::runtime_error(errmsg);
		}
	}
	// test frames
	uint32_t min_listCnt;
	uint32_t max_listCnt;
	uint64_t min_partCnt;
	uint64_t acc_partCnt(0);
	uint64_t max_partCnt;
	uint64_t frm_partCnt;
	uint64_t lst_partCnt;

	if (maxFrameCount > 0) {
		frame_count = std::min(maxFrameCount, frame_count);
	}

	for (uint32_t fi = frame_count - 1; fi < frame_count; fi++) {
		if (list_framedata) std::cout << "Frame #" << fi;
		file.seekg(frame_table[fi]);
		if (version == 102) {
			float theTime;
			file.read(reinterpret_cast<char*>(&theTime), 4);
			if (list_framedata) std::cout << " Simulation time " << theTime;
		}
		// number of lists
		uint32_t lists_cnt;
		file.read(reinterpret_cast<char*>(&lists_cnt), 4);
		if (list_framedata) std::cout << " - " << lists_cnt << " list" << ((lists_cnt != 1) ? "s" : "") << std::endl;
		if (fi == 0) {
			min_listCnt = max_listCnt = lists_cnt;
		}
		else {
			if (min_listCnt > lists_cnt) min_listCnt = lists_cnt;
			if (max_listCnt < lists_cnt) max_listCnt = lists_cnt;
		}
		frm_partCnt = 0;
		//auto particlePostionsInsertIterator = std::back_inserter(particlePositions);
		for (uint32_t li = 0; li < lists_cnt; li++) {
			// list data format info
			uint8_t vert_type, col_type;
			size_t vrt_size;
			size_t col_size;
			file.read(reinterpret_cast<char*>(&vert_type), 1);
			file.read(reinterpret_cast<char*>(&col_type), 1);
			if (list_framedata) std::cout << "    #" << li << ": ";
			switch (vert_type) {
			case 1: vrt_size = 12; if (list_framedata) std::cout << "VERTDATA_FLOAT_XYZ"; break;
			case 2: vrt_size = 16; if (list_framedata) std::cout << "VERTDATA_FLOAT_XYZR"; break;
			case 3: vrt_size = 6; if (list_framedata) std::cout << "VERTDATA_SHORT_XYZ"; break;
			case 0: // falls through
			default: vrt_size = 0; if (list_framedata) std::cout << "VERTDATA_NONE"; break;
			}
			if (list_framedata) std::cout << ", ";
			if (vert_type == 0) col_type = 0;
			switch (col_type) {
			case 1: col_size = 3; if (list_framedata) std::cout << "COLDATA_UINT8_RGB"; break;
			case 2: col_size = 4; if (list_framedata) std::cout << "COLDATA_UINT8_RGBA"; break;
			case 3: col_size = 4; if (list_framedata) std::cout << "COLDATA_FLOAT_I"; break;
			case 4: col_size = 12; if (list_framedata) std::cout << "COLDATA_FLOAT_RGB"; break;
			case 5: col_size = 16; if (list_framedata) std::cout << "COLDATA_FLOAT_RGBA"; break;
			case 0: // falls through
			default: col_size = 0; if (list_framedata) std::cout << "COLDATA_NONE"; break;
			}
			if (list_framedata) std::cout << std::endl;
			size_t stride = vrt_size + col_size;
			if (list_framedata) std::cout << "        " << stride << " byte" << ((stride != 1) ? "s" : "") << " per particle" << std::endl;
			float glob_rad(0.05f);
			if ((vert_type == 1) || (vert_type == 3)) {
				file.read(reinterpret_cast<char*>(&glob_rad), 4);
				if (list_framedata) std::cout << "        global radius: " << glob_rad << std::endl;
			}
			float col_range[2];
			if (col_type == 0) {
				uint8_t col[4];
				file.read(reinterpret_cast<char*>(col), 4);
				if (list_framedata) std::cout << "        global color: (" << col[0] << ", " << col[1] << ", " << col[2] << ", " << col[3] << ")" << std::endl;
			}
			else if (col_type == 3) {
				file.read(reinterpret_cast<char*>(col_range), 8);
				if (list_framedata) std::cout << "        intensity color range: [" << col_range[0] << ", " << col_range[1] << "]" << std::endl;
			}
			col_range[0] = 0.0f;
			col_range[1] = 1.0f;
			file.read(reinterpret_cast<char*>(&lst_partCnt), 8);
			if (list_framedata) std::cout << "        " << lst_partCnt << " particle" << ((lst_partCnt != 1) ? "s" : "") << std::endl;
			frm_partCnt += lst_partCnt;




			//////////////////////////////////////////////////////////////////////////////////////////
			// own code
			//
			// read particle data
			//assert(vrt_size == 12);

			if (vrt_size == 12) {
				outGlobalRadius = glob_rad;

				auto particleCount = std::min(maxParticles, lst_partCnt);

				std::unique_ptr<char[]> rawData = std::unique_ptr<char[]>(new char[stride]);
				if (particleCount > 0)
				{
					file.read(reinterpret_cast<char*>(rawData.get()), stride);
					auto particle = *reinterpret_cast<ParticleType*>(rawData.get());

					//debug
					if (lists_cnt > 1)
					{
						if (li == 0)
							particle.w = 2.011f;
						else
							particle.w = 1.78f;
					}
					else
						particle.w = glob_rad;
					//end debug

					//particlePostionsInsertIterator = particle;

					clippingBox[0] = glm::vec3(particle);
					clippingBox[1] = glm::vec3(particle);
				}

				for (uint64_t particleI = 1; particleI < particleCount; ++particleI) 
				{
					file.read(reinterpret_cast<char*>(rawData.get()), stride);

					// ignore additional data
					auto particle = *reinterpret_cast<ParticleType*>(rawData.get());

					//debug

					if (lists_cnt > 1)
					{
						if (li == 0)
							particle.w = 2.011f;
						else
							particle.w = 1.78f;
					}
					else
						particle.w = glob_rad;
					//end debug


		
					//particlePostionsInsertIterator = particle;
					// calculate bounding box
					clippingBox[0] = { std::min(clippingBox[0].x, particle.x), std::min(clippingBox[0].y, particle.y), std::min(clippingBox[0].z, particle.z) };
					clippingBox[1] = { std::max(clippingBox[1].x, particle.x), std::max(clippingBox[1].y, particle.y), std::max(clippingBox[1].z, particle.z) };
				}
			}

			//////////////////////////////////////////////////////////////////////////////////////////



			// list data
			// skip (for now; in a later version, we could check for faulty data: particles leaving clipbox or intensity color outside the color range)
			//file.seekg(lst_partCnt * stride, std::ios_base::cur);
			//break;
		}



		if (static_cast<uint64_t>(file.tellg()) != frame_table[fi + 1]) {
			std::cerr << "WRN: trailing data after frame " << fi << std::endl;
		}
		// collect info for particle summary
		if (fi == 0) {
			min_partCnt = max_partCnt = frm_partCnt;
		}
		else {
			if (min_partCnt > frm_partCnt) min_partCnt = frm_partCnt;
			if (max_partCnt < frm_partCnt) max_partCnt = frm_partCnt;
		}
		acc_partCnt += frm_partCnt;
	}
	acc_partCnt /= frame_count;
	// particle summary
	std::cout << "Data Summary" << std::endl;
	std::cout << "    " << frame_count << " time frame" << ((frame_count != 1) ? "s" : "") << std::endl;
	if (min_listCnt == max_listCnt) {
		std::cout << "    " << min_listCnt << " particle list" << ((min_listCnt != 1) ? "s" : "") << " per frame" << std::endl;
	}
	else {
		std::cout << "    " << min_listCnt << " .. " << max_listCnt << " particle lists per frame" << std::endl;
	}
	if (min_partCnt == max_partCnt) {
		std::cout << "    " << min_partCnt << " particle" << ((min_partCnt != 1) ? "s" : "") << " per frame" << std::endl;
	}
	else {
		std::cout << "    " << min_partCnt << " .. " << max_partCnt << " particles per frame" << std::endl;
		std::cout << "    " << acc_partCnt << " on average" << std::endl;
	}
}
void Helpers::IO::placeDataInTree(std::string filePath, uint64_t maxParticles, float &outGlobalRadius, glm::mat2x3 areaOfInterest, uint32_t maxFrameCount, glm::mat2x3 &outBoundingBox, octree& tree, long long in_node_max_memory_consumption, bool rescale, glm::mat2x3 *ownBBox)
{
	//////////////////////////////////////////////////////////////////////////////////////////
	// copied from megamol source code: https://svn.vis.uni-stuttgart.de/trac/megamol/attachment/wiki/mmpld/mmpldinfo.cpp
	//////////////////////////////////////////////////////////////////////////////////////////


	glm::mat2x3 &clippingBox = outBoundingBox;

	//compute scale parameters of bounding box
	const glm::vec3 clippingBoxCenter = (clippingBox[1] + clippingBox[0]) * 0.5f;
	const glm::vec3 invClippingBoxSize = 1.0f / (clippingBox[1] - clippingBox[0]);
	const float invLongestEdge = std::min(std::min(invClippingBoxSize[0], invClippingBoxSize[1]), invClippingBoxSize[2]);

	//create root of tree as scaled bounding box
	glm::mat2x3 scaledBox;
	scaledBox[0] = clippingBox[0] - clippingBoxCenter;
	scaledBox[0].x *= invLongestEdge;
	scaledBox[0].y *= invLongestEdge;
	scaledBox[0].z *= invLongestEdge;

	scaledBox[1] = clippingBox[1] - clippingBoxCenter;
	scaledBox[1].x *= invLongestEdge;
	scaledBox[1].y *= invLongestEdge;
	scaledBox[1].z *= invLongestEdge;

	glm::vec3 modelExtent = scaledBox[1] - scaledBox[0];

	glm::vec3 halfextent = 0.5f*modelExtent;
	//halfextent = std::max(modelExtent.y, halfextent);
	//halfextent = std::max(modelExtent.z, halfextent);
	//halfextent *= 0.5f;

	tree.init(clippingBoxCenter - clippingBoxCenter, halfextent, 0, 0, in_node_max_memory_consumption);



	const bool list_framedata = true;

	// open file
	std::ifstream file(filePath, std::ifstream::in | std::ifstream::binary);
	if (!file.is_open()) {
		throw std::runtime_error("Unable to open file");
	}
	// request errors
	file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	// read and check header
	char magicid[6];
	file.read(magicid, 6);
	if (::memcmp(magicid, "MMPLD", 6) != 0) {
		throw std::runtime_error("File does not seem to be of MMPLD format");
	}
	// read version
	uint16_t version;
	file.read(reinterpret_cast<char*>(&version), 2);
	if (version == 100) {
		std::cout << "    mmpld - version 1.0" << std::endl;
	}
	else if (version == 101) {
		std::cout << "    mmpld - version 1.1" << std::endl;
	}
	else if (version == 102) {
		std::cout << "    mmpld - version 1.2" << std::endl;
	}
	else {
		std::cerr << "    mmpld - version " << (version / 100) << "." << (version % 100) << std::endl;
		//throw std::runtime_error("Unsupported mmpld version encountered");
	}
	// read number of frames
	uint32_t frame_count;
	file.read(reinterpret_cast<char*>(&frame_count), 4);
	std::cout << "Number of frames: " << frame_count << std::endl;
	// read bounding boxes

	glm::mat2x3 boundingBox;
	file.read(reinterpret_cast<char*>(&boundingBox[0]), sizeof(float)* 6);
	std::cout << "Bounding box: (" << boundingBox[0][0] << ", " << boundingBox[0][1] << ", " << boundingBox[0][2] << ", "
		<< boundingBox[1][0] << ", " << boundingBox[1][1] << ", " << boundingBox[1][2] << ")" << std::endl;

	
	file.read(reinterpret_cast<char*>(&clippingBox[0]), sizeof(float)* 6);
	std::cout << "Clipping box: (" << clippingBox[0][0] << ", " << clippingBox[0][1] << ", " << clippingBox[0][2] << ", "
		<< clippingBox[1][0] << ", " << clippingBox[1][1] << ", " << clippingBox[1][2] << ")" << std::endl;
	// abort if no data
	if (frame_count <= 0) {
		throw std::runtime_error("No data");
	}
	// frame index table
	std::vector<uint64_t> frame_table(frame_count + 1);
	file.read(reinterpret_cast<char*>(frame_table.data()), (frame_count + 1) * 8);
	if (static_cast<uint64_t>(file.tellg()) != frame_table[0]) {
		std::cerr << "WRN: dead data trailing head" << std::endl;
	}
	file.seekg(0, std::ios_base::end);
	if (static_cast<uint64_t>(file.tellg()) < frame_table[frame_count]) {
		std::cerr << "WRN: file truncated" << std::endl;
	}
	if (static_cast<uint64_t>(file.tellg()) > frame_table[frame_count]) {
		std::cerr << "WRN: dead data trailing body" << std::endl;
	}
	{
		std::string errmsg;
		for (unsigned int fi = 0; fi < maxFrameCount; fi++) {
			if (frame_table[fi + 1] <= frame_table[fi]) {
				errmsg += "Frame table corrupted at frame ";
				errmsg += fi;
				errmsg += "\n";
			}
		}
		if (!errmsg.empty()) {
			throw std::runtime_error(errmsg);
		}
	}
	// test frames
	uint32_t min_listCnt;
	uint32_t max_listCnt;
	uint64_t min_partCnt;
	uint64_t acc_partCnt(0);
	uint64_t max_partCnt;
	uint64_t frm_partCnt;
	uint64_t lst_partCnt;

	if (maxFrameCount > 0) {
		frame_count = std::min(maxFrameCount, frame_count);
	}

	for (uint32_t fi = frame_count - 1; fi < frame_count; fi++) {
		if (list_framedata) std::cout << "Frame #" << fi;
		file.seekg(frame_table[fi]);
		if (version == 102) {
			float theTime;
			file.read(reinterpret_cast<char*>(&theTime), 4);
			if (list_framedata) std::cout << " Simulation time " << theTime;
		}
		// number of lists
		uint32_t lists_cnt;
		file.read(reinterpret_cast<char*>(&lists_cnt), 4);
		if (list_framedata) std::cout << " - " << lists_cnt << " list" << ((lists_cnt != 1) ? "s" : "") << std::endl;
		if (fi == 0) {
			min_listCnt = max_listCnt = lists_cnt;
		}
		else {
			if (min_listCnt > lists_cnt) min_listCnt = lists_cnt;
			if (max_listCnt < lists_cnt) max_listCnt = lists_cnt;
		}
		frm_partCnt = 0;
//		auto particlePostionsInsertIterator = std::back_inserter(particlePositions);
		for (uint32_t li = 0; li < lists_cnt; li++) {
			// list data format info
			uint8_t vert_type, col_type;
			size_t vrt_size;
			size_t col_size;
			file.read(reinterpret_cast<char*>(&vert_type), 1);
			file.read(reinterpret_cast<char*>(&col_type), 1);
			if (list_framedata) std::cout << "    #" << li << ": ";
			switch (vert_type) {
			case 1: vrt_size = 12; if (list_framedata) std::cout << "VERTDATA_FLOAT_XYZ"; break;
			case 2: vrt_size = 16; if (list_framedata) std::cout << "VERTDATA_FLOAT_XYZR"; break;
			case 3: vrt_size = 6; if (list_framedata) std::cout << "VERTDATA_SHORT_XYZ"; break;
			case 0: // falls through
			default: vrt_size = 0; if (list_framedata) std::cout << "VERTDATA_NONE"; break;
			}
			if (list_framedata) std::cout << ", ";
			if (vert_type == 0) col_type = 0;
			switch (col_type) {
			case 1: col_size = 3; if (list_framedata) std::cout << "COLDATA_UINT8_RGB"; break;
			case 2: col_size = 4; if (list_framedata) std::cout << "COLDATA_UINT8_RGBA"; break;
			case 3: col_size = 4; if (list_framedata) std::cout << "COLDATA_FLOAT_I"; break;
			case 4: col_size = 12; if (list_framedata) std::cout << "COLDATA_FLOAT_RGB"; break;
			case 5: col_size = 16; if (list_framedata) std::cout << "COLDATA_FLOAT_RGBA"; break;
			case 0: // falls through
			default: col_size = 0; if (list_framedata) std::cout << "COLDATA_NONE"; break;
			}
			if (list_framedata) std::cout << std::endl;
			size_t stride = vrt_size + col_size;
			if (list_framedata) std::cout << "        " << stride << " byte" << ((stride != 1) ? "s" : "") << " per particle" << std::endl;
			float glob_rad(0.05f);
			if ((vert_type == 1) || (vert_type == 3)) {
				file.read(reinterpret_cast<char*>(&glob_rad), 4);
				if (list_framedata) std::cout << "        global radius: " << glob_rad << std::endl;
			}
			float col_range[2];
			if (col_type == 0) {
				uint8_t col[4];
				file.read(reinterpret_cast<char*>(col), 4);
				if (list_framedata) std::cout << "        global color: (" << col[0] << ", " << col[1] << ", " << col[2] << ", " << col[3] << ")" << std::endl;
			}
			else if (col_type == 3) {
				file.read(reinterpret_cast<char*>(col_range), 8);
				if (list_framedata) std::cout << "        intensity color range: [" << col_range[0] << ", " << col_range[1] << "]" << std::endl;
			}
			col_range[0] = 0.0f;
			col_range[1] = 1.0f;
			file.read(reinterpret_cast<char*>(&lst_partCnt), 8);
			if (list_framedata) std::cout << "        " << lst_partCnt << " particle" << ((lst_partCnt != 1) ? "s" : "") << std::endl;
			frm_partCnt += lst_partCnt;




			//////////////////////////////////////////////////////////////////////////////////////////
			// own code
			//
			// read particle data
			//assert(vrt_size == 12);

			if (vrt_size == 12) {
				outGlobalRadius = glob_rad;

				auto particleCount = std::min(maxParticles, lst_partCnt);

				std::unique_ptr<char[]> rawData = std::unique_ptr<char[]>(new char[stride]);
				if (particleCount > 0)
				{
					file.read(reinterpret_cast<char*>(rawData.get()), stride);
					auto particle = *reinterpret_cast<ParticleType*>(rawData.get());

					//debug
					if (lists_cnt > 1)
					{
						if (li == 0)
							particle.w = 2.011f;
						else
							particle.w = 1.78f;
					}
					else
						particle.w = glob_rad;
					//end debug

					//shift and scale particle
					particle -= glm::vec4(clippingBoxCenter, 0);
					particle.x *= invLongestEdge;
					particle.y *= invLongestEdge;
					particle.z *= invLongestEdge;
					particle.w *= invLongestEdge;

					tree.placeParticle(0, particle,false);
				}

				for (uint64_t particleI = 1; particleI < particleCount; ++particleI) {
					file.read(reinterpret_cast<char*>(rawData.get()), stride);

					// ignore additional data
					auto particle = *reinterpret_cast<ParticleType*>(rawData.get());

					//debug

					if (lists_cnt > 1)
					{
						if (li == 0)
							particle.w = 2.011f;
						else
							particle.w = 1.78f;
					}
					else
						particle.w = glob_rad;
					//end debug

					//shift and scale particle
					particle -= glm::vec4(clippingBoxCenter, 0);
					particle.x *= invLongestEdge;
					particle.y *= invLongestEdge;
					particle.z *= invLongestEdge;
					particle.w *= invLongestEdge;

					tree.placeParticle(0,particle,false);

				}
			}

			//////////////////////////////////////////////////////////////////////////////////////////



			// list data
			// skip (for now; in a later version, we could check for faulty data: particles leaving clipbox or intensity color outside the color range)
			//file.seekg(lst_partCnt * stride, std::ios_base::cur);
			//break;
		}


		//update clipping box to be the scaled one
		clippingBox = scaledBox;

		if (static_cast<uint64_t>(file.tellg()) != frame_table[fi + 1]) {
			std::cerr << "WRN: trailing data after frame " << fi << std::endl;
		}
		// collect info for particle summary
		if (fi == 0) {
			min_partCnt = max_partCnt = frm_partCnt;
		}
		else {
			if (min_partCnt > frm_partCnt) min_partCnt = frm_partCnt;
			if (max_partCnt < frm_partCnt) max_partCnt = frm_partCnt;
		}
		acc_partCnt += frm_partCnt;
	}
	acc_partCnt /= frame_count;
	// particle summary
	std::cout << "Data Summary" << std::endl;
	std::cout << "    " << frame_count << " time frame" << ((frame_count != 1) ? "s" : "") << std::endl;
	if (min_listCnt == max_listCnt) {
		std::cout << "    " << min_listCnt << " particle list" << ((min_listCnt != 1) ? "s" : "") << " per frame" << std::endl;
	}
	else {
		std::cout << "    " << min_listCnt << " .. " << max_listCnt << " particle lists per frame" << std::endl;
	}
	if (min_partCnt == max_partCnt) {
		std::cout << "    " << min_partCnt << " particle" << ((min_partCnt != 1) ? "s" : "") << " per frame" << std::endl;
	}
	else {
		std::cout << "    " << min_partCnt << " .. " << max_partCnt << " particles per frame" << std::endl;
		std::cout << "    " << acc_partCnt << " on average" << std::endl;
	}
}