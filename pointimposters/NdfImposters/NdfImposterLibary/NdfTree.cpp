#include "NdfTree.h"

#include "FileIO.h"

#include <fstream>
#include <iostream>
#include <memory>

#include <limits>

using namespace NdfImposters;

// instatiate for static linking
template class NdfTree<float>;
template class NdfTreeLevel<float>;
template class NdfHistogramRef<float>;

template <typename BinType>
NdfHistogramRef<BinType>::NdfHistogramRef(BinType *const ndfDataPointer, glm::ivec2 spatial, glm::ivec2 view, glm::ivec2 histogramResolution)
	: NdfDataPointer(ndfDataPointer), View(view), Spatial(spatial), HistogramResolution(histogramResolution) {
}

template <typename BinType>
NdfTreeLevel<BinType>::NdfTreeLevel(glm::ivec2 spatialResolution, glm::ivec2 histogramBinResolution, glm::ivec2 viewDirectionResolution) :
	spatialResolution_(spatialResolution), histogramBinResolution_(histogramBinResolution), viewDirectionResolution_(viewDirectionResolution) {

	// initialize the storage for the whole tree level
	auto binCount = histogramBinResolution.x * histogramBinResolution.y * viewDirectionResolution.x * viewDirectionResolution.y * spatialResolution.x * spatialResolution.y;
	rawContigousData.resize(binCount, 0.0f);

	// initialize the view meta data
	{
		auto viewScale = glm::vec2(
			1.0f / static_cast<float>(viewDirectionResolution.x-1),
			1.0f / static_cast<float>(viewDirectionResolution.y-1));

		metaData.reserve(viewDirectionResolution.x * viewDirectionResolution.y);
		for(auto viewY = 0; viewY < viewDirectionResolution.y; ++viewY) {
			auto viewDirectionY = 0.0f;
			if(viewDirectionResolution.y > 1) {
				viewDirectionY = viewY * viewScale.y - 0.5f;
			}

			for(auto viewX = 0; viewX < viewDirectionResolution.x; ++viewX) {
				// TODO: half on the front, half on the back hemisphere
				// generate equially dsitributed normalized view directions on the hemisphere.
				auto viewDirectionX = 0.0f;
				if(viewDirectionResolution.x > 1) {
					viewDirectionX = viewX * viewScale.x - 0.5f;
				}

				auto viewDirectionZ = std::sqrt(1.0f - viewDirectionX*viewDirectionX - viewDirectionY*viewDirectionY);

				metaData.emplace_back(glm::vec3(viewDirectionX, viewDirectionY, viewDirectionZ), glm::ivec2(viewX, viewY));
			}
		}
	}
}

template <typename BinType>
glm::ivec2 NdfTreeLevel<BinType>::GetHistogramResolution() const {
	return histogramBinResolution_;
}

template <typename BinType>
glm::ivec2 NdfTreeLevel<BinType>::GetViewDirectionResolution() const {
	return viewDirectionResolution_;
}

template <typename BinType>
glm::ivec2 NdfTreeLevel<BinType>::GetSpatialResolution() const {
	return spatialResolution_;		
}

template <typename BinType>
NdfHistogramRef<BinType> NdfTreeLevel<BinType>::GetHistogram(glm::ivec2 spatial, glm::ivec2 view) const {
	auto dataIndex = GetDataIndex(spatial, view, {0, 0});

	assert(rawContigousData.size() > dataIndex);

	// FIXME: why is the const cast required here? And is it safe?
	auto ndfDataPointer = const_cast<BinType*>(&rawContigousData[dataIndex]);

	return NdfHistogramRef<BinType>(ndfDataPointer, spatial, view, histogramBinResolution_);
}

template <typename BinType>
int NdfTreeLevel<BinType>::GetDataIndex(glm::ivec2 spatial, glm::ivec2 view, glm::ivec2 bin) const {
	return (spatial.y * spatialResolution_.x + spatial.x) * viewDirectionResolution_.y * viewDirectionResolution_.x * histogramBinResolution_.y * histogramBinResolution_.x
		+ (view.y * viewDirectionResolution_.x + view.x) * histogramBinResolution_.y * histogramBinResolution_.x
		+ (bin.y * histogramBinResolution_.x + bin.x);
}

template <typename BinType>
GLuint NdfTreeLevel<BinType>::GetShaderStorageBufferOject() const {
	return ssbo.GlSsbo;
}

template <typename BinType>
void NdfTreeLevel<BinType>::Clear() {
	std::fill(rawContigousData.begin(), rawContigousData.end(), 0);
}

template <typename BinType>
void NdfTreeLevel<BinType>::UploadData(GLuint ssboBindingPointIndex) {
	std::cout << "Uploading SSBO to binding point " << ssboBindingPointIndex << std::endl;

	assert(glGenBuffers);
	assert(glBindBuffer);
	assert(glBufferData);
	assert(glBindBufferBase);

	// check if ssbo already exists and has the right size
	if(ssbo.GlSsbo == 0) {
		glGenBuffers(1, &ssbo.GlSsbo);
	}
	
	// upload normalized data to ssbo
	ssbo.Size = rawContigousData.size() * sizeof(*rawContigousData.begin());

#if 1
	// check if ssbo size is supported
	if (!glGetInteger64v) {
		glGetInteger64v = (PFNGLGETINTEGER64VPROC)wglGetProcAddress("glGetInteger64v");
	}
	if (!glGetInteger64i_v) {
		glGetInteger64i_v = (PFNGLGETINTEGER64I_VPROC)wglGetProcAddress("glGetInteger64i_v");
	}

	GLint64 maxComputeStorageBlocks = 0;
	glGetInteger64v(GL_MAX_COMPUTE_SHADER_STORAGE_BLOCKS, &maxComputeStorageBlocks);
	
	GLint64 maxFramgentStorageBlocks = 0;
	glGetInteger64v(GL_MAX_FRAGMENT_SHADER_STORAGE_BLOCKS, &maxFramgentStorageBlocks);

	GLint64 maxStorageBlockSize = 0;
	glGetInteger64v(GL_MAX_SHADER_STORAGE_BLOCK_SIZE, &maxStorageBlockSize);

	assert(ssbo.Size < maxStorageBlockSize);
	assert(maxFramgentStorageBlocks >= 2);
	assert(maxComputeStorageBlocks >= 2);
#endif

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo.GlSsbo);
	glBufferData(GL_SHADER_STORAGE_BUFFER, ssbo.Size, rawContigousData.data(), GL_DYNAMIC_COPY);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, ssbo.GlSsbo);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	auto glErr = glGetError();
    if (glErr != GL_NO_ERROR) {
		std::cout << "glError: " << __FILE__ << " " << __LINE__ << std::endl;
    }
}

template <typename BinType>
void NdfTreeLevel<BinType>::DownloadData() {
	std::cout << "Donwloading SSBO" << std::endl;

	assert(glBindBuffer);
	assert(glBufferData);
	assert(glBindBufferBase);

	assert(ssbo.GlSsbo != 0);

	// check if ssbo already exists and has the right size
	if(ssbo.GlSsbo == 0) {
		return;
	}
	
	// upload normalized data to ssbo
	ssbo.Size = rawContigousData.size() * sizeof(*rawContigousData.begin());

	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo.GlSsbo);
	auto readMap = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
	// FIXME: memcpy is unsafe
	memcpy(rawContigousData.data(), readMap, ssbo.Size);
	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	auto glErr = glGetError();
    if (glErr != GL_NO_ERROR) {
		std::cout << "glError: " << __FILE__ << " " << __LINE__ << std::endl;
    }
}

template <typename BinType>
NdfTree<BinType>::NdfTree() {
	histogramBinResolution = glm::ivec2(0, 0);
}

template <typename BinType>
void NdfTree<BinType>::InitializeStorage(int treeLevels, glm::ivec2 histogramBinResolution, glm::ivec2 viewDirectionResolution, glm::ivec2 spatialResolution) {
	this->histogramBinResolution = histogramBinResolution;
	this->viewDirectionResolution = viewDirectionResolution;
	this->spatialResolution = spatialResolution;

	auto bytesPerSpatial = sizeof(BinType) * histogramBinResolution.x * histogramBinResolution.y * viewDirectionResolution.x * viewDirectionResolution.y;
	
	// calculate memory requirement
	decltype(bytesPerSpatial) memoryRequirement = 0;
	{
		auto currentSpatialResolution = spatialResolution;
		for(auto mipIndex = 0; mipIndex < treeLevels; ++mipIndex) {
			if(currentSpatialResolution.x % 2 != 0 || currentSpatialResolution.y % 2 != 0) {
				std::cerr << "Warning: only spatial resolution that are powers of two are tested" << std::endl;
			}

			memoryRequirement += bytesPerSpatial * currentSpatialResolution.x * currentSpatialResolution.y;

			currentSpatialResolution /= 2;
		}
	}
	std::cout << "Ndf Tree Memory requirement: " << memoryRequirement << " bytes = " << memoryRequirement / 1048576 << " mBytes" << std::endl;

	// allocate tree memory
	auto currentSpatialResolution = spatialResolution;
	for(auto mipIndex = 0; mipIndex < treeLevels; ++mipIndex) {
		levels.emplace_back(currentSpatialResolution, histogramBinResolution, viewDirectionResolution);

		currentSpatialResolution /= 2;
	}
}

template <typename BinType>
glm::ivec2 NdfTree<BinType>::GetHistogramResolution() const {
	return histogramBinResolution;
}

template <typename BinType>
glm::ivec2 NdfTree<BinType>::GetViewDirectionResolution() const {
	return viewDirectionResolution;
}

template <typename BinType>
glm::ivec2 NdfTree<BinType>::GetSpatialResolution() const {
	return spatialResolution;		
}

template <typename BinType>
std::vector<NdfTreeLevel<BinType>> &NdfTree<BinType>::GetLevels() {
	return levels;
}

template <typename BinType>
int NdfTree<BinType>::GetLevelCount() const {
//	assert(levels.size() <= std::numeric_limits<int>::max());
	assert(levels.size() >= 0);

	return static_cast<int>(levels.size());
}

template <typename BinType>
void NdfTree<BinType>::Clear() {
	for(auto &level : levels) {
		level.Clear();
	}
}

template <typename BinType>
void NdfTree<BinType>::UploadData() {
	GLuint ssboBindingPointIndex = 0;
	for(auto &level : levels) {
		level.UploadData(ssboBindingPointIndex);

		++ssboBindingPointIndex;
	}
}

template <typename BinType>
void NdfTree<BinType>::Downsample() {
	// TODO: NYI
	/*for(auto &level : levels) {
		
	}

	
	for(int sourceLevelIndex = 0; sourceLevelIndex < MipLevelCount; ++sourceLevelIndex) {
		// downsample
		hierarchy.emplace_back(new NdfHierarchyLevel(
			std::max(1ll, hierarchy[sourceLevelIndex]->Width / 2),
			std::max(1ll, hierarchy[sourceLevelIndex]->Height / 2),
			std::max(1ll, hierarchy[sourceLevelIndex]->Depth / 2)));
		auto &targetLevel = hierarchy[sourceLevelIndex+1];

		for (int z = 0; z < std::max(1ll, targetLevel->Depth); ++z) {
			for (int y = 0; y < std::max(1ll, targetLevel->Height); ++y) {
				for (int x = 0; x < std::max(1ll, targetLevel->Width); ++x) {
					auto &ndf = GetNormalDistributionFunctionRef({ x, y, z }, sourceLevelIndex+1);
					
					auto &leftUpper = GetNormalDistributionFunctionRef({	x * 2,		y * 2,		z * 2 }, sourceLevelIndex);
					auto &rightUpper = GetNormalDistributionFunctionRef({	x * 2 + 1,	y * 2,		z * 2 }, sourceLevelIndex);
					auto &topUpper = GetNormalDistributionFunctionRef({		x * 2,		y * 2 + 1,	z * 2 }, sourceLevelIndex);
					auto &bottomUpper = GetNormalDistributionFunctionRef({	x * 2 + 1,	y * 2 + 1,	z * 2 }, sourceLevelIndex);
					// ignoring z...

					for(auto viewIndex = 0; viewIndex < ndf.ViewDependentHistograms1D().size(); ++viewIndex) {
						Ndf::NdfProbabilityType normalization = 0.0f;
						for(auto binIndex = 0; binIndex < ndf.ViewDependentHistograms1D()[viewIndex].HistogramData1D.size(); ++binIndex) {
							ndf.ViewDependentHistograms1D()[viewIndex].HistogramData1D[binIndex] = 
								leftUpper.ViewDependentHistograms1D()[viewIndex].HistogramData1D[binIndex] +
								rightUpper.ViewDependentHistograms1D()[viewIndex].HistogramData1D[binIndex] +
								topUpper.ViewDependentHistograms1D()[viewIndex].HistogramData1D[binIndex] +
								bottomUpper.ViewDependentHistograms1D()[viewIndex].HistogramData1D[binIndex];

							//normalization += ndf.ViewDependentHistograms1D()[viewIndex].HistogramData1D[binIndex];
						}

						normalization = 1.0f / 4.0f;
						for(auto &bin : ndf.ViewDependentHistograms1D()[viewIndex].HistogramData1D) {
							bin *= normalization;
						}
					}
				}
			}
		}

		targetLevel->UploadSSBO(sourceLevelIndex+1);
	}*/
}

template <typename BinType>
void NdfTree<BinType>::WriteToFile(std::string filePath, bool compressed) {
	//for(auto &level : levels) {
	
	// store highest resolution level only - the other ones can be easily calculated using downsampling
	auto &level = *levels.begin(); {
		if(!compressed) {
			Helpers::IO::FileGuard<std::ofstream> fileWriter(filePath, std::ofstream::binary | std::ofstream::out);

			fileWriter.file.write(reinterpret_cast<const char*>(&level.GetHistogramResolution()), sizeof(level.GetHistogramResolution()));
			fileWriter.file.write(reinterpret_cast<const char*>(&level.GetViewDirectionResolution()), sizeof(level.GetViewDirectionResolution()));
			fileWriter.file.write(reinterpret_cast<const char*>(&level.GetSpatialResolution()), sizeof(level.GetSpatialResolution()));
			fileWriter.file.write(reinterpret_cast<const char*>(level.GetMetaData().data()), level.GetMetaData().size() * sizeof(*level.GetMetaData().begin()));
			fileWriter.file.write(reinterpret_cast<const char*>(level.GetRawData().data()), level.GetRawData().size() * sizeof(*level.GetRawData().begin()));
		} else {
			Helpers::IO::ArchiveWriter compressedFileWriter(filePath);

			compressedFileWriter.write(reinterpret_cast<const char*>(&level.GetHistogramResolution()), sizeof(level.GetHistogramResolution()));
			compressedFileWriter.write(reinterpret_cast<const char*>(&level.GetViewDirectionResolution()), sizeof(level.GetViewDirectionResolution()));
			compressedFileWriter.write(reinterpret_cast<const char*>(&level.GetSpatialResolution()), sizeof(level.GetSpatialResolution()));
			compressedFileWriter.write(reinterpret_cast<const char*>(level.GetMetaData().data()), level.GetMetaData().size() * sizeof(*level.GetMetaData().begin()));
			compressedFileWriter.write(reinterpret_cast<const char*>(level.GetRawData().data()), level.GetRawData().size() * sizeof(*level.GetRawData().begin()));
		}
	}
}

template <typename BinType>
void NdfTree<BinType>::ReadFromFile(std::string filePath, bool compressed) {
	//for(auto &level : levels) {
	
	// TODO: read batches separately into same data

	// store highest resolution level only - the other ones can be easily calculated using downsampling
	/*auto &level = *levels.begin(); {
		if(!compressed) {
			Helpers::IO::FileGuard<std::ifstream> fileWriter(filePath, std::ifstream::binary | std::ifstream::in);

			glm::ivec2 histogramResolution;
			glm::ivec2 viewResolution;
			glm::ivec2 spatialResolution;

			fileWriter.file.read(reinterpret_cast<char*>(&histogramResolution), sizeof(histogramResolution));
			fileWriter.file.read(reinterpret_cast<char*>(&viewResolution), sizeof(viewResolution));
			fileWriter.file.read(reinterpret_cast<char*>(&spatialResolution), sizeof(spatialResolution));

			fileWriter.file.read(reinterpret_cast<char*>(level.GetMetaData().data()), level.GetMetaData().size() * sizeof(*level.GetMetaData().begin()));
			fileWriter.file.read(reinterpret_cast<char*>(level.GetRawData().data()), level.GetRawData().size() * sizeof(*level.GetRawData().begin()));
		} else {
			Helpers::IO::ArchiveWriter compressedFileWriter(filePath);

			glm::ivec2 histogramResolution;
			glm::ivec2 viewResolution;
			glm::ivec2 spatialResolution;

			fileWriter.file.read(reinterpret_cast<char*>(&histogramResolution), sizeof(histogramResolution));
			fileWriter.file.read(reinterpret_cast<char*>(&viewResolution), sizeof(viewResolution));
			fileWriter.file.read(reinterpret_cast<char*>(&spatialResolution), sizeof(spatialResolution));

			fileWriter.file.read(reinterpret_cast<char*>(level.GetMetaData().data()), level.GetMetaData().size() * sizeof(*level.GetMetaData().begin()));
			fileWriter.file.read(reinterpret_cast<char*>(level.GetRawData().data()), level.GetRawData().size() * sizeof(*level.GetRawData().begin()));
		}
	}*/
}