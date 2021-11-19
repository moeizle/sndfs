#ifndef NDF_TREE_H_
#define NDF_TREE_H_

//#include "GlhSingle.h"
#include "glad/glad.h"

#include <vector>
#include <memory>

#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <string>

namespace NdfImposters {
	template <typename BinType> class NdfTreeLevel;
	
	/*
	Reference to a histogram
	*/
	template <typename BinType>
	class NdfHistogramRef {
	public:
		// FIXME: this should be const
		NdfHistogramRef(BinType *const ndfDataPointer, glm::ivec2 spatial,
			glm::ivec2 view, glm::ivec2 histogramResolution);
		//NdfHistogramRef(NdfViewDependentHistogram &viewDependentHistogram, glm::ivec2 position);

		auto GetBinValue(glm::ivec2 binIndex) const -> BinType {
			return NdfDataPointer[binIndex.y * HistogramResolution.x + binIndex.x];
		}

		void SetBinValue(glm::ivec2 binIndex, BinType value) {
			NdfDataPointer[binIndex.y * HistogramResolution.x + binIndex.x] = value;
		}

		const glm::ivec2 Spatial;
		const glm::ivec2 View;
		const glm::ivec2 HistogramResolution;

		// non-owning raw pointer
		BinType *const NdfDataPointer;
	};

	/*
	Class simplifying access to a specific view dependent ndf within an Ndf Tree.
	Does not hold any data, only references to the data stored contigously in the tree level.
	This is done for performance reasons since the sizes are not known during compile time thus
	cannot be instantiated 
	*/
	/*template <typename BinType>
	class NdfViewDependentHistogram {
	public:
		NdfViewDependentHistogram(NdfTreeLevel &treeLevel, glm::ivec2 view);

		//NdfHistogram GetHistogram(glm::ivec2 position) const;

	private:
		// non-owning raw pointer
		const NonOwningPointer<BinType> ndfData;
	};*/

	/*
	Stores normal distributions functions (NDF) contigously in memory.
	The meta data is stored seperately. The histogram resolution is uniform over all NDFs.
	*/
	template <typename BinType>
	class NdfTreeLevel {
	public:
		NdfTreeLevel(glm::ivec2 spatialResolution, glm::ivec2 histogramBinResolution, glm::ivec2 viewDirectionResolution);

		glm::ivec2 GetHistogramResolution() const;
		glm::ivec2 GetViewDirectionResolution() const;
		glm::ivec2 GetSpatialResolution() const;

		glm::vec3 GetViewDirection(glm::ivec2 view) const {
			return metaData[view.y * viewDirectionResolution_.x + view.x].ViewDirection; 
		}

		// TODO: implement convience functions that transform the data into form that allows
		// easier access, whithout meta data in each object that take up lots of memory in a
		// six-dimensional grid. Also the data should not be copied by referenced to in a safe way.
		//NdfViewDependentHistogram GetViewDependentHistogram(glm::ivec2 view) const;
		NdfHistogramRef<BinType> GetHistogram(glm::ivec2 spatial, glm::ivec2 view) const;

		int GetDataIndex(glm::ivec2 spatial, glm::ivec2 view, glm::ivec2 bin) const;

		const std::vector<BinType> &GetRawData() const {
			return rawContigousData;
		}

		class ViewMetaData;
		std::vector<ViewMetaData> &GetMetaData() const {
			return const_cast<std::vector<ViewMetaData>&>(metaData);
		}


		GLuint GetShaderStorageBufferOject() const;
		
		void Clear();
		void UploadData(GLuint ssboBindingPointIndex);
		void DownloadData();

	private:
		// Stores the directions of the samplesd rays outside of the actual data to remain it SoA.
		class ViewMetaData {
		public:
			ViewMetaData(glm::vec3 viewDirection, glm::ivec2 viewSlicePosition)
				: ViewDirection(viewDirection), ViewSlicePosition(viewSlicePosition) {}
			// sampled direction
			glm::vec3 ViewDirection;

			// spatial location on the view plane
			glm::ivec2 ViewSlicePosition;
		};

		// NOTE: resizing this data possibly invalidates weak references
		struct {
			std::vector<BinType> rawContigousData;
			std::vector<ViewMetaData> metaData;
		};

		// resolutions cannot be changed after initialization
		const glm::ivec2 spatialResolution_;
		const glm::ivec2 viewDirectionResolution_;
		const glm::ivec2 histogramBinResolution_;

		class SsboHandle {
		public:
			SsboHandle() : GlSsbo(0), Size(0) {}
			~SsboHandle() {
				DeleteBuffer();	
			}

			void DeleteBuffer() {
				if(GlSsbo) {
					assert(glDeleteBuffers);

					glDeleteBuffers(1, &GlSsbo);
					GlSsbo = 0;
					Size = 0;
				}
			}

			GLuint GlSsbo;
			size_t Size;
		};
		SsboHandle ssbo;
	};

	/*
	Hierarchy organized in a quad tree.
	*/
	template <typename BinType = float>
	class NdfTree {
	public:
		NdfTree();

		// TODO: split into more reasonable subroutines
		/*
		Initializes the storage.
		*/
		void InitializeStorage(int levels, glm::ivec2 histogramBinResolution, glm::ivec2 viewDirectionResolution, glm::ivec2 spatialResolution);

		glm::ivec2 GetHistogramResolution() const;
		glm::ivec2 GetViewDirectionResolution() const;
		glm::ivec2 GetSpatialResolution() const;
		std::vector<NdfTreeLevel<BinType>> &GetLevels();
		// Number levels. Each mip level divides each spatial dimension of the previous level by two.
		int GetLevelCount() const;

		void Clear();

		void UploadData();

		void Downsample();

		void WriteToFile(std::string filePath, bool compressed);
		void ReadFromFile(std::string filePath, bool compressed);

	private:
		std::vector<NdfTreeLevel<BinType>> levels;

		// spatial resolution of the highest resolution level.
		glm::ivec2 spatialResolution = glm::ivec2(0, 0);
		
		glm::ivec2 viewDirectionResolution = glm::ivec2(0, 0);

		glm::ivec2 histogramBinResolution = glm::ivec2(0, 0);
	};

} // namespace NdfImposters

#endif // NDF_TREE_H_