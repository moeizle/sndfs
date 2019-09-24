#ifndef HELPERS_FILE_IO_H_
#define HELPERS_FILE_IO_H_

//#include "GlhSingle.h"
#include "glad/glad.h"

#include <memory>
#include <string>
#include <array>
#include <iostream>

#include <zlib.h>
#include <vector>

namespace Helpers {
namespace IO {
	class ImageData {
	public:
		ImageData(int width, int height) : width_(width), height_(height) {
			std::array<GLubyte, 4> clearColor = { 0, 0, 0, 255 };
			imageData = std::vector<std::array<GLubyte, 4>> (width * height, clearColor);
		}

		std::vector<std::array<GLubyte, 4>> imageData;
		int width_;
		int height_;
	};

	ImageData ReadPngFile(std::string filePath);

	/*
	Ensure closure of the file and exception safety.
	*/
	template <class FileStreamType = std::fstream>
	//typename std::enable_if<std::is_base_of<std::fstream, FileStreamType>::type>
	class FileGuard {
	public:
		FileGuard(std::string filePath, std::ios_base::openmode openModeFlag) {
			file = FileStreamType(filePath, openModeFlag);
		}

		~FileGuard() {
			if(file.is_open()) {
				file.close();
			}
		}
				
		FileStreamType file;

	private:
		// copy ctor and copy assignment are not allowed
		FileGuard(const FileGuard &other) = delete; //{}
		FileGuard &operator=(const FileGuard &other) = delete; //{ return *this; }
	};

	class ArchiveWriter {
	public:
		ArchiveWriter(std::string filePath) {
			outfile = gzopen(filePath.c_str(), "wb");
			// TODO: adjust compression - no need for lossless
			//gzsetparams(outfile, 0, 0);
		}

		~ArchiveWriter() {
			gzclose(outfile);
		}

		void write(const char* data, unsigned int dataSize) {
			if (!outfile) return;
			gzwrite(outfile, data, dataSize);
		}

	private:
		gzFile outfile;
	};

	// TODO: insert zlib functions
	class Archive {
	public:
		Archive(std::string filePath) {
			this->filePath = filePath;	
		}

		void write(const char* data, unsigned int dataSize) {
			gzFile outfile = gzopen(filePath.c_str(), "wb");
			if (!outfile) return;

			gzwrite(outfile, data, dataSize);
			gzclose(outfile);
		}

		std::vector<char> ReadFile() {
			// FIXME: use fileguard - not exception safe
			gzFile infile = gzopen(filePath.c_str(), "rb");
			
			static const int bufferSize = 4096;
			char buffer[bufferSize];
			int num_read = 0;
			std::vector<char> outputBuffer;
			while ((num_read = gzread(infile, buffer, sizeof(buffer))) > 0) {
				std::copy_n(buffer, num_read, std::inserter(outputBuffer, outputBuffer.end()));
			}

			gzclose(infile);

			return outputBuffer;
		}

		void CompressFile(std::string compressedFilePath) {
			compress_one_file(filePath.c_str(), compressedFilePath.c_str());
		}

	private:
		std::string filePath;

		// taken from
		// http://www.codeguru.com/cpp/cpp/algorithms/compression/article.php/c11735/zlib-Add-Industrial-Strength-Compression-to-Your-CC-Apps.htm
		int compress_one_file(const char *infilename, const char *outfilename) {
		   FILE *infile;
		   fopen_s(&infile, infilename, "rb");
		   gzFile outfile = gzopen(outfilename, "wb");
		   if (!infile || !outfile) return -1;

		   char inbuffer[128];
		   unsigned int num_read = 0;
		   unsigned int total_read = 0;
		   unsigned int total_wrote = 0;
		   while ((num_read = fread(inbuffer, 1, sizeof(inbuffer), infile)) > 0) {
			  total_read += num_read;
			  gzwrite(outfile, inbuffer, num_read);
		   }
		   fclose(infile);
		   gzclose(outfile);
		}
	};

} // namespace IO
} // namespace Helpers

#endif // HELPERS_FILE_IO_H_