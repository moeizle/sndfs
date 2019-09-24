#ifndef FILE_COMPRESSION_H_
#define FILE_COMPRESSION_H_

#include <zlib.h>

#include <vector>

namespace Helpers {
namespace IO {

	class Archive {
	public:
		Archive(std::string filePath) {
			this->filePath = filePath;	
		}

		void write() {
		
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
		   int num_read = 0;
		   unsigned long total_read = 0, total_wrote = 0;
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

#endif // FILE_COMPRESSION_H_