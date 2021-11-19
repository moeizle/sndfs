//#define USE_ATB_ROTATION
#include "glad/glad.h"
#include "SparseApproximator.h"
#include "BrdfIntegrator.h"
#include "NdfRenderer.h"
#include "NdfTree.h"
#include "ProgressiveParticleSampler.h"
#include "MegaMolParticleFile.h"
#include "FileIO.h"
#include "ExpectationMaximization.h"
#include "NormalTransferFunction.h"
#include "Procedural.h"
#include "DebugHelpers.h"

#include <fstream>
#include <iostream>
#include <memory>
#define _USE_MATH_DEFINES
#define CORE_FUNCTIONALITY_ONLY
#define NO_BRICKING
#include <math.h>
#include <thread>
#include <mutex>
#include <future>
#include <chrono>
#include <forward_list>
#include <algorithm>

#include "FreeImage.h"
#include "region.h"


#include <glm/vec2.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/norm.hpp>


//#include "GlhSingle.h"
#include "GlHelpers.h"

#include <GL/glut.h>

#include "octree.h"
#include "frustum.h"
#include "lod.h"
#include "tiles.h"

#include <time.h>
#include <iomanip>
#include "StopWatch.h"

#include "slice.h"
#include "disk.h"
#include "convexHull.h"

#include "AntTweakBar.h"

#include "cxxopts.hpp"

float quat[4] = { 0, 0, 0, 1 };

int downsamplingFactor = 1;

//#define DOWNSAMPLE_NDF
//#define HIGH_RES_BUFFER
//#define SPARSE_APPROXIMATION
//#define DOWNSAMPLE_SSBO
//#define WRITE_TO_FILE
#define SET_ID 0
//#define HEADLESS
//#define READ_FROM_FILE
#ifdef READ_FROM_FILE
#undef HEADLESS
#endif
//#define WRITE_SPARSE_STATISTICS

// Set 1
#if SET_ID == 1
const std::string fileFolder = "E:/Datasets/VisVideo/";
const std::string fileSparseSubFolder = "Set1/";
const std::string filePrefix = "laser.000";
// Set 1 (40-65)
const auto timeStepUpperLimit = int(65);
const auto timeStepLowerLimit = int(40);
//const auto timeStepUpperLimit = int(65);
//const auto timeStepLowerLimit = int(51);
bool skip43 = true;
#endif

// Set 2
#if SET_ID == 2
const std::string fileFolder = "E:/Datasets/VisVideo/";
const std::string fileSparseSubFolder = "Set2/";
const std::string filePrefix = "laser.000";
// Set 2 (40-50)
const auto timeStepUpperLimit = int(50);
const auto timeStepLowerLimit = int(40);
bool skip43 = true;
# endif // SET 2

// Set 3
#if SET_ID == 3
const std::string fileFolder = "E:/Datasets/expl30m/";
//const std::string fileFolder = "C:/";
const std::string fileSparseSubFolder = "Set3/";
const std::string filePrefix = "expl30m_bin_fix_a.mmspd_";
const auto timeStepUpperLimit = int(90);
//const auto timeStepLowerLimit = int(0);
const auto timeStepLowerLimit = int(35);
bool skip43 = false;
#endif

float ndfIntensityCorrection = 1.0f;

const std::string fileSuffix = ".mmpld";

// may be overwritten bei cmd line
#if SET_ID > 0
auto timeStep = timeStepLowerLimit;
#else
auto timeStep = int(0);
#endif

std::string cmdlineFileName = "";
std::string cmdlineImageName = "dump.png";
bool cmdlineSaveImage = false;
bool cmdlineAutoQuit = false;
int cmdlineIterations = 0;
size_t cmdlineNumParticles = 0;

#define INITIAL_WINDOW_WIDTH 1280//1920// 1280 //854 // 1280  
#define INITIAL_WINDOW_HEIGHT 720//1080 //720 //480 //720


auto windowSize = glm::ivec2(INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT);
GLuint globalSsboBindingPointIndex = 1;

auto aspectRatio = static_cast<float>(windowSize.x) / static_cast<float>(windowSize.y);
std::vector<std::vector<double>> A;
std::vector<std::vector<glm::vec3>> projectedNormalBinningAreas;
std::vector<std::vector<glm::vec4> > sampleTexData;
float samplingAspectRatio;

bool op0 = false, op1 = false, op2 = false, op3 = false, op4 = false, op5 = false, op6 = false,op7=false,op8=false,op9=false;
StopWatch w, w2, w3,s,e;
std::vector<std::pair<int, double>> iterationTimePair;
std::vector<double> timings;
std::vector<float> t1, t2, t3, t4, t5, t6, t7, t8, t9;
int new_tiles = 0, cached_tiles = 0;
float samplingScalingFactor = 3;
int samplingTextureSaves = 0;
bool save_SamplingTexture = false;
std::vector<region> regions;
int similarityMeasure = 0;

float circularPatternRadius = sqrt(0.5*0.5+0.5*0.5);
bool circularPattern = false;

int highestSampleCount, lowestSampleCount;
glm::vec2 sPos;
float sampleW;
float simPercentage = .05;
float ** gk;  //gauss kernel
int gkDim;
float filterRadius = sqrt(.5*.5 + .5*.5);
float gkStdv = 1.0f;
bool buildingHOM = false;
bool buildHOMFlag = false;
bool enableOcclusionCulluing = false;
bool ndfOverlayMode = false;
bool ndfExplorerMode=false;
bool diskPicked = false;
bool screenSaved = false;

std::vector<int> cellsToRender;
std::vector<int> cellsTbe;  //cells to be removed
std::queue<int> cellsTbt;  //cells to be tested
std::vector<int> cellsTbr;
std::vector<Helpers::Gl::RenderTargetHandle> homFbos;
Helpers::Gl::RenderTargetHandle ndfExplorer;

//new_tiles = cached_tiles = 0;
bool renderBarMode = true;
bool streaming = false;

inline void sampleData();
bool renderCell(int cellIndx);
inline void computeSampledRegion();
inline void debugfunction();
inline void computeSampleParameters();
inline void quantizeSamples();
inline void computeRenderedRegion();
inline void renderData();
inline void renderTransferFunction();
inline void renderSelection();
inline void renderAvgNDF();
inline void updateSimilarityLimits();
inline void createFilter(float dim, double stdv);
inline void renderBar();
inline void removeOccludedCells();
inline void renderNdfExplorer();

inline void computeDisk(GLfloat x, GLfloat y, GLfloat radius, std::vector<glm::vec3>& verts,float depth);
inline void computeSlice(slice& s, std::vector<glm::vec3>& verts,float radius,float depth);

//hierarchical occlusion map functions
inline void buildHOM();
inline void drawHOM();
inline void drawHOM(std::vector<int>& inds);
inline void copyHOMtoTexture();
inline void compareHOMtoTexture();
inline void updateHOM();
inline void testAgainstHOM(int nodeIndx, std::vector<int>& tbe, std::vector<int>& tbr);
inline void projectCell(int nodeIndx, std::vector<glm::vec2>& projectedCorners, int HOMlevel);
inline int getHOMTestLevel(int nodeIndx);
inline void getHOMOverlappingPixels(int HOMlevel, glm::vec2& cell_blc, glm::vec2& cell_trc,std::vector<int>& pixels);
inline void initializeHOMtestFbo(Helpers::Gl::RenderTargetHandle& homFbo, int HOMlevel);
inline void setHOMRenderingState(Helpers::Gl::RenderTargetHandle& homFbo);
inline void drawDepthBuffer(Helpers::Gl::RenderTargetHandle& homFbo);
inline void renderCelltoHOM(std::vector<int> Inds);


std::vector<glm::vec3> obb;
std::vector<glm::vec2> selectedPixels;
std::vector<slice> slices;
std::vector<disk> disks;
std::vector<std::pair<glm::vec3,int>> sliceColors;
int sliceIndx;

void reset();
void uploadGeometry();
void reshape(int w, int h);
void display();
void idle();
void unInitTweakBar();
void compileShaders(std::string shaderPath);
void keyboard(unsigned char key, int mx, int my);
void keyboardUp(unsigned char key, int mx, int my);
void mouse(int button, int state, int x, int y);
void mouseMotionActive(int x, int y);
glm::vec3 Map_to_trackball(float x, float y);
void tile_based_culling(bool tile_flag);
void tile_based_panning(bool tile_flag);
void update_tiled_fbo();
//void render_tile(std::vector<int> tiles_to_render,float level_of_detail);
//bool found_in_page_texture(glm::vec3 pos, float level_of_detail);
void computeNDFCache();
void computeNDFCache_Ceil();
void computeTiledRaycastingCache(const glm::mat4 & viewMatrix);
void computeTiledRaycastingCache_C(const glm::mat4 & viewMatrix);
void saveSamplingTexture();
void saveHOM();
void saveScreen();
inline void preIntegrateBins();
inline void preIntegrateBins_GPU();
inline void computeBinAreas();
inline void computeBinAreasSimple();
inline void computeBinAreasSpherical();
inline void computeBinAreasLambertAzimuthalEqualArea();
glm::mat3 rotationMatrix(glm::vec3 axis, float angle);
glm::vec3 blinnPhong(glm::vec3 normal, glm::vec3 light, glm::vec3 view, glm::vec3 diffuseColor, glm::vec3 specularColor, float specularExponent);
void mapToSphere(glm::vec2& p, glm::vec3& res);
glm::quat RotationBetweenVectors(glm::vec3 start, glm::vec3 dest);

void overlayNDFs();


void bindSSbos();
void adjustAvgNDF(glm::vec2 relativeMouse,bool clearFlag);

inline void update_page_texture();
void initialize();
void drawNDF();
void saveBmpImage(int w, int h, float* data, const std::string& name);
void drawNDFImage(int w, int h, float* data);
void updateExtents();
void get_visible_tiles_in_ceil_and_floor(float level_of_detail, std::vector<std::vector<int>>& visible_tiles_per_level);
//glm::vec2 pixel2obj(glm::vec2 pixel_coordinates, float lod);
//glm::vec2 obj2pixel(glm::vec2 obj_coordinates, float lod);

void remove_tile_from_physical_memory(std::vector<int>& locs);

void rotate_data(float x, float y);
void apply_permenant_rotation(float angle, bool axis);
void apply_permenant_rotation(float anglex, float angley);
void clear_NDF_Cache();
bool is_tile_in_cache(int level, int tile_indx);
void get_available_locations_in_cache(std::queue<int>& available, std::vector<int>& levels);
void get_one_available_location_in_cache(std::queue<int>& available, std::vector<int>& levels);
void erase_progressive_raycasting_ssbo();
void probeNDFs();
void computeAvgNDF(bool finished);
void computeAvgNDFGPU(bool finished);
void resetSimLimits(bool flag);
void drawSelection();
inline void remarkSimilarNDFs();
inline void markSimilarNDFs(region& r, std::vector<float>& NDFs, std::vector<float>& regionColor);
inline float similarNDFs(std::vector<float>& avgNDF, std::vector<float>& NDFs, int lookup);
inline float L2Norm(std::vector<float>& avgNDF, std::vector<float>& NDFs, int lookup);
inline float earthMoversDistance(std::vector<float>& avgNDF, std::vector<float>& NDFs, int lookup);

void setActiveChromeTexture(glm::vec3 color);

typedef unsigned short UInt16;  // [0..65535]
typedef float Float32;

glm::vec3 lastpoint;
int MaxAllowedSelectedPixels = 20000;



inline UInt16 Float32toFloat16(Float32 fValue)
{
	// Value -> [-1024 .. 3072] with precision of 1/64 after coma
	return ((fValue + 1024.f) / (4096.f / 0xFFFF));
}

inline Float32 Float16toFloat32(UInt16 fFloat16)
{
	return fFloat16 * (4096.0f / 65535.0f) - 1024.f;
}


std::vector<float> percentage_of_cached_tiles;

NdfImposters::NdfRenderer<> renderer;
NdfImposters::NdfTree<> ndfTree;

Helpers::Gl::BufferHandle boxHandle;
Helpers::Gl::BufferHandle quadHandle;
Helpers::Gl::BufferHandle pointsHandle;


glm::vec3 camPosi = { 0.0f, 0.0f, 3.415f };
glm::vec3 camPosiSampling;

bool printScreen = false;

glm::vec3 camTarget = { 0.0, 0.0f, 0.0f };
glm::vec3 camUp = { 0.0f, 1.0f, 0.0f };
glm::vec3 cameraOffset = { 0.0f, 0.0f, 0.0f };

int brick = 0;

StopWatch timeToConvergence;

const auto fieldOfView = 45.0f;

static const float toRadiants = static_cast<float>(M_PI) / 180.0f;
glm::vec3 LightDir = { 0.751368284, -0.452016711, -0.480756283 };// { 1.0f, -1.0f, 0.0f };
float lightRotationX = 145.0f;//104.0f;
float lightRotationY = 155.0f;//-120.0f;

auto cameraRotation = glm::vec2(0.0f, 0.0f);
float specularExponent = 8.0f;
float cameraDistance;// = 16.0f;// 0.0625f;//3.415f;

auto initialCameraDistance = 64;// 16.0;

float binDiscretizations = 8;

bool leftHeld = false;
bool rightHeld = false;
bool middleHeld = false;

int renderMode = 0;
int zoomMode = 0;
float zoomScale = 0.0f;
glm::vec2 zoomWindow = glm::vec2(0.5f, 0.5f);

static const float nearPlane = 0.0001f;
static const float farPlane = 100.0f;//1000.0f;
// NOTE: sampling rate currently has to be 4 because of high res ndf
static const auto multiSamplingRate = 1;

Helpers::Gl::RenderTargetHandle rayCastingSolutionRenderTarget;
Helpers::Gl::RenderTargetHandle homRenderTarget;
glm::ivec2 homHighestResolution=glm::ivec2(4096,4096);
glm::ivec2 origHomHighestResolution = homHighestResolution;
//Helpers::Gl::RenderTargetHandle MyRenderTarget;

std::ofstream out;

// Streaming SSBO
const GLsizeiptr bufSize = 32 * 1024 * 1024;
const GLint numBuffers = 3;
const GLuint streamingBufferCreationBits = GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT; 
const GLuint streamingBufferMappingBits = GL_MAP_PERSISTENT_BIT | GL_MAP_WRITE_BIT | GL_MAP_FLUSH_EXPLICIT_BIT;
GLuint theStreamingBuffer = 0;
const GLuint SSBObindingPoint = 31;
void *theStreamingMappedMem;
std::vector<GLsync> fences;
int currBuf = 0;

//debug
std::vector<int> lookups;

//end debug

//Mohamed's Fbo
GLuint TiledFbo = 0;
//GLuint PhysicalFbo = 0;
//GLuint glfinalfbo_depthbuffer = 0;
//GLuint PhysicalFbo_depthbuffer = 0;
GLuint pageTexture = 0;
GLuint HOMtexture = 0;
GLuint rendertarget = 0;
GLuint* TiledFbo_TextureBuffer = 0;
GLuint TiledFbo_DepthBuffer = 0;
GLenum* TiledFbo_DrawBuffer = 0;
//GLuint glfinalfbo = 0;
GLuint physicalTexture = 0;
int phys_tex_dim = -1;
float binLimit = 7;
int max_tiles = -1;
float highest_h_res, highest_w_res;
std::vector<std::vector<int>> visible_tiles_per_level;
int binning_mode = 1;
bool showEmptyTiles = false;

float* samples_data;
float* colorMapData;
//float* CircularSamples_data;
//float* CircularSamples_data_weights;

int sample_count;// = 2048 * 2048;
//int circular_pattern_sample_count;
//int circular_pattern_dim;
int square_pattern_sample_count;
int colorMapSize;


GLuint myArrayUBO;                         //samples ssbo
GLuint progressive_raycasting_ssbo;        //used for progressive raycasting
GLuint SampleCount_ssbo;
GLuint circularPatternSampleCount_ssbo;    //we need to keep track of how many samples were shot, for the circular pattern, since the samples are weighted, we can't just deduce that from the samplecount_ssbo
//GLuint region_ssbo;
GLuint simLimitsF_ssbo;
GLuint avgNDF_ssbo;
GLuint NDFImage_ssbo;
GLuint preIntegratedBins_ssbo;
GLuint binAreas_ssbo;
GLuint superPreIntegratedBins_ssbo;
GLuint simpleBinningAreas_ssbo;
GLuint colorMap_ssbo;
GLuint selectedPixels_ssbo;
GLuint ndfColors_ssbo;


unsigned char* screen_tex_data;
unsigned char* physical_tex_data;

glm::vec2 visible_region_blc;
glm::vec2 visible_region_trc;
glm::vec3 lod_blc;
glm::vec3 lod_trc;

glm::vec3 ceil_blc;
glm::vec3 floor_blc;

glm::vec2 blc_s;
glm::vec2 blc_l;
glm::vec2 trc_s;

glm::vec2 sampling_blc_s;
glm::vec2 sampling_blc_l;
glm::vec2 sampling_trc_s;

std::vector<glm::vec2> visible_tiles_resolutions;
std::vector<glm::vec2> floor_ceil_blc_l;

float OrigL, OrigR, OrigT, OrigB;



int Tile_index = 0;
int prev_tex_count = 0;
int MaxFboTexCount = 0;
float current_lod, prev_lod,downsampling_lod;
int lod_increment;
std::vector<std::pair<std::pair<bool, glm::vec2>, int>> occupied;

int* visible_spheres;
int visible_spheres_count;
//end Mohamed's Fbo

//Mohamed's Ndf tiling variables
GLuint Transfer_Texture = 0;
GLuint Page_Texture = 0;
GLenum Page_Texture_type = GL_FLOAT;
GLint  Page_texture_internalFormat = GL_RGBA32F;
GLenum Page_Texture_format = GL_RGBA;
typedef GLfloat Page_Texture_Datatype;

//end Mohamed's Ndf tiling variables

glm::vec2 calculate_lookup(int lod_w, int lod_h, glm::vec2 Coordinate, float F_lod, float E_lod, Page_Texture_Datatype* temp_texture);


bool PGMwrite(const std::string& name, size_t width, size_t height, uint8_t* img) {
	FILE* output = NULL;
	if (fopen_s(&output, name.c_str(), "wb")) return false;
	char hdr[1024];
	sprintf_s(hdr, 1024, "P5 %i %i 255\n", width, height);
	size_t n = strlen(hdr);
	if (fwrite(hdr, sizeof(char), n, output) != n) {
		fclose(output);
		return false;
	}
	size_t N = width*height;
	if (fwrite(img, sizeof(uint8_t), N, output) != N) {
		fclose(output);
		return false;
	}
	fclose(output);
	std::cout << "saved image: " << name << std::endl;
	return true;
}

glm::vec3 CameraPosition(glm::vec2 rotation, float distance)
{
	rotation += glm::vec2(90.0f, -90.0f);
	return distance * glm::normalize(glm::vec3(std::cos(toRadiants * rotation.x), std::cos(toRadiants * rotation.y), std::sin(toRadiants * rotation.x)));
}
int maxSubBins = 8;
static const auto angleRange = 360.0f;
static const bool gpuReduction = true;
static const glm::mat2x3 areaOfInterest = { 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f };
static const int MaxFrameCount = 1;
static const int MaxParticles = std::numeric_limits<int>::max();
static const auto shaderPath = std::string("./");

// NOTE: the local work group size in the convolution shader has to be adjusted as well!
static auto histogramResolution = glm::ivec2(8, 8);//glm::ivec2(4, 8);
static auto origHistogramResolution = histogramResolution;
//static const auto histogramResolution = glm::ivec2(12, 12);
//static const auto histogramResolution = glm::ivec2(16, 16);
//static const auto histogramResolution = glm::ivec2(32, 32);
static const auto viewResolution = glm::ivec2(1, 1);
// 2GB ssbo = 724.0773 pixels for 8x8 histograms and 4x4 multi sampling
//static const auto spatialResolution = glm::ivec2(720, 720);
static const auto spatialResolution = glm::ivec2(INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT);

//std::once_flag approximationFlag

static std::vector<float> areaHits;
std::vector<float> cpuAvgNDF(histogramResolution.x*histogramResolution.y, 0);

//std::future<SparseApproximation::ApproximationResult> futureApproximation;
std::future<SparseApproximation::GpuGmm> futureApproximation;
std::chrono::system_clock::time_point approximationStart;

float quantizationVarianceScale = 1.0f;

//const int emGaussians = 16;
//const int emIterations = 5;

// Wokring settings with 8x8 histograms
//const int emGaussians = 8;
//const int emIterations = 5;

const int emGaussians = 8;
const int emIterations = 5;
// + 1 lambertian
const int maxGaussians() {
	return emGaussians + 1;
}
GLuint eNtfGaussianSsbo = 0;
GLuint eNtfOffsetSsbo = 0;
std::string sparseENtfFilePath = "sparseENtfFile.entf";
class SparseENtfHeader
{
public:
	SparseENtfHeader() {}
	SparseENtfHeader(int64_t gaussianCount, int32_t histogramWidth, int32_t histogramHeight, int32_t imageWidth, int32_t imageHeight, float quantizationVarianceScale, int64_t approximationMilliseconds) :
		GaussianCount(gaussianCount), HistogramWidth(histogramWidth), HistogramHeight(histogramHeight), ImageWidth(imageWidth), ImageHeight(imageHeight), QuantizationVarianceScale(quantizationVarianceScale), ApproximationMilliseconds(approximationMilliseconds) {}

	int64_t GaussianCount;
	int32_t HistogramWidth;
	int32_t HistogramHeight;
	int32_t ImageWidth;
	int32_t ImageHeight;
	float QuantizationVarianceScale;
	int64_t ApproximationMilliseconds;
};

class SparseENtfDataMemo {
public:
	SparseENtfHeader fileHeader;
	SparseApproximation::GpuGmm sparseData;
};

bool renderSparse = false;
bool renderTransfer = true;
bool paused = false;

auto camRotation = angleRange / 60.0f;
auto lastMouse = glm::ivec2(0, 0);

glm::ivec2 prevLoc = glm::ivec2(0, 0);


glm::quat prevQ = glm::quat(1, 0, 0, 0);

glm::vec3 origModelMin;
glm::vec3 origModelMax;
glm::vec3 origModelExtent;

glm::vec3 modelMin;
glm::vec3 modelMax;
glm::vec3 modelExtent;

#ifdef DOWNSAMPLE_SSBO
GLuint downsampledSsbo = 0;
#endif // DOWNSAMPLE_SSBO

#ifdef HIGH_RES_BUFFER
GLuint highResNdf = 0;
#endif // HIGH_RES_BUFFER

//Mohamed's variables
float nh, nw, fh, fw;
octree global_tree;
int totalParticlesRemoved = 0;
std::vector<std::pair<int, std::vector<int>>> renderBricks;
__int64 max_node_memory_consumption;
octree global_tree_projection;
int octree_node_indx = 0;
std::vector<glm::vec3> temp_particleCenters;
std::vector<int> bf_tree_traversal;
std::vector<int> visible_indices;
std::vector<int>::iterator it;
float oclude_percentage_frustum = 1;
int frustum_indx = 0;
lod LOD;
float lodDelta = 0.0f;
int cur_w, cur_h;
int tile_h=128, tile_w=128;
float tw_obj, th_obj;
float sl, sr, st, sb;
float sh = windowSize.y, sw = windowSize.x;
std::vector<tiles> FloorCeilTiles;
tiles Tiles;     //tiles in object space
tiles Tiles_s;   //tiles in screen space
tiles vis_Tiles;
std::ofstream outfile;
glm::vec4 viewportMat;
glm::mat4 modelviewMat;
glm::mat4 projectionMat;
glm::mat4 viewMat, modelMat, viewprojectionMat;

glm::mat4 samplingModelMat, samplingModelViewMat, samplingViewMat, samplingViewProjectionMat;
glm::mat4 samplingProjectionMat;
glm::mat4 samplingViewMat_noPanning, samplingModelViewMat_noPanning;
glm::mat4         viewMat_noPanning, modelviewMat_noPanning;

glm::mat3 GlobalRotationMatrix;
float mynear, myfar;
glm::vec3 myup, mytarget;
int visualize_tiles = -1;
bool show_tree = false;
int tile_no = 0;
std::vector<int> orig_visible_tiles;
bool tile_flag;
glm::vec3 zoom_camPosi;
bool cull = true;
//temporary for one tile testing

std::vector<frustum> test_frustum;
//end temporary for one tile testing

//end Mohamed's variables

//Mohamed's functions
void setCamInternals_ortho(float left, float right, float top, float bottom, float nearpl, float farpl);
void setCamInternals(float angle, float ratio, float nearD, float farD);
void setCamDef(glm::vec3 &p, glm::vec3 &l, glm::vec3 &u);
void setCamDef_ortho(float left, float right, float top, float bottom, float nearpl, float farpl);
void setCamDef_2(const glm::mat4 &mat, float left, float right, float top, float bottom, float nearpl, float farpl);
void extract_planes(const glm::mat4 &mat);

void setCamInternals_ortho(float left, float right, float top, float bottom, float nearpl, float farpl)
{
	nh = fh = abs(top - bottom);
	nw = fw = abs(left - right);
}
void setCamInternals(float angle, float ratio, float nearD, float farD)
{
	//from: http://www.lighthouse3d.com/tutorials/view-frustum-culling/geometric-approach-implementation/

	float ANG2RAD = 3.14159 / 180.0;

	// compute width and height of the near and far plane sections
	float tang = (float)tan(ANG2RAD * angle * 0.5);
	nh = nearD * tang;
	nw = nh * ratio;
	fh = farD  * tang;
	fw = fh * ratio;
}
void extract_planes(const glm::mat4 &mat)
{
	test_frustum.clear();
	frustum temp;
	test_frustum.push_back(temp);

	//left plane
	glm::vec3 n = glm::vec3(mat[0][3] + mat[0][0], mat[1][3] + mat[1][0], mat[2][3] + mat[2][0]);
	float l = glm::length(n);

	n = glm::normalize(n);

	float d = (mat[3][3] + mat[3][0]) / l;

	glm::vec3 p = glm::vec3(-d / n.x, -d / n.y, -d / n.z);

	test_frustum[0].planes.push_back(frustum::plane(p, n));

	//right plane
	n = glm::vec3(mat[0][3] - mat[0][0], mat[1][3] - mat[1][0], mat[2][3] - mat[2][0]);
	l = glm::length(n);

	n = glm::normalize(n);

	d = (mat[3][3] - mat[3][0]) / l;

	p = glm::vec3(-d / n.x, -d / n.y, -d / n.z);

	test_frustum[0].planes.push_back(frustum::plane(p, n));

	//bottom plane
	n = glm::vec3(mat[0][3] + mat[0][1], mat[1][3] + mat[1][1], mat[2][3] + mat[2][1]);
	l = glm::length(n);

	n = glm::normalize(n);

	d = (mat[3][3] + mat[3][1]) / l;

	p = glm::vec3(-d / n.x, -d / n.y, -d / n.z);

	test_frustum[0].planes.push_back(frustum::plane(p, n));

	//top plane
	n = glm::vec3(mat[0][3] - mat[0][1], mat[1][3] - mat[1][1], mat[2][3] - mat[2][1]);
	l = glm::length(n);

	n = glm::normalize(n);

	d = (mat[3][3] - mat[3][1]) / l;

	p = glm::vec3(-d / n.x, -d / n.y, -d / n.z);

	test_frustum[0].planes.push_back(frustum::plane(p, n));

	//near plane
	n = glm::vec3(mat[0][3] + mat[0][2], mat[1][3] + mat[1][2], mat[2][3] + mat[2][2]);
	l = glm::length(n);

	n = glm::normalize(n);

	d = (mat[3][3] + mat[3][2]) / l;

	p = glm::vec3(-d / n.x, -d / n.y, -d / n.z);

	test_frustum[0].planes.push_back(frustum::plane(p, n));

	//far plane
	n = glm::vec3(mat[0][3] - mat[0][2], mat[1][3] - mat[1][2], mat[2][3] - mat[2][2]);
	l = glm::length(n);

	n = glm::normalize(n);

	d = (mat[3][3] - mat[3][2]) / l;

	p = glm::vec3(-d / n.x, -d / n.y, -d / n.z);

	test_frustum[0].planes.push_back(frustum::plane(p, n));


}
void setCamDef_ortho(float left, float right, float top, float bottom, float nearpl, float farpl)
{
	test_frustum.clear();
	frustum temp;
	test_frustum.push_back(temp);

	//front and back planes
	test_frustum[0].planes.push_back(frustum::plane(glm::vec3(left, top, nearpl), glm::vec3(0, 0, 1)));
	test_frustum[0].planes.push_back(frustum::plane(glm::vec3(left, top, farpl), glm::vec3(0, 0, -1)));

	//right and left planes
	test_frustum[0].planes.push_back(frustum::plane(glm::vec3(right, top, nearpl), glm::vec3(-1, 0, 0)));
	test_frustum[0].planes.push_back(frustum::plane(glm::vec3(left, top, farpl), glm::vec3(1, 0, 0)));

	//top and bottom planes
	test_frustum[0].planes.push_back(frustum::plane(glm::vec3(right, top, nearpl), glm::vec3(0, -1, 0)));
	test_frustum[0].planes.push_back(frustum::plane(glm::vec3(left, bottom, farpl), glm::vec3(0, 1, 0)));
}
void setCamDef_2(const glm::mat4 &mat, float left, float right, float top, float bottom, float nearpl, float farpl)
{
	test_frustum.clear();
	frustum temp;
	test_frustum.push_back(temp);

	glm::vec4 p1, p2, res;

	//compute new 'z' axis
	res = glm::transpose(glm::inverse(mat))*glm::vec4(0, 0, 1, 0);
	//compute new point on plane
	p1 = mat*glm::vec4(left, top, nearpl, 1);
	p2 = mat*glm::vec4(left, top, farpl, 1);

	//front and back planes
	test_frustum[0].planes.push_back(frustum::plane(glm::vec3(p1.x, p1.y, p1.z), glm::vec3(res.x, res.y, res.z)));
	test_frustum[0].planes.push_back(frustum::plane(glm::vec3(p2.x, p2.y, p2.z), glm::vec3(-res.x, -res.y, -res.z)));

	//compute new 'x' axis
	res = glm::transpose(glm::inverse(mat))*glm::vec4(-1, 0, 0, 0);
	//compute new point on plane
	p1 = mat*glm::vec4(right, top, nearpl, 1);
	p2 = mat*glm::vec4(left, top, farpl, 1);

	//right and left planes
	test_frustum[0].planes.push_back(frustum::plane(glm::vec3(p1.x, p1.y, p1.z), glm::vec3(res.x, res.y, res.z)));
	test_frustum[0].planes.push_back(frustum::plane(glm::vec3(p2.x, p2.y, p2.z), glm::vec3(-res.x, -res.y, -res.z)));

	//compute new 'x' axis
	res = glm::transpose(glm::inverse(mat))*glm::vec4(0, -1, 0, 0);
	//compute new point on plane
	p1 = mat*glm::vec4(right, top, nearpl, 1);
	p2 = mat*glm::vec4(left, bottom, farpl, 1);

	//top and bottom planes
	test_frustum[0].planes.push_back(frustum::plane(glm::vec3(p1.x, p1.y, p1.z), glm::vec3(res.x, res.y, res.z)));
	test_frustum[0].planes.push_back(frustum::plane(glm::vec3(p2.x, p2.y, p2.z), glm::vec3(-res.x, -res.y, -res.z)));

}
void setCamDef(glm::vec3 &p, glm::vec3 &l, glm::vec3 &u)  //cam position, cam target, cam up
{
	//from: http://www.lighthouse3d.com/tutorials/view-frustum-culling/geometric-approach-implementation/
	glm::vec3 dir, nc, fc, X, Y, Z, ntl, ntr, nbl, nbr, ftl, ftr, fbl, fbr;
	glm::vec3 a, b, n;
	float tile_count_in_x, tile_count_in_y;

	tile_count_in_x = tile_count_in_y = 2;

	test_frustum.clear();


	// compute the Z axis of camera
	// this axis points in the opposite direction from
	// the looking direction
	Z = p - l;
	Z = glm::normalize(Z);

	// X axis of camera with given "up" vector and Z axis
	X = glm::cross(u, Z);
	X = glm::normalize(X);

	// the real "up" vector is the cross product of Z and X
	Y = glm::cross(Z, X);

	// compute the centers of the near and far planes
	nc = p - Z * nearPlane;
	fc = p - Z * farPlane;

	// compute the 4 corners of the frustum on the near plane
	ntl = nc + Y * nh - X * nw;
	ntr = nc + Y * nh + X * nw;
	nbl = nc - Y * nh - X * nw;
	nbr = nc - Y * nh + X * nw;

	// compute the 4 corners of the frustum on the far plane
	ftl = fc + Y * fh - X * fw;
	ftr = fc + Y * fh + X * fw;
	fbl = fc - Y * fh - X * fw;
	fbr = fc - Y * fh + X * fw;


	//compute centers of tiles
	glm::vec3 wv = ntr - ntl;
	glm::vec3 hv = nbl - ntl;
	glm::vec3 nci, fci;
	float wt, ht;
	glm::vec3 ntli, ntri, nbli, nbri, ftli, ftri, fbli, fbri;
	float tile_w, tile_h;

	tile_w = glm::length(wv) / tile_count_in_x;
	tile_h = glm::length(hv) / tile_count_in_y;


	for (float i = tile_w / 2.0; i < glm::length(wv); i = i + tile_w)
	{
		wt = float(i / glm::length(wv));
		for (float j = tile_h / 2.0; j < glm::length(hv); j = j + tile_h)
		{
			frustum temp;

			ht = float(j / glm::length(hv));

			//computer center of tile
			nci = ntl + wt*wv + ht*hv;
			fci = ftl + wt*wv + ht*hv;

			//compute corners of frustum
			// compute the 4 corners of the frustum on the near plane
			ntli = nci + Y * 0.5f*tile_h - X * 0.5f*tile_w;
			ntri = nci + Y * 0.5f*tile_h + X * 0.5f*tile_w;
			nbli = nci - Y * 0.5f*tile_h - X * 0.5f*tile_w;
			nbri = nci - Y * 0.5f*tile_h + X * 0.5f*tile_w;

			// compute the 4 corners of the frustum on the far plane
			ftli = fci + Y * 0.5f*tile_h - X * 0.5f*tile_w;
			ftri = fci + Y * 0.5f*tile_h + X * 0.5f*tile_w;
			fbli = fci - Y * 0.5f*tile_h - X * 0.5f*tile_w;
			fbri = fci - Y * 0.5f*tile_h + X * 0.5f*tile_w;

			//compute planes of frustum
			//top plane
			a = ntri - ntli;
			b = ftli - ntli;
			n = glm::cross(a, b);
			n = glm::normalize(n);

			temp.planes.push_back(frustum::plane(ntri, -n));

			//bottom plane
			temp.planes.push_back(frustum::plane(fbri, n));

			//left plane
			a = ntli - nbli;
			b = fbli - nbli;

			n = glm::cross(a, b);
			n = glm::normalize(n);

			temp.planes.push_back(frustum::plane(nbli, -n));

			//right plane
			temp.planes.push_back(frustum::plane(nbri, n));

			//near plane
			a = ntli - ntri;
			b = nbri - ntri;

			n = glm::cross(a, b);
			n = glm::normalize(n);

			temp.planes.push_back(frustum::plane(ntri, -n));

			//far plane
			temp.planes.push_back(frustum::plane(ftri, n));

			test_frustum.push_back(temp);
		}
	}
}

//end Mohamed's functions

void reshape(int w, int h)
{
   //TwWindowSize(w, h);
   return;

	{

		windowSize = { w, h };
		aspectRatio = static_cast<float>(windowSize.x) / static_cast<float>(windowSize.y);

		initialize();

		//// reset render target
		Helpers::Gl::DeleteRenderTarget(rayCastingSolutionRenderTarget);
		rayCastingSolutionRenderTarget = Helpers::Gl::CreateRenderTarget(windowSize.x * 2 * pow(2, lodDelta) + windowSize.x, windowSize.y * 2 * pow(2, lodDelta) + windowSize.y, GL_RGBA32F, GL_RGBA, GL_FLOAT);


		tile_based_culling(false);
		update_page_texture();
		reset();
		preIntegrateBins();

		auto glErr = glGetError();
		if (glErr != GL_NO_ERROR) {
			std::cout << "glError: " << __FILE__ << " " << __LINE__ << std::endl;
		}

#ifdef HIGH_RES_BUFFER
		{
			static const auto ssboBindingPointIndex = GLuint(1);
			auto ssboSize = (windowSize.x * multiSamplingRate) * (windowSize.y * multiSamplingRate) * histogramResolution.x * histogramResolution.y * sizeof(float);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, highResNdf);
			glBufferData(GL_SHADER_STORAGE_BUFFER, ssboSize, nullptr, GL_DYNAMIC_COPY);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		}

		glErr = glGetError();
		if (glErr != GL_NO_ERROR) {
			std::cout << "glError: " << __FILE__ << " " << __LINE__ << std::endl;
		}
#endif

		//reset();
	}
}

// FIXME: multisampling NYI - update reduction progressive shader as well
int lastMeasuredRunIndex = 0;
auto samplingRunIndex = 0;
auto progressiveSamplesCount = 0;
auto prevSamplingRunIndex = 0;
auto origSamplingRuns = 1024;
auto maxSamplingRuns = 1024;
auto maxSamplingRunsFactor = 1.0f;
auto mymaxsamplingruns = .25;
auto tempmaxSamplingRuns = 1024;
bool singleRay = false;
bool noCaching =  false;
bool multipleRay_raycasting = false;
bool plainRayCasting = false;
bool noTilesVisible = false;
bool samplingFlag = true;
bool quantizeFlag = true;
bool cachedRayCasting = false;// false;
bool disable_binding = false;
bool pointCloudRendering = false;
bool singleRay_NDF = false;
//static const auto samplesPerRun = 1;
auto samplesPerRun = 1;
bool switchToRaycasting = true;
auto recompute = true;
int probeNDFsMode = 0;

auto fullScreen = false;

int circularPatternIterator = 0;

std::vector<std::vector<int>> pHits;
std::vector<std::string> measurements;

void reset()
{
	//std::cout << "NDF cache invalidated" << std::endl;
	lastMeasuredRunIndex = 0;
	samplingRunIndex = 0;
	progressiveSamplesCount = 0;
	lowestSampleCount = 0;
	//measurements.clear();


#ifndef READ_FROM_FILE
	if (renderSparse)
	{
		std::cout << "eNTF cache invalidated" << std::endl;
	}
	renderSparse = false;
#endif // READ_FROM_FILE
}

static const auto viewBatches = glm::ivec2(1, 1);

//Mohamed's shaders
std::unique_ptr<Helpers::Gl::ShaderProgram> tileShader;
//end Mohamed's shaders

std::unique_ptr<Helpers::Gl::ShaderProgram> computeAvgNDF_C;
std::unique_ptr<Helpers::Gl::ShaderProgram> drawHOMShader;
std::unique_ptr<Helpers::Gl::ShaderProgram> drawQuadMeshShader;
std::unique_ptr<Helpers::Gl::ShaderProgram> samplingShader;
std::unique_ptr<Helpers::Gl::ShaderProgram> samplingShader_PointCloud;
std::unique_ptr<Helpers::Gl::ShaderProgram> reductionShader;
std::unique_ptr<Helpers::Gl::ShaderProgram> reductionShader_C;
//std::unique_ptr<Helpers::Gl::ShaderProgram> downsamplingShader;
std::unique_ptr<Helpers::Gl::ShaderProgram> renderingShader;
//std::unique_ptr<Helpers::Gl::ShaderProgram> eNtfRenderingShader;
std::unique_ptr<Helpers::Gl::ShaderProgram> transferShader;
std::unique_ptr<Helpers::Gl::ShaderProgram> avgNDFRenderingShader;
std::unique_ptr<Helpers::Gl::ShaderProgram> barRenderingShader;
std::unique_ptr<Helpers::Gl::ShaderProgram> ndfExplorerShader;
std::unique_ptr<Helpers::Gl::ShaderProgram> selectionShader;
std::unique_ptr<Helpers::Gl::ShaderProgram> homDepthTransferShader;
std::unique_ptr<Helpers::Gl::ShaderProgram> drawDepthBufferShader;
//std::unique_ptr<Helpers::Gl::ShaderProgram> convolutionShader;
std::unique_ptr<Helpers::Gl::ShaderProgram> tileResetShader;
std::unique_ptr<Helpers::Gl::ShaderProgram> drawNDFShader;
std::unique_ptr<Helpers::Gl::ShaderProgram> buildHOMShader_C;
std::unique_ptr<Helpers::Gl::ShaderProgram> preIntegrateBins_C;
std::unique_ptr<Helpers::Gl::ShaderProgram> sumPreIntegratedBins_C;
std::unique_ptr<Helpers::Gl::ShaderProgram> overlayDownsampledNDFColor_C;
//std::unique_ptr<Helpers::Gl::ShaderProgram> rotationShader;
std::unique_ptr<Helpers::Gl::ShaderProgram> tiledRaycastingShader;
std::unique_ptr<Helpers::Gl::ShaderProgram> tiledRaycastingShader_C;
NdfImposters::NormalTransferFunction<float> *activeTransferFunction;
std::vector<NdfImposters::NormalTransferFunction<float>> transferFunctions;
std::vector<NdfImposters::NormalTransferFunction<float>> barStages;
std::vector<NdfImposters::NormalTransferFunction<float>> chromeTexture;
NdfImposters::NormalTransferFunction<float> *activeBarStage;
NdfImposters::NormalTransferFunction<float> *activeChromeTexture;
std::vector<bool> chromeTextureBits;
int activeTransferFunctionIndex = 0;

GLuint selectionTexture;
GLuint avgNDFTexture;

GLuint ndfExplorerTexture;

GLuint selectedPixelsVbo;
GLuint selectedPixelsVao;

GLuint avgNDFVbo;
GLuint avgNDFVao;

auto particleRadius = 1.0f;
auto particleScale = 1.0f;
auto particleCenters = std::vector<glm::vec4>();
std::vector<float> rawParticleCenters;

auto particleRadiusLarger = 2.0f;
auto particleCentersLarger = std::vector<glm::vec3>();

auto particleInstances = glm::ivec3(1, 1, 1);

std::unique_ptr<NdfImposters::NdfProgressiveParticleSampler<>> particleSampler;

// particles as quads
Helpers::Gl::BufferHandle particlesGlBuffer;
//Helpers::Gl::BufferHandle particlesGlBuffer_back;

//std::vector<Helpers::Gl::BufferHandle> particlesGLBufferArray

int bufferId = 0;

//array of particlebuffers, each entry in the vector is <id of node in spatial subdivision structure (ie id of octree or grid node), buffer of vertecies corresponding to this node>
std::vector< Helpers::Gl::BufferHandle> particlesGLBufferArray;

std::vector<std::pair<std::string, SparseENtfDataMemo>> sparseENtfCache;

TwBar *theBar;
bool tweakbarInitialized = false;
unsigned int currentTime = 0;

#include "callbacks.inl"

void loadSparseENtf(std::string filePath) {
	auto it = std::find_if(sparseENtfCache.begin(), sparseENtfCache.end(), [=](decltype(*sparseENtfCache.begin())& element) {
		return (element.first.compare(filePath) == 0);
	});

	// not thread safe - mutation of cache invalidates pointer
	std::pair<std::string, SparseENtfDataMemo> *sparseENtfDataMemo = nullptr;
	if (it != sparseENtfCache.end()) {
		sparseENtfDataMemo = &(*it);
	}
	else {
		auto loadingStart = std::chrono::system_clock::now();

		SparseENtfHeader header;
		SparseApproximation::GpuGmm gpuRepresentation;
		auto offsetMapSize = size_t(0);
		auto gpuGaussianSize = size_t(0);
		{
			std::cout << "Loading sparse eNTF file " << filePath << std::endl;
			Helpers::IO::FileGuard<std::ifstream> sparseFile(filePath, std::ifstream::in | std::ifstream::binary);
			if (!sparseFile.file.is_open()) {
				std::cout << "File not found!" << std::endl;
			}

			sparseFile.file.read(reinterpret_cast<char*>(&header), sizeof(header));

			offsetMapSize = header.ImageWidth * header.ImageHeight * sizeof(*gpuRepresentation.offsets.begin());
			gpuRepresentation.offsets.resize(header.ImageWidth * header.ImageHeight);

			gpuGaussianSize = header.GaussianCount * sizeof(*gpuRepresentation.gaussians.begin());
			gpuRepresentation.gaussians.resize(header.GaussianCount);
			gpuRepresentation.quantizationVarianceScale = header.QuantizationVarianceScale;
			quantizationVarianceScale = gpuRepresentation.quantizationVarianceScale;
			if (histogramResolution.x != header.HistogramWidth || histogramResolution.y != header.HistogramHeight) {
				std::cout << "Histogram resolution of sparse eNTF does not fit!" << std::endl;
			}
			//if(windowSize.x != header.ImageWidth || windowSize.y != header.ImageHeight) {
			//	std::cout << "Image resolution of sparse eNTF does not fit!" << std::endl;
			//}
			//glutReshapeWindow(header.ImageWidth, header.ImageHeight);

			sparseFile.file.read(reinterpret_cast<char*>(&gpuRepresentation.offsets[0]), offsetMapSize);
			sparseFile.file.read(reinterpret_cast<char*>(&gpuRepresentation.gaussians[0]), gpuGaussianSize);
		}

		glFlush();
		glFinish();
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - loadingStart);
		std::cout << "Sparse eNTF loaded and uploaded to GPU in " << duration.count() << " ms" << std::endl;

		// add to cache
		SparseENtfDataMemo newMemo;
		newMemo.fileHeader = std::move(header);
		newMemo.sparseData = std::move(gpuRepresentation);
		sparseENtfCache.emplace_back(filePath, newMemo);

		sparseENtfDataMemo = &sparseENtfCache.back();
	}

	// upload approximated data				
	if (!eNtfOffsetSsbo) {
		glGenBuffers(1, &eNtfOffsetSsbo);

		const auto ssboBindingPointIndex = GLuint(0);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, eNtfOffsetSsbo);
		{
			//auto blockIndex = glGetProgramResourceIndex(eNtfRenderingShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "OffsetMap");
			//glShaderStorageBlockBinding(eNtfRenderingShader->GetGlShaderProgramHandle(), blockIndex, ssboBindingPointIndex);
		}
	}

	if (!eNtfGaussianSsbo) {
		glGenBuffers(1, &eNtfGaussianSsbo);

		const auto ssboBindingPointIndex = GLuint(1);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, eNtfGaussianSsbo);
		{
			//auto blockIndex = glGetProgramResourceIndex(eNtfRenderingShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "Gaussians");
			//glShaderStorageBlockBinding(eNtfRenderingShader->GetGlShaderProgramHandle(), blockIndex, ssboBindingPointIndex);
		}
	}

	{
		auto offsetMapSize = sparseENtfDataMemo->second.fileHeader.ImageWidth * sparseENtfDataMemo->second.fileHeader.ImageHeight * sizeof(*(sparseENtfDataMemo->second.sparseData.offsets.begin()));

		const auto ssboBindingPointIndex = GLuint(0);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, eNtfOffsetSsbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, offsetMapSize, sparseENtfDataMemo->second.sparseData.offsets.data(), GL_DYNAMIC_COPY);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	{
		auto gpuGaussianSize = sparseENtfDataMemo->second.fileHeader.GaussianCount * sizeof(*(sparseENtfDataMemo->second.sparseData.gaussians.begin()));

		const auto ssboBindingPointIndex = GLuint(1);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, eNtfGaussianSsbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, gpuGaussianSize, sparseENtfDataMemo->second.sparseData.gaussians.data(), GL_DYNAMIC_COPY);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	renderSparse = true;
	samplingRunIndex = maxSamplingRuns;
}
void updateExtents()
{
	//rotate model extents by global rotation matrix
	std::vector<glm::vec3> corners;
	glm::vec3 tv;

	//update obb
	obb.clear();

	modelMax = origModelMax;
	modelMin = origModelMin;


	corners.push_back(glm::vec3(origModelMin.x, origModelMin.y, origModelMax.z));
	corners.push_back(glm::vec3(origModelMax.x, origModelMin.y, origModelMax.z));
	corners.push_back(origModelMax);
	corners.push_back(glm::vec3(origModelMin.x, origModelMax.y, origModelMax.z));
	corners.push_back(glm::vec3(origModelMax.x, origModelMin.y, origModelMin.z));
	corners.push_back(glm::vec3(origModelMax.x, origModelMax.y, origModelMin.z));
	corners.push_back(glm::vec3(origModelMin.x, origModelMax.y, origModelMin.z));
	corners.push_back(origModelMin);

	for (int i = 0; i < corners.size(); i++)
	{
		tv = GlobalRotationMatrix*corners[i];
		//update obb
		obb.push_back(tv);

		modelMax.x = std::max(tv.x, modelMax.x);
		modelMax.y = std::max(tv.y, modelMax.y);
		modelMax.z = std::max(tv.z, modelMax.z);

		modelMin.x = std::min(tv.x, modelMin.x);
		modelMin.y = std::min(tv.y, modelMin.y);
		modelMin.z = std::min(tv.z, modelMin.z);
	}
	modelExtent = modelMax - modelMin;


}
void initialize()
{
	//clear ndf
	if (phys_tex_dim>0)
		clear_NDF_Cache();

	//update extents as per rotations
	updateExtents();

	//set highest resolution
	float pixelsPerObjectSpaceUnit = 50000.0f;//   std::min(50000.0f, 100000.0f / modelExtent.x);  //object space unit is '1'
	float model_aspect_ratio = modelExtent.x / modelExtent.y;

	//set width and height with same ratio as aspect ratio of model to have as few as possible empty tiles
	int w = pixelsPerObjectSpaceUnit*modelExtent.x;
	int h = w / model_aspect_ratio;

	//make sure width and height are powers of 2
	highest_w_res = pow(2, std::ceil(log2(w)));
	highest_h_res = pow(2, std::ceil(log2(h)));

	//update the bounding box of the data such that it has the same ratio as higheest_w_res and highest_h_res
	modelExtent.x *= highest_w_res / w;
	modelExtent.y *= highest_h_res / h;

	//make sure model extent is between 0 and 1
	//float factor = std::max(modelExtent.x, modelExtent.y);
	//if (factor > 1)
	//{
	//	modelExtent.x = modelExtent.x / factor;
	//	modelExtent.y = modelExtent.y / factor;
	//	modelExtent.z = modelExtent.z / factor;
	//}

	//update dimensions of HOM
	homHighestResolution.y = origHomHighestResolution.x / model_aspect_ratio;
	homHighestResolution.x = pow(2, std::ceil(log2(origHomHighestResolution.x)));
	homHighestResolution.y = pow(2, std::ceil(log2(homHighestResolution.y)));
	Helpers::Gl::DeleteRenderTarget(homRenderTarget);
	homRenderTarget = Helpers::Gl::CreateRenderTarget(homHighestResolution.x, homHighestResolution.y, GL_RGBA32F, GL_RGBA, GL_FLOAT);

	//for drawing one sphere, this should be enabled
	//origModelMin = -.5f*modelExtent;
	//origModelMax = .5f*modelExtent;

	//updatemodel extent
	//float ratio = highest_w_res / highest_h_res;
	//modelExtent.y = modelExtent.x / ratio;

	////decide on tile width and height
	//tile_h = 128;// sqrt(res);
	//tile_w = 128;// sqrt(res);


	//debug
	//std::cout << std::endl;
	//std::cout << std::endl;
	//std::cout << "aspect ratio " << model_aspect_ratio<<std::endl;
	//std::cout << "model extent " << modelExtent.x << ", " << modelExtent.y << std::endl;
	//std::cout << "w and h" << w << ", " << h << std::endl;
	//std::cout << "highest w and h" << highest_w_res << ", " << highest_h_res << std::endl;
	//std::cout << std::endl;
	//std::cout << std::endl;

	//end debug

	//initizlize lod
	LOD = lod(highest_w_res, highest_h_res, initialCameraDistance, windowSize.x, windowSize.y, tile_w, tile_h, modelExtent, pixelsPerObjectSpaceUnit,lodDelta);
	//LOD = lod(highest_w_res, highest_h_res, initialCameraDistance, windowSize.x, windowSize.y, tile_w, tile_h, glm::vec3(OrigR-OrigL,OrigT-OrigB,10000), pixelsPerObjectSpaceUnit);

	//initialize the camera distance
	if (phys_tex_dim < 0)
	{
		int closest = -1;
		int lw, lh;
		int cw = 10000000000;
		for (int i = 0; i < LOD.myTiles.size(); i++)
		{
			LOD.get_lod_width_and_hight(i, lw, lh);
			if (abs(lw - windowSize.x)< cw)
			{
				closest = i;
				cw = abs(lw - windowSize.x);
			}
		}

		//**************************************new
		//adjust initial lod to and initial camera distance
		LOD.initial_lod = closest;
		initialCameraDistance = LOD.get_cam_dist(closest);
		//**************************************end new

		//adust camera distance to be at middle lod
		cameraDistance = LOD.get_cam_dist(closest);
		cameraDistance = std::max(LOD.min_cam_dist, std::min(cameraDistance, LOD.max_cam_dist - .3f));
		camPosi = CameraPosition(cameraRotation, cameraDistance);
		camPosi += cameraOffset;
		camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;
	}

	//generate the physical texture
	phys_tex_dim = 2816;// 3968;// 2816;//1408; res;  1408 for 16x16 bins and 2816 for 8x8 bins

	//set max tile number
	max_tiles = pow(phys_tex_dim / tile_w, 2);



	//
	//4-> initialize occupied array
	//
	occupied.clear();
	for (int i = 0; i < pow(phys_tex_dim / tile_w, 2); i++)
		occupied.push_back(std::make_pair(std::make_pair(false, glm::vec2(0, 0)), 0));
	//
	//5-> initialize page texture
	//
	glDeleteTextures(1, &Page_Texture);
	Page_Texture = 0;
	glGenTextures(1, &Page_Texture);
	glBindTexture(GL_TEXTURE_2D, Page_Texture);
	
	Page_Texture_Datatype* temp_page = new Page_Texture_Datatype[int(std::ceil(highest_w_res / float(tile_w))*std::ceil(highest_h_res / float(tile_h)) * 4)];

	for (int i = 0; i < std::ceil(highest_w_res / float(tile_w))*std::ceil(highest_h_res / float(tile_h)) * 4; i++)
		temp_page[i] = -1;


	glTexImage2D(GL_TEXTURE_2D, 0, Page_texture_internalFormat, std::ceil(highest_w_res / float(tile_w)), std::ceil(highest_h_res / float(tile_h)), 0, Page_Texture_format, Page_Texture_type, temp_page);
	glGenerateMipmap(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, 0);
	delete[] temp_page;

	//reset cells to be tested
	for (int i = 0; i < cellsTbt.size(); i++)
		cellsTbt.pop();

	//intialize cells to be tested with 'root' node
	cellsTbt.push(0);

	//buildhomflag reset
	//buildHOMFlag = true;
}

typedef void(*__PFNWGLSWAPINTERVALEXTPROC)(int);

int main(int argc, char* argv[]) {

    cxxopts::Options options("NDFRenderer", "NDFRenderer");
    options.add_options()
        ("frame", "select frame from dataset", cxxopts::value<int>())
        ("filename", "set the dataset to load", cxxopts::value<std::string>())
        ("imagename", "set the name of the screen dump", cxxopts::value<std::string>())
        ("numparticles", "set the number of particles to load", cxxopts::value<size_t>())
        ("save", "set whether to capture a screenshot when converged")
        ("iterations", "override the number of rays that indicates convergence", cxxopts::value<int>())
        ("quit", "set whether to quit after saving the screenshot")
        ("help", "print help")
        ;
    options.parse(argc, argv);
    timeStep = options["frame"].as<int>();
    cmdlineFileName = options["filename"].as<std::string>();
    cmdlineImageName = options["imagename"].as<std::string>();
    cmdlineSaveImage = options["save"].as<bool>();
    cmdlineAutoQuit = options["quit"].as<bool>();
    cmdlineIterations = options["iterations"].as<int>();
    cmdlineNumParticles = options["numparticles"].as<size_t>();
    if (options.count("help")) {
        std::cout << options.help({ "" }) << std::endl;
        exit(0);
    }

	//debug
	//outfile.open("data.txt");

	//end debug

	glutInit(&argc, argv);

	glutInitWindowSize(INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	glutCreateWindow("NDF Imposters");

	glutDisplayFunc(display);
	glutIdleFunc(idle);
	glutKeyboardFunc(keyboard);
	glutKeyboardUpFunc(keyboardUp);
	glutMouseFunc(mouse);
	glutMotionFunc(mouseMotionActive);
	glutReshapeFunc(reshape);

	//int cx, cy, cz;
	//glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &cx);
	//glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &cy);
	//glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &cz);

	//std::cout << "The max work groups for compute shader in x, y and z are: " << cx << ", " << cy << ", " << cz << std::endl;

	//Mohamed's
	//s.open("debuging_time.txt");
	//end Mohamed's

	// paser command line
	//if (argc > 1) {
	//	timeStep = std::atoi(argv[1]);
	//}

	__PFNWGLSWAPINTERVALEXTPROC mySwap = (__PFNWGLSWAPINTERVALEXTPROC)wglGetProcAddress("wglSwapIntervalEXT");
	//mySwap(0);

	//wglSwapIntervalEXT(0); // disable Vsync

	auto datasetPath = std::string("./");
	datasetPath = "../../data/";

	//auto result = Helpers::InitializeExtensions();
	//assert(result);
    if (!gladLoadGL()) {
        printf("Something went wrong!\n");
        exit(-1);
    }
    if (!GLAD_GL_VERSION_4_5) {
        printf("We think we need OpenGL 4.5\n");
        exit(-2);
    }
    if (GLAD_GL_KHR_debug) {
        //glEnable(GL_DEBUG_OUTPUT);
        //glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
        //glDebugMessageCallback(DebugCallback, NULL);
        //glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, NULL, GL_TRUE);
        //GLuint ignorethis;
        //// Buffer detailed info: Buffer object # (bound to GL_ARRAY_BUFFER_ARB, usage hint is GL_STATIC_DRAW) will use VIDEO memory as the source for buffer object operations.
        //ignorethis = 131185;
        //glDebugMessageControl(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_OTHER, GL_DONT_CARE, 1, &ignorethis, GL_FALSE);
        //// The driver allocated storage for renderbuffer #
        //ignorethis = 131169;
        //glDebugMessageControl(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_OTHER, GL_DONT_CARE, 1, &ignorethis, GL_FALSE);
        //// Texture state usage warning: Texture # is base level inconsistent. Check texture size.
        //ignorethis = 131204;
        //glDebugMessageControl(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_OTHER, GL_DONT_CARE, 1, &ignorethis, GL_FALSE);
        //// Buffer performance warning: Buffer object 2 (bound to GL_SHADER_STORAGE_BUFFER, GL_SHADER_STORAGE_BUFFER (1), GL_SHADER_STORAGE_BUFFER (2), GL_SHADER_STORAGE_BUFFER (3), GL_SHADER_STORAGE_BUFFER (4), GL_SHADER_STORAGE_BUFFER (5), GL_SHADER_STORAGE_BUFFER (6), and GL_SHADER_STORAGE_BUFFER (7), usage hint is GL_DYNAMIC_COPY) is being copied/moved from VIDEO memory to HOST memory.
        //ignorethis = 131186;
        //glDebugMessageControl(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_PERFORMANCE, GL_DONT_CARE, 1, &ignorethis, GL_FALSE);
    }

    glGetString(GL_VERSION);
    std::cout << "OpenGL version is: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL version is: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;

	compileShaders(shaderPath);

#ifdef CORE_FUNCTIONALITY_ONLY
#else
	//create texture to be used for selection
	glGenTextures(1, &selectionTexture);
	glBindTexture(GL_TEXTURE_2D, selectionTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, windowSize.x, windowSize.y, 0, GL_RGBA, GL_FLOAT, nullptr);
	glBindTexture(GL_TEXTURE_2D, 0);

	//create vbo to be used for selection
	{
		glGenVertexArrays(1, &selectedPixelsVao);
		glBindVertexArray(selectedPixelsVao);

		glGenBuffers(1, &selectedPixelsVbo);
		glBindBuffer(GL_ARRAY_BUFFER, selectedPixelsVbo);

		glBufferData(GL_ARRAY_BUFFER, MaxAllowedSelectedPixels, reinterpret_cast<const GLvoid*>(selectedPixels.data()), GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	//create texture to be used for avgndf
	glGenTextures(1, &avgNDFTexture);
	glBindTexture(GL_TEXTURE_2D, avgNDFTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, windowSize.x/6, windowSize.y/6, 0, GL_RGBA, GL_FLOAT, nullptr);
	glBindTexture(GL_TEXTURE_2D, 0);

	//create vbo to be used for selection
	{
		glGenVertexArrays(1, &avgNDFVao);
		glBindVertexArray(avgNDFVao);

		glGenBuffers(1, &avgNDFVbo);
		glBindBuffer(GL_ARRAY_BUFFER, avgNDFVbo);

		glBufferData(GL_ARRAY_BUFFER, (windowSize.y / 6)*(windowSize.x/6), NULL, GL_DYNAMIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
#endif




	transferFunctions.resize(4, glm::ivec2(64, 64));
	transferFunctions[0].FromPicture("../../NTF/NTF64 reflection2.png");
	transferFunctions[1].FromPicture("../../NTF/NTF64 Glass2.png");  //E: / Code Repositry / PointImposters;
	transferFunctions[2].FromPicture("../../NTF/NTF64 Gold.png");
	transferFunctions[3].FromPicture("../../NTF/NTF64 Metal3.png");//"D:/KAUST_2015_02_06/Datensatz/NTF32 Weird.png");
	activeTransferFunction = &transferFunctions[0];

	chromeTexture.resize(32, glm::ivec2(64, 64));
	std::string chromePath = "../../NTF/simple64-";
	for (int i = 0; i < chromeTexture.size(); i++)
	{
		chromeTexture[i].FromPicture(chromePath + std::to_string(i) + ".png");
	}
	chromeTextureBits.resize(5, true);
	activeChromeTexture = &chromeTexture[0];


	barStages.resize(28, glm::ivec2(31, 456));
	std::string barPath = "../../Bar/";
	for (int i = 0; i < barStages.size(); i++)
	{
		barStages[i].FromPicture(barPath + std::to_string(i) + ".png");
	}
	activeBarStage = &barStages[0];

	for (auto &transferFunction : transferFunctions)
	{
		transferFunction.UpdateTexture();
	}

	for (auto &barStage : barStages)
	{
		barStage.UpdateTexture();
	}

	for (auto &chromeT : chromeTexture)
	{
		chromeT.UpdateTexture();
	}

	//int imw, imh;
	//Transfer_Texture = 0;
	//glGenTextures(1, &Transfer_Texture);
	//glBindTexture(GL_TEXTURE_2D, Transfer_Texture);reg
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 32, 32, 0, GL_RGBA, GL_UNSIGNED_BYTE, &transferFunctions[0]);

#if SET_ID > 0
	const std::string filePath = fileFolder + filePrefix + std::to_string(timeStep) + fileSuffix;
	sparseENtfFilePath = filePath + ".entf";
#endif

#ifndef READ_FROM_FILE

	auto fileParticleRadius = 1.0f;
	glm::mat2x3 originalBoundingBox(0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f);
	bool overwriteRadius = false; // uses particleRadius instead of file radius if true
	//max_node_memory_consumption = pow(2, 32);// set a maximum size of a node, say 200mb (2^28)/ or 2GB (2^31) for our card.

	//particleCenters = Helpers::IO::ParticlesFromMmpld(datasetPath + "CuAg_55er_fixed.mmpld", MaxParticles, fileParticleRadius, areaOfInterest, 100, originalBoundingBox, true);

	//Helpers::IO::readParticleData(datasetPath + "laser.big.mmpld", MaxParticles, fileParticleRadius, areaOfInterest, MaxFrameCount, originalBoundingBox, global_tree, max_node_memory_consumption, true);
	//Helpers::IO::readParticleData(datasetPath + "Halle 10.mmpld", 500000000, fileParticleRadius, areaOfInterest, MaxFrameCount, originalBoundingBox, global_tree, max_node_memory_consumption, true);


	//particleCenters = Helpers::IO::ParticlesFromMmpld(datasetPath + "Halle 10.mmpld", 500000000, fileParticleRadius, areaOfInterest, MaxFrameCount, originalBoundingBox, true);
    //particleCenters = Helpers::IO::ParticlesFromMmpld(datasetPath + "laser.big.mmpld",MaxParticles, fileParticleRadius, areaOfInterest, MaxFrameCount, originalBoundingBox, true);

    if (cmdlineFileName.empty()) {
        cmdlineFileName = "ccmv-1.0.mmpld";
    }
	particleCenters = Helpers::IO::ParticlesFromMmpld(datasetPath+cmdlineFileName, cmdlineNumParticles > 0 ? cmdlineNumParticles : MaxParticles, fileParticleRadius, areaOfInterest, timeStep, originalBoundingBox, true);


	//<<<<<<< HEAD
	//particleCenters = Helpers::IO::ParticlesFromMmpld(datasetPath + "laser.big.mmpld",500000, fileParticleRadius, areaOfInterest, MaxFrameCount, originalBoundingBox, true);

	//=======
	//	//particleCenters = Helpers::IO::ParticlesFromMmpld(datasetPath + "laser.big.mmpld", MaxParticles, fileParticleRadius, areaOfInterest, MaxFrameCount, originalBoundingBox, true);
	//    //particleCenters = Helpers::IO::ParticlesFromMmpld("S:\\Daten\\Partikel\\FARO\\Skulptur1.mmpld", MaxParticles, fileParticleRadius, areaOfInterest, MaxFrameCount, originalBoundingBox, true);
	//    //particleCenters = Helpers::IO::ParticlesFromMmpld("S:\\Daten\\Partikel\\mheinen\\Phasenzerfall\\simVL\\run05\\megamol_0000050000.mmpld", MaxParticles, fileParticleRadius, areaOfInterest, MaxFrameCount, originalBoundingBox, true);
	//    particleCenters = Helpers::IO::ParticlesFromMmpld("S:\\Daten\\Partikel\\Gaugler+Lutz\\Halle 10.mmpld", 40000000, fileParticleRadius, areaOfInterest, MaxFrameCount, originalBoundingBox, true);
	//>>>>>>> eda623985e5e61314e0e5c39315fcf0dbc5acd91
	//particleCenters = Helpers::IO::ParticlesFromMmpld(datasetPath + "laser.big.mmpld", MaxParticles, fileParticleRadius, areaOfInterest, MaxFrameCount, originalBoundingBox, true);
	//particleCenters = Helpers::IO::ParticlesFromMmpld(datasetPath+"scivis2015largeintensity.mmpld",    50000000, fileParticleRadius, areaOfInterest, MaxFrameCount, originalBoundingBox, true);


	//std::cout << "max frame count " << MaxFrameCount << std::endl;
	//debug
	//glm::vec4 temp = particleCenters[0];
	//particleCenters.clear();
	//particleCenters.push_back(temp);
	//end debug



#if 0
	{
		//originalBoundingBox[1][1] = originalBoundingBox[1][0];
		//originalBoundingBox[0][1] = originalBoundingBox[0][0];
		
		particleCenters.clear();
		glm::vec3 count = glm::vec3(100, 100, 100);
		
		glm::vec3 s = glm::vec3(-1, -1, -1);
		glm::vec3 e = s*-1.0f;

		originalBoundingBox[0] = s;
		originalBoundingBox[1] = e;

		float Prad = 0.4f*((e.x-s.x)/count.x);

		for (float i = s.x; i < e.x; i = i + (e.x - s.x) / count.x)
		{
			for (float j = s.y; j < e.y; j = j + (e.y - s.y) / count.y)
			{
				for (float k = s.z; k < e.z; k = k + (e.z - s.z) / count.z)
				{
					particleCenters.push_back(glm::vec4(i, j, k, Prad));
				}
			}
		}
	}
#endif

#if 0
	{
		originalBoundingBox[1][1] = originalBoundingBox[1][0];
		originalBoundingBox[0][1] = originalBoundingBox[0][0];
		float Prad = 20 * particleCenters[0].w;
		particleCenters.clear();
		glm::vec3 count = glm::vec3(1, 1, 1);

		glm::vec3 ex = glm::vec3(originalBoundingBox[1][0] - originalBoundingBox[0][0], originalBoundingBox[1][1] - originalBoundingBox[0][1], originalBoundingBox[1][2] - originalBoundingBox[0][2]);
		glm::vec3 s = glm::vec3(-1, -1 * ex.y / ex.x, -1 * ex.z / ex.x);
		glm::vec3 e = s*-1.0f;

		for (float i = s.x; i < e.x; i = i + (e.x - s.x) / count.x)
		{
			for (float j = s.y; j < e.y; j = j + (e.y - s.y) / count.y)
			{
				for (float k = s.z; k < e.z; k = k + (e.z - s.z) / count.z)
				{
					particleCenters.push_back(glm::vec4(i, j, k, Prad));
				}
			}
		}
	}
#endif
	//end debug





	//particleCenters = Helpers::Procedural::GenerateSawToothSphere(0.001f,5000,5000);
	//particleRadius = 0.001f;
	
	
	//std::cout << "the number particles is: " <<particleCenters.size()<< std::endl;*/
	//particleRadius = 0.00075f;//0.001f;

	//std::cout << "before data" << std::endl;
	//while (true)
	//{
	//}
	//particleCenters = Helpers::IO::ParticlesFromMmpld(datasetPath + "expl30m_bin_b.mmpld", MaxParticles, fileParticleRadius, areaOfInterest, MaxFrameCount, originalBoundingBox, true);
	//particleCenters = Helpers::IO::ParticlesFromMmpld(datasetPath + "expl30m_bin_fix_a.mmpld", MaxParticles, fileParticleRadius, areaOfInterest, 40, originalBoundingBox, true);
	//particleCenters = Helpers::IO::ParticlesFromMmpld(datasetPath + "cyclone_25M_t137.mmpld", MaxParticles, fileParticleRadius, areaOfInterest, MaxFrameCount, originalBoundingBox, true);

	{
		//particleCenters = Helpers::IO::ParticlesFromMmpld(datasetPath + "laser.00059.mmpld", MaxParticles, fileParticleRadius, areaOfInterest, MaxFrameCount, originalBoundingBox, true);



		//origModelMin = glm::vec3(*particleCenters.begin());
		//origModelMax = origModelMin;
		//for (auto& particle : particleCenters) {
		//	origModelMin.x = std::min(origModelMin.x, particle.x);
		//	origModelMin.y = std::min(origModelMin.y, particle.y);
		//	origModelMin.z = std::min(origModelMin.z, particle.z);

		//	origModelMax.x = std::max(origModelMax.x, particle.x);
		//	origModelMax.y = std::max(origModelMax.y, particle.y);
		//	origModelMax.z = std::max(origModelMax.z, particle.z);
		//}
		//origModelExtent = origModelMax - origModelMin;

		////add four more instances
		//int s = particleCenters.size();
		//glm::vec4 p;
		//for (int i = 0; i < s; i++)
		//{
		//	p = particleCenters[i];
		//	p.y += origModelExtent.y;
		//	particleCenters.push_back(p);
		//	p.y += origModelExtent.y;
		//	particleCenters.push_back(p);
		//	p.y += origModelExtent.y;
		//	particleCenters.push_back(p);
		//	p.y += origModelExtent.y;
		//	particleCenters.push_back(p);
		//	p.y += origModelExtent.y;
		//	particleCenters.push_back(p);

		//}
	}
	//particleCenters = Helpers::IO::ParticlesFromMmpld(datasetPath + "laser.00000.mmpld", MaxParticles, fileParticleRadius, areaOfInterest, MaxFrameCount, originalBoundingBox, true);



	//particleCenters = Helpers::IO::ParticlesFromMmpld(datasetPath+"zug0-3.00004.chkpt.mmpld", MaxParticles, fileParticleRadius, areaOfInterest, MaxFrameCount, originalBoundingBox, true);
	//particleCenters = Helpers::IO::ParticlesFromMmpld(datasetPath + "cyclone_50k.mmpld", MaxParticles, fileParticleRadius, areaOfInterest, MaxFrameCount, originalBoundingBox, true);
	particleRadius = 0.000075f;//0.000075f;



	//debug 'synthesized data set'
#if 0
	{
		//originalBoundingBox[1][1] = originalBoundingBox[1][0];
		//originalBoundingBox[0][1] = originalBoundingBox[0][0];
		
		particleCenters.clear();
		glm::vec3 count = glm::vec3(100, 100, 100);

		//glm::vec3 ex = glm::vec3(originalBoundingBox[1][0] - originalBoundingBox[0][0], originalBoundingBox[1][1] - originalBoundingBox[0][1], originalBoundingBox[1][2] - originalBoundingBox[0][2]);
		glm::vec3 s = glm::vec3(-.3,-.3,-.3);
		glm::vec3 e = glm::vec3(.3, .3, .3);;

		float Prad = 0.5f*((e.x-s.x)/count.x);

		for (float i = s.x; i < e.x; i = i + (e.x - s.x) / count.x)
		{
			for (float j = s.y; j < e.y; j = j + (e.y - s.y) / count.y)
			{
				for (float k = s.z; k < e.z; k = k + (e.z - s.z) / count.z)
				{
					float rad = Prad;// / pow(2, abs(k - e.z));   //here we want particles along the z axis to get smaller as they get farter from viewer, this is a test case
					//that is, if HOM is correctly implemented, and if we have say 8 leaves, then 4 leaves (with farther z value) must be culled
					particleCenters.push_back(glm::vec4(i, j, k, rad));
				}
			}
		}
	}
#endif





	//std::cout << "done with load data" << std::endl;
	//while (true)
	//{
	//}

	//particleCenters = Helpers::IO::ParticlesFromMmpld(datasetPath+"Space_Al_114J/laser.00059.chkpt.mmpld", MaxParticles, fileParticleRadius, areaOfInterest, MaxFrameCount, originalBoundingBox, true);
	//particleRadius = 0.000008f;
	//particleInstances = glm::ivec3(1, 5, 1);
	//particleInstances = glm::ivec3(1, 8, 1);

	//particleCenters = Helpers::IO::ParticlesFromMmpld("E:/Datasets/expl30m/expl30m_bin_fix_a.mmspd_90.mmpld", MaxParticles, fileParticleRadius, areaOfInterest, MaxFrameCount, originalBoundingBox, true);
	//particleRadius = 0.000008f;
	//particleInstances = glm::ivec3(1, 2, 2);


	//particleCenters = Helpers::Procedural::GenerateCheckerboardSphere();
	//particleRadius = 0.001f;

	/*particleCenters = Helpers::Procedural::GenerateSawToothSphere();
	particleRadius = 0.00015f;
	overwriteRadius = true;
	maxSamplingRuns = 4096 * 4;
	cameraDistance = 3.5f;
	glutReshapeWindow(1920, 1080);
	lightRotationX = 140.0f;
	lightRotationY = 144.0f;*/

	//particleCenters = Helpers::Procedural::GenerateVGrooveSphere();
	//particleRadius = 0.00025f;


	// aliasing if spatial resolution is too low
	/*particleCenters = Helpers::IO::ParticlesFromMmpld("F:/Datasets/sonntag214/mmpld/laser.00150.mmpld", MaxParticles, fileParticleRadius, areaOfInterest, MaxFrameCount, originalBoundingBox, true);
	for(auto &particle : particleCenters) {
	auto temp = particle.x;
	particle.x = particle.y;
	particle.y = temp;

	temp = particle.z;
	particle.z = -particle.y;
	particle.y = temp;
	}
	//particleRadius = 0.0025f;
	particleRadius = 0.00125f;*/

	//particleCenters = Helpers::IO::ParticlesFromMmpld("D:/patrick/expl30m_bin_fix_a.mmspd.mmpld", 500000, fileParticleRadius, areaOfInterest, MaxFrameCount, true);
	//particleRadius = 0.0075f;

	// NOTE: seems to be too few particles to really benefit from this approach...
	// 0.001f -> > 128 samples
	//particleCenters = Helpers::IO::ParticlesFromMmpld(datasetPath+"zug0-3.00004.chkpt.mmpld", MaxParticles, fileParticleRadius, areaOfInterest, MaxFrameCount, true);
	// segmented
	//particleCenters = Helpers::IO::ParticlesFromMmpld(datasetPath+"zug4ku.00005.chkpt.mmpld", MaxParticles, fileParticleRadius, areaOfInterest, MaxFrameCount, true);
	// segmented
	//particleCenters = Helpers::IO::ParticlesFromMmpld(datasetPath+"zug4ku-6.00005.chkpt.mmpld", MaxParticles, fileParticleRadius, areaOfInterest, MaxFrameCount, true);
	//particleRadius = 0.0025f;//0.005f;//0.015f;//0.004f;//0.00175f;


#if 1
	origModelMin = glm::vec3(*particleCenters.begin());
	origModelMax = origModelMin;
	int sizeCount = 0;
	for (auto& particle : particleCenters) {
		origModelMin.x = std::min(origModelMin.x, particle.x);
		origModelMin.y = std::min(origModelMin.y, particle.y);
		origModelMin.z = std::min(origModelMin.z, particle.z);

		origModelMax.x = std::max(origModelMax.x, particle.x);
		origModelMax.y = std::max(origModelMax.y, particle.y);
		origModelMax.z = std::max(origModelMax.z, particle.z);

		sizeCount++;
	}


	if (sizeCount == 1)
	{
		origModelMin = glm::vec3(particleCenters.front().x, particleCenters.front().y, particleCenters.front().z) - glm::vec3(particleCenters.front().w, particleCenters.front().w, particleCenters.front().w);
		origModelMax = glm::vec3(particleCenters.front().x, particleCenters.front().y, particleCenters.front().z) + glm::vec3(particleCenters.front().w, particleCenters.front().w, particleCenters.front().w);

		originalBoundingBox[0] = origModelMin;
		originalBoundingBox[1] = origModelMax;
	}

	origModelExtent = origModelMax - origModelMin;
#else
	origModelMax = originalBoundingBox[1];
	origModelMin = originalBoundingBox[0];
	origModelExtent = origModelMax - origModelMin;
#endif


#if 0   //adding more instances
	{
		particleInstances = glm::ivec3(1, 2, 2);
		int s = particleCenters.size();
		glm::vec4 p;

		for (int l = 0; l < s; l++)
		{
			p = particleCenters[l];
			for (int i = 0; i < particleInstances.x; i++)
			{
				for (int j = 0; j < particleInstances.y; j++)
				{
					for (int k = 0; k < particleInstances.z; k++)
					{
						p += glm::vec4(origModelExtent.x*i, origModelExtent.y*j, origModelExtent.z*k, 0);
						particleCenters.push_back(p);
					}
				}
			}
		}


		//update bounding box and extent
		origModelExtent += glm::vec3(origModelExtent.x*(particleInstances.x-1), origModelExtent.y*(particleInstances.y-1), origModelExtent.z*(particleInstances.z-1));
		origModelMax = origModelMin + origModelExtent;

		//we want the extent to be a maximum of 1 in all dimensions
		float factor = std::max(std::max(origModelExtent.x, origModelExtent.y), origModelExtent.z);

		//get center of model
		glm::vec3 c = origModelMin + 0.5f*origModelExtent;
		glm::vec3 diff = glm::vec3(0, 0, 0) - c;

		for (int i = 0; i < particleCenters.size(); i++)
		{
			particleCenters[i] += glm::vec4(diff,0);
			particleCenters[i] *= 1.0f/factor;
		}

		//update extent, and min and max
		origModelExtent *= 1.0f / factor;
		origModelMin += diff;
		origModelMin *= 1.0f / factor;

		origModelMax += diff;
		origModelMax *= 1.0f / factor;

		originalBoundingBox[0] = origModelMin;
		originalBoundingBox[1] = origModelMax;
	}
#endif

	
	//glGetIntegerv(GL_MAX_COLOR_ATTACHMENTS, &MaxFboTexCount);

	if (!overwriteRadius) {
		// calculate particle radius scaling
		auto originalExtent = originalBoundingBox[1] - originalBoundingBox[0];
		auto longestEdge = std::max(std::max(std::max(originalExtent.x, 0.0f), originalExtent.y), originalExtent.z);

		const auto fileParticleRadiusScaling = (1.0f / longestEdge);
		for (auto& particle : particleCenters)
		{
			particle.w = particle.w * fileParticleRadiusScaling;
		}
	}





	//initialize lod
	//initialize physical texture
	//initialize page texture
	//initialize tile size
	//initialize ndf tree

	initialize();




	//end initialize

#ifdef NO_BRICKING
#else
#if 1
	//put data in octree
	{
		std::cout << "Initalizing Octree ..." << std::endl;
		//std::vector<glm::vec3> centers;
		glm::vec3 c = modelMin + 0.5f*(modelMax - modelMin);

		//make the bounding box have equal lengths
		//so 'halfextent' is maximum of modelmax.x...modelmax.z and same for model min

		glm::vec3 halfextent = 0.5f*(modelMax - modelMin);

		//halfextent = std::max(modelExtent.y, halfextent);
		//halfextent = std::max(modelExtent.z, halfextent);
		//halfextent *= 0.5f;


		//std::vector<int> indicies;
		//for (int i = 0; i < particleCenters.size(); i++)
		//{
		//	indicies.push_back(i);
		//	centers.push_back(glm::vec3(particleCenters[i].x, particleCenters[i].y, particleCenters[i].z));
		//}

		glm::vec2 v;
		//{
		//	int w, h;
		//	LOD.get_lod_width_and_hight(LOD.initial_lod, w, h);
		//	glm::vec2 s, e;
		//	std::cout << "the initial lod is " << LOD.initial_lod << std::endl;
		//	//get equivalent of half a pixel in object space
		//	s = LOD.pixel2obj(glm::vec2(0, 0), current_lod);

		//	//add that to the model min to make sure we are in the bounding box
		//	//s += glm::vec2(origModelMin.x,origModelMin.y);

		//	//get the equivalent of one pixel in obje ct space
		//	e = LOD.pixel2obj(glm::vec2(1, 1), current_lod);

		//	//add that to the model min to make sure we are in the bounding box
		//	//e += glm::vec2(origModelMin.x, origModelMin.y);

		//	//get radius
		//	v = e - s;
		//}


		

		
		//float pixel_dim = 10000 * glm::length(v);
		max_node_memory_consumption = pow(2, 30);// set a maximum size of a node, say 500 million particles.

		//debug
		//int s = particleCenters.size();
		//end debug
		unsigned int maxLevels = 0;


		global_tree.init(c, halfextent, sizeCount, 0, max_node_memory_consumption, maxLevels);
		std::cout << "Generated a tree with " << maxLevels << " levels" << std::endl;
		std::cout << "Pre-computing memory requirements per node" << std::endl;
		do{
			//reset memory flag
			global_tree.exceedMemory = false;

			//initiazlize octree storage requirements
			for (int i = 0; i < particleCenters.size(); i++)
			{
				global_tree.placeParticle(0, particleCenters[i], true);
				if (global_tree.exceedMemory)
				{
					std::cout << "memory exceeded, creating more levels for tree, increasing levels at dense parts of the tree "<<std::endl;
					break;
				}
			}
		} while (global_tree.exceedMemory);



		//regroup children sets (sets of 8 nodes that are children of one node) that occupy memory less than max_node_memory consumption
		//global_tree.groupSmallChildren(max_node_memory_consumption);
		//std::cout << "grouped smaller nodes into bigger ones" << std::endl;


		//allocate space for each leaf node with count >0
		global_tree.allocateStorage();

		std::cout << "Populating tree" << std::endl;
		for (int i = 0; i < particleCenters.size();i++)
		{
			global_tree.placeParticle(0,particleCenters[i],false);
		}
		particleCenters.clear();

		//global_tree.get_leaves();

		//debug
		//int count = 0;
		//for (int i = 0; i < global_tree.leaves.size(); i++)
		//{
		//	count += global_tree.nodes[global_tree.leaves[i]].Points.size();
		//}
		//std::cout << "count is: " << count << ", and s is: " << s << std::endl;
		//end debug
		std::cout << "Finished initializing Octree " << std::endl;
		std::cout << "Maximum node size in bytes: " << max_node_memory_consumption << std::endl;
		std::cout << "Number of leaves(bricks): " << global_tree.leaves.size() << std::endl;

		//global_tree.sort_nodes_backToFront(global_tree.leaves, GlobalRotationMatrix);
		global_tree.sort_nodes_frontToBack(global_tree.leaves, GlobalRotationMatrix);

		std::cout << "finished sorting leaves " << std::endl;

		//now we divide the leaves into bricks of size <=max_node_memory_consumption
		for (int i = 0; i < global_tree.leaves.size(); i++)
		{
			int divisions = global_tree.nodes[global_tree.leaves[i]].nodeCount / (max_node_memory_consumption / (4 * 4));
			int remaining = global_tree.nodes[global_tree.leaves[i]].nodeCount % (max_node_memory_consumption / (4 * 4));

			renderBricks.push_back(std::make_pair(global_tree.leaves[i], std::vector<int>(0)));

			for (int j = 0; j < divisions; j++)
				renderBricks[renderBricks.size() - 1].second.push_back(max_node_memory_consumption / (4 * 4));

			if (remaining>0)
				renderBricks[renderBricks.size() - 1].second.push_back(remaining);
		}
		
		//update cells to render
		cellsToRender = global_tree.leaves;

		
		//print stats in a file


		
		//{
		//	out.open("octreeStats.txt");
		//	std::vector<int> leaves;
		//	global_tree.get_leaves(leaves);

		//	for (int i = 0; i < leaves.size(); i++)
		//	{
		//		//print leaf index, and number of particles in leaf
		//		if (global_tree.nodes[leaves[i]].indicies.size()>0)
		//			out << leaves[i] << " " << global_tree.nodes[leaves[i]].indicies.size()<<std::endl;
		//	}
		//	out.close();
		//}



		//end of octree
	}
#else


		global_tree.get_leaves();
		std::cout << "Finished initializing Octree " << std::endl;
		std::cout << "Maximum node size in bytes: " << max_node_memory_consumption << std::endl;
		std::cout << "Number of leaves(bricks): " << global_tree.leaves.size() << std::endl;

		int sizeCount = 0;
		for (int i = 0; i < global_tree.leaves.size(); i++)
		{
			sizeCount += global_tree.nodes[global_tree.leaves[i]].Points.size();
		}
		std::cout << "Total Number of particles: " << sizeCount / 1000000.0f << " million particles" << std::endl;

		//global_tree.sort_nodes_backToFront(global_tree.leaves, GlobalRotationMatrix);
		global_tree.sort_nodes_frontToBack(global_tree.leaves, GlobalRotationMatrix);

		std::cout << "finished sorting leaves " << std::endl;
#endif
#endif

	//pass only visible particles to shader
	//particleCenters.erase(particleCenters.begin(), particleCenters.begin() + 100000);
	//end pass only visible particles to shader
#ifdef NO_BRICKING
		Helpers::Gl::MakeBuffer(particleCenters, particlesGlBuffer);
#else
#if 1

	

    
	
		if (renderBricks.size() > 1)
		{
			particlesGLBufferArray.push_back(Helpers::Gl::BufferHandle());
			particlesGLBufferArray.push_back(Helpers::Gl::BufferHandle());
			Helpers::Gl::MakeBuffer(max_node_memory_consumption, particlesGLBufferArray[0], global_tree.nodes[global_tree.leaves[0]].Points[0]);
			Helpers::Gl::MakeBuffer(max_node_memory_consumption, particlesGLBufferArray[1], global_tree.nodes[global_tree.leaves[0]].Points[0]);

			{
				auto glErr = glGetError();
				if (glErr != GL_NO_ERROR) {
					std::cout << "glError: " << __FILE__ << " " << __LINE__ << std::endl;
				}
			}
		}
		else
		{
			particlesGLBufferArray.push_back(Helpers::Gl::BufferHandle());
			Helpers::Gl::MakeBuffer(global_tree.nodes[global_tree.leaves[0]].Points.size() * sizeof(*global_tree.nodes[global_tree.leaves[0]].Points.begin()), particlesGLBufferArray[0], global_tree.nodes[global_tree.leaves[0]].Points[0]);
			particlesGLBufferArray[0].VertexCount_ = global_tree.nodes[global_tree.leaves[0]].Points.size();

			glBindBuffer(GL_ARRAY_BUFFER, particlesGLBufferArray[0].Vbo_);
			GLvoid* p = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
			memcpy(p, reinterpret_cast<char*>(&global_tree.nodes[global_tree.leaves[0]].Points[0]), global_tree.nodes[global_tree.leaves[0]].Points.size() * sizeof(*global_tree.nodes[global_tree.leaves[0]].Points.begin()));
			glUnmapBuffer(GL_ARRAY_BUFFER);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}

#else
	//{
	//	auto glErr = glGetError();
	//	if (glErr != GL_NO_ERROR) {
	//		std::cout << "glError: " << __FILE__ << " " << __LINE__ << std::endl;
	//	}
	//}
	//Helpers::Gl::MakeBuffer(particleCenters, global_tree, particlesGLBufferArray);         //bricking
	//{
	//	auto glErr = glGetError();
	//	if (glErr != GL_NO_ERROR) {
	//		std::cout << "glError: " << __FILE__ << " " << __LINE__ << std::endl;
	//	}
	//}
	//for (int i = 0; i < global_tree_leaves.size(); i++)
	//{
	//	particlesGLBufferArray.push_back(std::make_pair(global_tree_leaves[i], Helpers::Gl::BufferHandle()));
	//	Helpers::Gl::MakeBuffer(particleCenters, global_tree.nodes[global_tree_leaves[i]], particlesGLBufferArray[i].second);
	//}
#endif
#endif

	//std::cout << "Sampling " << particleCenters.size() << " particles" << std::endl;

	particleSampler = std::unique_ptr<NdfImposters::NdfProgressiveParticleSampler<>>(new NdfImposters::NdfProgressiveParticleSampler<>(ndfTree));

	//rayCastingSolutionRenderTarget = Helpers::Gl::CreateRenderTarget(spatialResolution.x, spatialResolution.y, GL_RGBA8, GL_RGBA, GL_UNSIGNED_BYTE);
	//rayCastingSolutionRenderTarget = Helpers::Gl::CreateRenderTarget(spatialResolution.x * multiSamplingRate, spatialResolution.y * multiSamplingRate, GL_RG8, GL_RG, GL_UNSIGNED_BYTE);
	//glViewport(0, 0, samplingScalingFactor*windowSize.x, samplingScalingFactor*windowSize.y);
	Helpers::Gl::DeleteRenderTarget(rayCastingSolutionRenderTarget);
	Helpers::Gl::DeleteRenderTarget(homRenderTarget);
	//rayCastingSolutionRenderTarget = Helpers::Gl::CreateRenderTarget(windowSize.x*3, windowSize.y*3, GL_RG8, GL_RG, GL_UNSIGNED_BYTE);
	rayCastingSolutionRenderTarget = Helpers::Gl::CreateRenderTarget(windowSize.x * 2 * pow(2,lodDelta)+ windowSize.x, windowSize.y * 2*pow(2,lodDelta)+windowSize.y, GL_RGBA32F, GL_RGBA, GL_FLOAT);
	homRenderTarget = Helpers::Gl::CreateRenderTarget(homHighestResolution.x, homHighestResolution.y, GL_RGBA32F, GL_RGBA, GL_FLOAT);
	//initizlize ndfexplorer
	ndfExplorer=Helpers::Gl::CreateRenderTarget(512,512, GL_RGBA32F, GL_RGBA, GL_FLOAT);
	disk d;
	d.radius = 0.2f;
	d.color = glm::vec3(.7, .7, .7);// 1, 1, 179 / 255.f);
	disks.push_back(d);

	//add slice colors
	sliceColors.push_back(std::make_pair(glm::vec3(141, 211, 199)*1.0f / 255.0f, -1));
	sliceColors.push_back(std::make_pair(glm::vec3(190, 186, 218)*1.0f / 255.0f, -1));
	sliceColors.push_back(std::make_pair(glm::vec3(251, 128, 114)*1.0f / 255.0f, -1));
	sliceColors.push_back(std::make_pair(glm::vec3(128, 177, 211)*1.0f / 255.0f, -1));
	sliceColors.push_back(std::make_pair(glm::vec3(253, 180, 98) *1.0f / 255.0f, -1));
	sliceColors.push_back(std::make_pair(glm::vec3(179, 222, 105)*1.0f / 255.0f, -1));
	sliceColors.push_back(std::make_pair(glm::vec3(252, 205, 229)*1.0f / 255.0f, -1));
	sliceColors.push_back(std::make_pair(glm::vec3(188, 128, 189)*1.0f / 255.0f, -1));
	sliceColors.push_back(std::make_pair(glm::vec3(204, 235, 197)*1.0f / 255.0f, -1));
	sliceColors.push_back(std::make_pair(glm::vec3(255, 237, 111)*1.0f / 255.0f, -1));

	//create texture to be used for ndfexplorer
	//glGenTextures(1, &ndfExplorerTexture);
	//glBindTexture(GL_TEXTURE_2D, ndfExplorerTexture);
	//glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, ndfExplorer.Width, ndfExplorer.Height, 0, GL_RGBA, GL_FLOAT, nullptr);
	//glBindTexture(GL_TEXTURE_2D, 0);

	//MyRenderTarget = Helpers::Gl::CreateRenderTarget(spatialResolution.x * multiSamplingRate, spatialResolution.y * multiSamplingRate, GL_RGBA16, GL_RGBA, GL_UNSIGNED_SHORT);
#endif // READ_FROM_FILE

	//glutReshapeWindow(1920, 1080);
	//glutFullScreen();

	//mohamed's code
	ndfTree.InitializeStorage(1, histogramResolution, viewResolution, glm::vec2(phys_tex_dim, phys_tex_dim));
	//end mohamed's code

	//patrick's code
	//ndfTree.InitializeStorage(1, histogramResolution, viewResolution, spatialResolution);
	//end patrick's code

	// TODO: put in extra folder
	std::string ndfFilePath = std::string("ndf") + std::to_string(spatialResolution.x);

	const auto histogramString = "_h" + std::to_string(histogramResolution.x) + "_" + std::to_string(histogramResolution.y);
	const auto viewString = "_v" + std::to_string(viewBatches.x) + "_" + std::to_string(viewBatches.y);
	const auto spatialString = "_s" + std::to_string(spatialResolution.x) + "_" + std::to_string(spatialResolution.y);

	// write batch file
	{
		const auto batchListFileName = ndfFilePath + std::string("ndf") + spatialString + histogramString + viewString;

		Helpers::IO::FileGuard<std::ofstream> fileWriter(batchListFileName + ".batchlist", std::ofstream::binary | std::ofstream::out);

		fileWriter.file.write(reinterpret_cast<const char*>(&viewBatches.x), sizeof(viewBatches.x));
		fileWriter.file.write(reinterpret_cast<const char*>(&viewBatches.y), sizeof(viewBatches.y));
	}

	const auto totalViewResolution = glm::ivec2(viewResolution.x*viewBatches.x, viewResolution.y*viewBatches.y);
	auto viewScale = glm::vec2(
		1.0f / static_cast<float>((totalViewResolution.x) - 1),
		1.0f / static_cast<float>((totalViewResolution.y) - 1));

	for (auto viewBatchY = 0; viewBatchY < viewBatches.y; ++viewBatchY) {
		for (auto viewBatchX = 0; viewBatchX < viewBatches.x; ++viewBatchX) {
			std::cout << "Batch " << viewBatchY << ", " << viewBatchX << std::endl;

			std::string batchString = "_batch" + std::to_string(viewBatchY) + "," + std::to_string(viewBatchX);

			// TODO: compute view directions of batch
			// store view directions uniformly distributed in a way the shader can look up
			// NOTE: naive representation has invalid directions outside unit circle and covers only one hemisphere
			// TODO: generate directions based on geodesic sphere
			for (auto &level : ndfTree.GetLevels()) {
				auto &meta = level.GetMetaData();

				for (auto viewMetaY = 0; viewMetaY < viewResolution.y; ++viewMetaY) {
					for (auto viewMetaX = 0; viewMetaX < viewResolution.x; ++viewMetaX) {
						auto &viewMeta = meta[viewMetaY * viewResolution.x + viewMetaX];
						viewMeta.ViewSlicePosition = glm::ivec2(viewBatchX * viewResolution.x + viewMetaX, viewBatchY * viewResolution.y + viewMetaY);



						// angles
						static const auto cameraDistance = 1.0f;
						auto cameraRotation = angleRange * glm::vec2(
							static_cast<float>(viewMeta.ViewSlicePosition.x) / static_cast<float>(totalViewResolution.x),
							static_cast<float>(viewMeta.ViewSlicePosition.y) / static_cast<float>(totalViewResolution.y));
						//auto camPosi = cameraDistance * glm::normalize(glm::vec3(std::cos(toRadiants * cameraRotation.x), std::cos(toRadiants * cameraRotation.y), std::sin(toRadiants * cameraRotation.x)));
						auto camPosi = CameraPosition(cameraRotation, cameraDistance);
						camPosi.x = -camPosi.x;

						viewMeta.ViewDirection = -camPosi;


					}
				}
			}

			/*particleSampler->ComputeFromParticles(particleCenters, particleRadius, *samplingShader.get(), samplingRate, gpuReduction, *reductionShader);

			const auto filename = ndfFilePath + std::string("ndf") + spatialString + histogramString + viewString + batchString;

			std::cout << "Writing to file " << filename << std::endl;
			ndfTree.WriteToFile(filename + ".ndf", false);
			ndfTree.WriteToFile(filename + ".ndf.zlib", true);
			std::cout << "Writing to file finished" << filename << std::endl;*/
		}
	}


	// approximate neigbouring views
	// 5 neighbour at a time with overlap
	//viewResolution.x 
	// TODO: rather approximate in toroidal space -> view coordinate repeats so highest view index is a neighbour to the lowest
	/*static const auto viewApproximationBatches = 4;
	static const auto viewMultiplication = 2;
	const auto totalApproximations = static_cast<int>(std::ceil(static_cast<float>(viewResolution.x) / static_cast<float>(viewApproximationBatches)));
	for(auto viewIndex = 0; viewIndex < totalApproximations; ++viewIndex) {
	auto viewStartIndex = viewIndex * viewApproximationBatches;
	auto viewEndIndex = viewStartIndex + viewApproximationBatches;
	std::cout << "EM approximating views " << viewStartIndex << " to " << viewEndIndex << std::endl;

	std::array<int, 3> approximatedDataSizes = { 100, 100, 100 };
	std::vector<float> approximatedDataCopy(std::accumulate(approximatedDataSizes.begin(), approximatedDataSizes.end(), 0), 0.0f);
	const int initialClusterCount = 16;

	// TODO: toroidal approximation
	Helpers::Approximation::ExpectationMaximization<3> em(approximatedDataCopy, approximatedDataSizes, initialClusterCount);

	// TODO: splat to more views than were actually calculated
	for(auto splattingViewIndex = viewStartIndex * viewMultiplication; splattingViewIndex < viewEndIndex * viewMultiplication; ++splattingViewIndex) {

	}
	}*/
	{
		


		//read square samples
		{
			std::ifstream infile;
			infile.open("samples1024x1024.txt");

			infile >> square_pattern_sample_count;
			samples_data = new float[square_pattern_sample_count];
			for (int i = 0; i < square_pattern_sample_count; i++)
			{
				infile >> samples_data[i];
				samples_data[i]--;
			}
			infile.close();

			//initialize sample count to square sample count
			sample_count = square_pattern_sample_count;
		}
#ifdef CORE_FUNCTIONALITY_ONLY
#else
		{
			//for circular pattern
			//we want to include all sample positions so we compute the stdv such the data with higher weight account for most of the data.
			//so we compute the standard deviation as 3*stdv=sqrt(dim), knowing that 3*stdv means the 99% of data have high (whatever that means) wight.
			gkStdv = filterRadius/3.0f;
			createFilter(filterRadius, gkStdv);
		}
#endif

		//read square samples 
		{
			//std::ifstream infile;
			//infile.open("circularPattern128x128.txt");
			//infile >> circularPatternRadius;
			//infile >> circular_pattern_dim;
			//infile >> circular_pattern_sample_count;
			//CircularSamples_data = new float[circular_pattern_sample_count];
			//for (int i = 0; i < circular_pattern_sample_count; i++)
			//{
			//	infile >> CircularSamples_data[i];
			//	CircularSamples_data[i]--;
			//}
			//infile.close();

			////read weights
			//infile.open("circularPatternWeights128x128.txt");
			//CircularSamples_data_weights = new float[circular_pattern_sample_count];
			//for (int i = 0; i < circular_pattern_sample_count; i++)
			//{
			//	infile >> CircularSamples_data_weights[i];
			//}
			//infile.close();
		}
		
	}

	//ssbo for sample count
	{
		glGenBuffers(1, &SampleCount_ssbo);

		const int MY_ARRAY_SIZE = (phys_tex_dim)*(phys_tex_dim);
		std::vector<float> data(MY_ARRAY_SIZE,0);

		// Allocate storage for the UBO
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, SampleCount_ssbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float)* MY_ARRAY_SIZE, data.data(), GL_DYNAMIC_COPY);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


		//glBindBuffer(GL_SHADER_STORAGE_BUFFER, SampleCount_ssbo);
		//GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);  
		//memcpy(p, &data, sizeof(data));
		//glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, SampleCount_ssbo);
		GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
		memcpy(p, reinterpret_cast<char*>(&data[0]), data.size() * sizeof(*data.begin()));
		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

	}

	//ssbo for NDF colors
	{
		glGenBuffers(1, &ndfColors_ssbo);

		const int MY_ARRAY_SIZE = (phys_tex_dim)*(phys_tex_dim)*3;
		std::vector<float> data(MY_ARRAY_SIZE, 1);

		// Allocate storage for the UBO
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ndfColors_ssbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float)* MY_ARRAY_SIZE, data.data(), GL_DYNAMIC_COPY);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ndfColors_ssbo);
		GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
		memcpy(p, reinterpret_cast<char*>(&data[0]), data.size() * sizeof(*data.begin()));
		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

	}

	////ssbo for colored regions
	//{
	//	glGenBuffers(1, &region_ssbo);

	//	const int MY_ARRAY_SIZE = 3*phys_tex_dim*phys_tex_dim;
	//	float* data = new float[MY_ARRAY_SIZE];

	//	for (int i = 0; i < MY_ARRAY_SIZE / 3; i++)
	//	{
	//		data[i * 3    ] = 1;// rand() % 2;
	//		data[i * 3 + 1] = 1;
	//		data[i * 3 + 2] = 1;
	//	}

	//	// Allocate storage for the UBO
	//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, region_ssbo);
	//	glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof (GLfloat)* MY_ARRAY_SIZE, data, GL_DYNAMIC_COPY);
	//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


	//	glBindBuffer(GL_SHADER_STORAGE_BUFFER, region_ssbo);
	//	GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
	//	memcpy(p, &data, sizeof(data));
	//	glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

	//	{
	//		unsigned int block_index = glGetProgramResourceIndex(renderingShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "regionColor");
	//		GLuint binding_point_index = 6;
	//		glShaderStorageBlockBinding(renderingShader->GetGlShaderProgramHandle(), block_index, binding_point_index);
	//		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, region_ssbo);
	//	}

	//	{
	//		unsigned int block_index = glGetProgramResourceIndex(tileResetShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "regionColor");
	//		GLuint binding_point_index = 3;
	//		glShaderStorageBlockBinding(tileResetShader->GetGlShaderProgramHandle(), block_index, binding_point_index);
	//		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, region_ssbo);
	//	}

	//	delete[]data;

	//}
#ifdef CORE_FUNCTIONALITY_ONLY
	{
		std::ifstream infile;
		infile.open("colorcube1000.txt");

		infile >> colorMapSize;
		colorMapData = new float[colorMapSize * 3];

		//this is wired, i don't know why it's read

		for (int i = 0; i < colorMapSize * 3; i++)
		{
			infile >> colorMapData[i];
		}
		infile.close();
	}

	//ssbo for color maps
	{
		glGenBuffers(1, &colorMap_ssbo);

		const int MY_ARRAY_SIZE = 3 * colorMapSize;
		float* data = new float[MY_ARRAY_SIZE];

		for (int i = 0; i < MY_ARRAY_SIZE; i++)
		{
			data[i] = colorMapData[i];
		}

		// Allocate storage for the UBO
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, colorMap_ssbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof (GLfloat)* MY_ARRAY_SIZE, data, GL_DYNAMIC_COPY);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


		glBindBuffer(GL_SHADER_STORAGE_BUFFER, colorMap_ssbo);
		GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
		memcpy(p, &data, sizeof(data));
		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

		delete[]data;

	}
#else
	

	//ssbo for selected pixels
	{
		glGenBuffers(1, &selectedPixels_ssbo);

		const int MY_ARRAY_SIZE = 2 * MaxAllowedSelectedPixels;
		float* data = new float[MY_ARRAY_SIZE];

		// Allocate storage for the UBO
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, selectedPixels_ssbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof (GLfloat)* MY_ARRAY_SIZE, data, GL_DYNAMIC_COPY);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


		glBindBuffer(GL_SHADER_STORAGE_BUFFER, selectedPixels_ssbo);
		GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
		memcpy(p, &data, sizeof(data));
		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

		delete[]data;

	}


	//ssbo for min and max similarity metric
	{
		glGenBuffers(1, &simLimitsF_ssbo);

		const int MY_ARRAY_SIZE = 2 + windowSize.x*windowSize.y;
		float* data = new float[MY_ARRAY_SIZE];

		data[0] = 1000000; //min
		data[1] = 0;  //max

		for (int i = 0; i < windowSize.x*windowSize.y; i++)
			data[i + 2] = -1;

		// Allocate storage for the UBO
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, simLimitsF_ssbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof (float)* MY_ARRAY_SIZE, data, GL_DYNAMIC_COPY);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


		glBindBuffer(GL_SHADER_STORAGE_BUFFER, simLimitsF_ssbo);
		GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
		memcpy(p, &data, sizeof(data));
		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

		


		delete[]data;

	}

	//ssbo for average ndf 
	{
		glGenBuffers(1, &avgNDF_ssbo);

		const int MY_ARRAY_SIZE = histogramResolution.x*histogramResolution.y;
		float* data = new float[MY_ARRAY_SIZE];

		for (int i = 0; i < MY_ARRAY_SIZE; i++)
		{
			data[i] = 0;
		}

		// Allocate storage for the UBO
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, avgNDF_ssbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof (GLfloat)* MY_ARRAY_SIZE, data, GL_DYNAMIC_COPY);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


		glBindBuffer(GL_SHADER_STORAGE_BUFFER, avgNDF_ssbo);
		GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
		memcpy(p, &data, sizeof(data));
		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

		


		delete[]data;

	}
#endif
	//ssbo for pre-integrated bins
	{
		glGenBuffers(1, &preIntegratedBins_ssbo);

		const int MY_ARRAY_SIZE = (histogramResolution.x)*(histogramResolution.y * 3);
		float* data = new float[MY_ARRAY_SIZE];

		for (int i = 0; i < MY_ARRAY_SIZE; i++)
			data[i] = 0;

		// Allocate storage for the UBO
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, preIntegratedBins_ssbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof (GLfloat)* MY_ARRAY_SIZE, data, GL_DYNAMIC_COPY);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


		glBindBuffer(GL_SHADER_STORAGE_BUFFER, preIntegratedBins_ssbo);
		GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
		memcpy(p, &data, sizeof(data));
		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);


		

		delete[]data;

	}

	//ssbo for super sampled pre-integrated bins
	{
		glGenBuffers(1, &superPreIntegratedBins_ssbo);

		const int MY_ARRAY_SIZE = maxSubBins*maxSubBins*histogramResolution.x*histogramResolution.y * 3;
		float* data = new float[MY_ARRAY_SIZE];

		for (int i = 0; i < MY_ARRAY_SIZE; i++)
			data[i] = 0;

		// Allocate storage for the UBO
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, superPreIntegratedBins_ssbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof (GLfloat)* MY_ARRAY_SIZE, data, GL_DYNAMIC_COPY);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


		glBindBuffer(GL_SHADER_STORAGE_BUFFER, superPreIntegratedBins_ssbo);
		GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
		memcpy(p, &data, sizeof(data));
		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

		delete[]data;

	}

	//ssbo for bin areas
	{
		glGenBuffers(1, &binAreas_ssbo);

		const int MY_ARRAY_SIZE = (histogramResolution.x)*(histogramResolution.y);
		double* data = new double[MY_ARRAY_SIZE];

		for (int i = 0; i < MY_ARRAY_SIZE; i++)
			data[i] = 0;

		// Allocate storage for the UBO
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, binAreas_ssbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof (GLdouble)* MY_ARRAY_SIZE, data, GL_DYNAMIC_COPY);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


		glBindBuffer(GL_SHADER_STORAGE_BUFFER, binAreas_ssbo);
		GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
		memcpy(p, &data, sizeof(data));
		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);





		delete[]data;

	}

	//ssbo for bin areas of simple binning technique
	{
		glGenBuffers(1, &simpleBinningAreas_ssbo);

		int MY_ARRAY_SIZE = 0;

		for (int i = maxSubBins; i >0; i = i / 2)
		{
			MY_ARRAY_SIZE += 3 * histogramResolution.x*histogramResolution.y*i*i;
		}
		double* data = new double[MY_ARRAY_SIZE];

		for (int i = 0; i < MY_ARRAY_SIZE; i++)
			data[i] = 0;

		// Allocate storage for the UBO
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, simpleBinningAreas_ssbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof (GLdouble)* MY_ARRAY_SIZE, data, GL_DYNAMIC_COPY);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


		glBindBuffer(GL_SHADER_STORAGE_BUFFER, simpleBinningAreas_ssbo);
		GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
		memcpy(p, &data, sizeof(data));
		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);


		

		delete[]data;

	}

#ifdef CORE_FUNCTIONALITY_ONLY
#else
	//ssbo to draw ndf image to
	{
		glGenBuffers(1, &NDFImage_ssbo);

		const int MY_ARRAY_SIZE = phys_tex_dim*phys_tex_dim*histogramResolution.x * histogramResolution.y;
		float* data = new float[MY_ARRAY_SIZE];

		for (int i = 0; i < MY_ARRAY_SIZE; i++)
			data[i] = 0;

		// Allocate storage for the UBO
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, NDFImage_ssbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof (GLfloat)* MY_ARRAY_SIZE, data, GL_DYNAMIC_COPY);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


		glBindBuffer(GL_SHADER_STORAGE_BUFFER, NDFImage_ssbo);
		GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
		memcpy(p, &data, sizeof(data));
		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);



		delete[]data;
	}
#endif

	//ssbo for progessive sampling done for raycasting rendering
	{
		glGenBuffers(1, &progressive_raycasting_ssbo);

		const int MY_ARRAY_SIZE = windowSize.x*windowSize.y * 3;
		float* screen_data = new float[MY_ARRAY_SIZE];

		for (int i = 0; i < MY_ARRAY_SIZE; i++)
			screen_data[i] = 0;

		// Allocate storage for the UBO
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, progressive_raycasting_ssbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof (GLfloat)* MY_ARRAY_SIZE, screen_data, GL_DYNAMIC_COPY);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


		glBindBuffer(GL_SHADER_STORAGE_BUFFER, progressive_raycasting_ssbo);
		GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
		memcpy(p, &screen_data, sizeof(screen_data));
		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

		
	}

	// TODO: downsample data - or directly during pre computation?
	//ndfTree.DownsampleData();
	ndfTree.UploadData();

	//renderer.BindSsbo(ndfTree, *renderingShader);
	//renderer.BindSsbo(ndfTree, *reductionShader);
	//renderer.BindSsbo(ndfTree, *reductionShader_C);
	//renderer.BindSsbo(ndfTree, *tiledRaycastingShader);
	//renderer.BindSsbo(ndfTree, *tiledRaycastingShader_C);

	//new
	bindSSbos();
	

	{
		//patrick's code
		//auto ssboSize = spatialResolution.x * spatialResolution.y * histogramResolution.x * histogramResolution.y * sizeof(float);
		// end paticks' code

		//mohamed's code
		auto ssboSize = phys_tex_dim * phys_tex_dim * histogramResolution.x * histogramResolution.y * sizeof(float);
		//patrick's code

		assert(glShaderStorageBlockBinding);
		assert(glGetProgramResourceIndex);

		GLuint ssboBindingPointIndex = 0;
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, ndfTree.GetLevels().front().GetShaderStorageBufferOject());
		//{
		//	auto blockIndex = glGetProgramResourceIndex(downsamplingShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "NdfVoxelData");
		//	glShaderStorageBlockBinding(downsamplingShader->GetGlShaderProgramHandle(), blockIndex, ssboBindingPointIndex);
		//}

		//{
		//	auto blockIndex = glGetProgramResourceIndex(convolutionShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "NdfVoxelData");
		//	glShaderStorageBlockBinding(eNtfRenderingShader->GetGlShaderProgramHandle(), blockIndex, ssboBindingPointIndex);
		//}

#ifdef DOWNSAMPLE_SSBO
		ssboBindingPointIndex = 1;
		glGenBuffers(1, &downsampledSsbo);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, downsampledSsbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, ssboSize, nullptr, GL_DYNAMIC_COPY);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, downsampledSsbo);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

		//{
		//	auto blockIndex = glGetProgramResourceIndex(downsamplingShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "NdfVoxelDataDownsampled");
		//	glShaderStorageBlockBinding(downsamplingShader->GetGlShaderProgramHandle(), blockIndex, ssboBindingPointIndex);
		//}
#endif

		/*{
		auto blockIndex = glGetProgramResourceIndex(renderingShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "NdfVoxelDataDownsampled");
		glShaderStorageBlockBinding(renderingShader->GetGlShaderProgramHandle(), blockIndex, ssboBindingPointIndex);
		}*/

		{



		}

		auto glErr = glGetError();
		if (glErr != GL_NO_ERROR) {
			std::cout << "glError: " << __FILE__ << " " << __LINE__ << std::endl;
		}
	}

	//	BrdfIntegrator<> integrator;

	//	SparseApproximator<> approximator;


	//if(!init())
	//	return 1;

    Helpers::Gl::MakeBuffer(Helpers::Gl::CreateBoxVertexPositions(1.0f, { 0.0f, 0.0f, 0.0f }), boxHandle);
    std::vector<glm::vec3> theQuad;
    theQuad.push_back(glm::vec3(-1.0f, -1.0f, 0.0f));
    theQuad.push_back(glm::vec3(1.0f, -1.0f, 0.0f));
    theQuad.push_back(glm::vec3(1.0f, 1.0f, 0.0f));
    theQuad.push_back(glm::vec3(1.0f, 1.0f, 0.0f));
    theQuad.push_back(glm::vec3(-1.0f, 1.0f, 0.0f));
    theQuad.push_back(glm::vec3(-1.0f, -1.0f, 0.0f));
    Helpers::Gl::MakeBuffer(theQuad, quadHandle);

#ifdef HEADLESS
	// does not call the display function anymore
	//glutHideWindow();
#endif

#ifdef READ_FROM_FILE
#if 0 // prefetching
	for (int i = timeStepLowerLimit; i < timeStepUpperLimit; ++i) {
		sparseENtfFilePath = fileFolder + fileSparseSubFolder + filePrefix + std::to_string(i) + fileSuffix + ".entf";

		loadSparseENtf(sparseENtfFilePath);
	}
#else
	sparseENtfFilePath = fileFolder + fileSparseSubFolder + filePrefix + std::to_string(timeStep) + fileSuffix + ".entf";
	loadSparseENtf(sparseENtfFilePath);
#endif // prefetch
#endif // READ_FROM_FILE

	std::cout << "32 bit float epsilon " << std::numeric_limits<float>::epsilon() << std::endl;

	//initialize tiles (it needs to be called twice for now to get the correct cam position and initialization of fbo), must be changed later

	//display();


	//glGenFramebuffers(1, &PhysicalFbo);
	//glGenFramebuffers(1, &glfinalfbo);
	//glGenFramebuffers(1, &TiledFbo);
	//
	//glGenRenderbuffers(1, &PhysicalFbo_depthbuffer);
	//glGenRenderbuffers(1, &glfinalfbo_depthbuffer);
	//glGenRenderbuffers(1, &TiledFbo_DepthBuffer);
	//
	//glBindFramebuffer(GL_FRAMEBUFFER, PhysicalFbo);

	//glBindRenderbuffer(GL_RENDERBUFFER, PhysicalFbo_depthbuffer);
	//glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, res,res);
	//glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, PhysicalFbo_depthbuffer);

	//glBindFramebuffer(GL_FRAMEBUFFER, glfinalfbo);

	//glBindRenderbuffer(GL_RENDERBUFFER, glfinalfbo_depthbuffer);
	//glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, windowSize.x, windowSize.y);
	//glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, glfinalfbo_depthbuffer);


	//glBindFramebuffer(GL_FRAMEBUFFER, TiledFbo);

	//glBindRenderbuffer(GL_RENDERBUFFER, TiledFbo_DepthBuffer);
	//glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, windowSize.x, windowSize.y);
	//glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, TiledFbo_DepthBuffer);

	//debug
	//{
	//	particleScale = 4.2;
	//}
	//end debug


	tile_based_culling(false);
	update_page_texture();
	reset();

	//set up bin areas

	A.push_back(std::vector<double>());
	A.push_back(std::vector<double>());
	A.push_back(std::vector<double>());

	if (histogramResolution.x == 8 && histogramResolution.y == 8)
	{
		//read areas and projected normal binning areas
		std::ifstream infile;
		infile.open("areasFile.txt");
		int size, size2;

		for (int i = 0; i < 3; i++)
		{
			infile >> size;
			A[i].resize(size, 0.0f);
			for (int j = 0; j < size; j++)
			{
				infile >> A[i][j];
			}
		}

		//upload areas to gpu
		for (int i = 0; i < 3; i++)
		{
			binning_mode = i;
			auto Size = size_t(histogramResolution.x * histogramResolution.y);
			auto ssbo = binAreas_ssbo;
			{
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
				glBufferData(GL_SHADER_STORAGE_BUFFER, Size* sizeof(double), A[binning_mode].data(), GL_DYNAMIC_COPY);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
			}
		}
		binning_mode = 0;

		//upload projected normal binning areas to gpu
		infile >> size;
		projectedNormalBinningAreas.resize(size);
		for (int i = 0; i < size; i++)
		{
			infile >> size2;
			projectedNormalBinningAreas[i].resize(size2);
			for (int j = 0; j < size2; j++)
			{
				infile >> projectedNormalBinningAreas[i][j].x;
				infile >> projectedNormalBinningAreas[i][j].y;
				infile >> projectedNormalBinningAreas[i][j].z;
			}
		}
		infile.close();

		//now we copy the areas to gpu
		{
			std::vector<double> data;
			size_t Size = 0;

			for (int j = 0; j < projectedNormalBinningAreas.size(); j++)
			{
				for (int i = 0; i < projectedNormalBinningAreas[j].size(); i++)
				{
					data.push_back(projectedNormalBinningAreas[j][i].x);
					data.push_back(projectedNormalBinningAreas[j][i].y);
					data.push_back(projectedNormalBinningAreas[j][i].z);
				}
				Size += size_t(projectedNormalBinningAreas[j].size() * 3);
			}

			auto ssbo = simpleBinningAreas_ssbo;
			{
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
				glBufferData(GL_SHADER_STORAGE_BUFFER, Size* sizeof(double), data.data(), GL_DYNAMIC_COPY);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
			}
		}

		std::cout << "Finished reading areas" << std::endl;

	}
	else
	{
		//compute areas
		for (int i = 0; i < 3; i++)
		{
			binning_mode = i;
			computeBinAreas();
		}
		binning_mode = 0;

#if 0	//write bin areas to a file
		std::ofstream areasFile;
		areasFile.open("areasFile.txt");
		for (int i = 0; i < 3; i++)
		{
			areasFile << A[i].size() << std::endl;
			for (int j = 0; j < A[i].size(); j++)
			{
				areasFile << A[i][j] << std::endl;
			}
		}

		//now we add the projectednormalsareas
		areasFile << projectedNormalBinningAreas.size() << std::endl;
		for (int i = 0; i < projectedNormalBinningAreas.size(); i++)
		{
			areasFile << projectedNormalBinningAreas[i].size() << std::endl;
			for (int j = 0; j < projectedNormalBinningAreas[i].size(); j++)
			{
				areasFile << projectedNormalBinningAreas[i][j].x << std::endl;
				areasFile << projectedNormalBinningAreas[i][j].y << std::endl;
				areasFile << projectedNormalBinningAreas[i][j].z << std::endl;
			}
		}

		areasFile.close();
		std::cout << "finished saving areas" << std::endl;
#endif	
	}
	preIntegrateBins();

	//build HOM
	buildHOMFlag = true;

	//debug
	//float r =  (.f / INITIAL_WINDOW_WIDTH);
	//particleCenters[0].w = r;
	//std::cout << "scaled to " << r << std::endl;

	//move particle 0 to the top left corner
	int w, h;
	LOD.get_lod_width_and_hight(LOD.initial_lod, w, h);
	glm::vec2 s, e;
	std::cout << "the initial lod is " << LOD.initial_lod << std::endl;
	//get equivalent of half a pixel in object space
	s = LOD.pixel2obj(glm::vec2(0, 0), current_lod);

	//add that to the model min to make sure we are in the bounding box
	//s += glm::vec2(origModelMin.x,origModelMin.y);

	//get the equivalent of one pixel in obje ct space
	e = LOD.pixel2obj(glm::vec2(1, 1), current_lod);

	//add that to the model min to make sure we are in the bounding box
	//e += glm::vec2(origModelMin.x, origModelMin.y);

	//get radius
	glm::vec2 v = e - s;


	//glm::vec2 p = glm::vec2(0.5*modelExtent.x / w, 0.5*modelExtent.y / h);
	//r = std::min(p.x,p.y);

	//to get hits per pixel
#if 0
	double r;
	glm::ivec2 blc, trc;
	glm::dvec2 blcObj, trcObj;
	std::vector<glm::ivec2> P;
	int lw, lh;
	LOD.get_lod_width_and_hight(current_lod, lw, lh);

	std::vector<int> hitCount(lw*lh, 0);
	glm::dvec2 d;
	glm::dvec2 pixelInObj;
	glm::dvec2 c1, c2, c3, c4;
	glm::dvec2 PC;
	double l;

	//get pixel width in obj
	double pixelDim;
	{
		glm::vec2 s, e;
		//debug
		s = LOD.pixel2obj(glm::vec2(.5*lw, .5*lh), current_lod);
		e = LOD.pixel2obj(glm::vec2(-.5*lw, -.5*lh), current_lod);
		//end debug
		s = LOD.pixel2obj(glm::vec2(0, 0), current_lod);
		e = LOD.pixel2obj(glm::vec2(1, 1), current_lod);
		glm::vec2 v = e - s;
		pixelDim= 0.5*std::min(v.x, v.y);
	}

	

	for (int i = 0; i < particleCenters.size(); i++)
	{
		P.clear();
		r = particleCenters[i].w;
		PC = glm::vec2(particleCenters[i]);



		if (i%1000000 == 0)
		{
			std::cout << "finished " << i << " million particles" << std::endl;
		}

		//to get just the glyph's blc and trc, we dont need to multiply the sqrt by 3, but we do so we have some padding, ie get a bigger neighbourhood just
		//so we don't miss anything due to inaccuracies in converting from object to pixel space.
		//in the intetesction test, pixels that don't intersect will be removed
		blcObj = glm::vec2(PC) - glm::vec2(sqrt(2 * r*r), sqrt(2 * r*r));
		trcObj = glm::vec2(PC) + glm::vec2(sqrt(2 * r*r), sqrt(2 * r*r));
		
		blc=LOD.obj2pixel(blcObj,current_lod);
		trc=LOD.obj2pixel(trcObj,current_lod);

		////do some padding on blc and trc just in case there is a floating point error.
		trc += glm::ivec2(1, 1);
		blc -= glm::ivec2(1, 1);

		for (int j = blc.x; j <= trc.x; j++)
		{
			for (int k = blc.y; k <= trc.y; k++)
			{
				P.push_back(glm::ivec2(j,k));
			}
		}

		for (int j = 0; j < P.size(); j++)
		{
			//test if pixel P[j] intersects PC
			//get bottom left corner of pixel in obj
			pixelInObj = LOD.pixel2obj(P[j], current_lod);

			//add dim to it, to get center of pixel in obj space
			//pixelInObj += glm::vec2(pixelDim, pixelDim);

			//old
			//d = pixelInObj - glm::vec2(PC);
			//l = glm::length(d);
			//if (l <= (PC.w + sqrt(2.0f*pow(pixelDim,2))))
			//{
			//	//P[j] given with respect to the origin being in the center of the lod, so we need to shift that
			//	P[j] += glm::ivec2(.5*lw, .5*lh);
			//	hitCount[P[j].y*lw + P[j].x]++;
			//}
			//end old

			//new
			double DeltaX = PC.x - std::max(pixelInObj.x, std::min(PC.x, pixelInObj.x + 2.0*pixelDim));
			double DeltaY = PC.y - std::max(pixelInObj.y, std::min(PC.y, pixelInObj.y + 2.0*pixelDim));
			if ((DeltaX * DeltaX + DeltaY * DeltaY) <= (r*r))
			{
				//	//P[j] given with respect to the origin being in the center of the lod, so we need to shift that
				P[j] += glm::ivec2(.5*lw, .5*lh);
				hitCount[P[j].y*lw + P[j].x]++;
			}
			//end new

			//newer
			//get four corners of pixel
			//glm::vec2 nearestPoint;
			//nearestPoint.x = std::max(pixelInObj())


			//end newer
		}
	}

	//print hits
	out.open("occupancyStats.txt");
	out << lw << " " << lh << std::endl;
	for (int i = 0; i < hitCount.size(); i++)
	{
		out << i << " " << hitCount[i] << std::endl;
	}
	out.close();

#endif

	//one sphere in one pixel
#if 0
	float r = 0.5*std::min(v.x, v.y);
	particleCenters[0].w = 0.999f*r;
	particleCenters[0].x = s.x + .5*v.x;
	particleCenters[0].y = s.y + .5*v.y;
	Helpers::Gl::CreateGlMeshFromBuffers(particleCenters, particlesGlBuffer);
#endif
	//two non-intersecting spheres in one pixel
#if 0
	float r = 0.25*std::min(v.x, v.y);

	particleCenters[0].w = 0.999f*r;
	particleCenters[0].x = s.x + .25*v.x;
	particleCenters[0].y = s.y + .25*v.y;

	particleCenters[1].w = 0.999f*r;
	particleCenters[1].x = s.x + .75*v.x;
	particleCenters[1].y = s.y + .75*v.y;
	Helpers::Gl::CreateGlMeshFromBuffers(particleCenters, particlesGlBuffer);
#endif
	//two intersecting spheres in one pixel
#if 0
	float r = 0.3*std::min(v.x, v.y);

	particleCenters[0].w = 0.999f*r;
	particleCenters[0].x = s.x + .4*v.x;
	particleCenters[0].y = s.y + .4*v.y;

	particleCenters[1].w = 0.999f*r;
	particleCenters[1].x = s.x + .6*v.x;
	particleCenters[1].y = s.y + .6*v.y;
	Helpers::Gl::CreateGlMeshFromBuffers(particleCenters, particlesGlBuffer);
#endif
#if 0
	float r = 0.3*std::min(v.x, v.y);

	particleCenters[0].w = .5;
	particleCenters[0].x = s.x;
	particleCenters[0].y = s.y;

	particleCenters[1].w = .4;
	particleCenters[1].x = s.x + 11 * v.x;
	particleCenters[1].y = s.y + 11 * v.y;
	Helpers::Gl::CreateGlMeshFromBuffers(particleCenters, particlesGlBuffer);
#endif 
	//end debug


#if 0
	LightDir = glm::vec3(-.298438,-.325,-.897391);
	keyboard('8',0,0);
	keyboard('m',0,0);
	for (int i = 0; i < 66; i++)
		keyboard('d', 0, 0);
	for (int i = 0; i < 6; i++)
		keyboard('+', 0, 0);
	{
		cameraDistance= 39.962;

		prev_lod = current_lod;

		camPosi = CameraPosition(cameraRotation, cameraDistance);
		camPosi += cameraOffset;
		camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

		zoom_camPosi = camPosi;


		tile_based_culling(false);
		tile_based_panning(false);


		//debug
		//phys_x = phys_y = 0;
		//end debug

		update_page_texture();

		//if (int(prev_lod)- int(current_lod)!=0)
		reset();
		glutPostRedisplay();
	}

	////change tile size
	//for (int i = 0; i < 4; i++)
	//	keyboard('z', 0, 0);

	////call display 8 times
	//for (int i = 0; i < 8; i++)
	//	display();

	////stop sampling
	//keyboard('0', 0, 0);

#endif

#if 0
	LightDir = glm::vec3(-.715625,-.241667,-.655346);
	keyboard('8', 0, 0);
	keyboard('m', 0, 0);
	for (int i = 0; i < 13; i++)
		keyboard('d', 0, 0);
	for (int i = 0; i < 6; i++)
		keyboard('q', 0, 0);
	for (int i = 0; i < 6; i++)
		keyboard('-', 0, 0);
	
	keyboard('w', 0, 0);

	////change tile size
	//for (int i = 0; i < 4; i++)
	//	keyboard('z', 0, 0);

	////call display 8 times
	//for (int i = 0; i < 8; i++)
	//	display();

	////stop sampling
	//keyboard('0', 0, 0);

#endif

#if 0
	
	LightDir = glm::vec3(-.00625,-.147222,-.989084);
	LightDir = glm::vec3(.963083, -.269206, -0);
	keyboard('8', 0, 0);
	keyboard('m', 0, 0);
	for (int i = 0; i < 13; i++)
		keyboard('d', 0, 0);
	for (int i = 0; i < 6; i++)
		keyboard('q', 0, 0);
	for (int i = 0; i < 6; i++)
		keyboard('-', 0, 0);

	keyboard('w', 0, 0);

	particleScale = .703447;
	clear_NDF_Cache();
	//initialize();

	tile_based_culling(false);
	tile_based_panning(false);
	update_page_texture();
	reset();
	display();

	////change tile size
	//for (int i = 0; i < 4; i++)
	//	keyboard('z', 0, 0);

	////call display 8 times
	//for (int i = 0; i < 8; i++)
	//	display();

	////stop sampling
	//keyboard('0', 0, 0);

#endif

#if 0
	for (int i = 0; i < 2;i++)
		keyboard('d', 0, 0);
	for (int i = 0; i < 2; i++)
		keyboard('e', 0, 0);

	//for (int i = 0; i < 5;i++)
#endif

#if 0   //for submitsion video  //liquid
	keyboard('8', 0, 0);

	for (int i = 0; i < 4; i++)
		keyboard('d', 0, 0);
	for (int i = 0; i < 4; i++)
		keyboard('q', 0, 0);
	for (int i = 0; i < 5; i++)
		keyboard('d', 0, 0);
#endif

#if 0  //for submitsion video  //laser
	keyboard('8', 0, 0);

	for (int i = 0; i < 1; i++)
		keyboard('d', 0, 0);
	for (int i = 0; i < 4; i++)
		keyboard('e', 0, 0);
	for (int i = 0; i < 2; i++)
		keyboard('d', 0, 0);

#endif

#if 0
	for (int i = 0; i < 50; i++)
		keyboard('+', 0, 0);
#endif

#if 0   //for revision video, laser ablation data set
	GlobalRotationMatrix[0] = glm::vec3(-.881396,-.280731,-.37991);
	GlobalRotationMatrix[1] = glm::vec3(-.462514, .34936, .814879);
	GlobalRotationMatrix[2] = glm::vec3(-.0960355, .893945, -.437767);

	//LightDir = glm::vec3(-.4125, .494444, -.765094);
	//LightDir = glm::vec3(-.785937,-.588889,-.188447);
	LightDir = glm::vec3(.604688,.0277778,-.795978);
	preIntegrateBins();

	initialize();

	cameraDistance = std::max(LOD.min_cam_dist, std::min(cameraDistance, LOD.max_cam_dist - .3f));
	camPosi = CameraPosition(cameraRotation, cameraDistance);
	camPosi += cameraOffset;
	camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

	tile_based_culling(false);
	//tile_based_panning(false);
	update_page_texture();
	reset();

	display();
	//glutPostRedisplay();
	
#endif

#if 0   //for revision video, copper silver
	GlobalRotationMatrix[0] = glm::vec3(.945735, .163886, .280585);
	GlobalRotationMatrix[1] = glm::vec3(.0458234, .787601, -.614479);
	GlobalRotationMatrix[2] = glm::vec3(-.321693, .593992, .737352);

	LightDir = glm::vec3(-.1875,.688889,-.700197);
	//LightDir = glm::vec3(-.465625,.266667,-.84385);
	preIntegrateBins();

	initialize();

	cameraDistance = std::max(LOD.min_cam_dist, std::min(cameraDistance, LOD.max_cam_dist - .3f));
	camPosi = CameraPosition(cameraRotation, cameraDistance);
	camPosi += cameraOffset;
	camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

	tile_based_culling(false);
	//tile_based_panning(false);
	update_page_texture();
	reset();

	display();
#endif

#if 0 //preset orientation for new laser dataset
	GlobalRotationMatrix[0] = glm::vec3(.62135,.360113,-.695872);
	GlobalRotationMatrix[1] = glm::vec3(.0874426,.850706,.518318);
	GlobalRotationMatrix[2] = glm::vec3(.778636,-.382905,.497096);

	LightDir = glm::vec3(.628125,.255556,-.734949);
	//LightDir = glm::vec3(-.465625,.266667,-.84385);
	preIntegrateBins();

	initialize();

	cameraDistance = std::max(LOD.min_cam_dist, std::min(cameraDistance, LOD.max_cam_dist - .3f));
	camPosi = CameraPosition(cameraRotation, cameraDistance);
	camPosi += cameraOffset;
	camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

	tile_based_culling(false);
	//tile_based_panning(false);
	update_page_texture();
	reset();

	display();

	for (int i = 0; i < 20; i++)
		keyboard('-', 0, 0);

#endif
	//Helpers::Gl::MakeBuffer(particleCenters, global_tree, particlesGLBufferArray);         //bricking
	glutMainLoop();
    //unInitTweakBar();
	//uninit();


	_fgetchar();

	return 0;
}
void bindSSbos()
{	
	globalSsboBindingPointIndex = 1;
	//samplecount ssbos
	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, SampleCount_ssbo);
		{
			unsigned int block_index = glGetProgramResourceIndex(reductionShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "sampleCount");

			reductionShader->setSsboBindingIndex(SampleCount_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = reductionShader->getSsboBiningIndex(SampleCount_ssbo);

			glShaderStorageBlockBinding(reductionShader->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, SampleCount_ssbo);
		}
		{
			unsigned int block_index = glGetProgramResourceIndex(reductionShader_C->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "sampleCount");

			reductionShader_C->setSsboBindingIndex(SampleCount_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = reductionShader_C->getSsboBiningIndex(SampleCount_ssbo);

			glShaderStorageBlockBinding(reductionShader_C->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, SampleCount_ssbo);
		}
		{
			unsigned int block_index = glGetProgramResourceIndex(tiledRaycastingShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "sampleCount");

			tiledRaycastingShader->setSsboBindingIndex(SampleCount_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = tiledRaycastingShader->getSsboBiningIndex(SampleCount_ssbo);

			glShaderStorageBlockBinding(tiledRaycastingShader->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, SampleCount_ssbo);
		}
		{
			unsigned int block_index = glGetProgramResourceIndex(tiledRaycastingShader_C->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "sampleCount");

			tiledRaycastingShader_C->setSsboBindingIndex(SampleCount_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = tiledRaycastingShader_C->getSsboBiningIndex(SampleCount_ssbo);

			glShaderStorageBlockBinding(tiledRaycastingShader_C->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, SampleCount_ssbo);
		}
		{
			unsigned int block_index = glGetProgramResourceIndex(renderingShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "sampleCount");

			renderingShader->setSsboBindingIndex(SampleCount_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = renderingShader->getSsboBiningIndex(SampleCount_ssbo);

			glShaderStorageBlockBinding(renderingShader->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, SampleCount_ssbo);
		}

		{
			unsigned int block_index = glGetProgramResourceIndex(tileResetShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "sampleCount");

			tileResetShader->setSsboBindingIndex(SampleCount_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = tileResetShader->getSsboBiningIndex(SampleCount_ssbo);

			glShaderStorageBlockBinding(tileResetShader->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, SampleCount_ssbo);
		}

		{
			unsigned int block_index = glGetProgramResourceIndex(overlayDownsampledNDFColor_C->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "sampleCount");

			overlayDownsampledNDFColor_C->setSsboBindingIndex(SampleCount_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = overlayDownsampledNDFColor_C->getSsboBiningIndex(SampleCount_ssbo);

			glShaderStorageBlockBinding(overlayDownsampledNDFColor_C->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, SampleCount_ssbo);
		}

#ifdef CORE_FUNCTIONALITY_ONLY
#else
		{
			unsigned int block_index = glGetProgramResourceIndex(computeAvgNDF_C->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "sampleCount");

			computeAvgNDF_C->setSsboBindingIndex(SampleCount_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = computeAvgNDF_C->getSsboBiningIndex(SampleCount_ssbo);

			glShaderStorageBlockBinding(computeAvgNDF_C->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, SampleCount_ssbo);
		}
#endif
	}
	//circular pattern sample count ssbo
#ifdef CORE_FUNCTIONALITY_ONLY
#else
	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, circularPatternSampleCount_ssbo);
		{
			unsigned int block_index = glGetProgramResourceIndex(reductionShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "circularSampleCount");

			reductionShader->setSsboBindingIndex(circularPatternSampleCount_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = reductionShader->getSsboBiningIndex(circularPatternSampleCount_ssbo);

			glShaderStorageBlockBinding(reductionShader->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, circularPatternSampleCount_ssbo);
		}

		{
			unsigned int block_index = glGetProgramResourceIndex(tileResetShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "circularSampleCount");

			tileResetShader->setSsboBindingIndex(circularPatternSampleCount_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = tileResetShader->getSsboBiningIndex(circularPatternSampleCount_ssbo);

			glShaderStorageBlockBinding(tileResetShader->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, circularPatternSampleCount_ssbo);
		}
		
		{
			unsigned int block_index = glGetProgramResourceIndex(reductionShader_C->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "circularSampleCount");

			reductionShader_C->setSsboBindingIndex(circularPatternSampleCount_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = reductionShader_C->getSsboBiningIndex(circularPatternSampleCount_ssbo);

			glShaderStorageBlockBinding(reductionShader_C->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, circularPatternSampleCount_ssbo);
		}
	}
#endif
	//colormap ssbo
#ifdef CORE_FUNCTIONALITY_ONLY
	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, colorMap_ssbo);

		{
			unsigned int block_index = glGetProgramResourceIndex(renderingShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "colorMap");

			renderingShader->setSsboBindingIndex(colorMap_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = renderingShader->getSsboBiningIndex(colorMap_ssbo);

			glShaderStorageBlockBinding(renderingShader->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, colorMap_ssbo);
		}

		{
			unsigned int block_index = glGetProgramResourceIndex(avgNDFRenderingShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "colorMap");

			avgNDFRenderingShader->setSsboBindingIndex(colorMap_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = avgNDFRenderingShader->getSsboBiningIndex(colorMap_ssbo);

			glShaderStorageBlockBinding(avgNDFRenderingShader->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, colorMap_ssbo);
		}

		{
			unsigned int block_index = glGetProgramResourceIndex(transferShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "colorMap");

			transferShader->setSsboBindingIndex(colorMap_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = transferShader->getSsboBiningIndex(colorMap_ssbo);

			glShaderStorageBlockBinding(transferShader->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, colorMap_ssbo);
		}

		{
			unsigned int block_index = glGetProgramResourceIndex(overlayDownsampledNDFColor_C->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "colorMap");

			overlayDownsampledNDFColor_C->setSsboBindingIndex(colorMap_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = overlayDownsampledNDFColor_C->getSsboBiningIndex(colorMap_ssbo);

			glShaderStorageBlockBinding(overlayDownsampledNDFColor_C->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, colorMap_ssbo);
		}
	}
#else
	
#endif
#ifdef CORE_FUNCTIONALITY_ONLY
#else
	//selected pixels ssbo
	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, selectedPixels_ssbo);
		{
			unsigned int block_index = glGetProgramResourceIndex(computeAvgNDF_C->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "selectedPixels");

			computeAvgNDF_C->setSsboBindingIndex(selectedPixels_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = computeAvgNDF_C->getSsboBiningIndex(selectedPixels_ssbo);

			glShaderStorageBlockBinding(computeAvgNDF_C->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, selectedPixels_ssbo);
		}
	}

	//minmax sim metric
	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, simLimitsF_ssbo);
		{
			unsigned int block_index = glGetProgramResourceIndex(renderingShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "similarityLimitsF");

			renderingShader->setSsboBindingIndex(simLimitsF_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = renderingShader->getSsboBiningIndex(simLimitsF_ssbo);

			glShaderStorageBlockBinding(renderingShader->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, simLimitsF_ssbo);
		}

		{
			unsigned int block_index = glGetProgramResourceIndex(computeAvgNDF_C->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "similarityLimitsF");

			computeAvgNDF_C->setSsboBindingIndex(simLimitsF_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = computeAvgNDF_C->getSsboBiningIndex(simLimitsF_ssbo);

			glShaderStorageBlockBinding(computeAvgNDF_C->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, simLimitsF_ssbo);
		}
	}
#endif
#ifdef CORE_FUNCTIONALITY_ONLY
#else
	//avgndf_ssbo
	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, avgNDF_ssbo);
		{
			unsigned int block_index = glGetProgramResourceIndex(renderingShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "avgNDF");

			renderingShader->setSsboBindingIndex(avgNDF_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = renderingShader->getSsboBiningIndex(avgNDF_ssbo);

			glShaderStorageBlockBinding(renderingShader->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, avgNDF_ssbo);
		}

		{
			unsigned int block_index = glGetProgramResourceIndex(computeAvgNDF_C->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "avgNDF");

			computeAvgNDF_C->setSsboBindingIndex(avgNDF_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = computeAvgNDF_C->getSsboBiningIndex(avgNDF_ssbo);

			glShaderStorageBlockBinding(computeAvgNDF_C->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, avgNDF_ssbo);
		}

		{
			unsigned int block_index = glGetProgramResourceIndex(avgNDFRenderingShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "avgNDF");

			avgNDFRenderingShader->setSsboBindingIndex(avgNDF_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = avgNDFRenderingShader->getSsboBiningIndex(avgNDF_ssbo);

			glShaderStorageBlockBinding(avgNDFRenderingShader->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, avgNDF_ssbo);
		}
	}
#endif

	//pre-integrated bins ssbo
	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, preIntegratedBins_ssbo);
		{
			unsigned int block_index = glGetProgramResourceIndex(renderingShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "preIntegratedBins");

			renderingShader->setSsboBindingIndex(preIntegratedBins_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = renderingShader->getSsboBiningIndex(preIntegratedBins_ssbo);

			glShaderStorageBlockBinding(renderingShader->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, preIntegratedBins_ssbo);
		}

		{
			unsigned int block_index = glGetProgramResourceIndex(sumPreIntegratedBins_C->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "preIntegratedBins");

			sumPreIntegratedBins_C->setSsboBindingIndex(preIntegratedBins_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = sumPreIntegratedBins_C->getSsboBiningIndex(preIntegratedBins_ssbo);

			glShaderStorageBlockBinding(sumPreIntegratedBins_C->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, preIntegratedBins_ssbo);
		}

		{
			unsigned int block_index = glGetProgramResourceIndex(overlayDownsampledNDFColor_C->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "preIntegratedBins");

			overlayDownsampledNDFColor_C->setSsboBindingIndex(preIntegratedBins_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = overlayDownsampledNDFColor_C->getSsboBiningIndex(preIntegratedBins_ssbo);

			glShaderStorageBlockBinding(overlayDownsampledNDFColor_C->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, preIntegratedBins_ssbo);
		}
	}
	//superpreintegratedbins ssbo
	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, superPreIntegratedBins_ssbo);
		{
			unsigned int block_index = glGetProgramResourceIndex(preIntegrateBins_C->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "superPreIntegratedBins");

			preIntegrateBins_C->setSsboBindingIndex(superPreIntegratedBins_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = preIntegrateBins_C->getSsboBiningIndex(superPreIntegratedBins_ssbo);

			glShaderStorageBlockBinding(preIntegrateBins_C->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, superPreIntegratedBins_ssbo);
		}

		{
			unsigned int block_index = glGetProgramResourceIndex(sumPreIntegratedBins_C->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "superPreIntegratedBins");

			sumPreIntegratedBins_C->setSsboBindingIndex(superPreIntegratedBins_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = sumPreIntegratedBins_C->getSsboBiningIndex(superPreIntegratedBins_ssbo);

			glShaderStorageBlockBinding(sumPreIntegratedBins_C->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, superPreIntegratedBins_ssbo);
		}
	}
	//bin areas ssbo
	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, binAreas_ssbo);
		{
			unsigned int block_index = glGetProgramResourceIndex(renderingShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "binAreas");

			renderingShader->setSsboBindingIndex(binAreas_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = renderingShader->getSsboBiningIndex(binAreas_ssbo);

			glShaderStorageBlockBinding(renderingShader->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, binAreas_ssbo);
		}

		{
			unsigned int block_index = glGetProgramResourceIndex(overlayDownsampledNDFColor_C->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "binAreas");

			overlayDownsampledNDFColor_C->setSsboBindingIndex(binAreas_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = overlayDownsampledNDFColor_C->getSsboBiningIndex(binAreas_ssbo);

			glShaderStorageBlockBinding(overlayDownsampledNDFColor_C->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, binAreas_ssbo);
		}
#ifdef CORE_FUNCTIONALITY_ONLY
#else
		{
			unsigned int block_index = glGetProgramResourceIndex(computeAvgNDF_C->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "binAreas");

			computeAvgNDF_C->setSsboBindingIndex(binAreas_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = computeAvgNDF_C->getSsboBiningIndex(binAreas_ssbo);

			glShaderStorageBlockBinding(computeAvgNDF_C->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, binAreas_ssbo);
		}

		{
			unsigned int block_index = glGetProgramResourceIndex(avgNDFRenderingShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "binAreas");

			avgNDFRenderingShader->setSsboBindingIndex(binAreas_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = avgNDFRenderingShader->getSsboBiningIndex(binAreas_ssbo);

			glShaderStorageBlockBinding(avgNDFRenderingShader->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, binAreas_ssbo);
		}
#endif
	}

	//siomplebinareas ssbo
	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, simpleBinningAreas_ssbo);
		{
			unsigned int block_index = glGetProgramResourceIndex(preIntegrateBins_C->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "simple_binAreas");

			preIntegrateBins_C->setSsboBindingIndex(simpleBinningAreas_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = preIntegrateBins_C->getSsboBiningIndex(simpleBinningAreas_ssbo);

			glShaderStorageBlockBinding(preIntegrateBins_C->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, simpleBinningAreas_ssbo);
		}
	}
	//ndfimage_ssbo
#ifdef CORE_FUNCTIONALITY_ONLY
#else
	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, NDFImage_ssbo);
		{
			unsigned int block_index = glGetProgramResourceIndex(drawNDFShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "NDFImage");

			drawNDFShader->setSsboBindingIndex(NDFImage_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = drawNDFShader->getSsboBiningIndex(NDFImage_ssbo);

			glShaderStorageBlockBinding(drawNDFShader->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, NDFImage_ssbo);
		}
	}
#endif

	//progressive raycasting ssbo
	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, progressive_raycasting_ssbo);
		{
			unsigned int block_index = glGetProgramResourceIndex(renderingShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "progressive_raycasting_data");

			renderingShader->setSsboBindingIndex(progressive_raycasting_ssbo, globalSsboBindingPointIndex);
			GLuint binding_point_index = renderingShader->getSsboBiningIndex(progressive_raycasting_ssbo);

			glShaderStorageBlockBinding(renderingShader->GetGlShaderProgramHandle(), block_index, binding_point_index);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, progressive_raycasting_ssbo);
		}
	}

	//ndf overlay ssbo
	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ndfColors_ssbo);
		{
			auto blockIndex = glGetProgramResourceIndex(overlayDownsampledNDFColor_C->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "ndfOverlay");

			overlayDownsampledNDFColor_C->setSsboBindingIndex(ndfColors_ssbo, globalSsboBindingPointIndex);
			GLuint ssboBindingPointIndex = overlayDownsampledNDFColor_C->getSsboBiningIndex(ndfColors_ssbo);

			glShaderStorageBlockBinding(overlayDownsampledNDFColor_C->GetGlShaderProgramHandle(), blockIndex, ssboBindingPointIndex);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, ndfColors_ssbo);
		}
		{
			auto blockIndex = glGetProgramResourceIndex(renderingShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "ndfOverlay");

			renderingShader->setSsboBindingIndex(ndfColors_ssbo, globalSsboBindingPointIndex);
			GLuint ssboBindingPointIndex = renderingShader->getSsboBiningIndex(ndfColors_ssbo);

			glShaderStorageBlockBinding(renderingShader->GetGlShaderProgramHandle(), blockIndex, ssboBindingPointIndex);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, ndfColors_ssbo);
		}
	}

	//ndf ssbo
	{
		{
			auto blockIndex = glGetProgramResourceIndex(renderingShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "NdfVoxelData");

			renderingShader->setSsboBindingIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject(), globalSsboBindingPointIndex);
			GLuint ssboBindingPointIndex = renderingShader->getSsboBiningIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject());

			glShaderStorageBlockBinding(renderingShader->GetGlShaderProgramHandle(), blockIndex, ssboBindingPointIndex);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, ndfTree.GetLevels().front().GetShaderStorageBufferOject());
		}
		{
		auto blockIndex = glGetProgramResourceIndex(overlayDownsampledNDFColor_C->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "NdfVoxelData");

		overlayDownsampledNDFColor_C->setSsboBindingIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject(), globalSsboBindingPointIndex);
		GLuint ssboBindingPointIndex = overlayDownsampledNDFColor_C->getSsboBiningIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject());

		glShaderStorageBlockBinding(overlayDownsampledNDFColor_C->GetGlShaderProgramHandle(), blockIndex, ssboBindingPointIndex);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, ndfTree.GetLevels().front().GetShaderStorageBufferOject());
	}
		{
		auto blockIndex = glGetProgramResourceIndex(reductionShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "NdfVoxelData");

		reductionShader->setSsboBindingIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject(), globalSsboBindingPointIndex);
		GLuint ssboBindingPointIndex = reductionShader->getSsboBiningIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject());

		glShaderStorageBlockBinding(reductionShader->GetGlShaderProgramHandle(), blockIndex, ssboBindingPointIndex);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, ndfTree.GetLevels().front().GetShaderStorageBufferOject());
	}
		{
			auto blockIndex = glGetProgramResourceIndex(reductionShader_C->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "NdfVoxelData");

			reductionShader_C->setSsboBindingIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject(), globalSsboBindingPointIndex);
			GLuint ssboBindingPointIndex = reductionShader_C->getSsboBiningIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject());

			glShaderStorageBlockBinding(reductionShader_C->GetGlShaderProgramHandle(), blockIndex, ssboBindingPointIndex);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, ndfTree.GetLevels().front().GetShaderStorageBufferOject());
		}
		{
			auto blockIndex = glGetProgramResourceIndex(tiledRaycastingShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "NdfVoxelData");

			tiledRaycastingShader->setSsboBindingIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject(), globalSsboBindingPointIndex);
			GLuint ssboBindingPointIndex = tiledRaycastingShader->getSsboBiningIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject());

			glShaderStorageBlockBinding(tiledRaycastingShader->GetGlShaderProgramHandle(), blockIndex, ssboBindingPointIndex);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, ndfTree.GetLevels().front().GetShaderStorageBufferOject());
		}
		{
			auto blockIndex = glGetProgramResourceIndex(tiledRaycastingShader_C->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "NdfVoxelData");

			tiledRaycastingShader_C->setSsboBindingIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject(), globalSsboBindingPointIndex);
			GLuint ssboBindingPointIndex = tiledRaycastingShader_C->getSsboBiningIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject());

			glShaderStorageBlockBinding(tiledRaycastingShader_C->GetGlShaderProgramHandle(), blockIndex, ssboBindingPointIndex);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, ndfTree.GetLevels().front().GetShaderStorageBufferOject());
		}
#ifdef CORE_FUNCTIONALITY_ONLY
#else
		{
			auto blockIndex = glGetProgramResourceIndex(drawNDFShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "NdfVoxelData");

			drawNDFShader->setSsboBindingIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject(), globalSsboBindingPointIndex);
			GLuint ssboBindingPointIndex = drawNDFShader->getSsboBiningIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject());

			glShaderStorageBlockBinding(drawNDFShader->GetGlShaderProgramHandle(), blockIndex, ssboBindingPointIndex);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, ndfTree.GetLevels().front().GetShaderStorageBufferOject());
		}
#endif
		{
			auto blockIndex = glGetProgramResourceIndex(tileResetShader->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "NdfVoxelData");

			tileResetShader->setSsboBindingIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject(), globalSsboBindingPointIndex);
			GLuint ssboBindingPointIndex = tileResetShader->getSsboBiningIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject());

			glShaderStorageBlockBinding(tileResetShader->GetGlShaderProgramHandle(), blockIndex, ssboBindingPointIndex);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, ndfTree.GetLevels().front().GetShaderStorageBufferOject());
		}
#ifdef CORE_FUNCTIONALITY_ONLY
#else
		{
			auto blockIndex = glGetProgramResourceIndex(computeAvgNDF_C->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "NdfVoxelData");

			computeAvgNDF_C->setSsboBindingIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject(), globalSsboBindingPointIndex);
			GLuint ssboBindingPointIndex = computeAvgNDF_C->getSsboBiningIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject());

			glShaderStorageBlockBinding(computeAvgNDF_C->GetGlShaderProgramHandle(), blockIndex, ssboBindingPointIndex);
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, ndfTree.GetLevels().front().GetShaderStorageBufferOject());
		}
#endif
	}
}
void overlayNDFs()
{
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, overlayDownsampledNDFColor_C->getSsboBiningIndex(SampleCount_ssbo), SampleCount_ssbo);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, overlayDownsampledNDFColor_C->getSsboBiningIndex(colorMap_ssbo), colorMap_ssbo);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, overlayDownsampledNDFColor_C->getSsboBiningIndex(ndfColors_ssbo), ndfColors_ssbo);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, overlayDownsampledNDFColor_C->getSsboBiningIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject()), ndfTree.GetLevels().front().GetShaderStorageBufferOject());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, overlayDownsampledNDFColor_C->getSsboBiningIndex(preIntegratedBins_ssbo), preIntegratedBins_ssbo);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, overlayDownsampledNDFColor_C->getSsboBiningIndex(binAreas_ssbo), binAreas_ssbo);

	if (LOD.myTiles[std::floor(current_lod - lodDelta)].visible.size() > 0)
	{
		//get sampling blc in lod we downsample to
		glm::vec2 downsampling_blc_s = glm::vec2(0, 0);
		glm::vec2 downsampling_trc_s = glm::vec2(sw*pow(2, current_lod - downsampling_lod), sh*pow(2, current_lod - downsampling_lod));

		int lw, lh;
		glm::vec3 c1, c2;
		LOD.get_lod_width_and_hight(std::floor(current_lod - lodDelta), lw, lh);
		LOD.myTiles[std::floor(current_lod - lodDelta)].get_blc_and_trc_of_viible_tiles(c1, c2, lw, lh);
		
		//get c1 in lod that we wish to downsample to
		c1 = glm::vec3(c1.x*pow(2, current_lod - downsampling_lod), c1.y*pow(2, current_lod - downsampling_lod), c1.z);
		glm::vec2 downsampling_blc_l = glm::vec2(c1 - (LOD.myTiles[std::floor(downsampling_lod - lodDelta)].T[0].c - glm::vec3(0.5f*tile_w, 0.5*tile_h, 0)));// +LOD.obj2pixel(glm::vec2(cameraOffset), std::floor(current_lod));

		//use shader
		auto Handle = overlayDownsampledNDFColor_C->GetGlShaderProgramHandle();
		glUseProgram(Handle);
		{
			glBindImageTexture(2, Page_Texture, std::floor(current_lod), false, 0, GL_READ_ONLY, Page_texture_internalFormat);
			glUniform1i(glGetUniformLocation(Handle, "currentLevel"), 2);

			glBindImageTexture(3, Page_Texture, std::floor(downsampling_lod), false, 0, GL_READ_ONLY, Page_texture_internalFormat);
			glUniform1i(glGetUniformLocation(Handle, "downsamplingLevel"), 3);

			glUniform2i(glGetUniformLocation(Handle, "histogramDiscretizations"), ndfTree.GetHistogramResolution().x, ndfTree.GetHistogramResolution().y);

			int cur_w, downsampled_w, tint, cur_h, downsampled_h;
			//do some code here
			LOD.get_lod_width_and_hight(std::floor(current_lod), cur_w, cur_h);
			downsampled_w = 0;

			LOD.get_lod_width_and_hight(std::floor(downsampling_lod), downsampled_w, downsampled_h);

			glUniform1i(glGetUniformLocation(Handle, "tile_w"), tile_w);
			glUniform1i(glGetUniformLocation(Handle, "tile_h"), tile_h);

			glUniform1i(glGetUniformLocation(Handle, "cur_w"), cur_w);
			glUniform1i(glGetUniformLocation(Handle, "downsampled_w"), downsampled_w);
			glUniform1i(glGetUniformLocation(Handle, "cur_h"), cur_h);
			glUniform1i(glGetUniformLocation(Handle, "downsampled_h"), downsampled_h);
			glUniform1f(glGetUniformLocation(Handle, "lod"), current_lod);
			glUniform1f(glGetUniformLocation(Handle, "dLod"), downsampling_lod);
			glUniform1i(glGetUniformLocation(Handle, "phys_tex_dim"), phys_tex_dim);

			glUniform1i(glGetUniformLocation(Handle, "circularPattern"), circularPattern);
			glUniform1f(glGetUniformLocation(Handle, "sampleW"), sampleW);
			glUniform1i(glGetUniformLocation(Handle, "colorMapSize"), colorMapSize);

			glUniform1i(glGetUniformLocation(Handle, "cachedRayCasting"), cachedRayCasting);

			glUniform2f(glGetUniformLocation(Handle, "trc_s"), downsampling_trc_s.x, downsampling_trc_s.y);
			glUniform2f(glGetUniformLocation(Handle, "blc_s"), downsampling_blc_s.x, downsampling_blc_s.y);
			glUniform2f(glGetUniformLocation(Handle, "blc_l"), downsampling_blc_l.x, downsampling_blc_l.y);

			//old
			glUniform2i(glGetUniformLocation(Handle, "spatialDiscretizations"), sw*pow(2, current_lod - downsampling_lod) / multiSamplingRate, sh*pow(2, current_lod - downsampling_lod) / multiSamplingRate);
		}

		glDispatchCompute(sw*pow(2, current_lod - downsampling_lod), sh*pow(2, current_lod - downsampling_lod), 1);
		//glDispatchCompute(1,1, 1);

		glUseProgram(0);
	}
}
void compileShaders(std::string shaderPath) {

	//Mohamed's code
	//{
		//auto tileVertex = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		//tileVertex->FromFile(shaderPath + "tileV.cpp", GL_VERTEX_SHADER);

		//auto tileFragment = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		//tileFragment->FromFile(shaderPath + "tileF.cpp", GL_FRAGMENT_SHADER);

		//tileShader = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
		//tileShader->Initialize();
		//tileShader->AttachVertexShader(tileVertex);
		//tileShader->AttachFragmentShader(tileFragment);
		//std::cout << "\tLinking program" << std::endl;
		//tileShader->LinkProgram();
	//}
	//globalSsboBindingPointIndex = 1;


	{

		std::cout << "Initializing computeAVGNDF shader" << std::endl;
		std::cout << "\tCompiling shaders" << std::endl;
		auto computeAVGNDFComputeShader = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		computeAVGNDFComputeShader->FromFile(shaderPath + "computeAVGNDF_C.cpp", GL_COMPUTE_SHADER);

		computeAvgNDF_C = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
		computeAvgNDF_C->Initialize();
		computeAvgNDF_C->AttachComputeShader(computeAVGNDFComputeShader);
		std::cout << "\tLinking program" << std::endl;
		computeAvgNDF_C->LinkProgram();

	}


	{

		std::cout << "Initializing draw NDF shader" << std::endl;
		std::cout << "\tCompiling shaders" << std::endl;
		auto drawNDFComputeShader = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		drawNDFComputeShader->FromFile(shaderPath + "drawNDFC.cpp", GL_COMPUTE_SHADER);

		drawNDFShader = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
		drawNDFShader->Initialize();
		drawNDFShader->AttachComputeShader(drawNDFComputeShader);
		std::cout << "\tLinking program" << std::endl;
		drawNDFShader->LinkProgram();

	}


	{

		std::cout << "Initializing build HOM shader" << std::endl;
		std::cout << "\tCompiling shaders" << std::endl;
		auto drawNDFComputeShader = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		drawNDFComputeShader->FromFile(shaderPath + "buildHOM_C.cpp", GL_COMPUTE_SHADER);

		buildHOMShader_C = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
		buildHOMShader_C->Initialize();
		buildHOMShader_C->AttachComputeShader(drawNDFComputeShader);
		std::cout << "\tLinking program" << std::endl;
		buildHOMShader_C->LinkProgram();

	}

	{

		std::cout << "Initializing preIntegrateNTF shader" << std::endl;
		std::cout << "\tCompiling shaders" << std::endl;
		auto preIntegrateBinsComputeShader = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		preIntegrateBinsComputeShader->FromFile(shaderPath + "preIntegrateBins_C.cpp", GL_COMPUTE_SHADER);

		preIntegrateBins_C = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
		preIntegrateBins_C->Initialize();
		preIntegrateBins_C->AttachComputeShader(preIntegrateBinsComputeShader);
		std::cout << "\tLinking program" << std::endl;
		preIntegrateBins_C->LinkProgram();

	}

	{

		std::cout << "Initializing sumPreIntegrateNTF shader" << std::endl;
		std::cout << "\tCompiling shaders" << std::endl;
		auto preIntegrateBinsComputeShader = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		preIntegrateBinsComputeShader->FromFile(shaderPath + "sumPreIntegratedBins_C.cpp", GL_COMPUTE_SHADER);

		sumPreIntegratedBins_C = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
		sumPreIntegratedBins_C->Initialize();
		sumPreIntegratedBins_C->AttachComputeShader(preIntegrateBinsComputeShader);
		std::cout << "\tLinking program" << std::endl;
		sumPreIntegratedBins_C->LinkProgram();

	}

	{

		std::cout << "Initializing overlay downsampled NDF colors shader" << std::endl;
		std::cout << "\tCompiling shaders" << std::endl;
		auto ndfOverlayComputeShader = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		ndfOverlayComputeShader->FromFile(shaderPath + "overlayDownsampledNDFColors.cpp", GL_COMPUTE_SHADER);

		overlayDownsampledNDFColor_C = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
		overlayDownsampledNDFColor_C->Initialize();
		overlayDownsampledNDFColor_C->AttachComputeShader(ndfOverlayComputeShader);
		std::cout << "\tLinking program" << std::endl;
		overlayDownsampledNDFColor_C->LinkProgram();

	}

	{

		std::cout << "Initializing tile reset shader" << std::endl;
		std::cout << "\tCompiling shaders" << std::endl;
		auto tileResetComputeShader = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		tileResetComputeShader->FromFile(shaderPath + "tileResetC.cpp", GL_COMPUTE_SHADER);

		tileResetShader = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
		tileResetShader->Initialize();
		tileResetShader->AttachComputeShader(tileResetComputeShader);
		std::cout << "\tLinking program" << std::endl;
		tileResetShader->LinkProgram();

	}

	//{
	//	std::cout << "Initializing tile rotation shader" << std::endl;
	//	std::cout << "\tCompiling shaders" << std::endl;
	//	auto rotationComputeShader = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
	//	rotationComputeShader->FromFile(shaderPath + "RotationC.cpp", GL_COMPUTE_SHADER);

	//	rotationShader = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
	//	rotationShader->Initialize();
	//	rotationShader->AttachComputeShader(rotationComputeShader);
	//	std::cout << "\tLinking program" << std::endl;
	//	rotationShader->LinkProgram();
	//}
	//end Mohamed's code

	//{
	//	std::cout << "Initializing pointCloud shader" << std::endl;
	//	std::cout << "\tCompiling shaders" << std::endl;
	//	auto particleVertex = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
	//	particleVertex->FromFile(shaderPath + "sphereSamplingV_PointCloud.cpp", GL_VERTEX_SHADER);

	//	auto particleFragment = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
	//	particleFragment->FromFile(shaderPath + "sphereSamplingF_PointCloud.cpp", GL_FRAGMENT_SHADER);

	//	auto particleGeometry = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
	//	particleGeometry->FromFile(shaderPath + "sphereSamplingG_PointCloud.cpp", GL_GEOMETRY_SHADER);

	//	samplingShader_PointCloud = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
	//	samplingShader_PointCloud->Initialize();
	//	samplingShader_PointCloud->AttachVertexShader(particleVertex);
	//	samplingShader_PointCloud->AttachFragmentShader(particleFragment);
	//	samplingShader_PointCloud->AttachGeometryShader(particleGeometry);
	//	std::cout << "\tLinking program" << std::endl;

	//	samplingShader_PointCloud->LinkProgram();
	//}

	{

		std::cout << "Initializing sampling shader" << std::endl;
		std::cout << "\tCompiling shaders" << std::endl;
		auto particleVertex = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		particleVertex->FromFile(shaderPath + "sphereSamplingV.cpp", GL_VERTEX_SHADER);

		auto particleFragment = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		particleFragment->FromFile(shaderPath + "sphereSamplingF.cpp", GL_FRAGMENT_SHADER);

		auto particleGeometry = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		particleGeometry->FromFile(shaderPath + "sphereSamplingG.cpp", GL_GEOMETRY_SHADER);

		samplingShader = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
		samplingShader->Initialize();
		samplingShader->AttachVertexShader(particleVertex);
		samplingShader->AttachFragmentShader(particleFragment);
		samplingShader->AttachGeometryShader(particleGeometry);
		std::cout << "\tLinking program" << std::endl;

		samplingShader->LinkProgram();


	}

	{

		std::cout << "Initializing draw HOM shader" << std::endl;
		std::cout << "\tCompiling shaders" << std::endl;
		auto particleVertex = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		particleVertex->FromFile(shaderPath + "sphereSamplingV_HOM.cpp", GL_VERTEX_SHADER);

		auto particleFragment = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		particleFragment->FromFile(shaderPath + "sphereSamplingF_HOM.cpp", GL_FRAGMENT_SHADER);

		auto particleGeometry = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		particleGeometry->FromFile(shaderPath + "sphereSamplingG_HOM.cpp", GL_GEOMETRY_SHADER);

		drawHOMShader = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
		drawHOMShader->Initialize();
		drawHOMShader->AttachVertexShader(particleVertex);
		drawHOMShader->AttachFragmentShader(particleFragment);
		drawHOMShader->AttachGeometryShader(particleGeometry);
		std::cout << "\tLinking program" << std::endl;

		drawHOMShader->LinkProgram();


	}

	{

		std::cout << "Initializing draw quad mesh shader" << std::endl;
		std::cout << "\tCompiling shaders" << std::endl;
		auto particleVertex = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		particleVertex->FromFile(shaderPath + "quadMeshV.cpp", GL_VERTEX_SHADER);

		auto particleFragment = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		particleFragment->FromFile(shaderPath + "quadMeshF.cpp", GL_FRAGMENT_SHADER);

		auto particleGeometry = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		particleGeometry->FromFile(shaderPath + "quadMeshG.cpp", GL_GEOMETRY_SHADER);

		drawQuadMeshShader = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
		drawQuadMeshShader->Initialize();
		drawQuadMeshShader->AttachVertexShader(particleVertex);
		drawQuadMeshShader->AttachFragmentShader(particleFragment);
		drawQuadMeshShader->AttachGeometryShader(particleGeometry);
		std::cout << "\tLinking program" << std::endl;

		drawQuadMeshShader->LinkProgram();


	}

	{

		std::cout << "Initializing ndf explorer shader" << std::endl;
		std::cout << "\tCompiling shaders" << std::endl;
		auto particleVertex = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		particleVertex->FromFile(shaderPath + "ndfExplorerV.cpp", GL_VERTEX_SHADER);

		auto particleFragment = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		particleFragment->FromFile(shaderPath + "ndfExplorerF.cpp", GL_FRAGMENT_SHADER);

		auto particleGeometry = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		particleGeometry->FromFile(shaderPath + "ndfExplorerG.cpp", GL_GEOMETRY_SHADER);

		ndfExplorerShader = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
		ndfExplorerShader->Initialize();
		ndfExplorerShader->AttachVertexShader(particleVertex);
		ndfExplorerShader->AttachFragmentShader(particleFragment);
		ndfExplorerShader->AttachGeometryShader(particleGeometry);
		std::cout << "\tLinking program" << std::endl;

		ndfExplorerShader->LinkProgram();


	}

	{

		std::cout << "Initializing reduction shader" << std::endl;
		std::cout << "\tCompiling shaders" << std::endl;
		auto reductionComputeShader = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		reductionComputeShader->FromFile(shaderPath + "reductionProgressiveC_clean.cpp", GL_COMPUTE_SHADER);

		reductionShader = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
		reductionShader->Initialize();
		reductionShader->AttachComputeShader(reductionComputeShader);
		std::cout << "\tLinking program" << std::endl;
		reductionShader->LinkProgram();

	}
	{

		std::cout << "Initializing reduction ceil shader" << std::endl;
		std::cout << "\tCompiling shaders" << std::endl;
		auto reductionComputeShader = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		reductionComputeShader->FromFile(shaderPath + "reductionProgressiveC_clean_C.cpp", GL_COMPUTE_SHADER);

		reductionShader_C = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
		reductionShader_C->Initialize();
		reductionShader_C->AttachComputeShader(reductionComputeShader);
		std::cout << "\tLinking program" << std::endl;
		reductionShader_C->LinkProgram();

	}

	{

		std::cout << "Initializing Tiled Raycasting shader" << std::endl;
		std::cout << "\tCompiling shaders" << std::endl;
		auto tiledRaycastingComputeShader = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		tiledRaycastingComputeShader->FromFile(shaderPath + "TiledRaycastingC.cpp", GL_COMPUTE_SHADER);

		tiledRaycastingShader = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
		tiledRaycastingShader->Initialize();
		tiledRaycastingShader->AttachComputeShader(tiledRaycastingComputeShader);
		std::cout << "\tLinking program" << std::endl;
		tiledRaycastingShader->LinkProgram();

	}

	{

		std::cout << "Initializing Tiled Raycasting Ceil shader" << std::endl;
		std::cout << "\tCompiling shaders" << std::endl;
		auto tiledRaycastingComputeShader = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		tiledRaycastingComputeShader->FromFile(shaderPath + "TiledRaycastingC_C.cpp", GL_COMPUTE_SHADER);

		tiledRaycastingShader_C = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
		tiledRaycastingShader_C->Initialize();
		tiledRaycastingShader_C->AttachComputeShader(tiledRaycastingComputeShader);
		std::cout << "\tLinking program" << std::endl;
		tiledRaycastingShader_C->LinkProgram();

	}

	//{
	//	std::cout << "Initializing sampling shader" << std::endl;
	//	std::cout << "\tCompiling shaders" << std::endl;
	//	auto downsamplingComputeShader = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
	//	downsamplingComputeShader->FromFile(shaderPath + "downsamplingC.cpp", GL_COMPUTE_SHADER);

	//	downsamplingShader = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
	//	downsamplingShader->Initialize();
	//	downsamplingShader->AttachComputeShader(downsamplingComputeShader);
	//	std::cout << "\tLinking program" << std::endl;
	//	downsamplingShader->LinkProgram();
	//}



	{
		

		std::cout << "Initializing rendering shader" << std::endl;
		std::cout << "\tCompiling shaders" << std::endl;
		auto renderigVertex = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		renderigVertex->FromFile("ndfRenderingV.cpp", GL_VERTEX_SHADER);

		auto renderingFragment = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		renderingFragment->FromFile("ndfRenderingF_clean.cpp", GL_FRAGMENT_SHADER);

		renderingShader = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
		renderingShader->Initialize();
		renderingShader->AttachVertexShader(renderigVertex);
		renderingShader->AttachFragmentShader(renderingFragment);
		std::cout << "\tLinking program" << std::endl;
		renderingShader->LinkProgram();

	}

	//{
	//	std::cout << "Initializing eNTF convolution shader" << std::endl;
	//	std::cout << "\tCompiling shaders" << std::endl;
	//	auto convolutionComputeShader = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
	//	convolutionComputeShader->FromFile("eNtfConvolutionC.cpp", GL_COMPUTE_SHADER);

	//	convolutionShader = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
	//	convolutionShader->Initialize();
	//	convolutionShader->AttachComputeShader(convolutionComputeShader);
	//	std::cout << "\tLinking program" << std::endl;
	//	convolutionShader->LinkProgram();
	//}

	//{
	//	std::cout << "Initializing eNTF rendering shader" << std::endl;
	//	std::cout << "\tCompiling shaders" << std::endl;
	//	auto renderigVertex = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
	//	renderigVertex->FromFile("eNtfRenderingV.cpp", GL_VERTEX_SHADER);

	//	auto renderingFragment = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
	//	renderingFragment->FromFile("eNtfRenderingF.cpp", GL_FRAGMENT_SHADER);

	//	eNtfRenderingShader = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
	//	eNtfRenderingShader->Initialize();
	//	eNtfRenderingShader->AttachVertexShader(renderigVertex);
	//	eNtfRenderingShader->AttachFragmentShader(renderingFragment);
	//	std::cout << "\tLinking program" << std::endl;
	//	eNtfRenderingShader->LinkProgram();
	//}

	{

		std::cout << "Initializing transfer function preview shader" << std::endl;
		std::cout << "\tCompiling shaders" << std::endl;
		auto transferVertex = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		transferVertex->FromFile("transferV.cpp", GL_VERTEX_SHADER);

		auto transferFragment = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		transferFragment->FromFile("transferF.cpp", GL_FRAGMENT_SHADER);

		transferShader = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
		transferShader->Initialize();
		transferShader->AttachVertexShader(transferVertex);
		transferShader->AttachFragmentShader(transferFragment);
		std::cout << "\tLinking program" << std::endl;
		transferShader->LinkProgram();

	}

	{

		std::cout << "Initializing average NDF preview shader" << std::endl;
		std::cout << "\tCompiling shaders" << std::endl;
		auto avgNDFVertex = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		avgNDFVertex->FromFile("avgNDFV.cpp", GL_VERTEX_SHADER);

		auto avgNDFFragment = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		avgNDFFragment->FromFile("avgNDFF.cpp", GL_FRAGMENT_SHADER);

		avgNDFRenderingShader = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
		avgNDFRenderingShader->Initialize();
		avgNDFRenderingShader->AttachVertexShader(avgNDFVertex);
		avgNDFRenderingShader->AttachFragmentShader(avgNDFFragment);
		std::cout << "\tLinking program" << std::endl;
		avgNDFRenderingShader->LinkProgram();

	}

	{

		std::cout << "Initializing sampling progress bar preview shader" << std::endl;
		std::cout << "\tCompiling shaders" << std::endl;
		auto barVertex = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		barVertex->FromFile("barV.cpp", GL_VERTEX_SHADER);

		auto barFragment = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		barFragment->FromFile("barF.cpp", GL_FRAGMENT_SHADER);

		barRenderingShader = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
		barRenderingShader->Initialize();
		barRenderingShader->AttachVertexShader(barVertex);
		barRenderingShader->AttachFragmentShader(barFragment);
		std::cout << "\tLinking program" << std::endl;
		barRenderingShader->LinkProgram();

	}

	{

		std::cout << "Initializing HOM depth transfer shader" << std::endl;
		std::cout << "\tCompiling shaders" << std::endl;
		auto homdtVertex = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		homdtVertex->FromFile("homdtV.cpp", GL_VERTEX_SHADER);

		auto homFragment = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		homFragment->FromFile("homdtF.cpp", GL_FRAGMENT_SHADER);

		homDepthTransferShader = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
		homDepthTransferShader->Initialize();
		homDepthTransferShader->AttachVertexShader(homdtVertex);
		homDepthTransferShader->AttachFragmentShader(homFragment);
		std::cout << "\tLinking program" << std::endl;
		homDepthTransferShader->LinkProgram();

	}

	{

		std::cout << "Initializing draw depth buffer shader" << std::endl;
		std::cout << "\tCompiling shaders" << std::endl;
		auto homdtVertex = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		homdtVertex->FromFile("drawDepthBufferV.cpp", GL_VERTEX_SHADER);

		auto homFragment = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		homFragment->FromFile("drawDepthBufferF.cpp", GL_FRAGMENT_SHADER);

		drawDepthBufferShader = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
		drawDepthBufferShader->Initialize();
		drawDepthBufferShader->AttachVertexShader(homdtVertex);
		drawDepthBufferShader->AttachFragmentShader(homFragment);
		std::cout << "\tLinking program" << std::endl;
		drawDepthBufferShader->LinkProgram();

	}
	{

		std::cout << "Initialzing selection preview shader" << std::endl;
		std::cout << "\tCompiling shaders" << std::endl;
		auto selectionVertex = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		selectionVertex->FromFile("selectionV.cpp", GL_VERTEX_SHADER);

		auto selectionFragment = std::shared_ptr<Helpers::Gl::Shader>(new Helpers::Gl::Shader());
		selectionFragment->FromFile("selectionF.cpp", GL_FRAGMENT_SHADER);

		selectionShader = std::unique_ptr<Helpers::Gl::ShaderProgram>(new Helpers::Gl::ShaderProgram());
		selectionShader->Initialize();
		selectionShader->AttachVertexShader(selectionVertex);
		selectionShader->AttachFragmentShader(selectionFragment);
		std::cout << "\tLinking program" << std::endl;
		selectionShader->LinkProgram();

	}

}

const auto timeDependent = true;
//auto frameIntervalMs = int(201);
auto frameIntervalMs = int(1);
const auto frameIntervalMsStep = int(20);
auto lastFrameChange = std::chrono::system_clock::now();

int frame = 0, Time, timebase = 0;

void idle() {
#ifdef READ_FROM_FILE
	if (timeDependent) {
		auto timeSinceLastFrame = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - lastFrameChange);
		int intervalMultiplier = 1;
		if (timeStep == timeStepLowerLimit || timeStep == timeStepUpperLimit) {
			intervalMultiplier = 2;
		}
		if (timeSinceLastFrame.count() >= (frameIntervalMs * intervalMultiplier)) {
			std::cout << "Time since last frame (includes rendering) " << timeSinceLastFrame.count() << std::endl;
			lastFrameChange = std::chrono::system_clock::now();

			auto timeTransformed = timeStep - timeStepLowerLimit;
			timeTransformed = (timeTransformed + 1) % ((timeStepUpperLimit + 1) - timeStepLowerLimit);
			timeStep = timeTransformed + timeStepLowerLimit;

			// FIXME: the time step was missing
			if (skip43 && timeStep == 43) {
				timeStep = 44;
			}

			sparseENtfFilePath = fileFolder + fileSparseSubFolder + filePrefix + std::to_string(timeStep) + fileSuffix + ".entf";
			//std::cout << "Changing frame to step " << timeStep << " " << sparseENtfFilePath << std::endl;

			loadSparseENtf(sparseENtfFilePath);
		}
	}
#endif // READ_FROM_FILE

	if (timebase == 0)
	{
		frame++;
		timebase = glutGet(GLUT_ELAPSED_TIME);
	}
	else{

		frame++;
		Time = glutGet(GLUT_ELAPSED_TIME);
		if (Time - timebase > 1000)
		{
			float fps = frame*1000.0 / (Time - timebase);
			//std::cout << "we are running at " << fps << " fps" << std::endl;
			timebase = Time;
			frame = 0;
		}
	}
	//display();
	glutPostRedisplay();
}

std::vector<float> currentNdfData;
std::vector<float> lastNdfData;
const int measurementInterval = 1;
std::vector<double> upscaledMeasurements;
std::vector<double> upscaledMeasurements2;

inline void drawHOM(std::vector<int>& inds)
{
	// render particles directly during rotation
	sw = homHighestResolution.x;
	sh = homHighestResolution.y;

	// calculate global projection matrix
	auto leftOriginal = -0.5f;
	auto rightOriginal = 0.5f;
	auto bottomOriginal = -0.5f / aspectRatio;
	auto topOriginal = 0.5f / aspectRatio;


	OrigL = leftOriginal;
	OrigR = rightOriginal;
	OrigB = bottomOriginal;
	OrigT = topOriginal;
	//auto bottomOriginal = -0.5f;
	//auto topOriginal = 0.5f;

	glm::mat4x4 modelMatrix, T, Tinv, S;
	const auto modelScale = initialCameraDistance / cameraDistance;
	// NOTE: turn scaling off for rendering the histograms
	//const auto modelScale = 1.0f;

	//T[0][4] = -lastMouse.x;
	//T[1][4] = -lastMouse.y;

	//Tinv[0][4] = lastMouse.x;
	//Tinv[1][4] = lastMouse.y;

	S[0][0] = modelScale;
	S[1][1] = modelScale;
	S[2][2] = modelScale;

	modelMatrix = Tinv*S*T;



	//modelMatrix[1][1] *= aspectRatio;

	camPosiSampling = camPosi;
	auto viewMatrix = glm::lookAt(camPosiSampling, camTarget, camUp);


	auto modeViewMatrix = modelMatrix * viewMatrix;
	auto projectionMatrix = glm::ortho(leftOriginal, rightOriginal, bottomOriginal, topOriginal, nearPlane, farPlane);

	auto viewProjectionMatrix = projectionMatrix * viewMatrix;
	//samplingAspectRatio = aspectRatio;

	glm::vec3 tempCam = CameraPosition(cameraRotation, cameraDistance);
	viewMat_noPanning = glm::lookAt(tempCam, glm::vec3(0, 0, 0), camUp);
	modelviewMat_noPanning = modelMatrix*viewMat_noPanning;

	//Mohamed's Code

	//at fractional lods, i want the sampling to be done at the ceil resolution to avoid race conditions,
	//therefore, i will edit the modelviewmat for this case only

	if ((!plainRayCasting) && (!pointCloudRendering))
	{
		//bring camera to ceil lod position
		float tempCamDist = LOD.get_cam_dist(std::floor(current_lod - lodDelta));
		float tempmodelscale = initialCameraDistance / tempCamDist;

		//update camposisampling
		camPosiSampling = CameraPosition(cameraRotation, tempCamDist);
		camPosiSampling += cameraOffset;

		//edit view matrix
		samplingViewMat = glm::lookAt(camPosiSampling, camTarget, camUp);

		//edit projection matrix
		samplingProjectionMat = glm::ortho(sl, sr, sb, st, nearPlane, farPlane);

		//edit viewprojectionmatrix
		samplingViewProjectionMat = samplingProjectionMat*samplingViewMat;

		//edit model matrix
		samplingModelMat = modelMatrix;
		samplingModelMat[0][0] = tempmodelscale;
		samplingModelMat[1][1] = tempmodelscale;
		samplingModelMat[2][2] = tempmodelscale;

		//glm::vec2 f = glm::vec2(sl + ((sr - sl) / 2.0f), sb + ((st - sb) / 2.0f));

		//T[0][4] = -f.x;
		//T[1][4] = -f.y;

		//Tinv[0][4] = f.x;
		//Tinv[1][4] = f.y;

		//samplingModelMat = Tinv*samplingModelMat*T;

		//edit model view matrix
		samplingModelViewMat = samplingModelMat*samplingViewMat;

		samplingAspectRatio = sw / sh;

		glm::vec3 tempCam = CameraPosition(cameraRotation, tempCamDist);
		samplingViewMat_noPanning = glm::lookAt(tempCam, glm::vec3(0, 0, 0), camUp);
		samplingModelViewMat_noPanning = samplingModelMat*samplingViewMat_noPanning;
	}
	else
	{
		//edit projection matrix
		samplingProjectionMat = projectionMatrix;
		samplingViewMat = viewMatrix;
		samplingViewProjectionMat = viewProjectionMatrix;
		samplingModelViewMat = modeViewMatrix;
		samplingModelMat = modelMatrix;
		samplingAspectRatio = aspectRatio;

		samplingViewMat_noPanning = viewMat_noPanning;
		samplingModelViewMat_noPanning = modelviewMat_noPanning;

		sw = windowSize.x;
		sh = windowSize.y;
		samplingAspectRatio = sw / sh;
	}




	modelviewMat = modeViewMatrix;
	projectionMat = projectionMatrix;
	//viewportMat = glm::vec4(0, 0,std::max(rayCastingSolutionRenderTarget.Width,windowSize.x),std::max( rayCastingSolutionRenderTarget.Height,windowSize.y));
	viewportMat = glm::vec4(0, 0, windowSize.x, windowSize.y);
	viewMat = viewMatrix;
	modelMat = modelMatrix;
	viewprojectionMat = viewProjectionMatrix;
	mynear = nearPlane;
	myfar = farPlane;
	myup = camUp;
	mytarget = camTarget;
	//end Mohamed's Code




	glBindFramebuffer(GL_FRAMEBUFFER, homRenderTarget.FrameBufferObject);
	glViewport(0, 0, sw, sh);




	//glBindFramebuffer(GL_FRAMEBUFFER, MyRenderTarget.FrameBufferObject);
	//glViewport(0, 0, MyRenderTarget.Width, MyRenderTarget.Height);


	//glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//glViewport(0, 0, windowSize.x, windowSize.y);

	// FIXME: viewport causes tears for some reason if sampling rate < 1024
	glClearColor(0.f, 0.f, 0.f, 1.0f);
	// clear render target
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	//glEnable(GL_POINT_SPRITE);
	//glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_DEPTH_TEST);
	glFrontFace(GL_CCW);

	//GLuint ssboBindingPointIndex = 2;
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, samplingShader->getSsboBiningIndex(myArrayUBO), myArrayUBO);


	GLuint samplingProgramHandle;

	//glPointSize(particleRadius);

	//if (pointCloudRendering)
	//	samplingProgramHandle = samplingShader_PointCloud->GetGlShaderProgramHandle();
	//else
	samplingProgramHandle = drawHOMShader->GetGlShaderProgramHandle();

	assert(glUseProgram);
	glUseProgram(samplingProgramHandle);

	glUniform1f(glGetUniformLocation(samplingProgramHandle, "far"), farPlane);
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "near"), nearPlane);
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "particleScale"), particleScale);
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "aspectRatio"), samplingAspectRatio);

	//get how much object space is represented in one pixel
	float objPerPixel;
	{
		glm::vec2 s, e;
		s = LOD.pixel2obj(glm::vec2(0, 0), std::floor(current_lod - lodDelta));
		e = LOD.pixel2obj(glm::vec2(1, 1), std::floor(current_lod - lodDelta));

		//get difference
		glm::vec2 v = e - s;
		objPerPixel = v.x;
	}
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "objPerPixel"), objPerPixel);

	glUniform1i(glGetUniformLocation(samplingProgramHandle, "buildingHOM"), buildingHOM);
	glUniform1i(glGetUniformLocation(samplingProgramHandle, "maxSamplingRuns"), tempmaxSamplingRuns);

	glUniform1f(glGetUniformLocation(samplingProgramHandle, "viewportWidth"), static_cast<float>(sw));
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "viewportHeight"), static_cast<float>(sh));
	glUniform1i(glGetUniformLocation(samplingProgramHandle, "plainRayCasting"), plainRayCasting);
	glUniform1i(glGetUniformLocation(samplingProgramHandle, "pointCloudRendering"), pointCloudRendering);


	glUniform1i(glGetUniformLocation(samplingProgramHandle, "samplescount"), sample_count);
	glUniform2fv(glGetUniformLocation(samplingProgramHandle, "sPos"), 1, &sPos[0]);


	glUniform1f(glGetUniformLocation(samplingProgramHandle, "maxZ"), modelMax.z);
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "minZ"), modelMin.z);

	auto right = glm::normalize(glm::cross(normalize(camTarget - camPosi), camUp));
	glUniform3fv(glGetUniformLocation(samplingProgramHandle, "right"), 1, &right[0]);
	glUniform3fv(glGetUniformLocation(samplingProgramHandle, "up"), 1, &camUp[0]);



	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "Model"), 1, GL_FALSE, &samplingModelMat[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "ModelView"), 1, GL_FALSE, &samplingModelViewMat[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "View"), 1, GL_FALSE, &samplingViewMat[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "Projection"), 1, GL_FALSE, &samplingProjectionMat[0][0]);
	glUniformMatrix3fv(glGetUniformLocation(samplingProgramHandle, "RotationMatrix"), 1, GL_FALSE, &GlobalRotationMatrix[0][0]);


	glm::mat4 ViewAlignmentMatrix;
	ViewAlignmentMatrix = glm::rotate(ViewAlignmentMatrix, cameraRotation.x, glm::vec3(0.0f, 1.0f, 0.0f));
	ViewAlignmentMatrix = glm::rotate(ViewAlignmentMatrix, cameraRotation.y, glm::vec3(1.0f, 0.0f, 0.0f));

	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "ViewAlignmentMatrix"), 1, GL_FALSE, &ViewAlignmentMatrix[0][0]);

	glUniform3fv(glGetUniformLocation(samplingProgramHandle, "ViewPosition"), 1, &camPosiSampling[0]);
	glUniform2i(glGetUniformLocation(samplingProgramHandle, "viewSlice"), 0, 0);

	// render particles using ray casting
	// set uniforms
	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "ViewProjection"), 1, GL_FALSE, &samplingViewProjectionMat[0][0]);

	auto initialOffset = glm::vec3(0.0f, 0.0f, 0.0f);

	// FIXME: fix magic numbers
	//if (particleInstances.x > 1) {
	//	initialOffset.x = -0.125f * static_cast<float>(particleInstances.x) * modelExtent.x;
	//}
	//if (particleInstances.y > 1) {
	//	initialOffset.y = -0.3f * static_cast<float>(particleInstances.y) * modelExtent.y;
	//}
	//if (particleInstances.z > 1) {
	//	initialOffset.z = -0.125f * static_cast<float>(particleInstances.z) * modelExtent.z;
	//}

	// render samples
#if 0  //no bricking
	for (int zInstance = 0; zInstance < particleInstances.z; ++zInstance)
	{
		for (int yInstance = 0; yInstance < particleInstances.y; ++yInstance)
		{
			for (int xInstance = 0; xInstance < particleInstances.x; ++xInstance)
			{
				auto modelOffset = initialOffset + glm::vec3(static_cast<float>(xInstance)* modelExtent.x, static_cast<float>(yInstance)* modelExtent.y, static_cast<float>(zInstance)* modelExtent.z);

				glUniform3fv(glGetUniformLocation(samplingProgramHandle, "modelOffset"), 1, &modelOffset[0]);
				particlesGlBuffer.Render(GL_POINTS);
				//particlesGlBuffer.Render_Selective(GL_POINTS, visible_spheres, visible_spheres_count);
				//int i[] = {0,1};
				//particlesGlBuffer.Render_Selective(GL_POINTS, i, 2);
			}
		}
	}
#else       
#if 1  //naive bricking
	//here we assume the bricks in 'global_tree_leaves' are sorted from back to front
	//if the depth buffer is not reset each time I render a brick, i don't need to sort, should do this!!!
	if (global_tree.leaves.size() > 1)
	{
#if 0
		//initially put data in buffer of index '1'
		bufferId = 0;
		glBindBuffer(GL_ARRAY_BUFFER, particlesGLBufferArray[1 - bufferId].Vbo_);
		GLvoid* p = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		memcpy(p, reinterpret_cast<char*>(&global_tree.nodes[global_tree.leaves[0]].Points[0]), global_tree.nodes[global_tree.leaves[0]].Points.size() * sizeof(*global_tree.nodes[global_tree.leaves[0]].Points.begin()));
		glUnmapBuffer(GL_ARRAY_BUFFER);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		for (int i = 1; i <= global_tree.leaves.size(); i++)
		{
			if (i < global_tree.leaves.size())
			{
				//1-copy data in one buffer (either particlesglbuffer or particlesglbuffer_back)
				glBindBuffer(GL_ARRAY_BUFFER, particlesGLBufferArray[bufferId].Vbo_);
				GLvoid* p = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
				memcpy(p, reinterpret_cast<char*>(&global_tree.nodes[global_tree.leaves[i]].Points[0]), global_tree.nodes[global_tree.leaves[i]].Points.size() * sizeof(*global_tree.nodes[global_tree.leaves[i]].Points.begin()));
				glUnmapBuffer(GL_ARRAY_BUFFER);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
			}

			//2- and render the other buffer
			particlesGLBufferArray[1 - bufferId].VertexCount_ = global_tree.nodes[global_tree.leaves[i - 1]].Points.size();
			particlesGLBufferArray[1 - bufferId].Render(GL_POINTS);

			//3- switch buffer id
			bufferId = 1 - bufferId;
		}
#else

		int s;
		for (int i = 0; i < renderBricks.size(); i++)
		{
			for (int k = 0; k < inds.size(); k++)
			{
				if (renderBricks[i].first == inds[k])  //render only the node passed
				{
					//copy first brick to buffer 1
					bufferId = 0;
					glBindBuffer(GL_ARRAY_BUFFER, particlesGLBufferArray[1 - bufferId].Vbo_);
					GLvoid* p = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
					memcpy(p, reinterpret_cast<char*>(&global_tree.nodes[renderBricks[i].first].Points[0]), renderBricks[i].second[0] * sizeof(*global_tree.nodes[global_tree.leaves[0]].Points.begin()));
					glUnmapBuffer(GL_ARRAY_BUFFER);
					glBindBuffer(GL_ARRAY_BUFFER, 0);

					for (int j = 1; j <= renderBricks[i].second.size(); j++)
					{
						if (j < renderBricks[i].second.size())
						{
							s = std::accumulate(renderBricks[i].second.begin(), renderBricks[i].second.begin() + j, 0);

							//1-copy brick in one buffer
							glBindBuffer(GL_ARRAY_BUFFER, particlesGLBufferArray[bufferId].Vbo_);
							GLvoid* p = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
							memcpy(p, reinterpret_cast<char*>(&global_tree.nodes[renderBricks[i].first].Points[s]), renderBricks[i].second[j] * sizeof(*global_tree.nodes[global_tree.leaves[0]].Points.begin()));
							glUnmapBuffer(GL_ARRAY_BUFFER);
							glBindBuffer(GL_ARRAY_BUFFER, 0);
						}

						//2- and render the other buffer
						particlesGLBufferArray[1 - bufferId].VertexCount_ = renderBricks[i].second[j - 1];
						particlesGLBufferArray[1 - bufferId].Render(GL_POINTS);

						//3- switch buffer id
						bufferId = 1 - bufferId;
					}
					break;
				}
			}
		}
#endif
	}
	else
	{
		//initially put data in buffer of index '1'
		bufferId = 0;
		glBindBuffer(GL_ARRAY_BUFFER, particlesGLBufferArray[1 - bufferId].Vbo_);
		GLvoid* p = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		memcpy(p, reinterpret_cast<char*>(&global_tree.nodes[global_tree.leaves[0]].Points[0]), global_tree.nodes[global_tree.leaves[0]].Points.size() * sizeof(*global_tree.nodes[global_tree.leaves[0]].Points.begin()));
		glUnmapBuffer(GL_ARRAY_BUFFER);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		particlesGLBufferArray[1 - bufferId].VertexCount_ = global_tree.nodes[global_tree.leaves[0]].Points.size();
		particlesGLBufferArray[1 - bufferId].Render(GL_POINTS);
	}
#else  //more efficient bricking
	//Helpers::Gl::MakeBuffer(particleCenters, global_tree, particlesGLBufferArray);
	//for (int zInstance = 0; zInstance < particleInstances.z; ++zInstance)
	//{
	//	for (int yInstance = 0; yInstance < particleInstances.y; ++yInstance)
	//	{
	//		for (int xInstance = 0; xInstance < particleInstances.x; ++xInstance)
	//		{
	//			auto modelOffset = glm::vec3(static_cast<float>(xInstance)* modelExtent.x, static_cast<float>(yInstance)* modelExtent.y, static_cast<float>(zInstance)* modelExtent.z);
	//			glUniform3fv(glGetUniformLocation(samplingProgramHandle, "modelOffset"), 1, &modelOffset[0]);

	for (int i = 0; i < particlesGLBufferArray.size(); i++)
	{
		particlesGLBufferArray[i].second.Render(GL_POINTS);
	}
	//		}
	//	}
	//}

#endif
#endif

	//experiment to get which particle I hit at each iteration
#if 0
	//create a vector of pairs <pixel index, indices of particles it sees>, call it 'pHits'
	//read sampling texture (read only porition specific to lod)
	//for each pixel compute the 1D index of the particle from the 4d index of hte color of the pixel
	//update 'pHits' of the pixel only if the index didn't show in the pixel before.

	//allocate pHits
	int lw, lh;
	LOD.get_lod_width_and_hight(current_lod, lw, lh);
	if (pHits.size() == 0)
	{
		for (int j = 0; j < lw*lh; j++)
		{
			pHits.push_back(std::vector<int>(0, 0));
		}
	}

	//read sampling texture
	float* fPixels = new float[4 * lw*lh];
	glm::ivec2 pIndx;
	glm::vec4 val;
	int particleIndx;
	int D1, D2, D3;
	D1 = D2 = D3 = 1000;
	bool flag;
	glm::ivec2 s = glm::ivec2(abs(.5f*rayCastingSolutionRenderTarget.Width - .5f*lw), abs(.5f*rayCastingSolutionRenderTarget.Height - .5f*lh));
	glReadPixels(0, 0, lw, lh, GL_RGBA, GL_FLOAT, fPixels);

	for (int i = 0; i < 4 * lw*lh; i = i + 4)
	{
		//get value of pixel
		val.x = fPixels[i] * D1;
		val.y = fPixels[i + 1] * D2;
		val.z = fPixels[i + 2] * D3;
		//val.w = fPixels[i + 3];

		//convert to a 1d index, ie the index of the particle hit
		particleIndx = val.x + val.y * D1 + val.z*D1*D2;

		if (particleIndx != 0)    //particleindx==0 if empyt pixel
		{
			//only add it if it was not hit before
			flag = false;
			for (int j = 0; j < pHits[i / 4].size(); j++)
			{
				if (pHits[i / 4][j] == particleIndx)
				{
					flag = true;
					break;
				}
			}

			if (!flag)
			{
				pHits[i / 4].push_back(particleIndx);
			}
		}
	}

	delete[] fPixels;
#endif

	if (save_SamplingTexture)
	{
		save_SamplingTexture = false;
		saveSamplingTexture();
	}
}

inline void drawHOM()
{
	// render particles directly during rotation
	sw = homHighestResolution.x;
	sh = homHighestResolution.y;

	// calculate global projection matrix
	auto leftOriginal = -0.5f;
	auto rightOriginal = 0.5f;
	auto bottomOriginal = -0.5f / aspectRatio;
	auto topOriginal = 0.5f / aspectRatio;


	OrigL = leftOriginal;
	OrigR = rightOriginal;
	OrigB = bottomOriginal;
	OrigT = topOriginal;
	//auto bottomOriginal = -0.5f;
	//auto topOriginal = 0.5f;

	glm::mat4x4 modelMatrix, T, Tinv, S;
	const auto modelScale = initialCameraDistance / cameraDistance;
	// NOTE: turn scaling off for rendering the histograms
	//const auto modelScale = 1.0f;

	//T[0][4] = -lastMouse.x;
	//T[1][4] = -lastMouse.y;

	//Tinv[0][4] = lastMouse.x;
	//Tinv[1][4] = lastMouse.y;

	S[0][0] = modelScale;
	S[1][1] = modelScale;
	S[2][2] = modelScale;

	modelMatrix = Tinv*S*T;



	//modelMatrix[1][1] *= aspectRatio;

	camPosiSampling = camPosi;
	auto viewMatrix = glm::lookAt(camPosiSampling, camTarget, camUp);


	auto modeViewMatrix = modelMatrix * viewMatrix;
	auto projectionMatrix = glm::ortho(leftOriginal, rightOriginal, bottomOriginal, topOriginal, nearPlane, farPlane);

	auto viewProjectionMatrix = projectionMatrix * viewMatrix;
	//samplingAspectRatio = aspectRatio;

	glm::vec3 tempCam = CameraPosition(cameraRotation, cameraDistance);
	viewMat_noPanning = glm::lookAt(tempCam, glm::vec3(0, 0, 0), camUp);
	modelviewMat_noPanning = modelMatrix*viewMat_noPanning;

	//Mohamed's Code

	//at fractional lods, i want the sampling to be done at the ceil resolution to avoid race conditions,
	//therefore, i will edit the modelviewmat for this case only

	if ((!plainRayCasting) && (!pointCloudRendering))
	{
		//bring camera to ceil lod position
		float tempCamDist = LOD.get_cam_dist(std::floor(current_lod - lodDelta));
		float tempmodelscale = initialCameraDistance / tempCamDist;

		//update camposisampling
		camPosiSampling = CameraPosition(cameraRotation, tempCamDist);
		camPosiSampling += cameraOffset;

		//edit view matrix
		samplingViewMat = glm::lookAt(camPosiSampling, camTarget, camUp);

		//edit projection matrix
		samplingProjectionMat = glm::ortho(sl, sr, sb, st, nearPlane, farPlane);

		//edit viewprojectionmatrix
		samplingViewProjectionMat = samplingProjectionMat*samplingViewMat;

		//edit model matrix
		samplingModelMat = modelMatrix;
		samplingModelMat[0][0] = tempmodelscale;
		samplingModelMat[1][1] = tempmodelscale;
		samplingModelMat[2][2] = tempmodelscale;

		//glm::vec2 f = glm::vec2(sl + ((sr - sl) / 2.0f), sb + ((st - sb) / 2.0f));

		//T[0][4] = -f.x;
		//T[1][4] = -f.y;

		//Tinv[0][4] = f.x;
		//Tinv[1][4] = f.y;

		//samplingModelMat = Tinv*samplingModelMat*T;

		//edit model view matrix
		samplingModelViewMat = samplingModelMat*samplingViewMat;

		samplingAspectRatio = sw / sh;

		glm::vec3 tempCam = CameraPosition(cameraRotation, tempCamDist);
		samplingViewMat_noPanning = glm::lookAt(tempCam, glm::vec3(0, 0, 0), camUp);
		samplingModelViewMat_noPanning = samplingModelMat*samplingViewMat_noPanning;
	}
	else
	{
		//edit projection matrix
		samplingProjectionMat = projectionMatrix;
		samplingViewMat = viewMatrix;
		samplingViewProjectionMat = viewProjectionMatrix;
		samplingModelViewMat = modeViewMatrix;
		samplingModelMat = modelMatrix;
		samplingAspectRatio = aspectRatio;

		samplingViewMat_noPanning = viewMat_noPanning;
		samplingModelViewMat_noPanning = modelviewMat_noPanning;

		sw = windowSize.x;
		sh = windowSize.y;
		samplingAspectRatio = sw / sh;
	}




	modelviewMat = modeViewMatrix;
	projectionMat = projectionMatrix;
	//viewportMat = glm::vec4(0, 0,std::max(rayCastingSolutionRenderTarget.Width,windowSize.x),std::max( rayCastingSolutionRenderTarget.Height,windowSize.y));
	viewportMat = glm::vec4(0, 0, windowSize.x, windowSize.y);
	viewMat = viewMatrix;
	modelMat = modelMatrix;
	viewprojectionMat = viewProjectionMatrix;
	mynear = nearPlane;
	myfar = farPlane;
	myup = camUp;
	mytarget = camTarget;
	//end Mohamed's Code



	
	glBindFramebuffer(GL_FRAMEBUFFER, homRenderTarget.FrameBufferObject);
	glViewport(0, 0, sw, sh);




	//glBindFramebuffer(GL_FRAMEBUFFER, MyRenderTarget.FrameBufferObject);
	//glViewport(0, 0, MyRenderTarget.Width, MyRenderTarget.Height);


	//glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//glViewport(0, 0, windowSize.x, windowSize.y);

	// FIXME: viewport causes tears for some reason if sampling rate < 1024
	glClearColor(0.f, 0.f, 0.f, 1.0f);
	// clear render target
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	//glEnable(GL_POINT_SPRITE);
	//glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_DEPTH_TEST);
	glFrontFace(GL_CCW);

	//GLuint ssboBindingPointIndex = 2;
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, samplingShader->getSsboBiningIndex(myArrayUBO), myArrayUBO);


	GLuint samplingProgramHandle;

	//glPointSize(particleRadius);

	//if (pointCloudRendering)
	//	samplingProgramHandle = samplingShader_PointCloud->GetGlShaderProgramHandle();
	//else
	samplingProgramHandle = drawHOMShader->GetGlShaderProgramHandle();

	assert(glUseProgram);
	glUseProgram(samplingProgramHandle);

	glUniform1f(glGetUniformLocation(samplingProgramHandle, "far"), farPlane);
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "near"), nearPlane);
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "particleScale"), particleScale);
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "aspectRatio"), samplingAspectRatio);

	//get how much object space is represented in one pixel
	float objPerPixel;
	{
		glm::vec2 s, e;
		s = LOD.pixel2obj(glm::vec2(0, 0), std::floor(current_lod - lodDelta));
		e = LOD.pixel2obj(glm::vec2(1, 1), std::floor(current_lod - lodDelta));

		//get difference
		glm::vec2 v = e - s;
		objPerPixel = v.x;
	}
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "objPerPixel"), objPerPixel);

	glUniform1i(glGetUniformLocation(samplingProgramHandle, "buildingHOM"), buildingHOM);
	glUniform1i(glGetUniformLocation(samplingProgramHandle, "maxSamplingRuns"), tempmaxSamplingRuns);

	glUniform1f(glGetUniformLocation(samplingProgramHandle, "viewportWidth"), static_cast<float>(sw));
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "viewportHeight"), static_cast<float>(sh));
	glUniform1i(glGetUniformLocation(samplingProgramHandle, "plainRayCasting"), plainRayCasting);
	glUniform1i(glGetUniformLocation(samplingProgramHandle, "pointCloudRendering"), pointCloudRendering);


	glUniform1i(glGetUniformLocation(samplingProgramHandle, "samplescount"), sample_count);
	glUniform2fv(glGetUniformLocation(samplingProgramHandle, "sPos"), 1, &sPos[0]);


	glUniform1f(glGetUniformLocation(samplingProgramHandle, "maxZ"), modelMax.z);
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "minZ"), modelMin.z);

	auto right = glm::normalize(glm::cross(normalize(camTarget - camPosi), camUp));
	glUniform3fv(glGetUniformLocation(samplingProgramHandle, "right"), 1, &right[0]);
	glUniform3fv(glGetUniformLocation(samplingProgramHandle, "up"), 1, &camUp[0]);



	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "Model"), 1, GL_FALSE, &samplingModelMat[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "ModelView"), 1, GL_FALSE, &samplingModelViewMat[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "View"), 1, GL_FALSE, &samplingViewMat[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "Projection"), 1, GL_FALSE, &samplingProjectionMat[0][0]);
	glUniformMatrix3fv(glGetUniformLocation(samplingProgramHandle, "RotationMatrix"), 1, GL_FALSE, &GlobalRotationMatrix[0][0]);


	glm::mat4 ViewAlignmentMatrix;
	ViewAlignmentMatrix = glm::rotate(ViewAlignmentMatrix, cameraRotation.x, glm::vec3(0.0f, 1.0f, 0.0f));
	ViewAlignmentMatrix = glm::rotate(ViewAlignmentMatrix, cameraRotation.y, glm::vec3(1.0f, 0.0f, 0.0f));

	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "ViewAlignmentMatrix"), 1, GL_FALSE, &ViewAlignmentMatrix[0][0]);

	glUniform3fv(glGetUniformLocation(samplingProgramHandle, "ViewPosition"), 1, &camPosiSampling[0]);
	glUniform2i(glGetUniformLocation(samplingProgramHandle, "viewSlice"), 0, 0);

	// render particles using ray casting
	// set uniforms
	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "ViewProjection"), 1, GL_FALSE, &samplingViewProjectionMat[0][0]);

	auto initialOffset = glm::vec3(0.0f, 0.0f, 0.0f);

	// FIXME: fix magic numbers
	//if (particleInstances.x > 1) {
	//	initialOffset.x = -0.125f * static_cast<float>(particleInstances.x) * modelExtent.x;
	//}
	//if (particleInstances.y > 1) {
	//	initialOffset.y = -0.3f * static_cast<float>(particleInstances.y) * modelExtent.y;
	//}
	//if (particleInstances.z > 1) {
	//	initialOffset.z = -0.125f * static_cast<float>(particleInstances.z) * modelExtent.z;
	//}

	// render samples
#if 0  //no bricking
	for (int zInstance = 0; zInstance < particleInstances.z; ++zInstance)
	{
		for (int yInstance = 0; yInstance < particleInstances.y; ++yInstance)
		{
			for (int xInstance = 0; xInstance < particleInstances.x; ++xInstance)
			{
				auto modelOffset = initialOffset + glm::vec3(static_cast<float>(xInstance)* modelExtent.x, static_cast<float>(yInstance)* modelExtent.y, static_cast<float>(zInstance)* modelExtent.z);

				glUniform3fv(glGetUniformLocation(samplingProgramHandle, "modelOffset"), 1, &modelOffset[0]);
				particlesGlBuffer.Render(GL_POINTS);
				//particlesGlBuffer.Render_Selective(GL_POINTS, visible_spheres, visible_spheres_count);
				//int i[] = {0,1};
				//particlesGlBuffer.Render_Selective(GL_POINTS, i, 2);
			}
		}
	}
#else       
#if 1  //naive bricking
	//here we assume the bricks in 'global_tree_leaves' are sorted from back to front
	//if the depth buffer is not reset each time I render a brick, i don't need to sort, should do this!!!
	if (global_tree.leaves.size() > 1)
	{
#if 0
		//initially put data in buffer of index '1'
		bufferId = 0;
		glBindBuffer(GL_ARRAY_BUFFER, particlesGLBufferArray[1 - bufferId].Vbo_);
		GLvoid* p = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		memcpy(p, reinterpret_cast<char*>(&global_tree.nodes[global_tree.leaves[0]].Points[0]), global_tree.nodes[global_tree.leaves[0]].Points.size() * sizeof(*global_tree.nodes[global_tree.leaves[0]].Points.begin()));
		glUnmapBuffer(GL_ARRAY_BUFFER);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		for (int i = 1; i <= global_tree.leaves.size(); i++)
		{
			if (i < global_tree.leaves.size())
			{
				//1-copy data in one buffer (either particlesglbuffer or particlesglbuffer_back)
				glBindBuffer(GL_ARRAY_BUFFER, particlesGLBufferArray[bufferId].Vbo_);
				GLvoid* p = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
				memcpy(p, reinterpret_cast<char*>(&global_tree.nodes[global_tree.leaves[i]].Points[0]), global_tree.nodes[global_tree.leaves[i]].Points.size() * sizeof(*global_tree.nodes[global_tree.leaves[i]].Points.begin()));
				glUnmapBuffer(GL_ARRAY_BUFFER);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
			}

			//2- and render the other buffer
			particlesGLBufferArray[1 - bufferId].VertexCount_ = global_tree.nodes[global_tree.leaves[i - 1]].Points.size();
			particlesGLBufferArray[1 - bufferId].Render(GL_POINTS);

			//3- switch buffer id
			bufferId = 1 - bufferId;
		}
#else

		int s;
		for (int i = 0; i < renderBricks.size(); i++)
		{
			//copy first brick to buffer 1
			bufferId = 0;
			glBindBuffer(GL_ARRAY_BUFFER, particlesGLBufferArray[1 - bufferId].Vbo_);
			GLvoid* p = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
			memcpy(p, reinterpret_cast<char*>(&global_tree.nodes[renderBricks[i].first].Points[0]), renderBricks[i].second[0] * sizeof(*global_tree.nodes[global_tree.leaves[0]].Points.begin()));
			glUnmapBuffer(GL_ARRAY_BUFFER);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			for (int j = 1; j <= renderBricks[i].second.size(); j++)
			{
				if (j < renderBricks[i].second.size())
				{
					s = std::accumulate(renderBricks[i].second.begin(), renderBricks[i].second.begin() + j, 0);

					//1-copy brick in one buffer
					glBindBuffer(GL_ARRAY_BUFFER, particlesGLBufferArray[bufferId].Vbo_);
					GLvoid* p = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
					memcpy(p, reinterpret_cast<char*>(&global_tree.nodes[renderBricks[i].first].Points[s]), renderBricks[i].second[j] * sizeof(*global_tree.nodes[global_tree.leaves[0]].Points.begin()));
					glUnmapBuffer(GL_ARRAY_BUFFER);
					glBindBuffer(GL_ARRAY_BUFFER, 0);
				}

				//2- and render the other buffer
				particlesGLBufferArray[1 - bufferId].VertexCount_ = renderBricks[i].second[j - 1];
				particlesGLBufferArray[1 - bufferId].Render(GL_POINTS);

				//3- switch buffer id
				bufferId = 1 - bufferId;
			}
		}
#endif
	}
	else
	{
		//initially put data in buffer of index '1'
		bufferId = 0;
		glBindBuffer(GL_ARRAY_BUFFER, particlesGLBufferArray[1 - bufferId].Vbo_);
		GLvoid* p = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		memcpy(p, reinterpret_cast<char*>(&global_tree.nodes[global_tree.leaves[0]].Points[0]), global_tree.nodes[global_tree.leaves[0]].Points.size() * sizeof(*global_tree.nodes[global_tree.leaves[0]].Points.begin()));
		glUnmapBuffer(GL_ARRAY_BUFFER);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		particlesGLBufferArray[1 - bufferId].VertexCount_ = global_tree.nodes[global_tree.leaves[0]].Points.size();
		particlesGLBufferArray[1 - bufferId].Render(GL_POINTS);
	}
#else  //more efficient bricking
	//Helpers::Gl::MakeBuffer(particleCenters, global_tree, particlesGLBufferArray);
	//for (int zInstance = 0; zInstance < particleInstances.z; ++zInstance)
	//{
	//	for (int yInstance = 0; yInstance < particleInstances.y; ++yInstance)
	//	{
	//		for (int xInstance = 0; xInstance < particleInstances.x; ++xInstance)
	//		{
	//			auto modelOffset = glm::vec3(static_cast<float>(xInstance)* modelExtent.x, static_cast<float>(yInstance)* modelExtent.y, static_cast<float>(zInstance)* modelExtent.z);
	//			glUniform3fv(glGetUniformLocation(samplingProgramHandle, "modelOffset"), 1, &modelOffset[0]);

	for (int i = 0; i < particlesGLBufferArray.size(); i++)
	{
		particlesGLBufferArray[i].second.Render(GL_POINTS);
	}
	//		}
	//	}
	//}

#endif
#endif

	//experiment to get which particle I hit at each iteration
#if 0
	//create a vector of pairs <pixel index, indices of particles it sees>, call it 'pHits'
	//read sampling texture (read only porition specific to lod)
	//for each pixel compute the 1D index of the particle from the 4d index of hte color of the pixel
	//update 'pHits' of the pixel only if the index didn't show in the pixel before.

	//allocate pHits
	int lw, lh;
	LOD.get_lod_width_and_hight(current_lod, lw, lh);
	if (pHits.size() == 0)
	{
		for (int j = 0; j < lw*lh; j++)
		{
			pHits.push_back(std::vector<int>(0, 0));
		}
	}

	//read sampling texture
	float* fPixels = new float[4 * lw*lh];
	glm::ivec2 pIndx;
	glm::vec4 val;
	int particleIndx;
	int D1, D2, D3;
	D1 = D2 = D3 = 1000;
	bool flag;
	glm::ivec2 s = glm::ivec2(abs(.5f*rayCastingSolutionRenderTarget.Width - .5f*lw), abs(.5f*rayCastingSolutionRenderTarget.Height - .5f*lh));
	glReadPixels(0, 0, lw, lh, GL_RGBA, GL_FLOAT, fPixels);

	for (int i = 0; i < 4 * lw*lh; i = i + 4)
	{
		//get value of pixel
		val.x = fPixels[i] * D1;
		val.y = fPixels[i + 1] * D2;
		val.z = fPixels[i + 2] * D3;
		//val.w = fPixels[i + 3];

		//convert to a 1d index, ie the index of the particle hit
		particleIndx = val.x + val.y * D1 + val.z*D1*D2;

		if (particleIndx != 0)    //particleindx==0 if empyt pixel
		{
			//only add it if it was not hit before
			flag = false;
			for (int j = 0; j < pHits[i / 4].size(); j++)
			{
				if (pHits[i / 4][j] == particleIndx)
				{
					flag = true;
					break;
				}
			}

			if (!flag)
			{
				pHits[i / 4].push_back(particleIndx);
			}
		}
	}

	delete[] fPixels;
#endif

	if (save_SamplingTexture)
	{
		save_SamplingTexture = false;
		saveSamplingTexture();
	}
}
inline void sampleData()
{
	// calculate global projection matrix
	auto leftOriginal = -0.5f;
	auto rightOriginal = 0.5f;
	auto bottomOriginal = -0.5f / aspectRatio;
	auto topOriginal = 0.5f / aspectRatio;


	OrigL = leftOriginal;
	OrigR = rightOriginal;
	OrigB = bottomOriginal;
	OrigT = topOriginal;
	//auto bottomOriginal = -0.5f;
	//auto topOriginal = 0.5f;

	glm::mat4x4 modelMatrix, T, Tinv, S;
	const auto modelScale = initialCameraDistance / cameraDistance;
	// NOTE: turn scaling off for rendering the histograms
	//const auto modelScale = 1.0f;

	//T[0][4] = -lastMouse.x;
	//T[1][4] = -lastMouse.y;

	//Tinv[0][4] = lastMouse.x;
	//Tinv[1][4] = lastMouse.y;

	S[0][0] = modelScale;
	S[1][1] = modelScale;
	S[2][2] = modelScale;

	modelMatrix = Tinv*S*T;



	//modelMatrix[1][1] *= aspectRatio;

	camPosiSampling = camPosi;
	auto viewMatrix = glm::lookAt(camPosiSampling, camTarget, camUp);


	auto modeViewMatrix = modelMatrix * viewMatrix;
	auto projectionMatrix = glm::ortho(leftOriginal, rightOriginal, bottomOriginal, topOriginal, nearPlane, farPlane);

	auto viewProjectionMatrix = projectionMatrix * viewMatrix;
	//samplingAspectRatio = aspectRatio;

	glm::vec3 tempCam = CameraPosition(cameraRotation, cameraDistance);
	viewMat_noPanning = glm::lookAt(tempCam, glm::vec3(0, 0, 0), camUp);
	modelviewMat_noPanning = modelMatrix*viewMat_noPanning;

	//Mohamed's Code

	//at fractional lods, i want the sampling to be done at the ceil resolution to avoid race conditions,
	//therefore, i will edit the modelviewmat for this case only

	if ((!plainRayCasting) && (!pointCloudRendering))
	{
		//bring camera to ceil lod position
		float tempCamDist = LOD.get_cam_dist(std::floor(current_lod-lodDelta));
		float tempmodelscale = initialCameraDistance / tempCamDist;

		//update camposisampling
		camPosiSampling = CameraPosition(cameraRotation, tempCamDist);
		camPosiSampling += cameraOffset;

		//edit view matrix
		samplingViewMat = glm::lookAt(camPosiSampling, camTarget, camUp);

		//edit projection matrix
		samplingProjectionMat = glm::ortho(sl, sr, sb, st, nearPlane, farPlane);

		//edit viewprojectionmatrix
		samplingViewProjectionMat = samplingProjectionMat*samplingViewMat;

		//edit model matrix
		samplingModelMat = modelMatrix;
		samplingModelMat[0][0] = tempmodelscale;
		samplingModelMat[1][1] = tempmodelscale;
		samplingModelMat[2][2] = tempmodelscale;

		//glm::vec2 f = glm::vec2(sl + ((sr - sl) / 2.0f), sb + ((st - sb) / 2.0f));

		//T[0][4] = -f.x;
		//T[1][4] = -f.y;

		//Tinv[0][4] = f.x;
		//Tinv[1][4] = f.y;

		//samplingModelMat = Tinv*samplingModelMat*T;

		//edit model view matrix
		samplingModelViewMat = samplingModelMat*samplingViewMat;

		samplingAspectRatio = sw / sh;

		glm::vec3 tempCam = CameraPosition(cameraRotation, tempCamDist);
		samplingViewMat_noPanning = glm::lookAt(tempCam, glm::vec3(0, 0, 0), camUp);
		samplingModelViewMat_noPanning = samplingModelMat*samplingViewMat_noPanning;
	}
	else
	{
		//edit projection matrix
		samplingProjectionMat = projectionMatrix;
		samplingViewMat = viewMatrix;
		samplingViewProjectionMat = viewProjectionMatrix;
		samplingModelViewMat = modeViewMatrix;
		samplingModelMat = modelMatrix;
		samplingAspectRatio = aspectRatio;

		samplingViewMat_noPanning = viewMat_noPanning;
		samplingModelViewMat_noPanning = modelviewMat_noPanning;

		sw = windowSize.x;
		sh = windowSize.y;
		samplingAspectRatio = sw / sh;
	}




	modelviewMat = modeViewMatrix;
	projectionMat = projectionMatrix;
	//viewportMat = glm::vec4(0, 0,std::max(rayCastingSolutionRenderTarget.Width,windowSize.x),std::max( rayCastingSolutionRenderTarget.Height,windowSize.y));
	viewportMat = glm::vec4(0, 0, windowSize.x, windowSize.y);
	viewMat = viewMatrix;
	modelMat = modelMatrix;
	viewprojectionMat = viewProjectionMatrix;
	mynear = nearPlane;
	myfar = farPlane;
	myup = camUp;
	mytarget = camTarget;
	//end Mohamed's Code




	glBindFramebuffer(GL_FRAMEBUFFER, rayCastingSolutionRenderTarget.FrameBufferObject);
	glViewport(0, 0, sw, sh);




	//glBindFramebuffer(GL_FRAMEBUFFER, MyRenderTarget.FrameBufferObject);
	//glViewport(0, 0, MyRenderTarget.Width, MyRenderTarget.Height);


	//glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//glViewport(0, 0, windowSize.x, windowSize.y);

	// FIXME: viewport causes tears for some reason if sampling rate < 1024
	glClearColor(0.f, 0.f, 0.f, 1.0f);
	// clear render target
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	//glEnable(GL_POINT_SPRITE);
	//glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_DEPTH_TEST);
	glFrontFace(GL_CCW);

	//GLuint ssboBindingPointIndex = 2;
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, samplingShader->getSsboBiningIndex(myArrayUBO), myArrayUBO);


	GLuint samplingProgramHandle;

	//glPointSize(particleRadius);

	//if (pointCloudRendering)
	//	samplingProgramHandle = samplingShader_PointCloud->GetGlShaderProgramHandle();
	//else
	samplingProgramHandle = samplingShader->GetGlShaderProgramHandle();

	assert(glUseProgram);
	glUseProgram(samplingProgramHandle);

	glUniform1f(glGetUniformLocation(samplingProgramHandle, "far"), farPlane);
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "near"), nearPlane);
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "particleScale"), particleScale);
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "aspectRatio"), samplingAspectRatio);

	//get how much object space is represented in one pixel
	float objPerPixel;
	{
		glm::vec2 s, e;
		s = LOD.pixel2obj(glm::vec2(0, 0), std::floor(current_lod-lodDelta));
		e = LOD.pixel2obj(glm::vec2(1, 1), std::floor(current_lod-lodDelta));

		//get difference
		glm::vec2 v = e - s;
		objPerPixel = v.x;
	}
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "objPerPixel"), objPerPixel);
	glUniform1i(glGetUniformLocation(samplingProgramHandle, "streamingFlag"), streaming);
	glUniform1i(glGetUniformLocation(samplingProgramHandle, "buildingHOM"), buildingHOM);
	glUniform1i(glGetUniformLocation(samplingProgramHandle, "maxSamplingRuns"), tempmaxSamplingRuns);

	glUniform1f(glGetUniformLocation(samplingProgramHandle, "viewportWidth"), static_cast<float>(sw));
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "viewportHeight"), static_cast<float>(sh));
	glUniform1i(glGetUniformLocation(samplingProgramHandle, "plainRayCasting"), plainRayCasting);
	glUniform1i(glGetUniformLocation(samplingProgramHandle, "pointCloudRendering"), pointCloudRendering);


	glUniform1i(glGetUniformLocation(samplingProgramHandle, "samplescount"), sample_count);
	glUniform2fv(glGetUniformLocation(samplingProgramHandle, "sPos"), 1, &sPos[0]);


	glUniform1f(glGetUniformLocation(samplingProgramHandle, "maxZ"), modelMax.z);
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "minZ"), modelMin.z);

	auto right = glm::normalize(glm::cross(normalize(camTarget - camPosi), camUp));
	glUniform3fv(glGetUniformLocation(samplingProgramHandle, "right"), 1, &right[0]);
	glUniform3fv(glGetUniformLocation(samplingProgramHandle, "up"), 1, &camUp[0]);



	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "Model"), 1, GL_FALSE, &samplingModelMat[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "ModelView"), 1, GL_FALSE, &samplingModelViewMat[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "View"), 1, GL_FALSE, &samplingViewMat[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "Projection"), 1, GL_FALSE, &samplingProjectionMat[0][0]);
	glUniformMatrix3fv(glGetUniformLocation(samplingProgramHandle, "RotationMatrix"), 1, GL_FALSE, &GlobalRotationMatrix[0][0]);


	glm::mat4 ViewAlignmentMatrix;
	ViewAlignmentMatrix = glm::rotate(ViewAlignmentMatrix, cameraRotation.x, glm::vec3(0.0f, 1.0f, 0.0f));
	ViewAlignmentMatrix = glm::rotate(ViewAlignmentMatrix, cameraRotation.y, glm::vec3(1.0f, 0.0f, 0.0f));

	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "ViewAlignmentMatrix"), 1, GL_FALSE, &ViewAlignmentMatrix[0][0]);

	glUniform3fv(glGetUniformLocation(samplingProgramHandle, "ViewPosition"), 1, &camPosiSampling[0]);
	glUniform2i(glGetUniformLocation(samplingProgramHandle, "viewSlice"), 0, 0);

	// render particles using ray casting
	// set uniforms
	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "ViewProjection"), 1, GL_FALSE, &samplingViewProjectionMat[0][0]);

	auto initialOffset = glm::vec3(0.0f, 0.0f, 0.0f);

	// FIXME: fix magic numbers
	//if (particleInstances.x > 1) {
	//	initialOffset.x = -0.125f * static_cast<float>(particleInstances.x) * modelExtent.x;
	//}
	//if (particleInstances.y > 1) {
	//	initialOffset.y = -0.3f * static_cast<float>(particleInstances.y) * modelExtent.y;
	//}
	//if (particleInstances.z > 1) {
	//	initialOffset.z = -0.125f * static_cast<float>(particleInstances.z) * modelExtent.z;
	//}

	// render samples
#ifdef NO_BRICKING   //no bricking
	//for (int zInstance = 0; zInstance < particleInstances.z; ++zInstance)
	//{
	//	for (int yInstance = 0; yInstance < particleInstances.y; ++yInstance)
	//	{
	//		for (int xInstance = 0; xInstance < particleInstances.x; ++xInstance)
	//		{
	//			auto modelOffset = initialOffset + glm::vec3(static_cast<float>(xInstance)* modelExtent.x, static_cast<float>(yInstance)* modelExtent.y, static_cast<float>(zInstance)* modelExtent.z);

	//			glUniform3fv(glGetUniformLocation(samplingProgramHandle, "modelOffset"), 1, &modelOffset[0]);
				particlesGlBuffer.Render(GL_POINTS);
				//particlesGlBuffer.Render_Selective(GL_POINTS, visible_spheres, visible_spheres_count);
				//int i[] = {0,1};
				//particlesGlBuffer.Render_Selective(GL_POINTS, i, 2);
	//		}
	//	}
	//}
#else       
#if 1  //naive bricking
	//here we assume the bricks in 'global_tree_leaves' are sorted from back to front
	//if the depth buffer is not reset each time I render a brick, i don't need to sort, should do this!!!
	if (global_tree.leaves.size() > 1)
	{
#if 0
		//initially put data in buffer of index '1'
		bufferId = 0;
		glBindBuffer(GL_ARRAY_BUFFER, particlesGLBufferArray[1 - bufferId].Vbo_);
		GLvoid* p = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
		memcpy(p, reinterpret_cast<char*>(&global_tree.nodes[global_tree.leaves[0]].Points[0]), global_tree.nodes[global_tree.leaves[0]].Points.size() * sizeof(*global_tree.nodes[global_tree.leaves[0]].Points.begin()));
		glUnmapBuffer(GL_ARRAY_BUFFER);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		for (int i = 1; i <= global_tree.leaves.size(); i++)
		{
			if (i < global_tree.leaves.size())
			{
				//1-copy data in one buffer (either particlesglbuffer or particlesglbuffer_back)
				glBindBuffer(GL_ARRAY_BUFFER, particlesGLBufferArray[bufferId].Vbo_);
				GLvoid* p = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
				memcpy(p, reinterpret_cast<char*>(&global_tree.nodes[global_tree.leaves[i]].Points[0]), global_tree.nodes[global_tree.leaves[i]].Points.size() * sizeof(*global_tree.nodes[global_tree.leaves[i]].Points.begin()));
				glUnmapBuffer(GL_ARRAY_BUFFER);
				glBindBuffer(GL_ARRAY_BUFFER, 0);
			}

			//2- and render the other buffer
			particlesGLBufferArray[1-bufferId].VertexCount_ = global_tree.nodes[global_tree.leaves[i-1]].Points.size();
			particlesGLBufferArray[1-bufferId].Render(GL_POINTS);

			//3- switch buffer id
			bufferId=1-bufferId;
		}
#else

		int s;
		for (int i = 0; i < renderBricks.size(); i++)
		{
			//check if renderBricks[i].first is found in cells to render
			if (renderCell(renderBricks[i].first))
			{

				//copy first brick to buffer 1
				bufferId = 0;
				glBindBuffer(GL_ARRAY_BUFFER, particlesGLBufferArray[1 - bufferId].Vbo_);
				GLvoid* p = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
				memcpy(p, reinterpret_cast<char*>(&global_tree.nodes[renderBricks[i].first].Points[0]), renderBricks[i].second[0] * sizeof(*global_tree.nodes[global_tree.leaves[0]].Points.begin()));
				glUnmapBuffer(GL_ARRAY_BUFFER);
				glBindBuffer(GL_ARRAY_BUFFER, 0);

				for (int j = 1; j <= renderBricks[i].second.size(); j++)
				{
					if (j < renderBricks[i].second.size())
					{
						s = std::accumulate(renderBricks[i].second.begin(), renderBricks[i].second.begin() + j, 0);

						//1-copy brick in one buffer
						glBindBuffer(GL_ARRAY_BUFFER, particlesGLBufferArray[bufferId].Vbo_);
						GLvoid* p = glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
						memcpy(p, reinterpret_cast<char*>(&global_tree.nodes[renderBricks[i].first].Points[s]), renderBricks[i].second[j] * sizeof(*global_tree.nodes[global_tree.leaves[0]].Points.begin()));
						glUnmapBuffer(GL_ARRAY_BUFFER);
						glBindBuffer(GL_ARRAY_BUFFER, 0);
					}

					//2- and render the other buffer
					particlesGLBufferArray[1 - bufferId].VertexCount_ = renderBricks[i].second[j - 1];
					particlesGLBufferArray[1 - bufferId].Render(GL_POINTS);

					//3- switch buffer id
					bufferId = 1 - bufferId;
				}
			}
		}
#endif
	}
	else
	{
		
		particlesGLBufferArray[0].Render(GL_POINTS);
	}
#else  //more efficient bricking
	//Helpers::Gl::MakeBuffer(particleCenters, global_tree, particlesGLBufferArray);
	//for (int zInstance = 0; zInstance < particleInstances.z; ++zInstance)
	//{
	//	for (int yInstance = 0; yInstance < particleInstances.y; ++yInstance)
	//	{
	//		for (int xInstance = 0; xInstance < particleInstances.x; ++xInstance)
	//		{
	//			auto modelOffset = glm::vec3(static_cast<float>(xInstance)* modelExtent.x, static_cast<float>(yInstance)* modelExtent.y, static_cast<float>(zInstance)* modelExtent.z);
	//			glUniform3fv(glGetUniformLocation(samplingProgramHandle, "modelOffset"), 1, &modelOffset[0]);

				for (int i = 0; i < particlesGLBufferArray.size(); i++)
				{
					particlesGLBufferArray[i].second.Render(GL_POINTS);
				}
	//		}
	//	}
	//}

#endif
#endif

	//experiment to get which particle I hit at each iteration
#if 0
	//create a vector of pairs <pixel index, indices of particles it sees>, call it 'pHits'
	//read sampling texture (read only porition specific to lod)
	//for each pixel compute the 1D index of the particle from the 4d index of hte color of the pixel
	//update 'pHits' of the pixel only if the index didn't show in the pixel before.

	//allocate pHits
	int lw, lh;
	LOD.get_lod_width_and_hight(current_lod, lw, lh);
	if (pHits.size() == 0)
	{
		for (int j = 0; j < lw*lh; j++)
		{
			pHits.push_back(std::vector<int>(0, 0));
		}
	}

	//read sampling texture
	float* fPixels = new float[4 * lw*lh];
	glm::ivec2 pIndx;
	glm::vec4 val;
	int particleIndx;
	int D1, D2, D3;
	D1 = D2 = D3 = 1000;
	bool flag;
	glm::ivec2 s = glm::ivec2(abs(.5f*rayCastingSolutionRenderTarget.Width - .5f*lw), abs(.5f*rayCastingSolutionRenderTarget.Height - .5f*lh));
	glReadPixels(0, 0, lw, lh, GL_RGBA, GL_FLOAT, fPixels);

	for (int i = 0; i < 4 * lw*lh; i = i + 4)
	{
		//get value of pixel
		val.x = fPixels[i]*D1;
		val.y = fPixels[i + 1]*D2;
		val.z = fPixels[i + 2]*D3;
		//val.w = fPixels[i + 3];

		//convert to a 1d index, ie the index of the particle hit
		particleIndx = val.x + val.y * D1+val.z*D1*D2;

		if (particleIndx != 0)    //particleindx==0 if empyt pixel
		{
			//only add it if it was not hit before
			flag = false;
			for (int j = 0; j < pHits[i / 4].size(); j++)
			{
				if (pHits[i / 4][j] == particleIndx)
				{
					flag = true;
					break;
				}
			}

			if (!flag)
			{
				pHits[i / 4].push_back(particleIndx);
			}
		}
	}

	delete[] fPixels;
#endif

	if (save_SamplingTexture)
	{
		save_SamplingTexture = false;
		saveSamplingTexture();
	}
}

void queueSignal(GLsync& syncObj) {
    if (syncObj) {
        glDeleteSync(syncObj);
    }
    syncObj = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}

void waitSignal(GLsync& syncObj) {
    if (syncObj) {
        while (1) {
            GLenum wait = glClientWaitSync(syncObj, GL_SYNC_FLUSH_COMMANDS_BIT, 1);
            if (wait == GL_ALREADY_SIGNALED || wait == GL_CONDITION_SATISFIED) {
                return;
            }
        }
    }
}

inline void sampleDataStreaming() {
    // calculate global projection matrix
    auto leftOriginal = -0.5f;
    auto rightOriginal = 0.5f;
    auto bottomOriginal = -0.5f / aspectRatio;
    auto topOriginal = 0.5f / aspectRatio;


    OrigL = leftOriginal;
    OrigR = rightOriginal;
    OrigB = bottomOriginal;
    OrigT = topOriginal;
    //auto bottomOriginal = -0.5f;
    //auto topOriginal = 0.5f;

    glm::mat4x4 modelMatrix, T, Tinv, S;
    const auto modelScale = initialCameraDistance / cameraDistance;
    // NOTE: turn scaling off for rendering the histograms
    //const auto modelScale = 1.0f;

    //T[0][4] = -lastMouse.x;
    //T[1][4] = -lastMouse.y;

    //Tinv[0][4] = lastMouse.x;
    //Tinv[1][4] = lastMouse.y;

    S[0][0] = modelScale;
    S[1][1] = modelScale;
    S[2][2] = modelScale;

    modelMatrix = Tinv*S*T;



    //modelMatrix[1][1] *= aspectRatio;

    camPosiSampling = camPosi;
    auto viewMatrix = glm::lookAt(camPosiSampling, camTarget, camUp);


    auto modeViewMatrix = modelMatrix * viewMatrix;
    auto projectionMatrix = glm::ortho(leftOriginal, rightOriginal, bottomOriginal, topOriginal, nearPlane, farPlane);

    auto viewProjectionMatrix = projectionMatrix * viewMatrix;
    //samplingAspectRatio = aspectRatio;

    glm::vec3 tempCam = CameraPosition(cameraRotation, cameraDistance);
    viewMat_noPanning = glm::lookAt(tempCam, glm::vec3(0, 0, 0), camUp);
    modelviewMat_noPanning = modelMatrix*viewMat_noPanning;

    //Mohamed's Code

    //at fractional lods, i want the sampling to be done at the ceil resolution to avoid race conditions,
    //therefore, i will edit the modelviewmat for this case only

    if ((!plainRayCasting) && (!pointCloudRendering)) {
        //bring camera to ceil lod position
        float tempCamDist = LOD.get_cam_dist(std::floor(current_lod - lodDelta));
        float tempmodelscale = initialCameraDistance / tempCamDist;

        //update camposisampling
        camPosiSampling = CameraPosition(cameraRotation, tempCamDist);
        camPosiSampling += cameraOffset;

        //edit view matrix
        samplingViewMat = glm::lookAt(camPosiSampling, camTarget, camUp);

        //edit projection matrix
        samplingProjectionMat = glm::ortho(sl, sr, sb, st, nearPlane, farPlane);

        //edit viewprojectionmatrix
        samplingViewProjectionMat = samplingProjectionMat*samplingViewMat;

        //edit model matrix
        samplingModelMat = modelMatrix;
        samplingModelMat[0][0] = tempmodelscale;
        samplingModelMat[1][1] = tempmodelscale;
        samplingModelMat[2][2] = tempmodelscale;

        //glm::vec2 f = glm::vec2(sl + ((sr - sl) / 2.0f), sb + ((st - sb) / 2.0f));

        //T[0][4] = -f.x;
        //T[1][4] = -f.y;

        //Tinv[0][4] = f.x;
        //Tinv[1][4] = f.y;

        //samplingModelMat = Tinv*samplingModelMat*T;

        //edit model view matrix
        samplingModelViewMat = samplingModelMat*samplingViewMat;

        samplingAspectRatio = sw / sh;

        glm::vec3 tempCam = CameraPosition(cameraRotation, tempCamDist);
        samplingViewMat_noPanning = glm::lookAt(tempCam, glm::vec3(0, 0, 0), camUp);
        samplingModelViewMat_noPanning = samplingModelMat*samplingViewMat_noPanning;
    } else {
        //edit projection matrix
        samplingProjectionMat = projectionMatrix;
        samplingViewMat = viewMatrix;
        samplingViewProjectionMat = viewProjectionMatrix;
        samplingModelViewMat = modeViewMatrix;
        samplingModelMat = modelMatrix;
        samplingAspectRatio = aspectRatio;

        samplingViewMat_noPanning = viewMat_noPanning;
        samplingModelViewMat_noPanning = modelviewMat_noPanning;

        sw = windowSize.x;
        sh = windowSize.y;
        samplingAspectRatio = sw / sh;
    }




    modelviewMat = modeViewMatrix;
    projectionMat = projectionMatrix;
    //viewportMat = glm::vec4(0, 0,std::max(rayCastingSolutionRenderTarget.Width,windowSize.x),std::max( rayCastingSolutionRenderTarget.Height,windowSize.y));
    viewportMat = glm::vec4(0, 0, windowSize.x, windowSize.y);
    viewMat = viewMatrix;
    modelMat = modelMatrix;
    viewprojectionMat = viewProjectionMatrix;
    mynear = nearPlane;
    myfar = farPlane;
    myup = camUp;
    mytarget = camTarget;
    //end Mohamed's Code




    glBindFramebuffer(GL_FRAMEBUFFER, rayCastingSolutionRenderTarget.FrameBufferObject);
    glViewport(0, 0, sw, sh);




    //glBindFramebuffer(GL_FRAMEBUFFER, MyRenderTarget.FrameBufferObject);
    //glViewport(0, 0, MyRenderTarget.Width, MyRenderTarget.Height);


    //glBindFramebuffer(GL_FRAMEBUFFER, 0);
    //glViewport(0, 0, windowSize.x, windowSize.y);

    // FIXME: viewport causes tears for some reason if sampling rate < 1024
    glClearColor(0.f, 0.f, 0.f, 1.0f);
    // clear render target
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


    //glEnable(GL_POINT_SPRITE);
    //glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_DEPTH_TEST);
    glFrontFace(GL_CCW);

    //GLuint ssboBindingPointIndex = 2;
    //glBindBufferBase(GL_SHADER_STORAGE_BUFFER, samplingShader->getSsboBiningIndex(myArrayUBO), myArrayUBO);


    GLuint samplingProgramHandle;

    //glPointSize(particleRadius);

    //if (pointCloudRendering)
    //	samplingProgramHandle = samplingShader_PointCloud->GetGlShaderProgramHandle();
    //else
    samplingProgramHandle = samplingShader->GetGlShaderProgramHandle();

    assert(glUseProgram);
    glUseProgram(samplingProgramHandle);

    glUniform1f(glGetUniformLocation(samplingProgramHandle, "far"), farPlane);
    glUniform1f(glGetUniformLocation(samplingProgramHandle, "near"), nearPlane);
    glUniform1f(glGetUniformLocation(samplingProgramHandle, "particleScale"), particleScale);
    glUniform1f(glGetUniformLocation(samplingProgramHandle, "aspectRatio"), samplingAspectRatio);

    //get how much object space is represented in one pixel
    float objPerPixel;
    {
        glm::vec2 s, e;
        s = LOD.pixel2obj(glm::vec2(0, 0), std::floor(current_lod - lodDelta));
        e = LOD.pixel2obj(glm::vec2(1, 1), std::floor(current_lod - lodDelta));

        //get difference
        glm::vec2 v = e - s;
        objPerPixel = v.x;
    }
    glUniform1f(glGetUniformLocation(samplingProgramHandle, "objPerPixel"), objPerPixel);

    glUniform1i(glGetUniformLocation(samplingProgramHandle, "buildingHOM"), buildingHOM);
	glUniform1i(glGetUniformLocation(samplingProgramHandle, "streamingFlag"), streaming);
    glUniform1i(glGetUniformLocation(samplingProgramHandle, "maxSamplingRuns"), tempmaxSamplingRuns);

    glUniform1f(glGetUniformLocation(samplingProgramHandle, "viewportWidth"), static_cast<float>(sw));
    glUniform1f(glGetUniformLocation(samplingProgramHandle, "viewportHeight"), static_cast<float>(sh));
    glUniform1i(glGetUniformLocation(samplingProgramHandle, "plainRayCasting"), plainRayCasting);
    glUniform1i(glGetUniformLocation(samplingProgramHandle, "pointCloudRendering"), pointCloudRendering);


    glUniform1i(glGetUniformLocation(samplingProgramHandle, "samplescount"), sample_count);
    glUniform2fv(glGetUniformLocation(samplingProgramHandle, "sPos"), 1, &sPos[0]);


    glUniform1f(glGetUniformLocation(samplingProgramHandle, "maxZ"), modelMax.z);
    glUniform1f(glGetUniformLocation(samplingProgramHandle, "minZ"), modelMin.z);

    auto right = glm::normalize(glm::cross(normalize(camTarget - camPosi), camUp));
    glUniform3fv(glGetUniformLocation(samplingProgramHandle, "right"), 1, &right[0]);
    glUniform3fv(glGetUniformLocation(samplingProgramHandle, "up"), 1, &camUp[0]);



    glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "Model"), 1, GL_FALSE, &samplingModelMat[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "ModelView"), 1, GL_FALSE, &samplingModelViewMat[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "View"), 1, GL_FALSE, &samplingViewMat[0][0]);
    glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "Projection"), 1, GL_FALSE, &samplingProjectionMat[0][0]);
    glUniformMatrix3fv(glGetUniformLocation(samplingProgramHandle, "RotationMatrix"), 1, GL_FALSE, &GlobalRotationMatrix[0][0]);


    glm::mat4 ViewAlignmentMatrix;
    ViewAlignmentMatrix = glm::rotate(ViewAlignmentMatrix, cameraRotation.x, glm::vec3(0.0f, 1.0f, 0.0f));
    ViewAlignmentMatrix = glm::rotate(ViewAlignmentMatrix, cameraRotation.y, glm::vec3(1.0f, 0.0f, 0.0f));

    glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "ViewAlignmentMatrix"), 1, GL_FALSE, &ViewAlignmentMatrix[0][0]);

    glUniform3fv(glGetUniformLocation(samplingProgramHandle, "ViewPosition"), 1, &camPosiSampling[0]);
    glUniform2i(glGetUniformLocation(samplingProgramHandle, "viewSlice"), 0, 0);

    // render particles using ray casting
    // set uniforms
    glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "ViewProjection"), 1, GL_FALSE, &samplingViewProjectionMat[0][0]);

    auto initialOffset = glm::vec3(0.0f, 0.0f, 0.0f);

    // TODO stream rendering
    //particlesGlBuffer.Render(GL_POINTS);

    if (theStreamingBuffer == 0) {
        fences.resize(numBuffers);
        glGenBuffers(1, &theStreamingBuffer);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, theStreamingBuffer);
        glBufferStorage(GL_SHADER_STORAGE_BUFFER, bufSize * numBuffers, nullptr, streamingBufferCreationBits);
        // TODO: returns zero. why?
        theStreamingMappedMem = glMapNamedBufferRangeEXT(theStreamingBuffer, 0, bufSize * numBuffers, streamingBufferMappingBits);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

        rawParticleCenters.resize(particleCenters.size() * 4);
        for (size_t v = 0; v < particleCenters.size(); v++) {
            rawParticleCenters[v * 4 + 0] = particleCenters[v].x;
            rawParticleCenters[v * 4 + 1] = particleCenters[v].y;
            rawParticleCenters[v * 4 + 2] = particleCenters[v].z;
            rawParticleCenters[v * 4 + 3] = particleCenters[v].w;
        }
    }

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, theStreamingBuffer);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, theStreamingBuffer);

    char *currVert = reinterpret_cast<char *>(rawParticleCenters.data());
    size_t vertCounter = 0;
    const size_t vertStride = 16;
    size_t numVerts = bufSize / vertStride;
    while (vertCounter < particleCenters.size()) {
        //GLuint vb = this->theBuffers[currBuf];
        void *mem = static_cast<char*>(theStreamingMappedMem) + bufSize * currBuf;
        const char *whence = currVert;
        UINT64 vertsThisTime = std::min(particleCenters.size() - vertCounter, numVerts);

        waitSignal(fences[currBuf]);
        //vislib::sys::Log::DefaultLog.WriteMsg(vislib::sys::Log::LEVEL_ERROR, "memcopying %u bytes from %016" PRIxPTR " to %016" PRIxPTR "\n", vertsThisTime * vertStride, whence, mem);
        memcpy(mem, whence, vertsThisTime * vertStride);
        glFlushMappedNamedBufferRangeEXT(theStreamingBuffer, bufSize * currBuf, vertsThisTime * vertStride);
        //glMemoryBarrier(GL_CLIENT_MAPPED_BUFFER_BARRIER_BIT);
        //glUniform1i(this->newShader->ParameterLocation("instanceOffset"), numVerts * currBuf);
        //glUniform1i(this->newShader->ParameterLocation("instanceOffset"), 0);

        //this->setPointers(parts, this->theSingleBuffer, reinterpret_cast<const void *>(currVert - whence), this->theSingleBuffer, reinterpret_cast<const void *>(currCol - whence));
        //glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindBufferRange(GL_SHADER_STORAGE_BUFFER, SSBObindingPoint, theStreamingBuffer, bufSize * currBuf, bufSize);
        glDrawArrays(GL_POINTS, 0, vertsThisTime);
        //glDrawArraysInstanced(GL_POINTS, 0, 1, vertsThisTime);
        queueSignal(fences[currBuf]);

        currBuf = (currBuf + 1) % numBuffers;
        vertCounter += vertsThisTime;
        currVert += vertsThisTime * vertStride;
    }

    //experiment to get which particle I hit at each iteration

    if (save_SamplingTexture) {
        save_SamplingTexture = false;
        saveSamplingTexture();
    }
}

bool renderCell(int cellIndx)
{
	for (int i = 0; i < cellsToRender.size(); i++)
	{
		if (cellsToRender[i] == cellIndx)
			return true;
	}
	return false;
}
inline void renderCelltoHOM(std::vector<int> Inds)
{
	float oldCameraDistance;
	glm::vec3 oldCamPosi, oldCamTarget;
	glm::mat3 oldMat;

	oldCameraDistance = cameraDistance;
	oldCamPosi = camPosi;
	oldCamTarget = camTarget;

	//1- get in a position where you can see the whole data set (assume that this the initial position the data set is loaded in).
	cameraDistance = LOD.get_cam_dist(LOD.initial_lod);//std::max(LOD.min_cam_dist, std::min(cameraDistance, LOD.max_cam_dist - .3f));
	camPosi = CameraPosition(cameraRotation, cameraDistance);
	camPosi += cameraOffset;
	camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

	tile_based_culling(false);
	update_page_texture();


	//2- call drawHOM to go
	buildingHOM = true;
	drawHOM(Inds);

	//3- copy drawnHom to the HOM texture
	compareHOMtoTexture();

	//5- return to the position at which the user was before issuing this call
	//reinitialize();

	cameraDistance = oldCameraDistance;
	camPosi = oldCamPosi;
	camTarget = oldCamTarget;

	tile_based_culling(false);
	update_page_texture();
}
inline void buildHOM()
{
	float oldCameraDistance;
	glm::vec3 oldCamPosi, oldCamTarget;
	glm::mat3 oldMat;

	oldCameraDistance = cameraDistance;
	oldCamPosi = camPosi;
	oldCamTarget = camTarget;

	//1- get in a position where you can see the whole data set (assume that this the initial position the data set is loaded in).
	cameraDistance = LOD.get_cam_dist(LOD.initial_lod);//std::max(LOD.min_cam_dist, std::min(cameraDistance, LOD.max_cam_dist - .3f));
	camPosi = CameraPosition(cameraRotation, cameraDistance);
	camPosi += cameraOffset;
	camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

	tile_based_culling(false);
	update_page_texture();


	//2- call drawHOM to go
	buildingHOM = true;
	sPos = glm::vec2(0,0);
	drawHOM();
	
	//3- copare drawnHom to the HOM texture
	copyHOMtoTexture();

	//4- call updateHOM to create the hierarchical map
	updateHOM();

	//5- return to the position at which the user was before issuing this call
	//initialize();

	cameraDistance = oldCameraDistance;
	camPosi = oldCamPosi;
	camTarget = oldCamTarget;

	tile_based_culling(false);
	update_page_texture();
}
inline void compareHOMtoTexture()
{
	//drawDepthBuffer(homRenderTarget);
	
	//1-> read depth of homFBO
	glBindFramebuffer(GL_FRAMEBUFFER, homRenderTarget.FrameBufferObject);

	float* fPixels = new float[homRenderTarget.Width*homRenderTarget.Height];
	glReadPixels(0, 0, homRenderTarget.Width, homRenderTarget.Height, GL_DEPTH_COMPONENT, GL_FLOAT, fPixels);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//2-> read lod 0 of hom texture
	auto imgSize = size_t(4 * (homHighestResolution.x)*(homHighestResolution.y));
	GLfloat* t = new GLfloat[imgSize];

	glBindTexture(GL_TEXTURE_2D, HOMtexture);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, t);

	//3-> compare  depth of homFBO to lod 0 of hom texture,
	//if depth in homFbo < lod 0 of hom texture, then we see a new sphere, and update lod 0
	bool homUpdated = false;
	for (int i = 0; i < imgSize; i = i + 4)
	{
		if (fPixels[i / 4] < t[i])
		{
			t[i] = fPixels[i / 4];
			if (!homUpdated)
				homUpdated = true;
		}
	}
	//update hom
	if (homUpdated)
	{
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, homHighestResolution.x, homHighestResolution.y, 0, GL_RGBA, GL_FLOAT, t);
		glGenerateMipmap(GL_TEXTURE_2D);

		glBindTexture(GL_TEXTURE_2D, 0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		std::cout << "HOM Updated!" << std::endl;
		//call updateHOM to create the updated hierarchical map
		updateHOM();
	}
	else
	{
		glBindTexture(GL_TEXTURE_2D, 0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
	

	delete[] fPixels;
	delete[] t;
}
inline void copyHOMtoTexture()
{
	//read whatever is in homFBO
	float* fPixels = new float[4 * homRenderTarget.Width*homRenderTarget.Height];
	glReadPixels(0, 0, homRenderTarget.Width, homRenderTarget.Height, GL_RGBA, GL_FLOAT, fPixels);

	//debug
	//as an initialization step, we should set the depth of all pixels to '1', ie farthest from viewer
	//this was not done in the 'drawhom' stage, i think we can just do it here, by replacting all pixles in 'fPixels' that are '0' with '1'
	for (int i = 0; i < 4 * homRenderTarget.Width*homRenderTarget.Height; i=i+4)
	{
		if (fPixels[i] == 0)
			fPixels[i] = 1;
	}
	//end debug

	//initialize hom texture
	glDeleteTextures(1, &HOMtexture);
	HOMtexture = 0;
	glGenTextures(1, &HOMtexture);
	glBindTexture(GL_TEXTURE_2D, HOMtexture);

	//copy what was read from fbo to texture
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, homHighestResolution.x, homHighestResolution.y, 0, GL_RGBA, GL_FLOAT, fPixels);
	glGenerateMipmap(GL_TEXTURE_2D);

	glBindTexture(GL_TEXTURE_2D, 0);
	delete[] fPixels;
}

inline void updateHOM()
{
	//run a compute shader to downsample the highest lod of HOM.
	//use shader
	auto buildHOMHandle = buildHOMShader_C->GetGlShaderProgramHandle();
	glUseProgram(buildHOMHandle);

	int mipmapCount = log2(std::min(homHighestResolution.x, homHighestResolution.y));

	for (int i = 0; i < mipmapCount; i++)
	{
		glBindImageTexture(0, HOMtexture, i, false, 0, GL_READ_ONLY, GL_RGBA32F);
		glUniform1i(glGetUniformLocation(buildHOMHandle, "floorLevel"), 0);

		glBindImageTexture(1, HOMtexture, i + 1, false, 0, GL_READ_WRITE, GL_RGBA32F);
		glUniform1i(glGetUniformLocation(buildHOMHandle, "ceilLevel"), 1);

		glDispatchCompute(homHighestResolution.x / pow(2, i + 1), homHighestResolution.y / pow(2, i + 1), 1);

		//put a synchronization function here so we are sure for the next iteration, there is something  read from and downsample
		glFinish();

#if 0		//download and save texture level
		{
			// download texture
			auto imgSize = size_t(4 * (homHighestResolution.x / pow(2, i))*(homHighestResolution.y / pow(2, i)));
			GLfloat* t = new GLfloat[imgSize];

			//now read it
			glBindTexture(GL_TEXTURE_2D, HOMtexture);
			glGetTexImage(GL_TEXTURE_2D, i, GL_RGBA, GL_FLOAT, t);

			//save to image
			{
				float* imgf = new float[imgSize / 4];
				for (int j = 0; j < imgSize; j = j + 4)
				{
					imgf[j / 4] = t[j];
					if (t[j] != 0)
					{
						t[j] = t[j];
					}
				}

				int w = homHighestResolution.x / pow(2, i);
				int h = homHighestResolution.y / pow(2, i);

				drawNDFImage(w, h, imgf);
				delete[] imgf;
			}
			delete[] t;
		}
#endif
	}

	//popluate homfbos if not aready popluated, and as long as homhighest resolution does not change no need to update it
	if (homFbos.size() == 0)
	{
		int homW, homH;
		homFbos.resize(mipmapCount);
		for (int i = 0; i < mipmapCount; i++)
		{
			homW = homHighestResolution.x / pow(2, i);
			homH = homHighestResolution.y / pow(2, i);
			homFbos[i]= Helpers::Gl::CreateRenderTarget(homW, homH, GL_RGBA32F, GL_RGBA, GL_FLOAT);
		}
	}

	glUseProgram(0);
}

inline void projectCell(int nodeIndx, std::vector<glm::vec2>& projectedCorners, int HOMlevel)
{
	sw = homHighestResolution.x / pow(2, HOMlevel);
	sh = homHighestResolution.y / pow(2, HOMlevel);

	octree::node n = global_tree.nodes[nodeIndx];

	// calculate global projection matrix
	auto leftOriginal = -0.5f;
	auto rightOriginal = 0.5f;
	auto bottomOriginal = -0.5f / aspectRatio;
	auto topOriginal = 0.5f / aspectRatio;


	OrigL = leftOriginal;
	OrigR = rightOriginal;
	OrigB = bottomOriginal;
	OrigT = topOriginal;


	glm::mat4x4 modelMatrix, T, Tinv, S;
	const auto modelScale = initialCameraDistance / cameraDistance;

	S[0][0] = modelScale;
	S[1][1] = modelScale;
	S[2][2] = modelScale;

	modelMatrix = Tinv*S*T;

	camPosiSampling = camPosi;
	auto viewMatrix = glm::lookAt(camPosiSampling, camTarget, camUp);


	auto modeViewMatrix = modelMatrix * viewMatrix;
	auto projectionMatrix = glm::ortho(leftOriginal, rightOriginal, bottomOriginal, topOriginal, nearPlane, farPlane);

	auto viewProjectionMatrix = projectionMatrix * viewMatrix;
	//samplingAspectRatio = aspectRatio;

	glm::vec3 tempCam = CameraPosition(cameraRotation, cameraDistance);
	viewMat_noPanning = glm::lookAt(tempCam, glm::vec3(0, 0, 0), camUp);
	modelviewMat_noPanning = modelMatrix*viewMat_noPanning;

	if ((!plainRayCasting) && (!pointCloudRendering))
	{
		//bring camera to ceil lod position
		float tempCamDist = LOD.get_cam_dist(std::floor(current_lod - lodDelta));
		float tempmodelscale = initialCameraDistance / tempCamDist;

		//update camposisampling
		camPosiSampling = CameraPosition(cameraRotation, tempCamDist);
		camPosiSampling += cameraOffset;

		//edit view matrix
		samplingViewMat = glm::lookAt(camPosiSampling, camTarget, camUp);

		//edit projection matrix
		samplingProjectionMat = glm::ortho(sl, sr, sb, st, nearPlane, farPlane);

		//edit viewprojectionmatrix
		samplingViewProjectionMat = samplingProjectionMat*samplingViewMat;

		//edit model matrix
		samplingModelMat = modelMatrix;
		samplingModelMat[0][0] = tempmodelscale;
		samplingModelMat[1][1] = tempmodelscale;
		samplingModelMat[2][2] = tempmodelscale;

		//edit model view matrix
		samplingModelViewMat = samplingModelMat*samplingViewMat;

		samplingAspectRatio = sw / sh;

		glm::vec3 tempCam = CameraPosition(cameraRotation, tempCamDist);
		samplingViewMat_noPanning = glm::lookAt(tempCam, glm::vec3(0, 0, 0), camUp);
		samplingModelViewMat_noPanning = samplingModelMat*samplingViewMat_noPanning;
	}
	viewportMat = glm::vec4(0, 0, sw,sh);


	//get corners of bounding box of cell, rotate them as the need arises
	std::vector<glm::vec3> bbCorners(8);

	bbCorners[0] = GlobalRotationMatrix*glm::vec3(n.center.x - n.extent.x, n.center.y - n.extent.y, n.center.z - n.extent.z); //-1,-1,-1
	bbCorners[1] = GlobalRotationMatrix*glm::vec3(n.center.x + n.extent.x, n.center.y - n.extent.y, n.center.z - n.extent.z); // 1,-1,-1
	bbCorners[2] = GlobalRotationMatrix*glm::vec3(n.center.x - n.extent.x, n.center.y + n.extent.y, n.center.z - n.extent.z); //-1, 1,-1
	bbCorners[3] = GlobalRotationMatrix*glm::vec3(n.center.x - n.extent.x, n.center.y - n.extent.y, n.center.z + n.extent.z); //-1,-1, 1
	bbCorners[4] = GlobalRotationMatrix*glm::vec3(n.center.x - n.extent.x, n.center.y + n.extent.y, n.center.z + n.extent.z); //-1, 1, 1
	bbCorners[5] = GlobalRotationMatrix*glm::vec3(n.center.x + n.extent.x, n.center.y - n.extent.y, n.center.z + n.extent.z); // 1,-1, 1
	bbCorners[6] = GlobalRotationMatrix*glm::vec3(n.center.x + n.extent.x, n.center.y + n.extent.y, n.center.z - n.extent.z); // 1, 1,-1
	bbCorners[7] = GlobalRotationMatrix*glm::vec3(n.center.x + n.extent.x, n.center.y + n.extent.y, n.center.z + n.extent.z); // 1, 1, 1

	for (int i = 0; i < 8; i++)
	{
		projectedCorners.push_back(glm::vec2(glm::project(bbCorners[i], samplingModelMat, samplingProjectionMat, viewportMat)));
	}

	convexHull h;
	std::vector<glm::vec2> hull;
	hull=h.convex_hull(projectedCorners);

	projectedCorners = hull;
}

inline int getHOMTestLevel(int nodeIndx)
{
	octree::node n = global_tree.nodes[nodeIndx];

	//get level of node
	int nodeLevel = global_tree.getNodeLevel(n);
	
	//solve 1=(w_HOMLevel*h_HOMlevel)/8^(level of node)  for HOMLevel
	int mipmapCount = log2(std::min(homHighestResolution.x, homHighestResolution.y));

#if 1 //old method
	{
		std::vector<std::pair<int, float>> levelsMeasures;
		int lw, lh;
		float measure;
		for (int i = 0; i <= mipmapCount; i++)
		{
			lw = homHighestResolution.x / pow(2, i);
			lh = homHighestResolution.y / pow(2, i);
			measure = abs((lw*lh) - pow(8, nodeLevel));
			levelsMeasures.push_back(std::make_pair(i, measure));
		}

		std::sort(levelsMeasures.begin(), levelsMeasures.end(), [](std::pair<int, float> &left, std::pair<int, float> &right)
		{
			return left.second < right.second;
		});

		return levelsMeasures[0].first;
	}
#else
	{
		//at level 0, the data set is composed of just one cell, therefore, the projected rectangle can take 1 pixel, therefore homLevel= max
		//at level 1, the data set is composed of 8 cells, therefore, we need at least 8 pixels, therefore homLevel


	}
#endif
}
inline void getHOMOverlappingPixels(int HOMlevel, glm::vec2& cell_blc, glm::vec2& cell_trc, std::vector<int>& pixels)
{
	//get pixel width and height in level HOMlevel
	glm::vec2 pdim = glm::vec2(pow(2, HOMlevel));
	std::vector<glm::vec2> pixelCorners(4);
	int pIndx;
	pixels.clear();

	for (int i = 0; i < homHighestResolution.x; i += pdim.x)
	{
		for (int j = 0; j < homHighestResolution.y; j += pdim.y)
		{
			pixelCorners[0] = glm::vec2(i, j);
			pixelCorners[1] = glm::vec2(i + pdim.x, j);
			pixelCorners[2] = glm::vec2(i + pdim.x, j + pdim.y);
			pixelCorners[3] = glm::vec2(i, j + pdim.y);

			for (int k = 0; k < 4; k++)
			{
				if (pixelCorners[k].x <= cell_trc.x && pixelCorners[k].y <= cell_trc.y && pixelCorners[k].x >= cell_blc.x && pixelCorners[k].y >= cell_blc.y)
				{
					//width is homhighestw/2^homlevel, height is homhighesth/2^homlevel
					pIndx = (j / pdim.y)*(homHighestResolution.x / pow(2, HOMlevel)) + (i / pdim.x);
					pixels.push_back(pIndx);
					break;
				}
			}
		}
	}
}

inline void setHOMRenderingState(Helpers::Gl::RenderTargetHandle& homFbo)
{
	// render particles directly during rotation
	int tsw = homFbo.Width;
	int tsh = homFbo.Height;

	// calculate global projection matrix
	auto leftOriginal = -0.5f;
	auto rightOriginal = 0.5f;
	auto bottomOriginal = -0.5f / aspectRatio;
	auto topOriginal = 0.5f / aspectRatio;


	OrigL = leftOriginal;
	OrigR = rightOriginal;
	OrigB = bottomOriginal;
	OrigT = topOriginal;
	//auto bottomOriginal = -0.5f;
	//auto topOriginal = 0.5f;

	glm::mat4x4 modelMatrix, T, Tinv, S;
	const auto modelScale = initialCameraDistance / cameraDistance;
	// NOTE: turn scaling off for rendering the histograms
	//const auto modelScale = 1.0f;

	//T[0][4] = -lastMouse.x;
	//T[1][4] = -lastMouse.y;

	//Tinv[0][4] = lastMouse.x;
	//Tinv[1][4] = lastMouse.y;

	S[0][0] = modelScale;
	S[1][1] = modelScale;
	S[2][2] = modelScale;

	modelMatrix = Tinv*S*T;



	//modelMatrix[1][1] *= aspectRatio;

	camPosiSampling = camPosi;
	auto viewMatrix = glm::lookAt(camPosiSampling, camTarget, camUp);


	auto modeViewMatrix = modelMatrix * viewMatrix;
	auto projectionMatrix = glm::ortho(leftOriginal, rightOriginal, bottomOriginal, topOriginal, nearPlane, farPlane);

	auto viewProjectionMatrix = projectionMatrix * viewMatrix;
	//samplingAspectRatio = aspectRatio;

	glm::vec3 tempCam = CameraPosition(cameraRotation, cameraDistance);
	viewMat_noPanning = glm::lookAt(tempCam, glm::vec3(0, 0, 0), camUp);
	modelviewMat_noPanning = modelMatrix*viewMat_noPanning;

	//Mohamed's Code

	//at fractional lods, i want the sampling to be done at the ceil resolution to avoid race conditions,
	//therefore, i will edit the modelviewmat for this case only

	if ((!plainRayCasting) && (!pointCloudRendering))
	{
		//bring camera to ceil lod position
		float tempCamDist = LOD.get_cam_dist(std::floor(current_lod - lodDelta));
		float tempmodelscale = initialCameraDistance / tempCamDist;

		//update camposisampling
		camPosiSampling = CameraPosition(cameraRotation, tempCamDist);
		camPosiSampling += cameraOffset;

		//edit view matrix
		samplingViewMat = glm::lookAt(camPosiSampling, camTarget, camUp);

		//edit projection matrix
		samplingProjectionMat = glm::ortho(sl, sr, sb, st, nearPlane, farPlane);

		//edit viewprojectionmatrix
		samplingViewProjectionMat = samplingProjectionMat*samplingViewMat;

		//edit model matrix
		samplingModelMat = modelMatrix;
		samplingModelMat[0][0] = tempmodelscale;
		samplingModelMat[1][1] = tempmodelscale;
		samplingModelMat[2][2] = tempmodelscale;

		//glm::vec2 f = glm::vec2(sl + ((sr - sl) / 2.0f), sb + ((st - sb) / 2.0f));

		//T[0][4] = -f.x;
		//T[1][4] = -f.y;

		//Tinv[0][4] = f.x;
		//Tinv[1][4] = f.y;

		//samplingModelMat = Tinv*samplingModelMat*T;

		//edit model view matrix
		samplingModelViewMat = samplingModelMat*samplingViewMat;

		samplingAspectRatio = tsw / tsh;

		glm::vec3 tempCam = CameraPosition(cameraRotation, tempCamDist);
		samplingViewMat_noPanning = glm::lookAt(tempCam, glm::vec3(0, 0, 0), camUp);
		samplingModelViewMat_noPanning = samplingModelMat*samplingViewMat_noPanning;
	}
	else
	{
		//edit projection matrix
		samplingProjectionMat = projectionMatrix;
		samplingViewMat = viewMatrix;
		samplingViewProjectionMat = viewProjectionMatrix;
		samplingModelViewMat = modeViewMatrix;
		samplingModelMat = modelMatrix;
		samplingAspectRatio = aspectRatio;

		samplingViewMat_noPanning = viewMat_noPanning;
		samplingModelViewMat_noPanning = modelviewMat_noPanning;

		tsw = windowSize.x;
		tsh = windowSize.y;
		samplingAspectRatio = tsw / tsh;
	}




	modelviewMat = modeViewMatrix;
	projectionMat = projectionMatrix;
	//viewportMat = glm::vec4(0, 0,std::max(rayCastingSolutionRenderTarget.Width,windowSize.x),std::max( rayCastingSolutionRenderTarget.Height,windowSize.y));
	viewportMat = glm::vec4(0, 0, windowSize.x, windowSize.y);
	viewMat = viewMatrix;
	modelMat = modelMatrix;
	viewprojectionMat = viewProjectionMatrix;
	mynear = nearPlane;
	myfar = farPlane;
	myup = camUp;
	mytarget = camTarget;
	//end Mohamed's Code




	glBindFramebuffer(GL_FRAMEBUFFER, homFbo.FrameBufferObject);
	glViewport(0, 0, tsw, tsh);




	//glBindFramebuffer(GL_FRAMEBUFFER, MyRenderTarget.FrameBufferObject);
	//glViewport(0, 0, MyRenderTarget.Width, MyRenderTarget.Height);


	//glBindFramebuffer(GL_FRAMEBUFFER, 0);
	//glViewport(0, 0, windowSize.x, windowSize.y);

	// FIXME: viewport causes tears for some reason if sampling rate < 1024
	//glClearColor(0.f, 0.f, 0.f, 1.0f);
	// clear render target
	// glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


	//glEnable(GL_POINT_SPRITE);
	//glEnable(GL_PROGRAM_POINT_SIZE);
	//glEnable(GL_DEPTH_TEST);
	//glFrontFace(GL_CCW);

	//GLuint ssboBindingPointIndex = 2;
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, samplingShader->getSsboBiningIndex(myArrayUBO), myArrayUBO);


	GLuint samplingProgramHandle;

	//glPointSize(particleRadius);

	//if (pointCloudRendering)
	//	samplingProgramHandle = samplingShader_PointCloud->GetGlShaderProgramHandle();
	//else
	samplingProgramHandle = drawQuadMeshShader->GetGlShaderProgramHandle();

	assert(glUseProgram);
	glUseProgram(samplingProgramHandle);

	glUniform1f(glGetUniformLocation(samplingProgramHandle, "far"), farPlane);
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "near"), nearPlane);
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "particleScale"), particleScale);
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "aspectRatio"), samplingAspectRatio);

	//get how much object space is represented in one pixel
	float objPerPixel;
	{
		glm::vec2 s, e;
		s = LOD.pixel2obj(glm::vec2(0, 0), std::floor(current_lod - lodDelta));
		e = LOD.pixel2obj(glm::vec2(1, 1), std::floor(current_lod - lodDelta));

		//get difference
		glm::vec2 v = e - s;
		objPerPixel = v.x;
	}
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "objPerPixel"), objPerPixel);

	glUniform1i(glGetUniformLocation(samplingProgramHandle, "buildingHOM"), buildingHOM);
	glUniform1i(glGetUniformLocation(samplingProgramHandle, "maxSamplingRuns"), tempmaxSamplingRuns);

	glUniform1f(glGetUniformLocation(samplingProgramHandle, "viewportWidth"), static_cast<float>(tsw));
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "viewportHeight"), static_cast<float>(tsh));
	glUniform1i(glGetUniformLocation(samplingProgramHandle, "plainRayCasting"), plainRayCasting);
	glUniform1i(glGetUniformLocation(samplingProgramHandle, "pointCloudRendering"), pointCloudRendering);


	glUniform1i(glGetUniformLocation(samplingProgramHandle, "samplescount"), sample_count);
	glUniform2fv(glGetUniformLocation(samplingProgramHandle, "sPos"), 1, &sPos[0]);


	glUniform1f(glGetUniformLocation(samplingProgramHandle, "maxZ"), modelMax.z);
	glUniform1f(glGetUniformLocation(samplingProgramHandle, "minZ"), modelMin.z);

	auto right = glm::normalize(glm::cross(normalize(camTarget - camPosi), camUp));
	glUniform3fv(glGetUniformLocation(samplingProgramHandle, "right"), 1, &right[0]);
	glUniform3fv(glGetUniformLocation(samplingProgramHandle, "up"), 1, &camUp[0]);



	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "Model"), 1, GL_FALSE, &samplingModelMat[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "ModelView"), 1, GL_FALSE, &samplingModelViewMat[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "View"), 1, GL_FALSE, &samplingViewMat[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "Projection"), 1, GL_FALSE, &samplingProjectionMat[0][0]);
	glUniformMatrix3fv(glGetUniformLocation(samplingProgramHandle, "RotationMatrix"), 1, GL_FALSE, &GlobalRotationMatrix[0][0]);


	glm::mat4 ViewAlignmentMatrix;
	ViewAlignmentMatrix = glm::rotate(ViewAlignmentMatrix, cameraRotation.x, glm::vec3(0.0f, 1.0f, 0.0f));
	ViewAlignmentMatrix = glm::rotate(ViewAlignmentMatrix, cameraRotation.y, glm::vec3(1.0f, 0.0f, 0.0f));

	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "ViewAlignmentMatrix"), 1, GL_FALSE, &ViewAlignmentMatrix[0][0]);

	glUniform3fv(glGetUniformLocation(samplingProgramHandle, "ViewPosition"), 1, &camPosiSampling[0]);
	glUniform2i(glGetUniformLocation(samplingProgramHandle, "viewSlice"), 0, 0);

	// render particles using ray casting
	// set uniforms
	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "ViewProjection"), 1, GL_FALSE, &samplingViewProjectionMat[0][0]);

	
}

inline void testAgainstHOM(int nodeIndx, std::vector<int>& tbe,std::vector<int>& tbr)
{
	octree::node n = global_tree.nodes[nodeIndx];

	Helpers::Gl::BufferHandle cellAABBbuffer;
	std::vector<glm::vec3> bbCorners(8);

	//given an octree node 'n':
	
	//1-> begin overlap test at a level in HOM where a pixel can fit inside the rectangle
	int HOMlevel = std::max(0,getHOMTestLevel(nodeIndx)-1);   //need to check why when i go one level below (by subtracting one) it's correct, otherwise, it's wrong, probably some
	                                                          //rounding error

	//debug
	//std::cout << "hom level is " <<HOMlevel <<std::endl;
	//HOMlevel = 0;
	//end debug

	////2-> project cell to screen, creating a polygon
	//std::vector<glm::vec2> cellProjectedConvexHull;
	//projectCell(n, cellProjectedConvexHull,HOMlevel);

	//3-> here we issue an occlusion query for the cell
	{
		//create an fbo of resolution equal to HOMlevel and attach to it HOMlevel as its depth buffer
		
		initializeHOMtestFbo(homFbos[HOMlevel], HOMlevel);

		//TODO: bind homFBO, set view port and opengl initializations...
		//same transformation as drawHOM (no scaling, no panning?) only rotation
		//draw vertices of bounding box of cell
		setHOMRenderingState(homFbos[HOMlevel]);
#if 0
		{
			//draw the bounding box, pass vertices of quads making up the box

			bbCorners[0] = glm::vec3(n.center.x - n.extent.x, n.center.y - n.extent.y, n.center.z - n.extent.z); //-1,-1,-1
			bbCorners[1] = glm::vec3(n.center.x + n.extent.x, n.center.y - n.extent.y, n.center.z - n.extent.z); // 1,-1,-1
			bbCorners[2] = glm::vec3(n.center.x - n.extent.x, n.center.y + n.extent.y, n.center.z - n.extent.z); //-1, 1,-1
			bbCorners[3] = glm::vec3(n.center.x - n.extent.x, n.center.y - n.extent.y, n.center.z + n.extent.z); //-1,-1, 1
			bbCorners[4] = glm::vec3(n.center.x - n.extent.x, n.center.y + n.extent.y, n.center.z + n.extent.z); //-1, 1, 1
			bbCorners[5] = glm::vec3(n.center.x + n.extent.x, n.center.y - n.extent.y, n.center.z + n.extent.z); // 1,-1, 1
			bbCorners[6] = glm::vec3(n.center.x + n.extent.x, n.center.y + n.extent.y, n.center.z - n.extent.z); // 1, 1,-1
			bbCorners[7] = glm::vec3(n.center.x + n.extent.x, n.center.y + n.extent.y, n.center.z + n.extent.z); // 1, 1, 1

			std::vector<glm::vec3> bbQuadMesh;
			//front
			bbQuadMesh.push_back(bbCorners[3]);
			bbQuadMesh.push_back(bbCorners[5]);
			bbQuadMesh.push_back(bbCorners[4]);
			bbQuadMesh.push_back(bbCorners[7]);

			//right
			bbQuadMesh.push_back(bbCorners[5]);
			bbQuadMesh.push_back(bbCorners[1]);
			bbQuadMesh.push_back(bbCorners[7]);
			bbQuadMesh.push_back(bbCorners[6]);

			//back
			bbQuadMesh.push_back(bbCorners[6]);
			bbQuadMesh.push_back(bbCorners[1]);
			bbQuadMesh.push_back(bbCorners[2]);
			bbQuadMesh.push_back(bbCorners[0]);

			//left
			bbQuadMesh.push_back(bbCorners[2]);
			bbQuadMesh.push_back(bbCorners[0]);
			bbQuadMesh.push_back(bbCorners[4]);
			bbQuadMesh.push_back(bbCorners[3]);

			//top
			bbQuadMesh.push_back(bbCorners[7]);
			bbQuadMesh.push_back(bbCorners[6]);
			bbQuadMesh.push_back(bbCorners[4]);
			bbQuadMesh.push_back(bbCorners[2]);

			//bottom
			bbQuadMesh.push_back(bbCorners[3]);
			bbQuadMesh.push_back(bbCorners[0]);
			bbQuadMesh.push_back(bbCorners[5]);
			bbQuadMesh.push_back(bbCorners[1]);


			Helpers::Gl::MakeBuffer(bbQuadMesh, cellAABBbuffer);
			//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			cellAABBbuffer.Render(GL_LINES_ADJACENCY);

			//debug
			drawDepthBuffer(homFbo);
		}
		//debug draw it
		glBindFramebuffer(GL_FRAMEBUFFER, homFbo.FrameBufferObject);
#if 1
		// Make the BYTE array, factor of 3 because it's RBG.
		BYTE* pixels = new BYTE[2 * homFbo.Width*homFbo.Height];
		float* fPixels = new float[4 * homFbo.Width*homFbo.Height];
		glReadPixels(0, 0, homFbo.Width, homFbo.Height, GL_RGBA, GL_FLOAT, fPixels);

		for (int i = 0; i < 4 * homFbo.Width*homFbo.Height; i = i + 4)
		{
			int j = i / 2;
			pixels[j] = fPixels[i] * 255;
			pixels[j + 1] = fPixels[i + 1] * 255;

			if (fPixels[i + 1] != 0)
			{
				fPixels[i + 1] = fPixels[i + 1];
			}
		}

		// Convert to FreeImage format & save to file
		FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, homFbo.Width, homFbo.Height, 2 * homFbo.Width, 16, 0x00FF, 0xFF00, 0xFFFF, true);

		std::string name = "samplingTexture" + std::to_string(samplingTextureSaves) + ".bmp";
		samplingTextureSaves++;
		FreeImage_Save(FIF_BMP, image, name.c_str(), 0);

		// Free resources
		FreeImage_Unload(image);
		delete[] pixels;
#endif
#endif

		//disable writting to color and depth buffer
		glColorMask(false, false, false, false);
		glDepthMask(GL_FALSE);

		glDepthFunc(GL_LEQUAL);

		GLuint uiQuery;
		int samplesPassed = 0;
		//generate queries
		glGenQueries(1, &uiQuery);
		
		//begin query
		//TODO: render bounding box of cell (DECIDE ON A DEPTH VALUE TO GIVE TO IT).
		glBeginQuery(GL_SAMPLES_PASSED, uiQuery);

		//draw the bounding box, pass vertices of quads making up the box

		bbCorners[0] = glm::vec3(n.center.x - n.extent.x, n.center.y - n.extent.y, n.center.z - n.extent.z); //-1,-1,-1
		bbCorners[1] = glm::vec3(n.center.x + n.extent.x, n.center.y - n.extent.y, n.center.z - n.extent.z); // 1,-1,-1
		bbCorners[2] = glm::vec3(n.center.x - n.extent.x, n.center.y + n.extent.y, n.center.z - n.extent.z); //-1, 1,-1
		bbCorners[3] = glm::vec3(n.center.x - n.extent.x, n.center.y - n.extent.y, n.center.z + n.extent.z); //-1,-1, 1
		bbCorners[4] = glm::vec3(n.center.x - n.extent.x, n.center.y + n.extent.y, n.center.z + n.extent.z); //-1, 1, 1
		bbCorners[5] = glm::vec3(n.center.x + n.extent.x, n.center.y - n.extent.y, n.center.z + n.extent.z); // 1,-1, 1
		bbCorners[6] = glm::vec3(n.center.x + n.extent.x, n.center.y + n.extent.y, n.center.z - n.extent.z); // 1, 1,-1
		bbCorners[7] = glm::vec3(n.center.x + n.extent.x, n.center.y + n.extent.y, n.center.z + n.extent.z); // 1, 1, 1

		std::vector<glm::vec3> bbQuadMesh;
		//front
		bbQuadMesh.push_back(bbCorners[3]);
		bbQuadMesh.push_back(bbCorners[5]);
		bbQuadMesh.push_back(bbCorners[4]);
		bbQuadMesh.push_back(bbCorners[7]);

		//right
		bbQuadMesh.push_back(bbCorners[5]);
		bbQuadMesh.push_back(bbCorners[1]);
		bbQuadMesh.push_back(bbCorners[7]);
		bbQuadMesh.push_back(bbCorners[6]);

		//back
		bbQuadMesh.push_back(bbCorners[6]);
		bbQuadMesh.push_back(bbCorners[1]);
		bbQuadMesh.push_back(bbCorners[2]);
		bbQuadMesh.push_back(bbCorners[0]);

		//left
		bbQuadMesh.push_back(bbCorners[2]);
		bbQuadMesh.push_back(bbCorners[0]);
		bbQuadMesh.push_back(bbCorners[4]);
		bbQuadMesh.push_back(bbCorners[3]);

		//top
		bbQuadMesh.push_back(bbCorners[7]);
		bbQuadMesh.push_back(bbCorners[6]);
		bbQuadMesh.push_back(bbCorners[4]);
		bbQuadMesh.push_back(bbCorners[2]);

		//bottom
		bbQuadMesh.push_back(bbCorners[3]);
		bbQuadMesh.push_back(bbCorners[0]);
		bbQuadMesh.push_back(bbCorners[5]);
		bbQuadMesh.push_back(bbCorners[1]);


		Helpers::Gl::MakeBuffer(bbQuadMesh, cellAABBbuffer);
		cellAABBbuffer.Render(GL_LINES_ADJACENCY);

		glEndQuery(GL_SAMPLES_PASSED);

		//get number of samples that passed
		glGetQueryObjectiv(uiQuery, GL_QUERY_RESULT, &samplesPassed);

		//enable writting to color and depth buffer
		glColorMask(true, true, true, true);
		glDepthMask(GL_TRUE);

		//use default frame buffer
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

		////delete rendertarget
		//Helpers::Gl::DeleteRenderTarget(homFbo);

		if (samplesPassed > 0)
		{
			if (n.children.size() > 0)
			{
				for (int i = 0; i < n.children.size(); i++)
				{
					testAgainstHOM(n.children[i],tbe,tbr);
				}
			}
			else
			{
				//render cell, or go down to individual particle level at highest HOM resolution
				//and update HOM
				//Optimization: call this function for all nodes that fall in the same level of the tree
				//renderCelltoHOM(nodeIndx);

				//add cell to be rendered to HOM
				tbr.push_back(nodeIndx);

				//add cell to be tested in sebsequent iterations
				cellsTbt.push(nodeIndx);
			}
		}
		else
		{
			tbe.push_back(nodeIndx);
		}
	}
}

inline void drawDepthBuffer(Helpers::Gl::RenderTargetHandle& homFbo)
{
	glBindFramebuffer(GL_FRAMEBUFFER, homFbo.FrameBufferObject);

#if 1
	float* fPixels = new float[homFbo.Width*homFbo.Height];
	glReadPixels(0, 0, homFbo.Width, homFbo.Height, GL_DEPTH_COMPONENT, GL_FLOAT, fPixels);

	drawNDFImage(homFbo.Width, homFbo.Height,fPixels);
#endif

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glUseProgram(0);
}

inline void initializeHOMtestFbo(Helpers::Gl::RenderTargetHandle& homFbo, int HOMlevel)
{
#if 1	
	glBindFramebuffer(GL_FRAMEBUFFER, homFbo.FrameBufferObject);

	//TODO: I think I need a shader that takes in the texture from hom, and just writes depth values from that texture to homFbo
	auto viewMatrix = glm::lookAt(camPosi, camTarget, camUp);
	auto programHandle = homDepthTransferShader->GetGlShaderProgramHandle();
	glUseProgram(programHandle);

	glViewport(0, 0, homFbo.Width, homFbo.Height);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	//glDisable(GL_DEPTH_TEST);

	glBindImageTexture(1, HOMtexture, HOMlevel, false, 0, GL_READ_ONLY, GL_RGBA32F);
	glUniform1i(glGetUniformLocation(programHandle, "homTexture"), 1);
	
	quadHandle.Render();

	//glEnable(GL_DEPTH_TEST);
#else
	//read hom texture at hom level
	glBindTexture(GL_TEXTURE_2D, HOMtexture);

	float* t = new float[4*homW*homH];
	glGetTexImage(GL_TEXTURE_2D, HOMlevel, GL_RGBA, GL_FLOAT, t);
	
	glBindTexture(GL_TEXTURE_2D, 0);
	
	//put in depth map
	glBindFramebuffer(GL_FRAMEBUFFER, homFbo.FrameBufferObject);
	glBindTexture(GL_TEXTURE_2D, homFbo.DepthMap);
	float* d = new float[homW*homH];

	for (int i = 0; i < 4 * homW*homH; i=i+4)
	{
		d[i / 4] = t[i];

		if (t[i] != 0 && t[i]!=1)
		{
			t[i] = t[i];
		}
	}

	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, homFbo.Width, homFbo.Height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, d);

	glBindTexture(GL_TEXTURE_2D, 0);

	delete[] t;
	delete[] d;
	
#endif


	//debug draw it
#if 0
	// Make the BYTE array, factor of 3 because it's RBG.
	BYTE* pixels = new BYTE[3 * homFbo.Width*homFbo.Height];
	float* fPixels = new float[4 * homFbo.Width*homFbo.Height];
	glReadPixels(0, 0, homFbo.Width, homFbo.Height, GL_RGBA, GL_FLOAT, fPixels);

	int j = 0;
	std::vector<float> vals;
	for (int i = 0; i < 4 * homFbo.Width*homFbo.Height; i = i + 4)
	{
		pixels[j] = fPixels[i] * 255;
		pixels[j + 1] = fPixels[i + 1] * 255;
		pixels[j + 2] = fPixels[i + 2] * 255;

		j = j + 3;

		if (fPixels[i] != 0 && fPixels[i]!=1)
		{
			bool found = false;
			for (int k = 0; k < vals.size(); k++)
			{
				if (vals[k] == fPixels[i])
				{
					found = true;
					break;
				}
			}
			if (!found)
				vals.push_back(fPixels[i]);
		}

		if (fPixels[i] == 1)
		{
			pixels[j] = pixels[j];
		}
	}

	// Convert to FreeImage format & save to file
	FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, homFbo.Width, homFbo.Height, 3 * homFbo.Width, 24, 0xFF0000, 0x00FF00, 0x0000FF, false);

	std::string name = "samplingTexture" + std::to_string(samplingTextureSaves) + ".bmp";
	samplingTextureSaves++;
	FreeImage_Save(FIF_BMP, image, name.c_str(), 0);

	// Free resources
	FreeImage_Unload(image);
	delete[] pixels;
#endif

	glUseProgram(0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	

	//debug: draw the depth buffer
//	drawDepthBuffer(homFbo);
}

inline void computeSampledRegion()
{
	if (LOD.myTiles[std::floor(current_lod-lodDelta)].visible.size() > 0 && !plainRayCasting)
	{
		sampling_blc_s = glm::vec2(0, 0);
		sampling_trc_s = glm::vec2(sw, sh);

		int lw, lh;
		glm::vec3 c1, c2;
		LOD.get_lod_width_and_hight(std::floor(current_lod-lodDelta), lw, lh);
		LOD.myTiles[std::floor(current_lod-lodDelta)].get_blc_and_trc_of_viible_tiles(c1, c2, lw, lh);
		sampling_blc_l = glm::vec2(c1 - (LOD.myTiles[std::floor(current_lod-lodDelta)].T[0].c - glm::vec3(0.5f*tile_w, 0.5*tile_h, 0)));// +LOD.obj2pixel(glm::vec2(cameraOffset), std::floor(current_lod));
	}
	else if (plainRayCasting)
	{
		sampling_blc_s = glm::vec2(0, 0);
		sampling_trc_s = glm::vec2(windowSize.x, windowSize.y);

		sampling_blc_l = glm::vec2(0, 0);
	}
}
inline void quantizeSamples()
{
	// reduction to histogram
	if ((LOD.myTiles[std::floor(current_lod)].visible.size() > 0) && (!plainRayCasting) && (!pointCloudRendering))
	{
		if (cachedRayCasting)
		{
			computeTiledRaycastingCache(samplingViewMat);
			if (lowestSampleCount%samplesPerRun == 0)   //we downsample in the ceil only once for each time we run the sampling loop
			{
#if 1
				computeTiledRaycastingCache_C(samplingViewMat);
#endif
#if 0
				{
					std::cout << "started test case" << std::endl;
					timings.clear();

					for (int i = 0; i < 1000; i++)
					{
						glFinish();
						w2.StartTimer();

						computeTiledRaycastingCache_C(samplingViewMat);

						glFinish();
						w2.StopTimer();
						timings.push_back(w2.GetElapsedTime());
					}

					std::cout << std::accumulate(timings.begin(), timings.end(), 0.0) / 1000.0 << std::endl;
				}
#endif
			}
		}
		else
		{
			if (quantizeFlag)
				computeNDFCache();
			//we downsample only when we zoom out
			//we downsample in the ceil only once for each time we run the sampling loop
			if ((std::floor(current_lod) > std::floor(prev_lod)) && (lowestSampleCount%samplesPerRun == 0))
			{

#if 1
				computeNDFCache_Ceil();
#endif
#if 0
			{
					std::cout << "started test case" << std::endl;
					timings.clear();

					for (int i = 0; i < 1000; i++)
					{
						glFinish();
						w2.StartTimer();

						computeNDFCache_Ceil();

						glFinish();
						w2.StopTimer();
						timings.push_back(w2.GetElapsedTime());
					}

					std::cout << std::accumulate(timings.begin(), timings.end(), 0.0) / 1000.0 << std::endl;
			}
#endif
			}
		}
	}
}

inline void computeRenderedRegion()
{
	if (LOD.myTiles[std::floor(current_lod)].visible.size() > 0 && !plainRayCasting)
	{
		//get obj per pixel
		glm::vec2 objPerPixel = glm::vec2((OrigR - OrigL) / windowSize.x, (OrigT - OrigB) / windowSize.y);
		glm::vec2 camTrcInInitialLod = glm::vec2(camPosi) + glm::vec2(OrigR, OrigT);
		camTrcInInitialLod.x *= 1.0f / objPerPixel.x;
		camTrcInInitialLod.y *= 1.0f / objPerPixel.y;

		glm::vec2 camBlcInInitialLod = glm::vec2(camPosi) + glm::vec2(OrigL, OrigB);
		camBlcInInitialLod.x *= 1.0f / objPerPixel.x;
		camBlcInInitialLod.y *= 1.0f / objPerPixel.y;

		//get camera position in current lod
		//glm::vec2 camTrcInLod = glm::vec2(camTrcInInitialLod.x*pow(2, LOD.initial_lod - current_lod), camTrcInInitialLod.y*pow(2, LOD.initial_lod - current_lod));
		//glm::vec2 camBlcInLod = glm::vec2(camBlcInInitialLod.x*pow(2, LOD.initial_lod - current_lod), camBlcInInitialLod.y*pow(2, LOD.initial_lod - current_lod));



		glm::vec2 camTrcInLod = camTrcInInitialLod;
		glm::vec2  camBlcInLod = camBlcInInitialLod;


		//get lod corners
		int lw, lh;
		LOD.get_lod_width_and_hight(current_lod, lw, lh);
		glm::vec2 lodBlc = glm::vec2(-.5*lw, -.5*lh);
		glm::vec2 lodTrc = glm::vec2(.5*lw, .5*lh);

		//if camera corners are within lod corners, we render whole screen, otherwise, we render only a part of it

		if (camBlcInLod.x < lodBlc.x)
		{
			blc_s.x = lodBlc.x - camBlcInLod.x;
		}
		else
		{
			blc_s.x = 0;
		}

		if (camBlcInLod.y < lodBlc.y)
		{
			blc_s.y = lodBlc.y - camBlcInLod.y;
		}
		else
		{
			blc_s.y = 0;
		}

		if (camTrcInLod.x <= lodTrc.x)
		{
			trc_s.x = windowSize.x;
		}
		else
		{
			trc_s.x = windowSize.x - (camTrcInLod.x - lodTrc.x);
		}


		if (camTrcInLod.y <= lodTrc.y)
		{
			trc_s.y = windowSize.y;
		}
		else
		{

			trc_s.y = windowSize.y - (camTrcInLod.y - lodTrc.y);
		}

		//now we compute in lod space
		glm::vec2 tvec = camBlcInLod - lodBlc;
		blc_l = glm::vec2(std::max(int(tvec.x), 0), std::max(int(tvec.y), 0));

	}
	else if (plainRayCasting && singleRay)
	{
		blc_s = glm::vec2(0, 0);
		trc_s = glm::vec2(windowSize.x, windowSize.y);
		blc_l = glm::vec2(0, 0);
	}
	else if (plainRayCasting)
	{
	}
}

bool initTweakBar() {
    bool ret = true;
    ret = (TwInit(TW_OPENGL_CORE, NULL) == 1);
    if (!ret) {
        printf("could not initialize AntTweakBar!\n");
    } else {
        theBar = TwNewBar("Options");
        TwAddVarCB(theBar, "rotation", TW_TYPE_QUAT4F, twSetQuat, twGetQuat, NULL, NULL);
        char buf[1024];
        sprintf_s(buf, 1024, "min=0 max=%u step=1", MaxFrameCount);
        TwAddVarRO(theBar, "maxTime", TW_TYPE_INT32, &MaxFrameCount, NULL);
        TwAddVarCB(theBar, "time", TW_TYPE_INT32, twSetTime, twGetTime, NULL, buf);
        
        //TwAddVarRW(theBar, "radius", TW_TYPE_FLOAT, &particleRadius, NULL);

        TwEnumVal binEnum[] = { { 0, "old" }, { 1, "spherical" }, { 2, "lambert AEA" } };
        TwType binType = TwDefineEnum("binType", binEnum, 3);
        TwAddVarCB(theBar, "binning", binType, twSetBinning, twGetBinning, NULL, "key=8");
        TwAddVarCB(theBar, "scaling", TW_TYPE_FLOAT, twSetRadiusScaling, twGetRadiusScaling, NULL, "keyincr=+ keydecr=- step=0.1 precision=5");
    }
    return ret;
}

void unInitTweakBar() {
    if (tweakbarInitialized) {
        TwDeleteBar(theBar);
        TwTerminate();
        tweakbarInitialized = false;
    }
}

void display()
{
	
	//if (!tweakbarInitialized) {
	//	tweakbarInitialized = initTweakBar();
	//}

	//check if a sphere is bigger than a pixel
	//if (switchToRaycasting)
	//{
	//	float tempCamDist = LOD.get_cam_dist(std::floor(current_lod));
	//	float tempmodelscale = initialCameraDistance / tempCamDist;

	//	glm::vec2 pixDim = glm::vec2(2.0f / windowSize.x, 2.0f / windowSize.y);
	//	if (particleCenters.size() > 0)
	//	{
	//		if (2.0f*tempmodelscale*particleScale* particleCenters[0].w > std::max(pixDim.x, pixDim.y))
	//		{
	//			if (!cachedRayCasting)
	//			{
	//				keyboard('u', 0, 0);
	//			}
	//		}
	//		else
	//		{
	//			if (cachedRayCasting)
	//			{
	//				keyboard('u', 0, 0);
	//			}
	//		}
	//	}
	//}


	//w2.StartTimer();
	if (singleRay || pointCloudRendering || singleRay_NDF)
	{
		maxSamplingRuns = 1;
	}
	else if (plainRayCasting)
	{
		//here we have no 'lod', this is for progressive raycasting
		//for now, we give it the maximum runs
		maxSamplingRuns = 1;
	}
	else
	{

		maxSamplingRuns = std::max(1.0*maxSamplingRunsFactor, pow(4, std::floor(current_lod - lodDelta))*maxSamplingRunsFactor);
        //printf("Setting maxSamplingRuns for screen dump to %u (%s)\n", cmdlineIterations > 0 ? cmdlineIterations : maxSamplingRuns, cmdlineIterations > 0 ? "from command line" : "from current lod");

		//debug
		//maxSamplingRuns = std::max(1.0, pow(2, std::floor(current_lod - lodDelta)));
		//end debug

		//debug
		//maxSamplingRuns = 16384;
		//end debug

		//if (!circularPattern)
		//{
			//if (maxSamplingRuns > square_pattern_sample_count)
			//{
			//	std::cout << "not enough samples!" << std::endl;
			//	std::cout << "sampling count should be " << maxSamplingRuns << " samples" << std::endl;
			//	std::cout << "sample buffer size is " << square_pattern_sample_count << std::endl;
			//}
		//}
		//else{
		//	if (maxSamplingRuns > circular_pattern_sample_count)
		//	{
		//		std::cout << "not enough samples!" << std::endl;
		//		std::cout << "sampling count should be " << maxSamplingRuns << " samples" << std::endl;
		//		std::cout << "sample buffer size is " << circular_pattern_sample_count << std::endl;
		//	}
		//}
		//maxSamplingRuns = 2;
			if (lowestSampleCount == maxSamplingRuns - 1)
			{
				//std::cout << "Finished sampling " << maxSamplingRuns << " samples" << std::endl;

				//out.open("visibilityStats.txt");
				//int lw, lh;
				//LOD.get_lod_width_and_hight(current_lod, lw, lh);
				//out << lw << " " << lh << std::endl;
				//for (int j = 0; j < pHits.size(); j++)
				//{
				//	out << j << " " << pHits[j].size() << std::endl;
				//}
				//out.close();

				std::cout << "Finished sampling " << maxSamplingRuns << " samples" << std::endl;
				//if (!screenSaved)
				//{
				//	//keyboard('n', 0, 0);
				//	//keyboard('M', 0, 0);
				//	saveScreen();
				//	//keyboard('n', 0, 0);
				//	//keyboard('M', 0, 0);
				//	screenSaved = true;
				//}
			}
	}



	//std::cout << "max s runs " << maxSamplingRuns << std::endl;
	//debug
	//maxSamplingRuns = 4;

	if (paused)
	{
		glutSwapBuffers();
		return;
	}

	// TODO: this is inefficient but whatever
	/*for (auto &transferFunction : transferFunctions)
	{
	transferFunction.UpdateTexture();
	}

	for (auto &barStage : barStages)
	{
	barStage.UpdateTexture();
	}*/

	//LightDir = CameraPosition({ lightRotationX, lightRotationY }, 1.0f);
	//LightDir.y = -LightDir.y;

	// update camera position
	camPosi = CameraPosition(cameraRotation, cameraDistance);
	camPosi += cameraOffset;
	camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;



	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);


	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glUseProgram(0);

#if 0
	{
		glFinish();
		w.StartTimer();
	}
#endif

	if (cull)
	{
		if (!plainRayCasting && lowestSampleCount < maxSamplingRuns)
		{			
			for (int i = 0; i < samplesPerRun; i++)
			{
				//compute the area to be downsampled
				computeSampledRegion();
				computeSampleParameters();
				//std::cout << "went in loop" << std::endl;
				if (lowestSampleCount <= maxSamplingRuns && noTilesVisible==false)
				{

					//debug
					//if (lowestSampleCount % 1000 == 0)
					//{
					//	std::cout << "finished " << lowestSampleCount << " out of " << maxSamplingRuns << std::endl;
					//}
					//end debug

					//if (samplingFlag)
					//{

					//HOM occlusion culling
					if (enableOcclusionCulluing)
					{
						removeOccludedCells();
						
					}
					/*if (lowestSampleCount == 0)
					{
					buildHOM();
					}
					else
					{
					for (int j = 0; j < cellsToRender.size(); j++)
					testAgainstHOM(cellsToRender[j]);

					std::cout << "sample number: " <<lowestSampleCount<< std::endl;
					}*/

					//sample the particles
					//save_SamplingTexture = true;
#if 1
					{
						glFinish();
						w.StartTimer();
					}
#endif
					if(streaming)
						sampleDataStreaming();
					else
						sampleData();

					//save_SamplingTexture = false;
#if 1  //to compute time taken per iteration
					{
						glFinish();
						w.StopTimer();
						if (timings.size() < 1000)
						{
							timings.push_back(w.GetElapsedTime());
							if (timings.size() == 1000)
							{
								double total_time = std::accumulate(timings.begin(), timings.end(), 0.0);
								std::cout << "total time is " << total_time / 1000.0 << std::endl;
							}
						}
					}
#endif
					//std::cout << "sampled data" << std::endl;
					//debugfunction();

					//}

					//if (quantizeFlag)
					//{
					//downsample
					quantizeSamples();
					//drawNDF();
					//}
				}
				else
				{
					

					break;
				}
			}
			//std::cout << "lowest sample count is " << lowestSampleCount << std::endl;
			//std::cout << "max is " << maxSamplingRuns << std::endl;
		}
		//else  //plain raycasing mode, 
		//{
			//computeSampledRegion();
			//computeSampleParameters();
			//sampleData();
		//}
	}

	if (noTilesVisible == false)
	{
		//compute area to be rendered
		computeRenderedRegion();




		//render data to screen
		renderData();
	}

#if 0  //to compute time taken per iteration
	{
		glFinish();
		w.StopTimer();
		if (timings.size() < 1000)
		{
			timings.push_back(w.GetElapsedTime());
			if (timings.size() == 1000)
			{
				double total_time = std::accumulate(timings.begin(), timings.end(), 0.0);
				std::cout << "total time is "<<total_time/1000.0<< std::endl;
			}
		}
	}
#endif

	//to generate the plot
#if 0
	{
		if (!op0)
		{
			op0 = true;
			if (cachedRayCasting && noCaching)
				outfile.open("dataSuperSampling_noCahe.txt");
			else if (cachedRayCasting)
				outfile.open("dataCachedRaycasting.txt");
			else
				outfile.open("dataNDFs.txt");

			w.StopTimer();
			iterationTimePair.push_back(std::make_pair(lowestSampleCount,w.GetElapsedTime()));
		}
		else
		{
			w.StopTimer();
			double accumlatedTime = iterationTimePair.back().second;
			iterationTimePair.push_back(std::make_pair(lowestSampleCount, accumlatedTime + w.GetElapsedTime()));


			accumlatedTime = iterationTimePair.back().second;

			//try different operations
			if (accumlatedTime > 1500 && !op1)  //change light direction and specular exponent
			{
				op1 = true;
				LightDir = LightDir + glm::vec3(.8, .6, .7);
				glm::normalize(LightDir);
				keyboard('^', 0, 0);

				//if (cachedRayCasting)
				//{
				//	initialize();
				//	preIntegrateBins();
				//	tile_based_culling(false);
				//	update_page_texture();
				//	reset();
				//}

			}

			if (accumlatedTime > 3000 && !op3)  //zoom in a lot
			{
				op3 = true;

				cameraDistance += 0.125f;

				prev_lod = current_lod;

				//debug
				//get cam distance that will take us to lower lod
				float target_lod = current_lod - 1.2;
				cameraDistance = LOD.get_cam_dist(target_lod);
				//end debug

				cameraDistance = std::max(LOD.min_cam_dist, std::min(cameraDistance, LOD.max_cam_dist - .3f));

				camPosi = CameraPosition(cameraRotation, cameraDistance);
				camPosi += cameraOffset;
				camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

				zoom_camPosi = camPosi;

				if (cachedRayCasting&&noCaching)
				{
					initialize();
				}

				tile_based_culling(false);
				tile_based_panning(false);

				update_page_texture();
				//if (int(prev_lod) - int(current_lod) != 0)
				reset();
				glutPostRedisplay();
			}

			if (accumlatedTime > 5000 && !op2)  //rotate
			{
				op2 = true;

				keyboard('q', 0, 0);
				keyboard('q', 0, 0);
				keyboard('q', 0, 0);
			}

			if (accumlatedTime > 7000 && !op4)  //zoom out a lot
			{
				op4 = true;

				cameraDistance += 0.125f;

				prev_lod = current_lod;

				//debug
				//get cam distance that will take us to lower lod
				float target_lod = current_lod +3.2;
				cameraDistance = LOD.get_cam_dist(target_lod);
				//end debug

				cameraDistance = std::max(LOD.min_cam_dist, std::min(cameraDistance, LOD.max_cam_dist - .3f));

				camPosi = CameraPosition(cameraRotation, cameraDistance);
				camPosi += cameraOffset;
				camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

				zoom_camPosi = camPosi;

				if (cachedRayCasting&&noCaching)
				{
					initialize();
				}

				tile_based_culling(false);
				tile_based_panning(false);

				update_page_texture();
				//if (int(prev_lod) - int(current_lod) != 0)
				reset();
				glutPostRedisplay();
			}


			if (accumlatedTime > 9000 && !op6)  //pan
			{
				op6 = true;
				if (cachedRayCasting&&noCaching)
				{
					initialize();
				}
				keyboard('g', 0, 0);
			}

			if (accumlatedTime > 12000 && !op5)  //change transfer function
			{
				op5 = true;

				keyboard('m', 0, 0);
			}
			

			if (accumlatedTime > 14500 && !op8)  //zoom out a bit
			{
				op8 = true;

				cameraDistance += 0.125f;

				prev_lod = current_lod;

				//debug
				//get cam distance that will take us to lower lod
				float target_lod = current_lod + .4;
				cameraDistance = LOD.get_cam_dist(target_lod);
				//end debug

				cameraDistance = std::max(LOD.min_cam_dist, std::min(cameraDistance, LOD.max_cam_dist - .3f));

				camPosi = CameraPosition(cameraRotation, cameraDistance);
				camPosi += cameraOffset;
				camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

				zoom_camPosi = camPosi;

				if (cachedRayCasting&&noCaching)
				{
					initialize();
				}

				tile_based_culling(false);
				tile_based_panning(false);

				update_page_texture();
				//if (int(prev_lod) - int(current_lod) != 0)
				reset();
				glutPostRedisplay();
			}
			

			if (accumlatedTime > 16000 && !op9)  //pan
			{
				op9 = true;
				if (cachedRayCasting&&noCaching)
				{
					initialize();
				}
				keyboard('g', 0, 0);
			}

			

			if (lowestSampleCount == maxSamplingRuns - 1 && !op7)  //write file
			{
				op7 = true;

				for (int i = 0; i < iterationTimePair.size(); i++)
				{
					outfile << iterationTimePair[i].first << " " << iterationTimePair[i].second << std::endl;
				}
				iterationTimePair.clear();
				outfile.close();
				std::cout << "finished writting data" << std::endl;
			}
		}
	}
#endif

#ifdef CORE_FUNCTIONALITY_ONLY
#else
	if (probeNDFsMode)
	{
		updateSimilarityLimits();
		
		//render section
		renderSelection();

		//render avg NDF
		renderAvgNDF();
	}


#endif


	if (lowestSampleCount == (cmdlineIterations > 0 ? cmdlineIterations : maxSamplingRuns - 1))
	{
		if (!screenSaved && cmdlineSaveImage)
		{
			saveScreen();
			screenSaved = true;
            if (cmdlineAutoQuit) {
                exit(0);
            }
		}
	}
	
	if (printScreen)
		saveScreen();

	//render transfer function
	renderTransferFunction();

	//render sampling bar
	if (noTilesVisible == false)
	{
		if (renderBarMode)
			renderBar();
	}

    if (tweakbarInitialized) 
	{
        TwRefreshBar(theBar);
        TwDraw();
    }

	glutSwapBuffers();



}
inline void computeSlice(slice& s, std::vector<glm::vec3>& verts,float radius,float depth)
{
	int i;
	float PI = 3.141592;
	int triangleAmount = s.radius*180.0f/PI; //# of triangles used to draw slice, 1 triangle per degree
	
	//GLfloat radius = 0.8f; //radius
	GLfloat twicePi = 2.0f * PI;
	verts.clear();

	//glBegin(GL_TRIANGLE_FAN);
	float deg = s.angle*180.0f / PI;


	//s.angle = 3*PI/4.0f;

	glm::mat3x3 R = glm::mat3x3(cos(s.angle - 0.5*s.radius), sin(s.angle - 0.5*s.radius), 0, -sin(s.angle - 0.5*s.radius), cos(s.angle - 0.5*s.radius), 0, 0, 0, 1);
	glm::vec2 vert;

	//draw slice
	for (i = 0; i <= triangleAmount; i = i + 2)
	{
		//center of circle
		verts.push_back(glm::vec3(0, 0, depth));

		if (i == 0)
		{
			vert = glm::vec2(radius * cos(i      *  s.radius / triangleAmount), radius * sin(i      * s.radius / triangleAmount));
			verts.push_back(glm::vec3(vert,depth));
			vert = glm::vec2(radius * cos((i +1)     *  s.radius / triangleAmount), radius * sin((i+1)      * s.radius / triangleAmount));
			verts.push_back(glm::vec3(vert,depth));
		}
		else
		{
			vert = glm::vec2(radius * cos((i-1)      *  s.radius / triangleAmount), radius * sin((i-1)      * s.radius / triangleAmount));
			verts.push_back(glm::vec3(vert, depth));
			vert = glm::vec2(radius * cos((i + 1)     *  s.radius / triangleAmount), radius * sin((i + 1)      * s.radius / triangleAmount));
			verts.push_back(glm::vec3(vert, depth));
		}
	}

	//rotate slice based on position
	for (int i = 0; i < verts.size(); i++)
	{
		verts[i] = R*verts[i];
	}
}
inline void computeDisk(GLfloat x, GLfloat y, GLfloat radius, std::vector<glm::vec3>& verts,float depth)
{

	int i;
	int triangleAmount = 360; //# of triangles used to draw circle
	float PI = 3.141592;
	//GLfloat radius = 0.8f; //radius
	GLfloat twicePi = 2.0f * PI;
	verts.clear();

	//glBegin(GL_TRIANGLE_FAN);
	

	for (i = 0; i <= triangleAmount; i=i+2) 
	{
		//center of circle
		verts.push_back(glm::vec3(x, y, depth));

		if (i == 0)
		{
			verts.push_back(glm::vec3(x + (radius * cos(i *  twicePi / triangleAmount)),
				y + (radius * sin(i * twicePi / triangleAmount)),
				depth));

			verts.push_back(glm::vec3(x + (radius * cos((i + 1) *  twicePi / triangleAmount)),
				y + (radius * sin((i + 1) * twicePi / triangleAmount)),
				depth));
		}
		else
		{
			verts.push_back(glm::vec3(x + (radius * cos((i - 1) *  twicePi / triangleAmount)),
				y + (radius * sin((i - 1) * twicePi / triangleAmount)),
				depth));
			verts.push_back(glm::vec3(x + (radius * cos((i + 1) *  twicePi / triangleAmount)),
				y + (radius * sin((i + 1) * twicePi / triangleAmount)),
				depth));
		}
	}
}

inline void renderNdfExplorer()
{

	auto leftOriginal = -0.5f;
	auto rightOriginal = 0.5f;
	auto bottomOriginal = -0.5f;// / aspectRatio;
	auto topOriginal = 0.5f;// / aspectRatio;

	//fetch data to render

	//first draw background disk
	Helpers::Gl::BufferHandle backGroundDiskBuffer;
	Helpers::Gl::BufferHandle foreGroundDiskBuffer;
	std::vector<glm::vec3> backGroundDisk;
	std::vector<glm::vec3> foreGroundDisk;
	
	computeDisk(0, 0, 0.5, backGroundDisk, 0);
	Helpers::Gl::MakeBuffer(backGroundDisk, backGroundDiskBuffer);

	//draw middle disk
	computeDisk(0, 0, disks[0].radius, foreGroundDisk, .2);
	Helpers::Gl::MakeBuffer(foreGroundDisk, foreGroundDiskBuffer);

	//then draw slices
	std::vector<Helpers::Gl::BufferHandle> slicesBuffer(slices.size());
	std::vector<std::vector<glm::vec3>> sliceVerts(slices.size());

	for (int i = 0; i < slices.size(); i++)
	{
		computeSlice(slices[i], sliceVerts[i], 0.5, 0.1);
		Helpers::Gl::MakeBuffer(sliceVerts[i], slicesBuffer[i]);
	}
	
	//render data
	{
		//bind fbo
		glBindFramebuffer(GL_FRAMEBUFFER, ndfExplorer.FrameBufferObject);
		glViewport(0, 0, ndfExplorer.Width, ndfExplorer.Height);

		auto projectionMatrix = glm::ortho(leftOriginal, rightOriginal, bottomOriginal, topOriginal, nearPlane, farPlane);
		auto viewMatrix       = glm::lookAt(glm::vec3(0,0,100), glm::vec3(0,0,0), glm::vec3(0,1,0));

		// clear render target
		glClearColor(0.f, 0.f, 0.f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


		glEnable(GL_DEPTH_TEST);
		glFrontFace(GL_CCW);

		GLuint programHandle;
		programHandle = ndfExplorerShader->GetGlShaderProgramHandle();

		assert(glUseProgram);
		glUseProgram(programHandle);

		glUniformMatrix4fv(glGetUniformLocation(programHandle, "ModelView"), 1, GL_FALSE, &viewMatrix[0][0]);
		glUniformMatrix4fv(glGetUniformLocation(programHandle, "Projection"), 1, GL_FALSE, &projectionMatrix[0][0]);

		//draw background disk
		glm::vec3 bcolor = glm::vec3(.9, .9, .9);// glm::vec3(217 / 255.f, 217 / 255.f, 217 / 255.f);
		glUniform3fv(glGetUniformLocation(programHandle, "color"), 1, &bcolor[0]);
		backGroundDiskBuffer.Render(GL_TRIANGLES);

		//draw slices
		for (int i = 0; i < slices.size(); i++)
		{
			glUniform3fv(glGetUniformLocation(programHandle, "color"), 1, &slices[i].color[0]);
			slicesBuffer[i].Render(GL_TRIANGLES);
		}

		//draw foreground disk
		glUniform3fv(glGetUniformLocation(programHandle, "color"), 1, &disks[0].color[0]);
		foreGroundDiskBuffer.Render(GL_TRIANGLES);

#if 0//debug
		// Make the BYTE array, factor of 3 because it's RBG.
		BYTE* pixels = new BYTE[2 * ndfExplorer.Width*ndfExplorer.Height];
		float* fPixels = new float[4 * ndfExplorer.Width*ndfExplorer.Height];
		glReadPixels(0, 0, ndfExplorer.Width, ndfExplorer.Height, GL_RGBA, GL_FLOAT, fPixels);

		for (int i = 0; i < 4 * ndfExplorer.Width*ndfExplorer.Height; i = i + 4)
		{
			int j = i / 2;
			pixels[j] = fPixels[i] * 255;
			pixels[j + 1] = fPixels[i + 1] * 255;

			if (fPixels[i + 1] != 0)
			{
				fPixels[i + 1] = fPixels[i + 1];
			}
		}

		// Convert to FreeImage format & save to file
		FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, ndfExplorer.Width, ndfExplorer.Height, 2 * ndfExplorer.Width, 16, 0x00FF, 0xFF00, 0xFFFF, true);

		std::string name = "samplingTexture" + std::to_string(samplingTextureSaves) + ".bmp";
		samplingTextureSaves++;
		FreeImage_Save(FIF_BMP, image, name.c_str(), 0);

		// Free resources
		FreeImage_Unload(image);
		delete[] pixels;
#endif

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}
}

inline void removeOccludedCells()
{
	//clear cells to be erased and cells to be rendered to HOM
	cellsTbe.clear();
	cellsTbr.clear();

	if (buildHOMFlag)
	{
		buildHOM();
		buildHOMFlag = false;
		std::cout << "HOM Built!" << std::endl;
	}
	else
	{
		std::cout << "Testing Against HOM!" << std::endl;
		//start by testing the root node
		int tbts = cellsTbt.size();

		for (int i = 0; i < tbts; i++)
		{
			testAgainstHOM(cellsTbt.front(), cellsTbe,cellsTbr);
			cellsTbt.pop();
		}

		//remove cells to be removed from cellsTbe
		{
 			std::vector<int> l;
			//cell is not visible and should be removed
			for (int j = 0; j < cellsTbe.size(); j++)
			{
				global_tree.getLeavesWithCommonAncestor(cellsTbe[j], l);
				for (int k = 0; k < l.size(); k++)
				{
					for (int i = 0; i < cellsToRender.size(); i++)
					{
						if (cellsToRender[i] == l[k])
						{	
							totalParticlesRemoved += global_tree.nodes[l[k]].Points.size();
							//std::cout << "cell to be erased center is: " << global_tree.nodes[cellsTbe[j]].center.x << ", " << global_tree.nodes[cellsTbe[j]].center.y << ", " << global_tree.nodes[cellsTbe[j]].center.z << std::endl;
							cellsToRender.erase(cellsToRender.begin() + i);
							break;
						}
					}
				}
			}

			std::cout <<"Removed "<< totalParticlesRemoved <<" particles, Cells reduced to : " << cellsToRender.size() <<" cell, originally were : " << global_tree.leaves.size() <<" cell" <<std::endl;
			std::cout << "sample number: " << lowestSampleCount << std::endl;
		}

		//render cells to HOM that should be rendered from cellsTbr
		renderCelltoHOM(cellsTbr);
		enableOcclusionCulluing = false;
	}
}

inline void renderAvgNDF()
{
	auto viewMatrix = glm::lookAt(camPosi, camTarget, camUp);

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, avgNDFRenderingShader->getSsboBiningIndex(avgNDF_ssbo), avgNDF_ssbo);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, avgNDFRenderingShader->getSsboBiningIndex(colorMap_ssbo), colorMap_ssbo);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, avgNDFRenderingShader->getSsboBiningIndex(binAreas_ssbo), binAreas_ssbo);

	auto programHandle = avgNDFRenderingShader->GetGlShaderProgramHandle();
	glUseProgram(programHandle);

	glUniform2i(glGetUniformLocation(programHandle, "histogramDiscretizations"), histogramResolution.x, histogramResolution.y);
	glUniform1i(glGetUniformLocation(programHandle, "binningMode"), binning_mode);
	glUniform1i(glGetUniformLocation(programHandle, "colorMapSize"), colorMapSize);

	glViewport(windowSize.x - windowSize.y / 4, windowSize.y - windowSize.y / 4 + -4 * windowSize.y / 6, windowSize.y / 6, windowSize.y / 6);
	glClear(GL_DEPTH_BUFFER_BIT);
	glBindTexture(GL_TEXTURE_2D,avgNDFTexture);


	glBindVertexArray(avgNDFVao);
	//glPointSize(0.1f);
	//glDrawArrays(GL_POINTS, 0, selectedPixels.size());
	quadHandle.Render();
	glBindVertexArray(0);


	glBindTexture(GL_TEXTURE_2D, 0);
	glUseProgram(0);
}

inline void renderBar()
{
	auto viewMatrix = glm::lookAt(camPosi, camTarget, camUp);
	auto programHandle = barRenderingShader->GetGlShaderProgramHandle();
	glUseProgram(programHandle);

	glm::ivec2 transferFunctionOffset = glm::ivec2(0, 0);
	int currentIndex = 0;

	glViewport(.025f*windowSize.x, .15f*windowSize.y,.025*windowSize.x,.7*windowSize.y);

	glClear(GL_DEPTH_BUFFER_BIT);

	//decide on which image to render
	int barStage = (float(lowestSampleCount) /float( maxSamplingRuns))*(barStages.size()-1);
	
	//std::cout <<lowestSampleCount<<", "<<maxSamplingRuns<<", "<<barStage << std::endl;
	activeBarStage = &barStages[barStage];

	glBindTexture(GL_TEXTURE_2D, activeBarStage->transferFunctionTexture.Texture);

	glUniform1i(glGetUniformLocation(programHandle, "barSampler"), 0);

	quadHandle.Render();

	glBindTexture(GL_TEXTURE_2D, 0);
	glUseProgram(0);
}
inline void updateSimilarityLimits()
{
	//download similarity ssbo
	std::vector<float> simLimits(2 + windowSize.x*windowSize.y, 0);

	auto ssbo = simLimitsF_ssbo;
	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
		auto readMap = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
		memcpy(reinterpret_cast<char*>(&simLimits[0]), readMap, simLimits.size() * sizeof(*simLimits.begin()));

		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	unsigned int minVal = 100000000;
	float maxVal = -1;

	for (int i = 0; i < windowSize.x*windowSize.y; i++)
	{
		if (simLimits[2 + i]>maxVal)
			maxVal = simLimits[2 + i];

		if (simLimits[2 + i] < minVal && simLimits[2 + i] >= 0)
			minVal = simLimits[2 + i];
	}


	simLimits[0] = minVal;
	simLimits[1] = maxVal;

	//upload limits_ssbo
	{
		auto Size = size_t(2 + windowSize.x*windowSize.y);
		auto ssboSize = Size*sizeof(float);
		auto ssbo = simLimitsF_ssbo;
		{
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
			GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
			memcpy(p, reinterpret_cast<char*>(&simLimits[0]), simLimits.size() * sizeof(*simLimits.begin()));
			glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
		}
	}

}
inline void debugfunction()
{
	return;
	////download sampling texture
	////if (lowestSampleCount > 2143)
	//{
	//	unsigned char* t = new unsigned char[2 * rayCastingSolutionRenderTarget.Width * rayCastingSolutionRenderTarget.Height];

	//	for (int i = 0; i < 2 * rayCastingSolutionRenderTarget.Width*rayCastingSolutionRenderTarget.Height; i++)
	//	{
	//		t[i] = 0;
	//	}

	//	glBindFramebuffer(GL_FRAMEBUFFER, rayCastingSolutionRenderTarget.FrameBufferObject);
	//	glBindTexture(GL_TEXTURE_2D, rayCastingSolutionRenderTarget.RenderTexture);
	//	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, t);

	//	sampleTexData.push_back(std::vector<glm::vec4>(0));
	//	for (int i = 0; i < rayCastingSolutionRenderTarget.Width*rayCastingSolutionRenderTarget.Height * 2; i = i + 2)
	//	{
	//		if (t[i]>0 || t[i + 1] > 0 || t[i + 2] > 0)
	//		{
	//			sampleTexData[sampleTexData.size() - 1].push_back(glm::vec4(t[i], t[i + 1], t[i + 2], t[i + 3]));
	//			//sample size
	//			out << sampleTexData[sampleTexData.size() - 1].size() << std::endl;
	//			for (int j = 0; j < sampleTexData[sampleTexData.size() - 1].size(); j++)
	//			{
	//				out << sampleTexData[sampleTexData.size() - 1][j].x << std::endl;
	//				out << sampleTexData[sampleTexData.size() - 1][j].y << std::endl;

	//				if (sampleTexData[sampleTexData.size() - 1][j].x<0 || sampleTexData[sampleTexData.size() - 1][j].x>1 || sampleTexData[sampleTexData.size() - 1][j].y<0 || sampleTexData[sampleTexData.size() - 1][j].y>1)
	//					out << "#############################################################################" << std::endl;
	//			}
	//		}
	//	}

	//	delete[]t;
	//	glBindFramebuffer(GL_FRAMEBUFFER, 0);


	//	//int bins[64];
	//	//glm::vec2 quantizedRay;
	//	//int histogramIndexCentral;

	//	//for (int i = 0; i < 64; i++) bins[i] = 0;

	//	//for (int i = 0; i < sampleTexData.size(); i++)
	//	//{
	//	//	//now we bin the samples
	//	//	quantizedRay = glm::vec2(sampleTexData[i][0].x * 8.0f, sampleTexData[i][0].y * 8.0f);

	//	//	histogramIndexCentral = int(quantizedRay.y) * 8 + int(quantizedRay.x);

	//	//	bins[histogramIndexCentral]++;
	//	//}

	//}
}

void createFilter(float radius, double stdv)
{
	//implement what matlap does for 'fspecial' function, given 'gaussian' as a parameter
	
	if (gk != NULL)
	{
		//delte previous filter
		for (int i = 0; i < gkDim; i++)
			delete[] gk[i];
		delete gk;
	}

	int dim = sqrt(square_pattern_sample_count);
	//alocate new filter
	gk = new float*[dim];
	for (int i = 0; i < dim; i++)
		gk[i] = new float[dim];

	gkDim = dim;
	double r, s = 2.0 * stdv * stdv;  // Assigning standard deviation to 1.0
	double sum = 0.0;   // Initialization of sun for normalization
	int halfDim = dim / 2;
	

	for (int i = 0; i < dim; i++) // Loop to generate 2*halfdimx2*halfdim kernel
	{
		for (int j = 0; j < dim; j++)
		{
			float x = ((float(i)+.5)/float(dim))*2*radius-radius;    //-radius<x<radius
			float y = ((float(j)+.5)/float(dim))*2*radius-radius;
			r = sqrt(x*x + y*y);
			gk[i][j] = (exp(-(r*r) / s))/ (M_PI * s);
			sum += gk[i][j];
		}
	}

	for (int i = 0; i < dim; ++i) // Loop to normalize the kernel
		for (int j = 0; j < dim; ++j)
			gk[i][j] /= sum;

		std::cout << "created a gaussian filter of " << dim*dim << " samples, and with standard deviation: " << stdv << std::endl;
}

inline void computeSampleParameters()
{
	if (!plainRayCasting)
	{
		//old

		//for all visible tiles, get the maximum number of samples in a tile,
		//that will be the sample index for all visible tiles so that we ensure that
		//we don't sample at previous locations

		int tile_indx;
		int sIndx;
		int n, nMax = std::numeric_limits<int>::min(), nMin = std::numeric_limits<int>::max();


		//download buffer
		//download img data
		std::vector<float> data;
		auto Size = size_t((phys_tex_dim)* (phys_tex_dim));
		data.resize(Size, 0.0f);

		if (circularPattern)
		{
			auto ssbo = circularPatternSampleCount_ssbo;
			{
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
				auto readMap = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
				memcpy(reinterpret_cast<char*>(&data[0]), readMap, data.size() * sizeof(*data.begin()));

				glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
			}
		}
		else
		{
			auto ssbo = SampleCount_ssbo;
			{
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
				auto readMap = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
				memcpy(reinterpret_cast<char*>(&data[0]), readMap, data.size() * sizeof(*data.begin()));

				glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
			}
		}

		noTilesVisible = true;

		for (int i = 0; i < occupied.size(); i++)
		{
			if (occupied[i].first.first)
			{
				if (occupied[i].first.second.x == std::floor(current_lod-lodDelta))
				{
					//search for tile in visible
					bool found = false;
					for (int j = 0; j < LOD.myTiles[std::floor(current_lod-lodDelta)].visible.size(); j++)
					{
						if (LOD.myTiles[std::floor(current_lod-lodDelta)].visible[j] == occupied[i].first.second.y)
						{
							found = true;
							break;
						}
					}

					if (found)
					{
						noTilesVisible = false;
						//get number of samples in tile from ssbo
						//glm::vec2 twoDIndx = glm::vec2(i%(phys_tex_dim/tile_w),i/(phys_tex_dim/tile_h));
						//twoDIndx = glm::vec2(twoDIndx.x*tile_w,twoDIndx.y*tile_h);
						//n = data[twoDIndx.y*phys_tex_dim+twoDIndx.x];

						//if (data[twoDIndx.y*phys_tex_dim + twoDIndx.x] != data[i*tile_h*tile_w])
						//	n = n;

						n = data[i*tile_w*tile_h];

						if (n > nMax)
							nMax = n;

						if (n < nMin)
							nMin = n;
					}
				}
			}
		}


		highestSampleCount = nMax % sample_count;
		lowestSampleCount = nMin;


		//std::cout << "lowest sample count is " << lowestSampleCount << std::endl;
		//std::cout << "max sampling runs is " << maxSamplingRuns << std::endl;

		//compute sample index to sample at
		//this can't only be 'nMax' since if the tiles with nMax finish sampling, the sample index will never go above maxsampling runs, so all new tiles that are visible with the tiles taht have nMax samples
		//will be sampled at a constant location, nMax.
		if (nMax == maxSamplingRuns)
			sIndx = (nMin + nMax) % sample_count;
		else
			sIndx = nMax%sample_count;

		//static const int M = 64;
		//float x = (float(int(samples_data[sIndx]+0.1f) % M) + 0.5f) / float(M) - 0.5f;
		//float y = (float(int(samples_data[sIndx]+0.1f) / M) + 0.5f) / float(M) - 0.5f;
		//sPos = glm::vec2(x, y);

		
		//sPos = glm::vec2((fmod(samples_data[sIndx], dim)+0.5f) / dim, (floor(samples_data[sIndx] / dim)+0.5f) / dim);
		//sPos = glm::vec2((float(sIndx%iDim) + 0.5f) / dim, (float(sIndx / iDim)+0.5f) / dim);

		if (circularPattern)
		{
			//lowestSampleCount=sIndx = circularPatternIterator;
			//circularPatternIterator++;

			//float dim = circular_pattern_dim;
			//int iDim = int(dim);
			//sPos = glm::vec2((float(int(CircularSamples_data[sIndx]) % iDim) + 0.5f) / dim, (float(CircularSamples_data[sIndx] / iDim) + 0.5f) / dim);

			//sPos -= glm::vec2(circularPatternRadius,circularPatternRadius);  //so we shift data by values to be between -r and r in both dimensions
			//sampleW = CircularSamples_data_weights[sIndx];

			

			//new
			//do{
				//lowestSampleCount = sIndx = circularPatternIterator;
				//circularPatternIterator++;
				float dim = sqrt(sample_count);
				int iDim = int(dim);
				sPos = glm::vec2((float(int(samples_data[sIndx]) % iDim) + 0.5f) / dim, (float(samples_data[sIndx] / iDim) + 0.5f) / dim);

				//sPos should now be from (-r,-r) to (r,r)
				//1-scale sPos by '2r'
				sPos *= 2 * filterRadius;
				//subtract 'r' from each dimension
				sPos -= glm::vec2(filterRadius, filterRadius);
				sampleW = gk[int(sIndx%iDim)][int(sIndx / iDim)];

				//std::cout << sPos.x << ", " << sPos.y << std::endl;
				//std::cout << sampleW << std::endl;
			//} while (sampleW==0);
		}
		else
		{
			float dim = sqrt(sample_count);
			int iDim = int(dim);
			sPos = glm::vec2((float(int(samples_data[sIndx]) % iDim) + 0.5f) / dim, (float(samples_data[sIndx] / iDim) + 0.5f) / dim);

			//new
			/*glm::vec2 s = glm::vec2(.01, .01);
			glm::vec2 e = glm::vec2(.9,.9);

			float u = (sIndx % 64) / 64.0f;
			float v = (sIndx / 64.0f) / 64.0f;

			sPos = s + glm::vec2(u, v)*(e - s);*/

			//if (lowestSampleCount == 2143)
			//	int x = 9;

			//int count = 0;
			//float s = 0;// .00001;
			//float e = 1;// 0.9999f;
			//for (float i = s; i <= e && count<=lowestSampleCount; i = i + (e - s) / 64.0f)
			//{
			//	for (float j = s; j <= e && count<=lowestSampleCount; j = j + (e - s) / 64.0f)
			//	{
			//		count++;
			//		sPos = glm::vec2(i, j);
			//	}
			//}
			//end new

			//sPos = glm::vec2((float(sIndx%int(dim)) ) / dim, (float(sIndx / dim)) / dim);
			sPos -= glm::vec2(0.5, 0.5);  //so we shift data by values of -.5 to .
		}
		

		//if (sIndx < maxSamplingRuns)
		//	printf("\r %i", sIndx);
		//else
		//	printf("\rdone        \n");
	}
	else if (plainRayCasting && singleRay)
	{
		sPos = glm::vec2(0, 0);
	}
	else if (plainRayCasting)
	{
		lowestSampleCount++;
		int sIndx = lowestSampleCount;

		//static const int M = 64;
		//float x = (float(int(samples_data[sIndx]+0.1f) % M) + 0.5f) / float(M) - 0.5f;
		//float y = (float(int(samples_data[sIndx]+0.1f) / M) + 0.5f) / float(M) - 0.5f;
		//sPos = glm::vec2(x, y);

		float dim = sqrt(sample_count);
		int iDim = int(dim);
		//sPos = glm::vec2((fmod(samples_data[sIndx], dim)+0.5f) / dim, (floor(samples_data[sIndx] / dim)+0.5f) / dim);
		//sPos = glm::vec2((float(sIndx%iDim) + 0.5f) / dim, (float(sIndx / iDim)+0.5f) / dim);
		sPos = glm::vec2((float(int(samples_data[sIndx]) % iDim) + 0.5f) / dim, (float(samples_data[sIndx] / iDim) + 0.5f) / dim);

		//new
		/*glm::vec2 s = glm::vec2(.01, .01);
		glm::vec2 e = glm::vec2(.9,.9);

		float u = (sIndx % 64) / 64.0f;
		float v = (sIndx / 64.0f) / 64.0f;

		sPos = s + glm::vec2(u, v)*(e - s);*/

		//if (lowestSampleCount == 2143)
		//	int x = 9;

		//int count = 0;
		//float s = 0;// .00001;
		//float e = 1;// 0.9999f;
		//for (float i = s; i <= e && count<=lowestSampleCount; i = i + (e - s) / 64.0f)
		//{
		//	for (float j = s; j <= e && count<=lowestSampleCount; j = j + (e - s) / 64.0f)
		//	{
		//		count++;
		//		sPos = glm::vec2(i, j);
		//	}
		//}
		//end new

		//sPos = glm::vec2((float(sIndx%int(dim)) ) / dim, (float(sIndx / dim)) / dim);
		sPos -= glm::vec2(0.5, 0.5);  //so we shift data by values of -.5 to .

		//if (sIndx < maxSamplingRuns)
		//	printf("\r %i", sIndx);
		//else
		//	printf("\rdone        \n");
	}
}

inline void renderData()
{
	// projection matrix

	auto viewMatrix = glm::lookAt(camPosi, camTarget, camUp);
	auto modelViewMatrix = viewMatrix;
	auto projectionMatrix = glm::perspective(fieldOfView, aspectRatio, nearPlane, farPlane);
	auto modelViewProjectionMatrix = projectionMatrix * viewMatrix;

	glFrontFace(GL_CW);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(0, 0, windowSize.x, windowSize.y);
	//glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	/*w.StopTimer();
	std::cout << "F elapsed time: " << w.GetElapsedTime() << " ms" << std::endl;
	w.StartTimer();*/

	//if (lowestSampleCount == maxSamplingRuns)
	//{
	//	w.StartTimer();
	//}


	auto &tree = ndfTree;
	auto &l0 = tree.GetLevels().front();

	//LARGE_INTEGER freq,t1,t2,elapsed;
	//QueryPerformanceFrequency(&freq);
	//QueryPerformanceCounter(&t1);

	//GLuint ssboBindingPointIndex = 0;
	//#define HIGH_RES_TEST

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, renderingShader->getSsboBiningIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject()), ndfTree.GetLevels().front().GetShaderStorageBufferOject());
	//ssboBindingPointIndex = 2;
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, renderingShader->getSsboBiningIndex(progressive_raycasting_ssbo), progressive_raycasting_ssbo);
	//ssboBindingPointIndex = 3;
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, renderingShader->getSsboBiningIndex(preIntegratedBins_ssbo), preIntegratedBins_ssbo);

	//ssboBindingPointIndex = 5;
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, renderingShader->getSsboBiningIndex(binAreas_ssbo), binAreas_ssbo);

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, renderingShader->getSsboBiningIndex(ndfColors_ssbo), ndfColors_ssbo);

	//ssboBindingPointIndex = 6;
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, region_ssbo);

#ifdef CORE_FUNCTIONALITY_ONLY
#else
	//ssboBindingPointIndex = 7;
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, renderingShader->getSsboBiningIndex(avgNDF_ssbo), avgNDF_ssbo);

	//ssboBindingPointIndex = 9;
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, renderingShader->getSsboBiningIndex(colorMap_ssbo), colorMap_ssbo);

	//ssboBindingPointIndex = 10;
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, renderingShader->getSsboBiningIndex(simLimitsF_ssbo), simLimitsF_ssbo);
#endif

	auto programHandle = renderingShader->GetGlShaderProgramHandle();
	//if (renderSparse) {
	//	programHandle = eNtfRenderingShader->GetGlShaderProgramHandle();
	//}

	glUseProgram(programHandle);

	for (auto &transferFunction : transferFunctions)
	{
		glBindTexture(GL_TEXTURE_2D, activeTransferFunction->transferFunctionTexture.Texture);// activeTransferFunction->transferFunctionTexture.Texture);
		glUniform1i(glGetUniformLocation(programHandle, "normalTransferSampler"), 0);
		break;
	}

	



	//Mohamed's Code

	//glBindImageTexture(0, MyRenderTarget.RenderTexture, 0, false, 0, GL_READ_ONLY, GL_RGBA16);
	//glUniform1i(glGetUniformLocation(programHandle, "tileTex"), 0);

	glBindImageTexture(3, rayCastingSolutionRenderTarget.RenderTexture, 0, false, 0, GL_READ_ONLY, GL_RGBA32F);
	glUniform1i(glGetUniformLocation(programHandle, "tex"), 3);

	glBindImageTexture(1, Page_Texture, std::floor(current_lod), false, 0, GL_READ_ONLY, Page_texture_internalFormat);
	glUniform1i(glGetUniformLocation(programHandle, "floorLevel"), 1);

	glBindImageTexture(2, Page_Texture, std::floor(current_lod + 1), false, 0, GL_READ_ONLY, Page_texture_internalFormat);
	glUniform1i(glGetUniformLocation(programHandle, "ceilLevel"), 2);


	//glBindImageTexture(0, Transfer_Texture, 0, false, 0, GL_READ_ONLY, GL_RGBA8);
	//glUniform1i(glGetUniformLocation(programHandle, "transtex"), 0);

	int floor_w, ceil_w, floor_h, ceil_h;
	LOD.get_lod_width_and_hight(std::floor(current_lod), floor_w, floor_h);
	glUniform1i(glGetUniformLocation(programHandle, "floor_w"), floor_w);
	glUniform1i(glGetUniformLocation(programHandle, "floor_h"), floor_h);


	LOD.get_lod_width_and_hight(std::floor(current_lod + 1), ceil_w, ceil_h);
	glUniform1i(glGetUniformLocation(programHandle, "ceil_w"), ceil_w);
	glUniform1i(glGetUniformLocation(programHandle, "ceil_h"), ceil_h);

	glUniform1f(glGetUniformLocation(programHandle, "lod"), current_lod);

	//new
	glUniform2i(glGetUniformLocation(programHandle, "visible_region_blc"), visible_region_blc.x, visible_region_blc.y);
	glUniform2i(glGetUniformLocation(programHandle, "visible_region_trc"), visible_region_trc.x, visible_region_trc.y);
	glUniform3i(glGetUniformLocation(programHandle, "lod_blc"), lod_blc.x, lod_blc.y, lod_blc.z);
	glUniform3i(glGetUniformLocation(programHandle, "lod_trc"), lod_trc.x, lod_trc.y, lod_trc.z);

	glUniform3i(glGetUniformLocation(programHandle, "floor_blc"), floor_blc.x, floor_blc.y, floor_blc.z);
	glUniform3i(glGetUniformLocation(programHandle, "ceil_blc"), ceil_blc.x, ceil_blc.y, ceil_blc.z);

	glUniform2f(glGetUniformLocation(programHandle, "trc_s"), trc_s.x, trc_s.y);
	glUniform2f(glGetUniformLocation(programHandle, "blc_s"), blc_s.x, blc_s.y);
	glUniform2f(glGetUniformLocation(programHandle, "blc_l"), blc_l.x, blc_l.y);

	glUniform2f(glGetUniformLocation(programHandle, "sampling_trc_s"), sampling_trc_s.x, sampling_trc_s.y);
	glUniform2f(glGetUniformLocation(programHandle, "sampling_blc_s"), sampling_blc_s.x, sampling_blc_s.y);
	glUniform2f(glGetUniformLocation(programHandle, "sampling_blc_l"), sampling_blc_l.x, sampling_blc_l.y);
	//end new

	glUniform1i(glGetUniformLocation(programHandle, "win_w"), windowSize.x);
	glUniform1i(glGetUniformLocation(programHandle, "win_h"), windowSize.y);

	glUniform1i(glGetUniformLocation(programHandle, "visualizeTiles"), visualize_tiles);
	glUniform1i(glGetUniformLocation(programHandle, "probeNDFsMode"), probeNDFsMode);
	glUniform1i(glGetUniformLocation(programHandle, "showEmptyTiles"), showEmptyTiles);
	glUniform1i(glGetUniformLocation(programHandle, "ndfOverlayMode"), ndfOverlayMode);

	glUniform1i(glGetUniformLocation(programHandle, "tile_w"), tile_w);
	glUniform1i(glGetUniformLocation(programHandle, "tile_h"), tile_h);
	glUniform1i(glGetUniformLocation(programHandle, "phys_tex_dim"), phys_tex_dim);
	glUniform1f(glGetUniformLocation(programHandle, "binLimit"), binLimit);
	glBindTexture(GL_TEXTURE_2D, 0);
	//end Mohamed's code

	glUniform1i(glGetUniformLocation(programHandle, "colorMapSize"), colorMapSize);
	glUniform1i(glGetUniformLocation(programHandle, "renderMode"), renderMode);
	glUniform1i(glGetUniformLocation(programHandle, "zoomMode"), zoomMode);
	glUniform1f(glGetUniformLocation(programHandle, "zoomScale"), zoomScale);
	glUniform1f(glGetUniformLocation(programHandle, "aspectRatio"), aspectRatio);
	glUniform1f(glGetUniformLocation(programHandle, "ndfIntensityCorrection"), ndfIntensityCorrection);

	glUniform1i(glGetUniformLocation(programHandle, "plainRayCasting"), plainRayCasting);
	glUniform1i(glGetUniformLocation(programHandle, "cachedRayCasting"), cachedRayCasting);

	if (renderSparse)
	{
		{
			const GLuint ssboBindingPointIndex = 0;
			glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, eNtfOffsetSsbo);
		}

		{
		const GLuint ssboBindingPointIndex = 1;
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, eNtfGaussianSsbo);
	}

		glUniform1f(glGetUniformLocation(programHandle, "quantizationVarianceScale"), quantizationVarianceScale);

		glUniform1i(glGetUniformLocation(programHandle, "maxGaussians"), maxGaussians());
	}

	glUniform2f(glGetUniformLocation(programHandle, "zoomWindow"), zoomWindow.x, zoomWindow.y);
	glUniform1i(glGetUniformLocation(programHandle, "multiSamplingRate"), multiSamplingRate);

	//glUniformMatrix4fv(glGetUniformLocation(programHandle, "MVP"), 1, false, &modelViewProjectionMatrix[0][0]);
	//glUniformMatrix4fv(glGetUniformLocation(programHandle, "MV"), 1, false, &modelViewMatrix[0][0]);

	glUniform3fv(glGetUniformLocation(programHandle, "camPosi"), 1, &camPosi[0]);
	glUniform2fv(glGetUniformLocation(programHandle, "camAngles"), 1, &cameraRotation[0]);

	auto camDirection = glm::normalize(camTarget - camPosi);
	glUniform3fv(glGetUniformLocation(programHandle, "camDirection"), 1, &camDirection[0]);

	auto viewSpaceCamDirection = viewMatrix * glm::vec4(camDirection, 0.0f);
	viewSpaceCamDirection = glm::normalize(viewSpaceCamDirection);
	glUniform3fv(glGetUniformLocation(programHandle, "viewSpaceCamDirection"), 1, &viewSpaceCamDirection[0]);

	auto viewSpaceLightDir = viewMatrix * glm::vec4(LightDir, 0.0f);
	viewSpaceLightDir = glm::normalize(viewSpaceLightDir);

	glUniform3fv(glGetUniformLocation(programHandle, "viewSpaceLightDir"), 1, &viewSpaceLightDir[0]);

	glUniform2i(glGetUniformLocation(programHandle, "viewDiscretizations"), l0.GetViewDirectionResolution().x, l0.GetViewDirectionResolution().y);
	glUniform2i(glGetUniformLocation(programHandle, "histogramDiscretizations"), l0.GetHistogramResolution().x, l0.GetHistogramResolution().y);

	//int vx = Ndf::volumeResolutionX / std::pow(2, Ndf::MipLevelIndex);
	//int vy = Ndf::volumeResolutionX / std::pow(2, Ndf::MipLevelIndex);

	/*int vx =rayCastingSolutionRenderTarget.Width / multiSamplingRate;
	int vy =rayCastingSolutionRenderTarget.Height/ multiSamplingRate;*/

	int vx = windowSize.x / multiSamplingRate;
	int vy = windowSize.y / multiSamplingRate;

	glUniform1i(glGetUniformLocation(programHandle, "binningMode"), binning_mode);
	glUniform3i(glGetUniformLocation(programHandle, "spatialDiscretizations"), vx, vx, 1);
	glUniform2i(glGetUniformLocation(programHandle, "viewportSize"), windowSize.x, windowSize.y);


	progressiveSamplesCount = std::min(progressiveSamplesCount, maxSamplingRuns);

	glUniform1i(glGetUniformLocation(programHandle, "progressiveSamplesCount"), progressiveSamplesCount);
	progressiveSamplesCount++;

	glUniform1i(glGetUniformLocation(programHandle, "maxSamplingRuns"), maxSamplingRuns);

	//glUniform1f(glGetUniformLocation(programHandle, "quantizationVarianceScale"), l0.quantizationVarianceScale);

	auto right = glm::normalize(glm::cross(normalize(camTarget - camPosi), camUp));
	auto upInverted = -camUp;
	glUniform3fv(glGetUniformLocation(programHandle, "right"), 1, &right[0]);
	glUniform3fv(glGetUniformLocation(programHandle, "up"), 1, &upInverted[0]);

	// FIXME: quad handle does not work for some reason
	//QuadHandle.Render();
	//BoxHandle.Render();

	static const auto instances = 1;//20;
	static const auto spacing = 0.75f * glm::vec3(0.175f, 0.115f, 0.15f);
	static const auto scale = 1.0f / static_cast<float>(instances);
	static const auto halfInstance = instances / 2;

	/*w.StopTimer();
	std::cout << "G elapsed time: " << w.GetElapsedTime() << " ms" << std::endl;
	w.StartTimer();*/

	glFrontFace(GL_CCW);

	//const auto viewProjectionMatrix = projectionMatrix * viewMatrix;
	auto modelMatrix = glm::mat4x4();
	modelMatrix[0][0] = modelMatrix[1][1] = modelMatrix[2][2] = scale;
	for (auto z = 0; z < instances; ++z)
	{
		modelMatrix[3][2] = static_cast<float>(z - instances) * spacing.z;
		for (auto y = 0; y < instances; ++y)
		{
			modelMatrix[3][1] = static_cast<float>(y - halfInstance) * spacing.y;
			for (auto x = 0; x < instances; ++x)
			{
				modelMatrix[3][0] = static_cast<float>(x - halfInstance) * spacing.x;

				auto viewMatrix = glm::lookAt(camPosi, camTarget, camUp);
				auto modeViewMatrix = modelMatrix * viewMatrix;
				auto viewProjectionMatrix = projectionMatrix * viewMatrix;
				auto modelViewProjectionMatrix = viewProjectionMatrix * modelMatrix;

				glUniformMatrix4fv(glGetUniformLocation(programHandle, "ModelView"), 1, GL_FALSE, &modeViewMatrix[0][0]);
				glUniformMatrix4fv(glGetUniformLocation(programHandle, "ViewProjection"), 1, GL_FALSE, &viewProjectionMatrix[0][0]);
				glUniformMatrix4fv(glGetUniformLocation(programHandle, "Projection"), 1, GL_FALSE, &projectionMatrix[0][0]);
				glUniformMatrix4fv(glGetUniformLocation(programHandle, "MVP"), 1, false, &modelViewProjectionMatrix[0][0]);
				glUniformMatrix4fv(glGetUniformLocation(programHandle, "M"), 1, false, &modelMatrix[0][0]);


				quadHandle.Render();
				//boxHandle.Render();
				// NOTE: quad handle need ccw culling
				//glClearColor(1.0f, 0.f, 0.f, 1.0f);
				// clear render target
				//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			}
		}
	}

	glBindTexture(GL_TEXTURE_2D, 0);
	glColor3f(1, 0, 0);
	quadHandle.Render();

}

inline void  renderTransferFunction()
{
	if (renderTransfer)
	{
		auto viewMatrix = glm::lookAt(camPosi, camTarget, camUp);

		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, transferShader->getSsboBiningIndex(colorMap_ssbo), colorMap_ssbo);

		auto programHandle = transferShader->GetGlShaderProgramHandle();
		glUseProgram(programHandle);
		glUniform1f(glGetUniformLocation(programHandle, "specularExp"), specularExponent);
		glUniform1i(glGetUniformLocation(programHandle, "colorMapSize"), colorMapSize);
		glUniform1i(glGetUniformLocation(programHandle, "ndfOverlayMode"), ndfOverlayMode);

		

		//glBindTexture(GL_TEXTURE_2D, activeChromeTexture->transferFunctionTexture.Texture);
		//glUniform1i(glGetUniformLocation(programHandle, "chromeTexture"), 0);

		glBindImageTexture(0, ndfExplorer.RenderTexture, 0, false, 0, GL_READ_ONLY, GL_RGBA32F);
		glUniform1i(glGetUniformLocation(programHandle, "tex"), 0);

		if (renderMode == 2)
		{
			glm::ivec2 transferFunctionOffset = glm::ivec2(0, 0);
			int currentIndex = 0;
			//for (auto &transferFunction : transferFunctions)
			{
				glViewport(windowSize.x - windowSize.y / 4 + transferFunctionOffset.x, windowSize.y - windowSize.y / 4 + transferFunctionOffset.y, windowSize.y / 6, windowSize.y / 6);
				transferFunctionOffset += glm::ivec2(0, -windowSize.y / 6);

				glClear(GL_DEPTH_BUFFER_BIT);

				glBindTexture(GL_TEXTURE_2D, activeTransferFunction->transferFunctionTexture.Texture);

				glUniform1i(glGetUniformLocation(programHandle, "normalTransferSampler"), 0);
				glUniform1i(glGetUniformLocation(programHandle, "renderMode"), renderMode);
				glUniform1i(glGetUniformLocation(programHandle, "active"), (activeTransferFunctionIndex == currentIndex && transferFunctions.size() > 1));

				auto viewSpaceLightDir = viewMatrix * glm::vec4(LightDir, 0.0f);
				viewSpaceLightDir = glm::normalize(viewSpaceLightDir);
				glUniform3fv(glGetUniformLocation(programHandle, "lightViewSpace"), 1, &viewSpaceLightDir[0]);

				quadHandle.Render();
				//				++currentIndex;
			}
		}
		else
		{
			glViewport(windowSize.x - windowSize.y / 4, windowSize.y - windowSize.y / 4, windowSize.y / 6, windowSize.y / 6);

			

			glClear(GL_DEPTH_BUFFER_BIT);

			glUniform1i(glGetUniformLocation(programHandle, "normalTransferSampler"), 0);
			glUniform1i(glGetUniformLocation(programHandle, "renderMode"), renderMode);

			auto viewSpaceLightDir = viewMatrix * glm::vec4(LightDir, 0.0f);
			viewSpaceLightDir = glm::normalize(viewSpaceLightDir);
			glUniform3fv(glGetUniformLocation(programHandle, "lightViewSpace"), 1, &viewSpaceLightDir[0]);

			quadHandle.Render();
		}
		glBindTexture(GL_TEXTURE_2D, 0);
		glUseProgram(0);
	}
}

inline void  renderSelection()
{
	auto viewMatrix = glm::lookAt(camPosi, camTarget, camUp);
	auto programHandle = selectionShader->GetGlShaderProgramHandle();
	glUseProgram(programHandle);

	glViewport(0, 0, windowSize.x, windowSize.y);
	glClear(GL_DEPTH_BUFFER_BIT);
	glBindTexture(GL_TEXTURE_2D, selectionTexture);


	glBindVertexArray(selectedPixelsVao);
	//glPointSize(0.1f);
	glDrawArrays(GL_POINTS, 0, selectedPixels.size());
	glBindVertexArray(0);


	glBindTexture(GL_TEXTURE_2D, 0);
	glUseProgram(0);
}



void drawSelection()
{
	if (selectedPixels.size() > MaxAllowedSelectedPixels)
	{
		std::cout << "number of selected pixels exceeds maximum limit" << std::endl;
	}
	else
	{
		glBindBuffer(GL_ARRAY_BUFFER, selectedPixelsVbo);
		glBufferData(GL_ARRAY_BUFFER, selectedPixels.size() * sizeof(selectedPixels[0]), reinterpret_cast<const GLvoid*>(selectedPixels.data()), GL_DYNAMIC_DRAW);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}
}
void saveSamplingTexture()
{
	// Make the BYTE array, factor of 3 because it's RBG.
	BYTE* pixels = new BYTE[2 * rayCastingSolutionRenderTarget.Width * rayCastingSolutionRenderTarget.Height];
	float* fPixels = new float[4 * rayCastingSolutionRenderTarget.Width*rayCastingSolutionRenderTarget.Height];
	glReadPixels(0, 0, rayCastingSolutionRenderTarget.Width, rayCastingSolutionRenderTarget.Height, GL_RGBA, GL_FLOAT, fPixels);

	for (int i = 0; i < 4 * rayCastingSolutionRenderTarget.Width*rayCastingSolutionRenderTarget.Height; i = i + 4)
	{
		int j = i / 2;
		pixels[j] = fPixels[i] * 255;
		pixels[j + 1] = fPixels[i + 1] * 255;
	}

	// Convert to FreeImage format & save to file
	FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, rayCastingSolutionRenderTarget.Width, rayCastingSolutionRenderTarget.Height, 2 * rayCastingSolutionRenderTarget.Width, 16, 0x00FF, 0xFF00, 0xFFFF, true);

	std::string name = "samplingTexture" + std::to_string(samplingTextureSaves) + ".bmp";
	samplingTextureSaves++;
	FreeImage_Save(FIF_BMP, image, name.c_str(), 0);

	// Free resources
	FreeImage_Unload(image);
	delete[] pixels;
}
void saveHOM()
{
	// Make the BYTE array, factor of 3 because it's RBG.
	BYTE* pixels = new BYTE[2 * homRenderTarget.Width*homRenderTarget.Height];
	float* fPixels = new float[4 * homRenderTarget.Width*homRenderTarget.Height];
	glReadPixels(0, 0, homRenderTarget.Width, homRenderTarget.Height, GL_RGBA, GL_FLOAT, fPixels);

	for (int i = 0; i < 4 * homRenderTarget.Width*homRenderTarget.Height; i = i + 4)
	{
		int j = i / 2;
		pixels[j] = fPixels[i] * 255;
		pixels[j + 1] = fPixels[i + 1] * 255;
	}

	// Convert to FreeImage format & save to file
	FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, homRenderTarget.Width, homRenderTarget.Height, 2 * homRenderTarget.Width, 16, 0x00FF, 0xFF00, 0xFFFF, true);

	std::string name = "samplingTexture" + std::to_string(samplingTextureSaves) + ".bmp";
	samplingTextureSaves++;
	FreeImage_Save(FIF_BMP, image, name.c_str(), 0);

	// Free resources
	FreeImage_Unload(image);
	delete[] pixels;
}
void saveScreen()
{
#if 0
	// Make the BYTE array, factor of 3 because it's RBG.
	BYTE* pixels = new BYTE[3 * windowSize.x*windowSize.y];
	glReadPixels(0, 0, windowSize.x, windowSize.y, GL_BGR, GL_BYTE, pixels);

	// Convert to FreeImage format & save to file
	FIBITMAP* image = FreeImage_ConvertFromRawBits(pixels, windowSize.x, windowSize.y, 3 * windowSize.x, 24, 0xFF0000, 0x00FF00, 0x0000FF, false);

	std::string name = "screen" + std::to_string(samplingTextureSaves) + ".bmp";
	samplingTextureSaves++;
	FreeImage_Save(FIF_BMP, image, name.c_str(), 0);

	// Free resources
	FreeImage_Unload(image);
	delete[] pixels;
#else

	float* fPixels = new float[windowSize.x*windowSize.y*4];
	float* pixels = new float[windowSize.x*windowSize.y];
	glReadPixels(0, 0, windowSize.x, windowSize.y, GL_RGBA, GL_FLOAT, fPixels);

	for (int i = 0; i < windowSize.x*windowSize.y; i++)
	{
		pixels[i] = fPixels[4 * i];
	}

	drawNDFImage(windowSize.x,windowSize.y, pixels);
	delete[] fPixels;

#endif
}
void computeNDFCache()
{
	/*TODO: compute how much the histograms have changed to figure out when its near enough
	taylor expansion to bound the error is impossible due to discontinuity
	stop if the difference between k consecutive iterations is below a threshold (consider normalization)
	or try variance reduction (but very complicated)
	instead of change in histogram (which is constant anyways) use the change in convolution (double integral of NDF over whole incoming light)
	-> sum of gradients of irradiance wrt incoming light directions
	error bounds not possible - errors are unbounded
	-> stochastic error bound*/



	{
		//GLuint ssboBindingPointIndex = 0;
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, reductionShader->getSsboBiningIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject()), ndfTree.GetLevels().front().GetShaderStorageBufferOject());
	}
	{
		//GLuint ssboBindingPointIndex = 2;
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, reductionShader->getSsboBiningIndex(SampleCount_ssbo),SampleCount_ssbo);
	}
#ifdef CORE_FUNCTIONALITY_ONLY
#else
	{
		//GLuint ssboBindingPointIndex = 2;
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, reductionShader->getSsboBiningIndex(circularPatternSampleCount_ssbo), circularPatternSampleCount_ssbo);
	}
#endif

	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, downsampledSsbo);

	auto gpuBinningHandle = reductionShader->GetGlShaderProgramHandle();
	glUseProgram(gpuBinningHandle);

	//glBindTexture(GL_TEXTURE_2D, rayCastingSolutionRenderTarget.RenderTexture);
	glBindImageTexture(0, rayCastingSolutionRenderTarget.RenderTexture, 0, false, 0, GL_READ_ONLY, GL_RGBA32F);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "tex"), 0);

	//glBindImageTexture(1, MyRenderTarget.RenderTexture, 0, false, 0, GL_READ_ONLY, GL_RGBA16);
	//glUniform1i(glGetUniformLocation(gpuBinningHandle, "tileTex"), 1);

	glBindImageTexture(2, Page_Texture, std::floor(current_lod), false, 0, GL_READ_ONLY, Page_texture_internalFormat);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "floorLevel"), 2);

	glBindImageTexture(3, Page_Texture, std::floor(current_lod + 1), false, 0, GL_READ_ONLY, Page_texture_internalFormat);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "ceilLevel"), 3);

	//int Maxbinding;
	//glGetIntegerv(GL_MAX_IMAGE_UNITS, &Maxbinding);

	//glActiveTexture(GL_TEXTURE0+4);
	//glBindTexture(GL_TEXTURE_2D, echo_tex);
	//glUniform1i(glGetUniformLocation(gpuBinningHandle, "echo_tex"), 4);

	// not necessary for each frame. Doesn't change anyway...
	glUniform2i(glGetUniformLocation(gpuBinningHandle, "viewDiscretizations"), ndfTree.GetViewDirectionResolution().x, ndfTree.GetViewDirectionResolution().y);
	glUniform2i(glGetUniformLocation(gpuBinningHandle, "histogramDiscretizations"), ndfTree.GetHistogramResolution().x, ndfTree.GetHistogramResolution().y);

	//glUniform2i(glGetUniformLocation(gpuBinningHandle, "spatialDiscretizations"), ndfTree.GetSpatialResolution().x, ndfTree.GetSpatialResolution().y);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "multiSamplingRate"), multiSamplingRate);

	glUniform1i(glGetUniformLocation(gpuBinningHandle, "maxSamplingRuns"), maxSamplingRuns);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "binningMode"), binning_mode);

	glUniform1i(glGetUniformLocation(gpuBinningHandle, "samplescount"), sample_count);
	glUniform2fv(glGetUniformLocation(gpuBinningHandle, "sPos"), 1, &sPos[0]);



	int floor_w, ceil_w, tint, floor_h, ceil_h;
	//do some code here
	LOD.get_lod_width_and_hight(std::floor(current_lod), floor_w, floor_h);
	ceil_w = 0;


	LOD.get_lod_width_and_hight(std::floor(current_lod + 1), ceil_w, ceil_h);


	//w.StartTimer();*/


	//debug
	//floor_w = 512;
	//end debug
	glUniform1f(glGetUniformLocation(gpuBinningHandle, "viewportWidth"), static_cast<float>(sw));

	glUniform1i(glGetUniformLocation(gpuBinningHandle, "tile_w"), tile_w);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "tile_h"), tile_h);

	glUniform1i(glGetUniformLocation(gpuBinningHandle, "floor_w"), floor_w);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "ceil_w"), ceil_w);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "floor_h"), floor_h);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "ceil_h"), ceil_h);
	glUniform1f(glGetUniformLocation(gpuBinningHandle, "lod"), current_lod);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "phys_tex_dim"), phys_tex_dim);


	glUniform1i(glGetUniformLocation(gpuBinningHandle, "circularPattern"), circularPattern);
	glUniform1f(glGetUniformLocation(gpuBinningHandle, "sampleW"), sampleW);

	glUniform1i(glGetUniformLocation(gpuBinningHandle, "cachedRayCasting"), cachedRayCasting);

	glUniform2f(glGetUniformLocation(gpuBinningHandle, "trc_s"), sampling_trc_s.x, sampling_trc_s.y);
	glUniform2f(glGetUniformLocation(gpuBinningHandle, "blc_s"), sampling_blc_s.x, sampling_blc_s.y);
	glUniform2f(glGetUniformLocation(gpuBinningHandle, "blc_l"), sampling_blc_l.x, sampling_blc_l.y);

	//old
	glUniform2i(glGetUniformLocation(gpuBinningHandle, "spatialDiscretizations"), sw / multiSamplingRate, sh / multiSamplingRate);
	glDispatchCompute(sw / multiSamplingRate, sh / multiSamplingRate, 1);

	glUseProgram(0);
	glBindTexture(GL_TEXTURE_2D, 0);
}
void computeNDFCache_Ceil()
{
	/*TODO: compute how much the histograms have changed to figure out when its near enough
	taylor expansion to bound the error is impossible due to discontinuity
	stop if the difference between k consecutive iterations is below a threshold (consider normalization)
	or try variance reduction (but very complicated)
	instead of change in histogram (which is constant anyways) use the change in convolution (double integral of NDF over whole incoming light)
	-> sum of gradients of irradiance wrt incoming light directions
	error bounds not possible - errors are unbounded
	-> stochastic error bound*/



	{
		//GLuint ssboBindingPointIndex = 0;
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, reductionShader_C->getSsboBiningIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject()), ndfTree.GetLevels().front().GetShaderStorageBufferOject());
	}
	{
		//GLuint ssboBindingPointIndex = 2;
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, reductionShader_C->getSsboBiningIndex(SampleCount_ssbo), SampleCount_ssbo);
	}
#ifdef CORE_FUNCTIONALITY_ONLY
#else
	{
		//GLuint ssboBindingPointIndex = 2;
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, reductionShader_C->getSsboBiningIndex(circularPatternSampleCount_ssbo), circularPatternSampleCount_ssbo);
	}
#endif

	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, downsampledSsbo);

	auto gpuBinningHandle = reductionShader_C->GetGlShaderProgramHandle();
	glUseProgram(gpuBinningHandle);

	//glBindTexture(GL_TEXTURE_2D, rayCastingSolutionRenderTarget.RenderTexture);
	glBindImageTexture(0, rayCastingSolutionRenderTarget.RenderTexture, 0, false, 0, GL_READ_ONLY, GL_RGBA32F);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "tex"), 0);

	//glBindImageTexture(1, MyRenderTarget.RenderTexture, 0, false, 0, GL_READ_ONLY, GL_RGBA16);
	//glUniform1i(glGetUniformLocation(gpuBinningHandle, "tileTex"), 1);

	glBindImageTexture(2, Page_Texture, std::floor(current_lod), false, 0, GL_READ_ONLY, Page_texture_internalFormat);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "floorLevel"), 2);

	glBindImageTexture(3, Page_Texture, std::floor(current_lod + 1), false, 0, GL_READ_ONLY, Page_texture_internalFormat);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "ceilLevel"), 3);

	int Maxbinding;
	glGetIntegerv(GL_MAX_IMAGE_UNITS, &Maxbinding);

	// not necessary for each frame. Doesn't change anyway...
	glUniform2i(glGetUniformLocation(gpuBinningHandle, "viewDiscretizations"), ndfTree.GetViewDirectionResolution().x, ndfTree.GetViewDirectionResolution().y);
	glUniform2i(glGetUniformLocation(gpuBinningHandle, "histogramDiscretizations"), ndfTree.GetHistogramResolution().x, ndfTree.GetHistogramResolution().y);

	//glUniform2i(glGetUniformLocation(gpuBinningHandle, "spatialDiscretizations"), ndfTree.GetSpatialResolution().x, ndfTree.GetSpatialResolution().y);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "multiSamplingRate"), multiSamplingRate);

	glUniform1i(glGetUniformLocation(gpuBinningHandle, "maxSamplingRuns"), maxSamplingRuns);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "binningMode"), binning_mode);

	glUniform1i(glGetUniformLocation(gpuBinningHandle, "samplescount"), sample_count);
	glUniform2fv(glGetUniformLocation(gpuBinningHandle, "sPos"), 1, &sPos[0]);



	int floor_w, ceil_w, tint, floor_h, ceil_h;
	//do some code here
	LOD.get_lod_width_and_hight(std::floor(current_lod), floor_w, floor_h);
	ceil_w = 0;

	LOD.get_lod_width_and_hight(std::floor(current_lod + 1), ceil_w, ceil_h);


	glUniform1f(glGetUniformLocation(gpuBinningHandle, "viewportWidth"), static_cast<float>(sw));

	glUniform1i(glGetUniformLocation(gpuBinningHandle, "tile_w"), tile_w);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "tile_h"), tile_h);

	glUniform1i(glGetUniformLocation(gpuBinningHandle, "floor_w"), floor_w);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "ceil_w"), ceil_w);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "floor_h"), floor_h);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "ceil_h"), ceil_h);
	glUniform1f(glGetUniformLocation(gpuBinningHandle, "lod"), current_lod);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "phys_tex_dim"), phys_tex_dim);

	glUniform1i(glGetUniformLocation(gpuBinningHandle, "cachedRayCasting"), cachedRayCasting);

	//new
	glUniform2i(glGetUniformLocation(gpuBinningHandle, "visible_region_blc"), visible_region_blc.x, visible_region_blc.y);
	glUniform2i(glGetUniformLocation(gpuBinningHandle, "visible_region_trc"), visible_region_trc.x, visible_region_trc.y);
	glUniform3i(glGetUniformLocation(gpuBinningHandle, "lod_blc"), lod_blc.x, lod_blc.y, lod_blc.z);
	glUniform3i(glGetUniformLocation(gpuBinningHandle, "lod_trc"), lod_trc.x, lod_trc.y, lod_trc.z);

	glUniform3i(glGetUniformLocation(gpuBinningHandle, "floor_blc"), floor_blc.x, floor_blc.y, floor_blc.z);
	glUniform3i(glGetUniformLocation(gpuBinningHandle, "ceil_blc"), ceil_blc.x, ceil_blc.y, ceil_blc.z);

	glUniform2f(glGetUniformLocation(gpuBinningHandle, "trc_s"), sampling_trc_s.x, sampling_trc_s.y);
	glUniform2f(glGetUniformLocation(gpuBinningHandle, "blc_s"), sampling_blc_s.x, sampling_blc_s.y);
	glUniform2f(glGetUniformLocation(gpuBinningHandle, "blc_l"), sampling_blc_l.x, sampling_blc_l.y);
	//end new

	//old
	glUniform2i(glGetUniformLocation(gpuBinningHandle, "spatialDiscretizations"), (sw / multiSamplingRate) / 2, (sh / multiSamplingRate) / 2);
	glDispatchCompute((sw / multiSamplingRate) / 2, (sh / multiSamplingRate) / 2, 1);

	glUseProgram(0);
	glBindTexture(GL_TEXTURE_2D, 0);
}
void computeTiledRaycastingCache(const glm::mat4 & viewMatrix)
{
	/*TODO: compute how much the histograms have changed to figure out when its near enough
	taylor expansion to bound the error is impossible due to discontinuity
	stop if the difference between k consecutive iterations is below a threshold (consider normalization)
	or try variance reduction (but very complicated)
	instead of change in histogram (which is constant anyways) use the change in convolution (double integral of NDF over whole incoming light)
	-> sum of gradients of irradiance wrt incoming light directions
	error bounds not possible - errors are unbounded
	-> stochastic error bound*/



	{
		//GLuint ssboBindingPointIndex = 0;
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, tiledRaycastingShader->getSsboBiningIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject()), ndfTree.GetLevels().front().GetShaderStorageBufferOject());
	}
	//{
	//GLuint ssboBindingPointIndex = 2;
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, myArrayUBO);
	//}

	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, downsampledSsbo);

	auto gpuBinningHandle = tiledRaycastingShader->GetGlShaderProgramHandle();
	glUseProgram(gpuBinningHandle);

	//glBindTexture(GL_TEXTURE_2D, rayCastingSolutionRenderTarget.RenderTexture);
	glBindImageTexture(0, rayCastingSolutionRenderTarget.RenderTexture, 0, false, 0, GL_READ_ONLY, GL_RGBA32F);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "tex"), 0);

	//glBindImageTexture(1, MyRenderTarget.RenderTexture, 0, false, 0, GL_READ_ONLY, GL_RGBA16);
	//glUniform1i(glGetUniformLocation(gpuBinningHandle, "tileTex"), 1);

	glBindImageTexture(2, Page_Texture, std::floor(current_lod), false, 0, GL_READ_ONLY, Page_texture_internalFormat);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "floorLevel"), 2);

	glBindImageTexture(3, Page_Texture, std::floor(current_lod + 1), false, 0, GL_READ_ONLY, Page_texture_internalFormat);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "ceilLevel"), 3);

	int Maxbinding;
	glGetIntegerv(GL_MAX_IMAGE_UNITS, &Maxbinding);

	auto viewSpaceLightDir = viewMatrix * glm::vec4(LightDir, 0.0f);
	viewSpaceLightDir = glm::normalize(viewSpaceLightDir);
	glUniform3fv(glGetUniformLocation(gpuBinningHandle, "viewSpaceLightDir"), 1, &viewSpaceLightDir[0]);

	glUniform1i(glGetUniformLocation(gpuBinningHandle, "renderMode"), renderMode);

	for (auto &transferFunction : transferFunctions)
	{
		glBindTexture(GL_TEXTURE_2D, activeTransferFunction->transferFunctionTexture.Texture);// activeTransferFunction->transferFunctionTexture.Texture);
		glUniform1i(glGetUniformLocation(gpuBinningHandle, "normalTransferSampler"), 0);
		break;
	}

	//glActiveTexture(GL_TEXTURE0+4);
	//glBindTexture(GL_TEXTURE_2D, echo_tex);
	//glUniform1i(glGetUniformLocation(gpuBinningHandle, "echo_tex"), 4);

	// not necessary for each frame. Doesn't change anyway...
	glUniform2i(glGetUniformLocation(gpuBinningHandle, "viewDiscretizations"), ndfTree.GetViewDirectionResolution().x, ndfTree.GetViewDirectionResolution().y);
	glUniform2i(glGetUniformLocation(gpuBinningHandle, "histogramDiscretizations"), ndfTree.GetHistogramResolution().x, ndfTree.GetHistogramResolution().y);

	//glUniform2i(glGetUniformLocation(gpuBinningHandle, "spatialDiscretizations"), ndfTree.GetSpatialResolution().x, ndfTree.GetSpatialResolution().y);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "multiSamplingRate"), multiSamplingRate);

	glUniform1f(glGetUniformLocation(gpuBinningHandle, "specularExp"), specularExponent);

	glUniform1i(glGetUniformLocation(gpuBinningHandle, "maxSamplingRuns"), maxSamplingRuns);

	glUniform1i(glGetUniformLocation(gpuBinningHandle, "samplingRunIndex"), samplingRunIndex);
	glUniform2fv(glGetUniformLocation(gpuBinningHandle, "sPos"), 1, &sPos[0]);





	int floor_w, ceil_w, tint, floor_h, ceil_h;
	//do some code here
	LOD.get_lod_width_and_hight(std::floor(current_lod), floor_w, floor_h);
	ceil_w = 0;

	LOD.get_lod_width_and_hight(std::floor(current_lod + 1), ceil_w, ceil_h);



	//debug
	//floor_w = 512;
	//end debug
	glUniform1f(glGetUniformLocation(gpuBinningHandle, "viewportWidth"), static_cast<float>(sw));

	glUniform1i(glGetUniformLocation(gpuBinningHandle, "tile_w"), tile_w);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "tile_h"), tile_h);

	glUniform1i(glGetUniformLocation(gpuBinningHandle, "floor_w"), floor_w);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "ceil_w"), ceil_w);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "floor_h"), floor_h);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "ceil_h"), ceil_h);

	glUniform1f(glGetUniformLocation(gpuBinningHandle, "lod"), current_lod);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "phys_tex_dim"), phys_tex_dim);

	glUniform1i(glGetUniformLocation(gpuBinningHandle, "cachedRayCasting"), cachedRayCasting);

	//new
	glUniform2i(glGetUniformLocation(gpuBinningHandle, "visible_region_blc"), visible_region_blc.x, visible_region_blc.y);
	glUniform2i(glGetUniformLocation(gpuBinningHandle, "visible_region_trc"), visible_region_trc.x, visible_region_trc.y);
	glUniform3i(glGetUniformLocation(gpuBinningHandle, "lod_blc"), lod_blc.x, lod_blc.y, lod_blc.z);
	glUniform3i(glGetUniformLocation(gpuBinningHandle, "lod_trc"), lod_trc.x, lod_trc.y, lod_trc.z);

	glUniform3i(glGetUniformLocation(gpuBinningHandle, "floor_blc"), floor_blc.x, floor_blc.y, floor_blc.z);
	glUniform3i(glGetUniformLocation(gpuBinningHandle, "ceil_blc"), ceil_blc.x, ceil_blc.y, ceil_blc.z);

	glUniform2f(glGetUniformLocation(gpuBinningHandle, "trc_s"), sampling_trc_s.x, sampling_trc_s.y);
	glUniform2f(glGetUniformLocation(gpuBinningHandle, "blc_s"), sampling_blc_s.x, sampling_blc_s.y);
	glUniform2f(glGetUniformLocation(gpuBinningHandle, "blc_l"), sampling_blc_l.x, sampling_blc_l.y);
	//end new

	//old
	glUniform2i(glGetUniformLocation(gpuBinningHandle, "spatialDiscretizations"), sw / multiSamplingRate, sh / multiSamplingRate);
	glDispatchCompute(sw / multiSamplingRate, sh / multiSamplingRate, 1);

	glUseProgram(0);
	glBindTexture(GL_TEXTURE_2D, 0);
}
void computeTiledRaycastingCache_C(const glm::mat4 & viewMatrix)
{
	/*TODO: compute how much the histograms have changed to figure out when its near enough
	taylor expansion to bound the error is impossible due to discontinuity
	stop if the difference between k consecutive iterations is below a threshold (consider normalization)
	or try variance reduction (but very complicated)
	instead of change in histogram (which is constant anyways) use the change in convolution (double integral of NDF over whole incoming light)
	-> sum of gradients of irradiance wrt incoming light directions
	error bounds not possible - errors are unbounded
	-> stochastic error bound*/



	{
		//GLuint ssboBindingPointIndex = 0;
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, tiledRaycastingShader_C->getSsboBiningIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject()), ndfTree.GetLevels().front().GetShaderStorageBufferOject());
	}
//	{
//	GLuint ssboBindingPointIndex = 2;
//	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, myArrayUBO);
//}

	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, downsampledSsbo);

	auto gpuBinningHandle = tiledRaycastingShader_C->GetGlShaderProgramHandle();
	glUseProgram(gpuBinningHandle);

	//glBindTexture(GL_TEXTURE_2D, rayCastingSolutionRenderTarget.RenderTexture);
	glBindImageTexture(0, rayCastingSolutionRenderTarget.RenderTexture, 0, false, 0, GL_READ_ONLY, GL_RGBA32F);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "tex"), 0);

	//glBindImageTexture(1, MyRenderTarget.RenderTexture, 0, false, 0, GL_READ_ONLY, GL_RGBA16);
	//glUniform1i(glGetUniformLocation(gpuBinningHandle, "tileTex"), 1);

	glBindImageTexture(2, Page_Texture, std::floor(current_lod), false, 0, GL_READ_ONLY, Page_texture_internalFormat);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "floorLevel"), 2);

	glBindImageTexture(3, Page_Texture, std::floor(current_lod + 1), false, 0, GL_READ_ONLY, Page_texture_internalFormat);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "ceilLevel"), 3);

	int Maxbinding;
	glGetIntegerv(GL_MAX_IMAGE_UNITS, &Maxbinding);

	auto viewSpaceLightDir = viewMatrix * glm::vec4(LightDir, 0.0f);
	viewSpaceLightDir = glm::normalize(viewSpaceLightDir);
	glUniform3fv(glGetUniformLocation(gpuBinningHandle, "viewSpaceLightDir"), 1, &viewSpaceLightDir[0]);

	glUniform1i(glGetUniformLocation(gpuBinningHandle, "renderMode"), renderMode);

	for (auto &transferFunction : transferFunctions)
	{
		glBindTexture(GL_TEXTURE_2D, activeTransferFunction->transferFunctionTexture.Texture);// activeTransferFunction->transferFunctionTexture.Texture);
		glUniform1i(glGetUniformLocation(gpuBinningHandle, "normalTransferSampler"), 0);
		break;
	}

	//glActiveTexture(GL_TEXTURE0+4);
	//glBindTexture(GL_TEXTURE_2D, echo_tex);
	//glUniform1i(glGetUniformLocation(gpuBinningHandle, "echo_tex"), 4);

	// not necessary for each frame. Doesn't change anyway...
	glUniform2i(glGetUniformLocation(gpuBinningHandle, "viewDiscretizations"), ndfTree.GetViewDirectionResolution().x, ndfTree.GetViewDirectionResolution().y);
	glUniform2i(glGetUniformLocation(gpuBinningHandle, "histogramDiscretizations"), ndfTree.GetHistogramResolution().x, ndfTree.GetHistogramResolution().y);

	//glUniform2i(glGetUniformLocation(gpuBinningHandle, "spatialDiscretizations"), ndfTree.GetSpatialResolution().x, ndfTree.GetSpatialResolution().y);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "multiSamplingRate"), multiSamplingRate);

	glUniform1i(glGetUniformLocation(gpuBinningHandle, "maxSamplingRuns"), maxSamplingRuns);
	//glUniform1i(glGetUniformLocation(gpuBinningHandle, "highestSampleCount"), highestSampleCount);
	glUniform2fv(glGetUniformLocation(gpuBinningHandle, "samplePos"), 1, &sPos[0]);

	glUniform1f(glGetUniformLocation(gpuBinningHandle, "specularExp"), specularExponent);


	int floor_w, ceil_w, tint, floor_h, ceil_h;
	//do some code here
	LOD.get_lod_width_and_hight(std::floor(current_lod), floor_w, floor_h);
	ceil_w = 0;

	LOD.get_lod_width_and_hight(std::floor(current_lod + 1), ceil_w, ceil_h);



	//debug
	//floor_w = 512;
	//end debug
	glUniform1f(glGetUniformLocation(gpuBinningHandle, "viewportWidth"), static_cast<float>(sw));

	glUniform1i(glGetUniformLocation(gpuBinningHandle, "tile_w"), tile_w);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "tile_h"), tile_h);

	glUniform1i(glGetUniformLocation(gpuBinningHandle, "floor_w"), floor_w);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "ceil_w"), ceil_w);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "floor_h"), floor_h);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "ceil_h"), ceil_h);

	glUniform1f(glGetUniformLocation(gpuBinningHandle, "lod"), current_lod);
	glUniform1i(glGetUniformLocation(gpuBinningHandle, "phys_tex_dim"), phys_tex_dim);

	glUniform1i(glGetUniformLocation(gpuBinningHandle, "cachedRayCasting"), cachedRayCasting);

	//new
	glUniform2i(glGetUniformLocation(gpuBinningHandle, "visible_region_blc"), visible_region_blc.x, visible_region_blc.y);
	glUniform2i(glGetUniformLocation(gpuBinningHandle, "visible_region_trc"), visible_region_trc.x, visible_region_trc.y);
	glUniform3i(glGetUniformLocation(gpuBinningHandle, "lod_blc"), lod_blc.x, lod_blc.y, lod_blc.z);
	glUniform3i(glGetUniformLocation(gpuBinningHandle, "lod_trc"), lod_trc.x, lod_trc.y, lod_trc.z);

	glUniform3i(glGetUniformLocation(gpuBinningHandle, "floor_blc"), floor_blc.x, floor_blc.y, floor_blc.z);
	glUniform3i(glGetUniformLocation(gpuBinningHandle, "ceil_blc"), ceil_blc.x, ceil_blc.y, ceil_blc.z);

	glUniform2f(glGetUniformLocation(gpuBinningHandle, "trc_s"), sampling_trc_s.x, sampling_trc_s.y);
	glUniform2f(glGetUniformLocation(gpuBinningHandle, "blc_s"), sampling_blc_s.x, sampling_blc_s.y);
	glUniform2f(glGetUniformLocation(gpuBinningHandle, "blc_l"), sampling_blc_l.x, sampling_blc_l.y);
	//end new

	//old
	glUniform2i(glGetUniformLocation(gpuBinningHandle, "spatialDiscretizations"), (sw / multiSamplingRate) / 2, (sh / multiSamplingRate) / 2);
	glDispatchCompute((sw / multiSamplingRate) / 2, (sh / multiSamplingRate) / 2, 1);

	glUseProgram(0);
	glBindTexture(GL_TEXTURE_2D, 0);
}
inline void update_page_texture()
{
	if (plainRayCasting || pointCloudRendering)
		return;

	//for all visible tiles, check if they have addresses to the physical in their corresponding page index
	float level_of_detail = LOD.get_lod(cameraDistance);
	std::vector<int> levels;
	int lodw, lodh;
	int size, offset;
	bool found;
	bool memflag = false;
	StopWatch t1, t2, t3, t4, t5;
	float st3, st5;
	int count_in, count_out;

	count_in = count_out = 0;
	//t1.StartTimer();

	int phys_tx, phys_ty, phys_tdim;

	Page_Texture_Datatype pgarbage[4] = { -1, -1, -1, -1 };
	Page_Texture_Datatype val[4] = { -1, -1, 0, 0 };

	float ssboClearColor = 0.0f;
	std::vector<int> toBeErased;






	int tilesum = LOD.myTiles[std::floor(current_lod-lodDelta)].visible.size() + LOD.myTiles[std::floor(current_lod + 1)].visible.size();

	if (tilesum>max_tiles)
	{
		std::cout << std::endl;
		std::cout << "Tiles needed, exceed Memory capacity!!" << std::endl;
		std::cout << std::endl;
		Tiles.visible.clear();
		Tiles.T.clear();
		return;
	}


	{
		//given some tiles to view, we do the following for each tile
		//1-> check if the tile, in the page texture, points to some place in memory, if so, do nothing
		//2-> else, compose a list of possible places to place the tile, 'locations':
		//            -> this list is composed of the places that are empty in the NDF (get that from 'occupied' vector
		//            -> or, places in the NDF that contain tiles other than those that should be currently visible
		//3-> fetch the first location in 'locations'
		//4-> if the location is empty, ie occupied of location is false
		//       -> update page texture
		//       -> update occupied list
		//       -> remove 'location' from locations
		//5-> else, 
		//       -> remove tile from NDF
		//       -> remove its reference from page texture
		//       -> update page texture
		//       -> update occupied array
		//	     -> remove 'location' from locations

		levels.push_back(std::floor(current_lod-lodDelta));
		levels.push_back(std::floor(current_lod + 1));

		std::vector<int> list;
		//glBindTexture(GL_TEXTURE_2D, Page_Texture);

		std::queue<int> available;
		bool incache;
		int loc;
		int tile_indx;
		glm::ivec2 tile_indx_2d;
		//glm::ivec2 cache_indx_2d;
		int level_h, level_w, tiles_in_w, tiles_in_h;

		//get possible locations to add new tiles that weren't visible before
		//t4.StartTimer();
		visible_tiles_per_level.clear();
		visible_tiles_per_level.push_back(LOD.myTiles[std::floor(current_lod-lodDelta)].visible);
		visible_tiles_per_level.push_back(LOD.myTiles[std::floor(current_lod + 1)].visible);
		get_available_locations_in_cache(available, levels);
		//t4.StopTimer();
		//std::cout << "search for avaialble locations : " << t4.GetElapsedTime() << " ms" << std::endl;

		//glBindTexture(GL_TEXTURE_2D, Page_Texture);

		st3 = st5 = 0;

		//debug
		if (visible_tiles_per_level[0].size() == 0 || occupied.size()==0)
			st3 = st5;

		for (int j = 0; j < visible_tiles_per_level.size(); j++)
		{
			LOD.get_lod_width_and_hight(levels[j], level_w, level_h);
			tiles_in_w = level_w / tile_w;
			tiles_in_h = level_h / tile_h;

			count_out = 0;

			for (int i = 0; i < visible_tiles_per_level[j].size(); i++)
			{
				//check tile is already in cache

				incache = is_tile_in_cache(levels[j], visible_tiles_per_level[j][i]);


				if (!incache)
				{
					new_tiles++;
					//get_one_available_location_in_cache(available, levels);

					//std::cout << "Cache miss occured" << std::endl;
					loc = available.front();
					available.pop();

					//get 2d cache indx
					//cache_indx_2d = glm::ivec2(loc % (phys_tex_dim / tile_w), loc / (phys_tex_dim / tile_w));

					//get 2d tile indx
					tile_indx = visible_tiles_per_level[j][i];
					tile_indx_2d = glm::ivec2(tile_indx%tiles_in_w, tile_indx / tiles_in_w);

					//check if a tile already exists in this location or not
					if (occupied[loc].first.first)
					{
						memflag = true;

						//1->erase tile from ndf

						toBeErased.push_back(loc);



						//std::cout << "a tile has been deleted" << std::endl;

						//2-> remove frefrence from page texture, since tile is removed
						int p_level = occupied[loc].first.second.x;
						int p_level_w, p_level_h;

						LOD.get_lod_width_and_hight(p_level, p_level_w, p_level_h);
						p_level_w = p_level_w / tile_w;
						p_level_h = p_level_h / tile_h;

						if (!disable_binding)
						{
							glBindTexture(GL_TEXTURE_2D, Page_Texture);
							glTexSubImage2D(GL_TEXTURE_2D, p_level, int(occupied[loc].first.second.y) % p_level_w, occupied[loc].first.second.y / p_level_w, 1, 1, Page_Texture_format, Page_Texture_type, pgarbage);
						}
					}

					//t5.StartTimer();
					//1-> update page texture
					val[0] = loc;// cache_indx_2d.x;
					val[1] = -1;
					val[2] = -1;
					val[3] = -1;
					//val[1] = cache_indx_2d.y;

					if (!disable_binding)
					{
						glBindTexture(GL_TEXTURE_2D, Page_Texture);
						glTexSubImage2D(GL_TEXTURE_2D, levels[j], tile_indx_2d.x, tile_indx_2d.y, 1, 1, Page_Texture_format, Page_Texture_type, val);
					}

					//2-> update occupied
					occupied[loc].first.first = true;
					occupied[loc].first.second = glm::vec2(levels[j], visible_tiles_per_level[j][i]);
					occupied[loc].second = 0;
					//t5.StopTimer();
					//std::cout << "update page texture : " << t5.GetElapsedTime() << " ms" << std::endl;
					//st5 += t5.GetElapsedTime();
				}
				else
				{
					cached_tiles++;
				}
			}
			//debug
			//if (visible_tiles_per_level[j].size() > 0)
			//	percentage_of_cached_tiles.push_back(1 - (count_out / visible_tiles_per_level[j].size()));
			//end debug
		}

		//erase any tiles that needs to be erased
		if (toBeErased.size() > 0)
		{
			//we need to send to the reset shader batches of tiles that are next to each other in memory
			std::sort(toBeErased.begin(), toBeErased.end());
			std::vector<int> locs;
			for (int i = 0; i < toBeErased.size(); i++)
			{
				locs.push_back(toBeErased[i]);
				if (i == toBeErased.size() - 1)
				{
					remove_tile_from_physical_memory(locs);
					locs.clear();
				}
				else if (toBeErased[i + 1] - toBeErased[i] != 1)
				{
					remove_tile_from_physical_memory(locs);
					locs.clear();
				}
			}
		}


	}


}

bool is_tile_in_cache(int level, int tile_indx)
{
	for (int i = 0; i < occupied.size(); i++)
	{
		if (occupied[i].first.first && occupied[i].first.second.x == level && occupied[i].first.second.y == tile_indx)
		{
			return true;
		}
	}
	return false;
}

void get_available_locations_in_cache(std::queue<int>& available, std::vector<int>& levels)
{
	//put in available, the locations in the cache that are either empty or occupied by tiles not in visibleperlevel

	std::queue<int> occupied_locations;
	bool found;

	//to avoid erasing tiles as much as possible, we first put the the empty locations
	for (int i = 0; i < occupied.size(); i++)
	{
		if (!occupied[i].first.first)
		{
			available.push(i);
		}
		else
		{
			//search for the tile in occupied if it's in visible tiles per level
			found = false;
			for (int j = 0; j < visible_tiles_per_level.size(); j++)
			{
				for (int k = 0; k < visible_tiles_per_level[j].size(); k++)
				{
					if (occupied[i].first.second.x == levels[j] && occupied[i].first.second.y == visible_tiles_per_level[j][k])
					{
						found = true;
						break;
					}
				}
				if (found)
					break;
			}
			if (!found)    //we didn't find the tile in occupied[i] in the visible tiles, so it could be safely removed for this iteration
				occupied_locations.push(i);
		}
	}

	//merge availabel and occupied
	while (!occupied_locations.empty())
	{
		available.push(occupied_locations.front());
		occupied_locations.pop();
	}
}

void get_one_available_location_in_cache(std::queue<int>& available, std::vector<int>& levels)
{
	//put in available, the locations in the cache that are either empty or occupied by tiles not in visibleperlevel

	std::queue<int> occupied_locations;
	bool found;

	//to avoid erasing tiles as much as possible, we first put the the empty locations
	for (int i = 0; i < occupied.size(); i++)
	{
		if (!occupied[i].first.first)
		{
			available.push(i);
			return;
		}
		else
		{
			//search for the tile in occupied if it's in visible tiles per level
			found = false;
			for (int j = 0; j < visible_tiles_per_level.size(); j++)
			{
				for (int k = 0; k < visible_tiles_per_level[j].size(); k++)
				{
					if (occupied[i].first.second.x == levels[j] && occupied[i].first.second.y == visible_tiles_per_level[j][k])
					{
						found = true;
						break;
					}
				}
				if (found)
					break;
			}
			if (!found)    //we didn't find the tile in occupied[i] in the visible tiles, so it could be safely removed for this iteration
				occupied_locations.push(i);
		}
	}

	//merge availabel and occupied
	if (!occupied_locations.empty())
	{
		available.push(occupied_locations.front());
		return;
	}
}


void remove_tile_from_physical_memory(std::vector<int>&locs)

{
	//run a compute shader to remove a tile from the physical memory at location x,y

	GLuint ssboBindingPointIndex_ndfCache = 0;

	GLuint ssboBindingPointIndex_sampleCount = 2;
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, tileResetShader->getSsboBiningIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject()), ndfTree.GetLevels().front().GetShaderStorageBufferOject());
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, tileResetShader->getSsboBiningIndex(SampleCount_ssbo), SampleCount_ssbo);
#ifdef CORE_FUNCTIONALITY_ONLY
#else
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, tileResetShader->getSsboBiningIndex(circularPatternSampleCount_ssbo), circularPatternSampleCount_ssbo);
#endif
	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, region_ssbo);

	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, ssboBindingPointIndex, downsampledSsbo);

	auto tileClearingHandle = tileResetShader->GetGlShaderProgramHandle();
	glUseProgram(tileClearingHandle);


	glUniform2i(glGetUniformLocation(tileClearingHandle, "histogramDiscretizations"), ndfTree.GetHistogramResolution().x, ndfTree.GetHistogramResolution().y);
	glUniform1i(glGetUniformLocation(tileClearingHandle, "phys_tex_dim"), phys_tex_dim);

	glUniform1i(glGetUniformLocation(tileClearingHandle, "firstTile"), locs[0]);

	glUniform1i(glGetUniformLocation(tileClearingHandle, "tile_w"), tile_w);

	glDispatchCompute(tile_h*tile_w, locs.size(), histogramResolution.x*histogramResolution.y);

	glUseProgram(0);
	glBindTexture(GL_TEXTURE_2D, 0);


	//std::cout << "erased " << locs.size() << " tiles" << std::endl;


}
glm::vec2 calculate_lookup(int lod_w, int lod_h, glm::vec2 Coordinate, float F_lod, float E_lod, Page_Texture_Datatype* temp_texture)
{
	//if (levelFlag)
	//int x = calculate_lookup_pixel(lod_w,lod_h,tile_sample_count, levelFlag, Coordinate, F_lod, E_lod,offset, offset);
	//mohamed's lookup
	int num_tiles_in_w = lod_w / tile_w;
	int num_tiles_in_h = lod_h / tile_h;


	glm::ivec2 PixelCoordIn_F_Lod = glm::ivec2(Coordinate - blc_s + blc_l);
	glm::ivec2 PixelCoordIn_E_Lod = glm::ivec2(PixelCoordIn_F_Lod.x*pow(2, F_lod - E_lod), PixelCoordIn_F_Lod.y*pow(2, F_lod - E_lod));

	//add offset to image in PixelCoordIN_E_Lod
	glm::uvec2 spatialCoordinate = glm::uvec2(PixelCoordIn_E_Lod);

	//the offset may get us a pixel outside the iamge

	if ((spatialCoordinate.x >= 0) && (spatialCoordinate.x < lod_w) && (spatialCoordinate.y >= 0) && (spatialCoordinate.y < lod_h))
	{
		//get tile of the pixel
		glm::ivec2 tileindx2D = glm::ivec2(spatialCoordinate.x / tile_w, spatialCoordinate.y / tile_h);
		int tile_indx = tileindx2D.y*num_tiles_in_w + tileindx2D.x;
		glm::vec2 withinTileOffset = glm::vec2(spatialCoordinate.x%tile_w, spatialCoordinate.y%tile_h);

		//get tile coordiantes in page texture

		glm::ivec2  tileCoords_InPageTex = glm::ivec2(tile_indx% num_tiles_in_w, tile_indx / num_tiles_in_w);

		//read physical texture coordinates from page texture
		glm::vec4 tileCoords_InPhysTex;
#if 1
		{
			// get this from the occupied array knowing the lod and the tile indx you can get the location in the physical texture;// imageLoad(floorLevel, tileCoords_InPageTex);
			for (int i = 0; i < occupied.size(); i++)
			{
				if (occupied[i].first.first && occupied[i].first.second.x == E_lod && occupied[i].first.second.y == tile_indx)
				{
					tileCoords_InPhysTex.x = i;
					break;
				}
			}
		}
#endif

		
		{
			tileCoords_InPhysTex.x=temp_texture[tile_indx * 4];
		}

		//location in ndf tree is the physical texture location + within tile offset
		//ivec2 Pixelcoord_InPhysTex = ivec2(tileCoords_InPhysTex.x + withinTileOffset.x, tileCoords_InPhysTex.y + withinTileOffset.y);
		int Pixelcoord_InPhysTex = int((tileCoords_InPhysTex.x*tile_w*tile_h) + (withinTileOffset.y*tile_w + withinTileOffset.x));

		//const unsigned int lookup = unsigned int((Pixelcoord_InPhysTex.y * VolumeResolutionX + Pixelcoord_InPhysTex.x) * HistogramHeight * HistogramWidth);
		int lookup = int(Pixelcoord_InPhysTex * histogramResolution.x * histogramResolution.y);
		return glm::vec2(tileCoords_InPhysTex.x, lookup);
	}
	else
	{
		return glm::vec2(-1, -1);
	}

}
void computeAvgNDFGPU(bool finished)
{
	//run a compute shader to remove a tile from the physical memory at location x,y
	if (selectedPixels.size() > 0)
	{

		//bind ndf cache
		//bind sample count ssbo
		//bind avg ndf ssbo
		//bind ssbo for selected pixels
		//bind bin area ssbo
		//bind simlimits ssbo


		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, computeAvgNDF_C->getSsboBiningIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject()), ndfTree.GetLevels().front().GetShaderStorageBufferOject());
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, computeAvgNDF_C->getSsboBiningIndex(SampleCount_ssbo), SampleCount_ssbo);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, computeAvgNDF_C->getSsboBiningIndex(avgNDF_ssbo), avgNDF_ssbo);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, computeAvgNDF_C->getSsboBiningIndex(binAreas_ssbo), binAreas_ssbo);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, computeAvgNDF_C->getSsboBiningIndex(simLimitsF_ssbo), simLimitsF_ssbo);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, computeAvgNDF_C->getSsboBiningIndex(selectedPixels_ssbo), selectedPixels_ssbo);

		auto programHandle = computeAvgNDF_C->GetGlShaderProgramHandle();
		glUseProgram(programHandle);

		glBindImageTexture(1, Page_Texture, std::floor(current_lod), false, 0, GL_READ_ONLY, Page_texture_internalFormat);
		glUniform1i(glGetUniformLocation(programHandle, "floorLevel"), 1);

		glBindImageTexture(2, Page_Texture, std::floor(current_lod + 1), false, 0, GL_READ_ONLY, Page_texture_internalFormat);
		glUniform1i(glGetUniformLocation(programHandle, "ceilLevel"), 2);

		//upload selected pixels
		{
			auto Size = size_t(2 * selectedPixels.size());
			auto ssboSize = Size*sizeof(float);
			auto ssbo = selectedPixels_ssbo;
			{
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
				GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
				memcpy(p, reinterpret_cast<char*>(&selectedPixels[0]), 2 * selectedPixels.size() * sizeof(*selectedPixels.begin()));
				glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
			}
		}




		glUniform2f(glGetUniformLocation(programHandle, "trc_s"), trc_s.x, trc_s.y);
		glUniform2f(glGetUniformLocation(programHandle, "blc_s"), blc_s.x, blc_s.y);
		glUniform2f(glGetUniformLocation(programHandle, "blc_l"), blc_l.x, blc_l.y);

		glUniform1i(glGetUniformLocation(programHandle, "tile_w"), tile_w);
		glUniform1i(glGetUniformLocation(programHandle, "tile_h"), tile_h);
		glUniform1i(glGetUniformLocation(programHandle, "phys_tex_dim"), phys_tex_dim);

		int floor_w, ceil_w, floor_h, ceil_h;
		LOD.get_lod_width_and_hight(std::floor(current_lod), floor_w, floor_h);
		glUniform1i(glGetUniformLocation(programHandle, "floor_w"), floor_w);
		glUniform1i(glGetUniformLocation(programHandle, "floor_h"), floor_h);


		LOD.get_lod_width_and_hight(std::floor(current_lod + 1), ceil_w, ceil_h);
		glUniform1i(glGetUniformLocation(programHandle, "ceil_w"), ceil_w);
		glUniform1i(glGetUniformLocation(programHandle, "ceil_h"), ceil_h);

		glUniform1f(glGetUniformLocation(programHandle, "selectedPixelsSize"), selectedPixels.size());

		glUniform1f(glGetUniformLocation(programHandle, "lod"), current_lod);

		glUniform2i(glGetUniformLocation(programHandle, "histogramDiscretizations"), histogramResolution.x, histogramResolution.y);

		glUniform1i(glGetUniformLocation(programHandle, "win_w"), windowSize.x);
		glUniform1i(glGetUniformLocation(programHandle, "win_h"), windowSize.y);

		//glUniform1i(glGetUniformLocation(computeAVGNDFHandle, "tile_w"), tile_w);

		glDispatchCompute(histogramResolution.x*histogramResolution.y, 1, 1);

		glUseProgram(0);
		glBindTexture(GL_TEXTURE_2D, 0);

		if (finished)
			selectedPixels.clear();
	}
}

void computeAvgNDF(bool finished)
{
	computeAvgNDFGPU(finished);
	return;
	//here, we are given the array of selectedpixels and we compute the average ndf and upload it to average ndf ssbo
	if (selectedPixels.size() > 0)
	{
		int fLOD = std::floor(current_lod);
		int floor_w, floor_h;
		LOD.get_lod_width_and_hight(fLOD, floor_w, floor_h);
		Page_Texture_Datatype* temp_texture = new Page_Texture_Datatype[4 * floor_h*floor_w];

		float samples;
		{
			glBindTexture(GL_TEXTURE_2D, Page_Texture);
			glGetTexImage(GL_TEXTURE_2D, fLOD, Page_Texture_format,Page_Texture_type, temp_texture);
		}

		//download ndf ssbo
		std::vector<float> NDF;
		{
			auto Size = size_t(histogramResolution.x*histogramResolution.y * phys_tex_dim* phys_tex_dim);
			NDF.resize(Size, 0.0f);

			auto ssbo = ndfTree.GetLevels().front().GetShaderStorageBufferOject();
			{
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
				auto readMap = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
				memcpy(reinterpret_cast<char*>(&NDF[0]), readMap, NDF.size() * sizeof(*NDF.begin()));

				glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
			}
		}

		//download sample count ssbo
		std::vector<float> sample_counts;
		{
			auto Size = size_t(phys_tex_dim* phys_tex_dim);
			sample_counts.resize(Size, 0.0f);

			auto ssbo = SampleCount_ssbo;
			{
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
				auto readMap = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
				memcpy(reinterpret_cast<char*>(&sample_counts[0]), readMap, sample_counts.size() * sizeof(*sample_counts.begin()));

				glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
			}
		}

		//download bin area ssbo
		std::vector<double> binArea(histogramResolution.x*histogramResolution.y);
		{
			//auto Size = size_t();

			auto ssbo = binAreas_ssbo;
			{
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
				auto readMap = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
				memcpy(reinterpret_cast<char*>(&binArea[0]), readMap, binArea.size() * sizeof(*binArea.begin()));

				glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
			}
		}

		//downoad simlimts ssbo
		std::vector<float> simLimits(2 + windowSize.x*windowSize.y, 0);
		{
			auto ssbo = simLimitsF_ssbo;
			{
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
				auto readMap = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
				memcpy(reinterpret_cast<char*>(&simLimits[0]), readMap, simLimits.size() * sizeof(*simLimits.begin()));

				glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
			}
		}

		std::vector<float> avgNDF(histogramResolution.x*histogramResolution.y, 0.0f);
		float hits,bA;
		for (int i = 0; i < selectedPixels.size(); i++)
		{
			selectedPixels[i].x = windowSize.x - selectedPixels[i].x;
			//get index of pixel in ndf vector, and index in occupied array, both indices are in 'indx' vector, indx.x is the index in occupied array, and indx.y is the 'lookup' into the ndf cache
			glm::vec2 indx = calculate_lookup(floor_w, floor_h, selectedPixels[i], current_lod, fLOD,temp_texture);

			samples = sample_counts[indx.y / (histogramResolution.x*histogramResolution.y)];

			//set color of selected pixels to red
			simLimits[2 + (windowSize.x*selectedPixels[i].y + selectedPixels[i].x)] = -2;
			//regionColor[3 * (indx.y / (histogramResolution.x*histogramResolution.y))] = 1;
			//regionColor[3 * (indx.y / (histogramResolution.x*histogramResolution.y))+1] = 0;
			//regionColor[3 * (indx.y / (histogramResolution.x*histogramResolution.y))+2] = 0;
			float sampleHits = 0;
			for (int j = 0; j < histogramResolution.x*histogramResolution.y; j++)
			{
				sampleHits += NDF[indx.y + j] ;  //add normalized ndf bins
			}

			//add ndf to avgNDF
			for (int j = 0; j < histogramResolution.x*histogramResolution.y; j++)
			{
				if (binArea[j] != 0.0f)
				{
					hits = NDF[indx.y + j];

					if (sampleHits>samples)
					{
						std::cout << "sample hits: " << sampleHits << " samples: " << samples << std::endl;
					}

					bA = binArea[j];
					avgNDF[j] += (hits/ samples) / bA;  //add normalized ndf bins
				}
			}

			//debug
			//float sum = std::accumulate(avgNDF.begin(), avgNDF.end(), 0.0f);
			//std::cout << "avg ndf sum " << sum << std::endl;
			//end debugc
		}

		//divide by number of selected pixels
		for (int j = 0; j < histogramResolution.x*histogramResolution.y; j++)
		{
			avgNDF[j] /= float(selectedPixels.size());
		}



		//upload avgNDF_ssbo
		{
			auto Size = size_t(histogramResolution.x*histogramResolution.y);
			auto ssboSize = Size*sizeof(float);
			auto ssbo = avgNDF_ssbo;
			{
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
				GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
				memcpy(p, reinterpret_cast<char*>(&avgNDF[0]), avgNDF.size() * sizeof(*avgNDF.begin()));
				glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
			}
		}

		//upload simlimits
		{
			auto ssbo = simLimitsF_ssbo;
			{
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
				GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
				memcpy(p, reinterpret_cast<char*>(&simLimits[0]), simLimits.size() * sizeof(*simLimits.begin()));
				glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
			}
		}


	}

	//finished means that we finished selecing all the pixels we want
	if (finished)
		selectedPixels.clear();
}
void probeNDFs()
{
	//if (selectedPixels.size() > 0)
	//{
	//	bool oneRegion = true;

	//	//for now, just have one region
	//	if (oneRegion)
	//		regions.clear();

	//	int fLOD = std::floor(current_lod);
	//	int floor_w, floor_h;
	//	LOD.get_lod_width_and_hight(fLOD, floor_w, floor_h);


	//	std::vector<glm::vec3> colors;
	//	colors.push_back(glm::vec3(1, 0, 0));
	//	colors.push_back(glm::vec3(0, 1, 0));
	//	colors.push_back(glm::vec3(0, 0, 1));


	//	//create a new region, assign its color and lod and selected pixels
	//	region r;
	//	if (oneRegion)
	//		r.color = glm::vec4(1, 0, 0, 1);
	//	else
	//		r.color = glm::vec4((rand() % 256) / 255.0f, (rand() % 256) / 255.0f, (rand() % 256) / 255.0f, 1);//glm::vec4(colors[rand()%3], 1);
	//	r.lod = fLOD;


	//	//download region ssbo
	//	std::vector<float> regionColor(3 * phys_tex_dim*phys_tex_dim, 1);// = new float[3 * phys_tex_dim*phys_tex_dim];

	//	//if (!oneRegion)
	//	//{
	//	//	auto Size = size_t(3 * phys_tex_dim* phys_tex_dim);
	//	//	//regionColor.resize(Size, 0.0f);

	//	//	auto ssbo = region_ssbo;
	//	//	{
	//	//		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
	//	//		auto readMap = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
	//	//		memcpy(reinterpret_cast<char*>(&regionColor[0]), readMap, regionColor.size() * sizeof(*regionColor.begin()));

	//	//		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
	//	//		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	//	//	}
	//	//}

	//	//download ndf ssbo
	//	std::vector<float> NDF;
	//	{
	//		auto Size = size_t(histogramResolution.x*histogramResolution.y * phys_tex_dim* phys_tex_dim);
	//		NDF.resize(Size, 0.0f);

	//		auto ssbo = ndfTree.GetLevels().front().GetShaderStorageBufferOject();
	//		{
	//			glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
	//			auto readMap = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
	//			memcpy(reinterpret_cast<char*>(&NDF[0]), readMap, NDF.size() * sizeof(*NDF.begin()));

	//			glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
	//			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	//		}
	//	}

	//	std::vector<float> avgNDF(histogramResolution.x*histogramResolution.y, 0.0f);

	//	for (int i = 0; i < selectedPixels.size(); i++)
	//	{
	//		selectedPixels[i].x = windowSize.x - selectedPixels[i].x;
	//		//get index of pixel in ndf vector, and index in occupied array, both indices are in 'indx' vector, indx.x is the index in occupied array, and indx.y is the 'lookup' into the ndf cache
	//		glm::vec2 indx = calculate_lookup(floor_w, floor_h, selectedPixels[i], current_lod, fLOD);

	//		//update color in regions ssbo
	//		regionColor[3 * (int(indx.y) / (histogramResolution.x*histogramResolution.y))] = r.color.x;
	//		regionColor[3 * (int(indx.y) / (histogramResolution.x*histogramResolution.y)) + 1] = r.color.y;
	//		regionColor[3 * (int(indx.y) / (histogramResolution.x*histogramResolution.y)) + 2] = r.color.z;

	//		//add ndf to avgNDF
	//		for (int j = 0; j < histogramResolution.x*histogramResolution.y; j++)
	//		{
	//			avgNDF[j] += NDF[indx.y + j];
	//		}
	//	}

	//	//put average NDF in Regions
	//	r.avgNDF = avgNDF;
	//	r.selectedPixels = selectedPixels;

	//	//divide by number of selected pixels
	//	for (int j = 0; j < histogramResolution.x*histogramResolution.y; j++)
	//	{
	//		avgNDF[j] /= selectedPixels.size();
	//	}


	//	markSimilarNDFs(r, NDF, regionColor);

	//	////upload region_ssbo
	//	//{
	//	//	auto Size = size_t(3 * phys_tex_dim* phys_tex_dim);
	//	//	auto ssboSize =  Size*sizeof(float);
	//	//	auto ssbo = region_ssbo;
	//	//	{
	//	//		glBindBuffer(GL_SHADER_STORAGE_BUFFER,ssbo);
	//	//		GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
	//	//		memcpy(p, reinterpret_cast<char*>(&regionColor[0]), regionColor.size() * sizeof(*regionColor.begin()));
	//	//		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
	//	//	}
	//	//}

	//	//clear selected pixels
	//	selectedPixels.clear();

	//	//add new region to regions
	//	regions.push_back(r);
	//}
}

inline void remarkSimilarNDFs()
{
	////here we reset the region ssbo, and call marksimilarNDFs for all regions
	//region r;
	//int floor_w, floor_h;

	////reset region ssbo
	//std::vector<float> regionColor(3 * phys_tex_dim*phys_tex_dim, 1);// = new float[3 * phys_tex_dim*phys_tex_dim];

	////download ndf ssbo
	//std::vector<float> NDF;
	//{
	//	auto Size = size_t(histogramResolution.x*histogramResolution.y * phys_tex_dim* phys_tex_dim);
	//	NDF.resize(Size, 0.0f);

	//	auto ssbo = ndfTree.GetLevels().front().GetShaderStorageBufferOject();
	//	{
	//		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
	//		auto readMap = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
	//		memcpy(reinterpret_cast<char*>(&NDF[0]), readMap, NDF.size() * sizeof(*NDF.begin()));

	//		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
	//		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	//	}
	//}

	//for (int k = 0; k < regions.size(); k++)
	//{
	//	r = regions[k];
	//	LOD.get_lod_width_and_hight(r.lod, floor_w, floor_h);

	//	for (int i = 0; i < r.selectedPixels.size(); i++)
	//	{
	//		//get index of pixel in ndf vector, and index in occupied array, both indices are in 'indx' vector, indx.x is the index in occupied array, and indx.y is the 'lookup' into the ndf cache
	//		glm::vec2 indx = calculate_lookup(floor_w, floor_h, selectedPixels[i], current_lod, r.lod);

	//		//update color in regions ssbo
	//		regionColor[3 * (int(indx.y) / (histogramResolution.x*histogramResolution.y))] = r.color.x;
	//		regionColor[3 * (int(indx.y) / (histogramResolution.x*histogramResolution.y)) + 1] = r.color.y;
	//		regionColor[3 * (int(indx.y) / (histogramResolution.x*histogramResolution.y)) + 2] = r.color.z;
	//	}

	//	markSimilarNDFs(r, NDF, regionColor);
	//}

	//////upload region_ssbo
	////{
	////	auto Size = size_t(3 * phys_tex_dim* phys_tex_dim);
	////	auto ssboSize = Size*sizeof(float);
	////	auto ssbo = region_ssbo;
	////	{
	////		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
	////		GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
	////		memcpy(p, reinterpret_cast<char*>(&regionColor[0]), regionColor.size() * sizeof(*regionColor.begin()));
	////		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
	////	}
	////}
}

inline void markSimilarNDFs(region& r, std::vector<float>& NDFs, std::vector<float>& regionColor)
{
	//for all ndfs that belong to regions floor(lod) of region r
	//check how similar they are with 'avgNDF' of region r
	//if similar enough
	//update region color ssbo

	int Pixelcoord_InPhysTex, lookup;
	float measure;
	std::vector<std::pair<float, int>> scores;
	//float simPercentage = .05;

	for (int i = 0; i < occupied.size(); i++)
	{
		//for each tile that is in physical memory, loop over its pixels and check if they are similar to the avgndf in 'r'
		if (occupied[i].first.first && occupied[i].first.second.x == r.lod)
		{
			Pixelcoord_InPhysTex = i*tile_w*tile_h;

			for (int j = 0; j < tile_w*tile_h; j++)
			{
				lookup = int((Pixelcoord_InPhysTex + j) * histogramResolution.x * histogramResolution.y);
				measure = similarNDFs(r.avgNDF, NDFs, lookup);
				scores.push_back(std::make_pair(measure, lookup));
			}
		}
	}

	//return the similar percentage
	std::sort(scores.begin(), scores.end());
	float minScore, maxScore;
	float t;
	glm::vec3 rColor;
	glm::vec3 red = glm::vec3(1, 0, 0);
	glm::vec3 blue = glm::vec3(0, 0, 1);

	minScore = scores[0].first;
	maxScore = (scores[scores.size() - 1].first + minScore) / 2.0;

	for (int i = 0; i < simPercentage*scores.size(); i++)
	{
		//we will color only the simPercentage percent of the scores, and for these scores, we'll color the closest ones with color closest to color of r, but we'll fade 'r' as the score increases
		if (minScore != maxScore)
		{
			t = (scores[i].first - minScore) / (maxScore - minScore);
			rColor = red + t*(blue - red);
		}
		else
		{
			t = 1;
			rColor = red + t*(blue - red);
		}
		//update color in regions ssbo
		//if (regionColor[3 * (int(scores[i].second) / (histogramResolution.x*histogramResolution.y))] == 1 &&
		//	regionColor[3 * (int(scores[i].second) / (histogramResolution.x*histogramResolution.y)) + 1 == 1] &&
		//	regionColor[3 * (int(scores[i].second) / (histogramResolution.x*histogramResolution.y)) + 2])
		{
			//if region is colorless, ie 1,1,1, then we color it with color of region
			regionColor[3 * (int(scores[i].second) / (histogramResolution.x*histogramResolution.y))] = rColor.x;
			regionColor[3 * (int(scores[i].second) / (histogramResolution.x*histogramResolution.y)) + 1] = rColor.y;
			regionColor[3 * (int(scores[i].second) / (histogramResolution.x*histogramResolution.y)) + 2] = rColor.z;
		}
		//else
		//{
		//	//else we blend region color with color already there
		//	regionColor[3 * (int(scores[i].second) / (histogramResolution.x*histogramResolution.y))] =0.5* (regionColor[3 * (int(scores[i].second) / (histogramResolution.x*histogramResolution.y))] + t*r.color.x);
		//	regionColor[3 * (int(scores[i].second) / (histogramResolution.x*histogramResolution.y)) + 1] = 0.5*(regionColor[3 * (int(scores[i].second) / (histogramResolution.x*histogramResolution.y))+1]+ t*r.color.y);
		//	regionColor[3 * (int(scores[i].second) / (histogramResolution.x*histogramResolution.y)) + 2] = 0.5*(regionColor[3 * (int(scores[i].second) / (histogramResolution.x*histogramResolution.y))+1] + t*r.color.z);
		//}
	}
}

inline float similarNDFs(std::vector<float>& avgNDF, std::vector<float>& NDFs, int lookup)
{
	float similarityThreshold = 5000;

	if (similarityMeasure == 0)
	{
		return L2Norm(avgNDF, NDFs, lookup);
		//return true;
	}
	else if (similarityMeasure == 1)
	{
		return earthMoversDistance(avgNDF, NDFs, lookup);// <= similarityThreshold)
		//	return true;
	}
	//return false;
}

inline float L2Norm(std::vector<float>& avgNDF, std::vector<float>& NDFs, int lookup)
{
	std::vector<float> diff(histogramResolution.x*histogramResolution.y, 0);
	float l = 0.0f;

	for (int i = 0; i < histogramResolution.x*histogramResolution.y; i++)
	{
		diff[i] = avgNDF[i] - NDFs[lookup + i];
		l += diff[i] * diff[i];
	}

	return std::sqrt(l);
}

inline float earthMoversDistance(std::vector<float>& avgNDF, std::vector<float>& NDFs, int lookup)
{
	return 0;
}

void drawNDF()
{
	//run a compute shader to draw NDF

	//only draw the part of the NDF that is occupied
	int highest_indx = 0;
	for (int i = 0; i < occupied.size(); i++)
	{
		if (occupied[i].first.first)
		{
			if (i>highest_indx)
			{
				highest_indx = i;
			}
		}
	}

	glm::ivec2 twoDIndx = glm::ivec2(highest_indx % (phys_tex_dim / tile_w), highest_indx / (phys_tex_dim / tile_w));
	glm::ivec2 iSize;
	iSize.y = (twoDIndx.y + 1)*tile_w;
	if (twoDIndx.y > 0)
		iSize.x = phys_tex_dim;
	else
		iSize.x = (twoDIndx.x + 1)*tile_w;


	//for now
	iSize = glm::ivec2(phys_tex_dim, phys_tex_dim);
	//end for now

	//bind ndf cache to shader
	//GLuint ssboBindingPointIndex_ndfCache = 0;
	//GLuint ssboBindingPointIndex_ndfImage = 1;

	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, drawNDFShader->getSsboBiningIndex(ndfTree.GetLevels().front().GetShaderStorageBufferOject()), ndfTree.GetLevels().front().GetShaderStorageBufferOject());

	//bind image ssbo to shader
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, drawNDFShader->getSsboBiningIndex(NDFImage_ssbo), NDFImage_ssbo);

	//use shader
	auto drawNDFHandle = drawNDFShader->GetGlShaderProgramHandle();
	glUseProgram(drawNDFHandle);

	glUniform2i(glGetUniformLocation(drawNDFHandle, "histogramDiscretizations"), ndfTree.GetHistogramResolution().x, ndfTree.GetHistogramResolution().y);
	glUniform1i(glGetUniformLocation(drawNDFHandle, "phys_tex_dim"), phys_tex_dim);
	glUniform1i(glGetUniformLocation(drawNDFHandle, "tile_w"), tile_w);
	glUniform1i(glGetUniformLocation(drawNDFHandle, "tile_h"), tile_h);


	glDispatchCompute(iSize.x*iSize.y, histogramResolution.x *histogramResolution.y, 1);

	glUseProgram(0);
	{
		auto glErr = glGetError();
		if (glErr != GL_NO_ERROR) {
			std::cout << "glError: " << __FILE__ << " " << __LINE__ << std::endl;
		}
	}

	//download save image
	{
		//download img data
		std::vector<float> imgData;
		auto imgSize = size_t(iSize.x*iSize.y * histogramResolution.x  * histogramResolution.y);
		imgData.resize(imgSize, 0.0f);

		// download data
		auto ssbo = NDFImage_ssbo;
		{
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
			auto readMap = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
			memcpy(reinterpret_cast<char*>(&imgData[0]), readMap, imgData.size() * sizeof(*imgData.begin()));

			glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		}

		//save to image
		{
			float* imgf = new float[iSize.x*iSize.y * histogramResolution.x * histogramResolution.y];
			for (int i = 0; i < iSize.x*iSize.y * histogramResolution.x * histogramResolution.y; i++)
				imgf[i] = imgData[i];

			int w = iSize.x* histogramResolution.x;
			int h = iSize.y *histogramResolution.y;

			drawNDFImage(w, h, imgf);

			delete[] imgf;
		}
	}

	{
		auto glErr = glGetError();
		if (glErr != GL_NO_ERROR) {
			std::cout << "glError: " << __FILE__ << " " << __LINE__ << std::endl;
		}
	}
}
void drawNDFImage(int w, int h, float* data)
{
	bool test =false;
	static int nbr = 0;
	char ch[256];
	sprintf_s(ch, 256, "img%.4i.pgm", nbr++);
	float vmax = data[0];
	for (size_t n = 1; n < size_t(w)*size_t(h); n++) vmax = std::max(vmax, data[n]);
	float scale = 255.0f / vmax;
	if (!test)
	{
		uint8_t* buf = new uint8_t[w*h];
		size_t n = 0;
		size_t H = (h - 1)*w;
		for (size_t j = 0; j < h; j++) {
			for (size_t i = 0; i <w; i++) {
				if (data[i + H - j*w]>0)
				{
					int x = 0;
				}
				buf[n++] = uint8_t(data[i + H - j*w] * scale);
			}
		}
        std::string name;
        if (cmdlineImageName.empty()) {
            name = "screenShots/Screen_" + std::to_string(samplingTextureSaves) + ".pgm";
            samplingTextureSaves++;
        } else {
            name = cmdlineImageName;
        }
		PGMwrite(name.c_str(), w, h, buf);
		delete[] buf;
	}
	else
	{
		uint8_t* crop = new uint8_t[histogramResolution.x*histogramResolution.y];
		std::vector<float> v;
		//NDF for one pixel

#if 0   //code for downsampling for zooming figure
		uint8_t* crop0 = new uint8_t[histogramResolution.x*histogramResolution.y];
		uint8_t* crop1 = new uint8_t[histogramResolution.x*histogramResolution.y];
		uint8_t* crop2 = new uint8_t[histogramResolution.x*histogramResolution.y];
		uint8_t* crop3 = new uint8_t[histogramResolution.x*histogramResolution.y];


		int fLOD = std::floor(current_lod);
		int floor_w, floor_h;
		LOD.get_lod_width_and_hight(fLOD, floor_w, floor_h);
		Page_Texture_Datatype* temp_texture = new Page_Texture_Datatype[4 * floor_h*floor_w];

		float samples;
		{
			glBindTexture(GL_TEXTURE_2D, Page_Texture);
			glGetTexImage(GL_TEXTURE_2D, fLOD, Page_Texture_format, Page_Texture_type, temp_texture);
		}

		//get index of pixel in ndf vector, and index in occupied array, both indices are in 'indx' vector, indx.x is the index in occupied array, and indx.y is the 'lookup' into the ndf cache
		glm::vec2 indx = calculate_lookup(floor_w, floor_h, glm::vec2(windowSize.x - 931, 378), current_lod, fLOD, temp_texture);

		//add ndf to avgNDF
		for (int j = 0; j < histogramResolution.x*histogramResolution.y; j++)
		{
			crop0[j] = data[int(indx.y) + j];

		}
		PGMwrite("img0.pgm", histogramResolution.x, histogramResolution.y, &crop0[0]);

		indx = calculate_lookup(floor_w, floor_h, glm::vec2(windowSize.x-930, 377), current_lod, fLOD, temp_texture);

		//add ndf to avgNDF
		for (int j = 0; j < histogramResolution.x*histogramResolution.y; j++)
		{
			crop1[j] = data[int(indx.y) + j];

		}
		PGMwrite("img1.pgm", histogramResolution.x, histogramResolution.y, &crop1[0]);


		indx = calculate_lookup(floor_w, floor_h, glm::vec2(windowSize.x - 930, 378), current_lod, fLOD, temp_texture);

		//add ndf to avgNDF
		for (int j = 0; j < histogramResolution.x*histogramResolution.y; j++)
		{
			crop2[j] = data[int(indx.y) + j];

		}
		PGMwrite("img2.pgm", histogramResolution.x, histogramResolution.y, &crop2[0]);


		indx = calculate_lookup(floor_w, floor_h, glm::vec2(windowSize.x - 931, 377), current_lod, fLOD, temp_texture);

		//add ndf to avgNDF
		for (int j = 0; j < histogramResolution.x*histogramResolution.y; j++)
		{
			crop3[j] = data[int(indx.y) + j];

		}
		PGMwrite("img3.pgm", histogramResolution.x, histogramResolution.y, &crop3[0]);


		//pring sum of ndfs
		for (int j = 0; j < histogramResolution.x*histogramResolution.y; j++)
		{
			crop[j] = crop0[j]+crop1[j]+crop2[j]+crop3[j];

		}
		PGMwrite("img4.pgm", histogramResolution.x, histogramResolution.y, &crop[0]);
#endif

#if 1
		for (size_t j = h - histogramResolution.y; j <h; j++)
		{
			for (size_t i = 0; i < histogramResolution.x; i++)
			{
				crop[i + (j - h + histogramResolution.x) * histogramResolution.y] = uint8_t(data[i + (h - 1 - j)*w] * scale); // / ((A[i + (j - h + 8) * 8] == 0) ? scale * 1 : scale*A[i + (j - h + 8) * 8]);
			}
		}
		PGMwrite(ch, histogramResolution.x, histogramResolution.y, &crop[0]);
#endif
		//draw NDF*binArea, should give uniform color
#if 0
		sprintf_s(ch, 256, "img%.4i.pgm", nbr++);
		{
			double vmax = 0;
			double total_hits = 0;
			double prob = 0;
			double s = 0.0;

			//normalize items in data first
			//for (size_t j = h - 8; j < h; j++)
			//{
			//	for (size_t i = 0; i < 8; i++)
			//	{
			//		data[i + (h - 1 - j)*w] *= scale; // / ((A[i + (j - h + 8) * 8] == 0) ? scale * 1 : scale*A[i + (j - h + 8) * 8]);
			//	}
			//}

			std::vector<double> p;
			for (size_t j = h - histogramResolution.x; j < h; j++)
			{
				for (size_t i = 0; i < histogramResolution.y; i++)
				{
					total_hits += data[i + (h - 1 - j)*w];
					p.push_back(data[i + (h - 1 - j)*w]);
				}
			}

			s = std::accumulate(p.begin(), p.end(), 0.0f);

			for (size_t j = h - histogramResolution.x; j < h; j++)
			{
				for (size_t i = 0; i < histogramResolution.y; i++)
				{
					int indx = i + (j - h + histogramResolution.x) * histogramResolution.y;
					double hitsRatio = (p[indx]) / total_hits;
					prob += hitsRatio;
					double Area = (A[binning_mode][indx] == 0) ? 1 : A[binning_mode][indx];
					vmax = std::max(hitsRatio / Area, vmax);

				}
			}

			std::cout << "sum of probabilities: " << prob << std::endl;
			std::cout << "total hits: " << total_hits << std::endl;

			for (int j = h - histogramResolution.x; j < h; j++)
			{
				for (int i = 0; i < histogramResolution.y; i++)
				{
					int indx = i + (j - h + histogramResolution.x) * histogramResolution.y;
					double hitsRatio = (p[indx]) / total_hits;

					double Area = (A[binning_mode][indx] == 0) ? 1 : A[binning_mode][indx];
					v.push_back(hitsRatio / Area);
					crop[indx] = uint8_t((hitsRatio / Area)*255.0 / vmax);
				}
			}


			//scale the crop
			//float y=9;
			//for (size_t n = 1; n < 64; n++) vmax = std::max(vmax, crop[n]);
		}
		PGMwrite(ch, histogramResolution.x, histogramResolution.y, &crop[0]);
#endif

	}
	return;
}
void saveBmpImage(int w, int h, float* data, const std::string& name)
{
	FILE *f;
	unsigned char *img = NULL;
	int filesize = 54 + 3 * w*h;  //w is your image width, h is image height, both int
	if (img)
		free(img);
	img = (unsigned char *)malloc(3 * w*h);
	memset(img, 0, sizeof(img));

	int x, y, r, g, b;

	//int hcolor = 0;
	//for (int i = 0; i < w; i++)
	//{
	//	for (int j = 0; j < h; j++)
	//	{
	//		if (data[j*w + i]>hcolor && data[j*w + i] != 10000)
	//			hcolor = data[j*w + i];
	//	}
	//}

	for (int i = 0; i < w; i++)
	{
		for (int j = 0; j<h; j++)
		{
			x = i;
			y = (h - 1) - j;

			r = data[j*w * 3 + i * 3];// *255 / hcolor;// *10000;//
			g = data[j*w * 3 + i * 3 + 1];// *255 / hcolor;// *10000;//
			b = data[j*w * 3 + i * 3 + 2];// *255 / hcolor;// *10000;//

			if (r > 255) r = 255;
			if (g > 255) g = 255;
			if (b > 255) b = 255;
			img[(x + y*w) * 3 + 2] = (unsigned char)(r);
			img[(x + y*w) * 3 + 1] = (unsigned char)(g);
			img[(x + y*w) * 3 + 0] = (unsigned char)(b);
		}
	}

	unsigned char bmpfileheader[14] = { 'B', 'M', 0, 0, 0, 0, 0, 0, 0, 0, 54, 0, 0, 0 };
	unsigned char bmpinfoheader[40] = { 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 24, 0 };
	unsigned char bmppad[3] = { 0, 0, 0 };

	bmpfileheader[2] = (unsigned char)(filesize);
	bmpfileheader[3] = (unsigned char)(filesize >> 8);
	bmpfileheader[4] = (unsigned char)(filesize >> 16);
	bmpfileheader[5] = (unsigned char)(filesize >> 24);

	bmpinfoheader[4] = (unsigned char)(w);
	bmpinfoheader[5] = (unsigned char)(w >> 8);
	bmpinfoheader[6] = (unsigned char)(w >> 16);
	bmpinfoheader[7] = (unsigned char)(w >> 24);
	bmpinfoheader[8] = (unsigned char)(h);
	bmpinfoheader[9] = (unsigned char)(h >> 8);
	bmpinfoheader[10] = (unsigned char)(h >> 16);
	bmpinfoheader[11] = (unsigned char)(h >> 24);

	f = fopen(name.c_str(), "wb");
	fwrite(bmpfileheader, 1, 14, f);
	fwrite(bmpinfoheader, 1, 40, f);
	for (int i = 0; i < h; i++)
	{
		fwrite(img + (w*(h - i - 1) * 3), 3, w, f);
		fwrite(bmppad, 1, (4 - (w * 3) % 4) % 4, f);
	}
	fclose(f);
	std::cout << "image saved" << std::endl;
	delete[] img;
}
void rotate_data(float anglex, float angley)
{

	//int cx, cy, cz;
	//glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, 0, &cx);

	//glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, particlesGlBuffer.Vbo_);


	//auto rotateDataHandle = rotationShader->GetGlShaderProgramHandle();
	//glUseProgram(rotateDataHandle);
	////glMemoryBarrier(GL_ALL_BARRIER_BITS);

	//int size = particleCenters.size();


	//here we'll transform the whole dataset permenately
	glm::mat3x3 Rx, Ry, R;

	anglex = toRadiants*anglex;
	angley = toRadiants*angley;


	Rx = glm::mat3x3(1, 0, 0, 0, cos(anglex), sin(anglex), 0, -sin(anglex), cos(anglex));
	Ry = glm::mat3x3(cos(angley), 0, -sin(angley), 0, 1, 0, sin(angley), 0, cos(angley));

	//std::cout << "rotation matrix: " << std::endl;
	//std::cout << GlobalRotationMatrix[0].x << ", " << GlobalRotationMatrix[0].y << ", " << GlobalRotationMatrix[0].z << std::endl;
	//std::cout << GlobalRotationMatrix[1].x << ", " << GlobalRotationMatrix[1].y << ", " << GlobalRotationMatrix[1].z << std::endl;
	//std::cout << GlobalRotationMatrix[2].x << ", " << GlobalRotationMatrix[2].y << ", " << GlobalRotationMatrix[2].z << std::endl;

	R = Ry*Rx;

#ifdef USE_ATB_ROTATION
    glm::quat rot(quat[0], quat[1], quat[2], quat[3]);
    GlobalRotationMatrix = glm::mat3_cast(rot);
#else
	GlobalRotationMatrix = GlobalRotationMatrix*R;
	//std::cout << "rotation matrix: " << std::endl;
	//std::cout << GlobalRotationMatrix[0].x << ", " << GlobalRotationMatrix[0].y << ", " << GlobalRotationMatrix[0].z << std::endl;
	//std::cout << GlobalRotationMatrix[1].x << ", " << GlobalRotationMatrix[1].y << ", " << GlobalRotationMatrix[1].z << std::endl;
	//std::cout << GlobalRotationMatrix[2].x << ", " << GlobalRotationMatrix[2].y << ", " << GlobalRotationMatrix[2].z << std::endl;
	//global_tree.sort_nodes_backToFront(global_tree.leaves, GlobalRotationMatrix);
#ifdef NO_BRICKING
#else
	global_tree.sort_nodes_frontToBack(global_tree.leaves, GlobalRotationMatrix);
	cellsToRender = global_tree.leaves;
#endif // NO_BRICKING

	

#endif

	//glUniformMatrix3fv(glGetUniformLocation(rotateDataHandle, "R"), 1, GL_FALSE, &R[0][0]);

	////glMemoryBarrier(GL_ALL_BARRIER_BITS);
	//glDispatchCompute(size - 1, 1 , 1);

	//glUseProgram(0);
	//glMemoryBarrier(GL_ALL_BARRIER_BITS);
}
void get_visible_tiles_in_ceil_and_floor(float level_of_detail, std::vector<std::vector<int>>& visible_tiles_per_level)
{
	////here, for floor(lod) and ceil(lod) we'll get the visible tiles and put them in visible_tiles_per_tile
	//visible_tiles_per_level.clear();
	//FloorCeilTiles.clear();
	////visible_tiles_resolutions.clear();
	////floor_ceil_blc_l.clear();

	//std::vector<int>lods;

	//int lodw, lodh;
	//glm::vec3 newpos;
	//glm::vec3 tempCamPosi, tempCamTarget;
	//float tempCameraDistance;
	//float pad_scale;    //this scale is to precompute tiles padded around the window being viewed.

	//if (std::floor(level_of_detail) == std::ceil(level_of_detail))
	//{
	//	lods.push_back(level_of_detail);
	//}
	//else
	//{
	//	lods.push_back(std::floor(level_of_detail));
	//	lods.push_back(std::ceil(level_of_detail));
	//}


	//for (int i = 0; i < lods.size(); i++)
	//{
	//	FloorCeilTiles.push_back(tiles::tiles());

	//	//1->get camera distance that will get you the lod in lod[i]
	//	tempCameraDistance = LOD.get_cam_dist(lods[i]);
	//	//2->update camera position accordingally (change variable names)
	//	tempCamPosi = CameraPosition(cameraRotation, tempCameraDistance);
	//	tempCamTarget = glm::vec3(0, 0, 0);

	//	//3-> update transformation matrices with new cam position
	//	glm::mat4x4 modelMatrix;

	//	const auto modelScale = float(initialCameraDistance) / tempCameraDistance;
	//	const auto scale = cameraDistance / tempCameraDistance;
	//	// NOTE: turn scaling off for rendering the histograms
	//	//const auto modelScale = 1.0f;
	//	modelMatrix[0][0] = modelScale;
	//	modelMatrix[1][1] = modelScale;
	//	modelMatrix[2][2] = modelScale;
	//	auto viewMatrix = glm::lookAt(tempCamPosi, tempCamTarget, camUp);
	//	auto modeViewMatrix = modelMatrix * viewMatrix;

	//	//std::cout << "The scale is " << modelScale << std::endl;

	//	//update projection matrix

	//	LOD.get_lod_width_and_hight(lods[i], lodw, lodh);

	//	//old
	//	//auto img_scale = scale;// lodw / cur_w;
	//	//auto projectionMatrix = glm::ortho(OrigL*img_scale, OrigR*img_scale, OrigB*img_scale, OrigT*img_scale, nearPlane, farPlane);
	//	//end old

	//	//new
	//	auto img_scale = std::pow(2, current_lod - lods[i]);// lodw / cur_w;	
	//	auto projectionMatrix = projectionMat;
	//	//end new

	//	//newer
	//	//compute pad scale so we add tiles just outsiste the border of the camera frustum tothe tiles in the cache so as to precomute tiles that we think the user will view with high probability
	//	pad_scale = ((windowSize.x*img_scale) + 2 * tile_w) / (windowSize.x*img_scale);
	//	//end newwer

	//	newpos = glm::project(tempCamPosi, modeViewMatrix, projectionMatrix, viewportMat);
	//	FloorCeilTiles[i].tile_image(newpos, lodw, lodh, tile_w, tile_h);

	//	//old
	//	newpos = glm::project(tempCamPosi + cameraOffset, modeViewMatrix, projectionMatrix, viewportMat);
	//	//FloorCeilTiles[i].intersect_with_camera(newpos, img_scale* windowSize.x,img_scale*  windowSize.y);
	//	glm::ivec2 lod_blc = glm::ivec2(FloorCeilTiles[i].T[0].c.x - .5*tile_w, FloorCeilTiles[i].T[0].c.y - .5*tile_h);
	//	glm::ivec2 cam_blc = glm::ivec2(newpos.x - .5*windowSize.x*img_scale*pad_scale, newpos.y - .5*windowSize.y*img_scale*pad_scale);
	//	FloorCeilTiles[i].intersect_with_camera(lod_blc, cam_blc, windowSize.x*img_scale*pad_scale, windowSize.y*img_scale*pad_scale, tile_w, tile_h, lodw, lodh);
	//	//end old





	//	////debug
	//	//FloorCeilTiles[i].visible.clear();
	//	//for (int j = 0; j < FloorCeilTiles[i].T.size(); j++)
	//	//	FloorCeilTiles[i].visible.push_back(j);
	//	////end debug

	//	visible_tiles_per_level.push_back(FloorCeilTiles[i].visible);

	//	//calculate the number of pixels to sample
	//	//we'll sample pixels that belong to the visible tiles in floor level
	//	//glm::vec3 tblc, ttrc;
	//	//FloorCeilTiles[i].get_blc_and_trc_of_viible_tiles(tblc, ttrc);
	//	//visible_tiles_resolutions.push_back(glm::vec2(ttrc.x - tblc.x, ttrc.y - tblc.y));

	//	//get blc_l
	//	//newpos = glm::vec3(FloorCeilTiles[i].T[0].c.x - .5*tile_w, FloorCeilTiles[i].T[0].c.y - .5*tile_h, newpos.z);
	//	//floor_ceil_blc_l.push_back( glm::vec2(std::max(int(-newpos.x - 0.5), 0), std::max(int(-newpos.y - 0.5), 0)));  //the .5 to make int act as ceil

	//	////calculate sampling_blc_s,trc_s and blc_l
	//	//glm::vec2 tempvec = glm::vec2(tempCamPosi.x,tempCamPosi.y) - glm::vec2(modelScale*windowSize.x*0.5f, modelScale*windowSize.y*0.5f);
	//	//sampling_blc_s = glm::vec2(std::max(0.0f,tempvec.x),std::max( 0.0f, tempvec.y));

	//	//tempvec = glm::vec2(tempCamPosi.x, tempCamPosi.y) + glm::vec2(modelScale*windowSize.x*0.5f, modelScale*windowSize.y*0.5f);
	//	//sampling_trc_s = glm::vec2(std::min(float(windowSize.x), tempvec.x), std::min(float(windowSize.y), tempvec.y));


	//}



	////make sure that for each visible ceil tile, it's 4 parents in the floor are visible as well
	//if (lods.size() > 1)
	//{
	//	int fw, fh;
	//	glm::ivec2 lod_blc = glm::ivec2(FloorCeilTiles[1].T[0].c.x - .5*tile_w, FloorCeilTiles[1].T[0].c.y - .5*tile_h);
	//	LOD.get_lod_width_and_hight(lods[1], lodw, lodh);
	//	LOD.get_lod_width_and_hight(lods[0], fw, fh);

	//	int num_tiles_in_h = lodh / tile_h;
	//	int num_tiles_in_w = lodw / tile_w;
	//	glm::ivec2 twoDindex, FtwoDindex;
	//	glm::vec2 tileC, FtileC;
	//	int tile_no;

	//	int Fnum_tiles_in_w = fw / tile_w;

	//	for (int i = 0; i < FloorCeilTiles[1].visible.size(); i++)
	//	{
	//		tile_no = FloorCeilTiles[1].visible[i];

	//		twoDindex = glm::ivec2(tile_no%num_tiles_in_w, tile_no / num_tiles_in_h);

	//		//add corresponding tiles to floor
	//		FtwoDindex = glm::ivec2(twoDindex.x * 2, twoDindex.y * 2);
	//		FloorCeilTiles[0].visible.push_back(FtwoDindex.y*Fnum_tiles_in_w + FtwoDindex.x);

	//		FtwoDindex = glm::ivec2(twoDindex.x * 2 + 1, twoDindex.y * 2);
	//		FloorCeilTiles[0].visible.push_back(FtwoDindex.y*Fnum_tiles_in_w + FtwoDindex.x);

	//		FtwoDindex = glm::ivec2(twoDindex.x * 2, twoDindex.y * 2 + 1);
	//		FloorCeilTiles[0].visible.push_back(FtwoDindex.y*Fnum_tiles_in_w + FtwoDindex.x);

	//		FtwoDindex = glm::ivec2(twoDindex.x * 2 + 1, twoDindex.y * 2 + 1);
	//		FloorCeilTiles[0].visible.push_back(FtwoDindex.y*Fnum_tiles_in_w + FtwoDindex.x);
	//	}
	//}

	//std::sort(FloorCeilTiles[0].visible.begin(), FloorCeilTiles[0].visible.end());
	//FloorCeilTiles[0].visible.erase(std::unique(FloorCeilTiles[0].visible.begin(), FloorCeilTiles[0].visible.end()), FloorCeilTiles[0].visible.end());
	//visible_tiles_per_level[0] = FloorCeilTiles[0].visible;


	////new
	//{
	//	//get corners of visible region
	//	glm::vec3 c1, c2;
	//	int lw, lh;
	//	LOD.get_lod_width_and_hight(lods[0], lw, lh);
	//	FloorCeilTiles[0].get_blc_and_trc_of_viible_tiles(c1, c2, lw, lh);

	//	//set viewable area resolution
	//	Helpers::Gl::DeleteRenderTarget(rayCastingSolutionRenderTarget);
	//	rayCastingSolutionRenderTarget = Helpers::Gl::CreateRenderTarget(c2.x - c1.x, c2.y - c1.y, GL_RGBA32F, GL_RGBA, GL_FLOAT);

	//	//map them to object space
	//	glm::vec2 c1_obj = LOD.pixel2obj(glm::vec2(c1), lods[0]);
	//	glm::vec2 c2_obj = LOD.pixel2obj(glm::vec2(c2), lods[0]);

	//	//sampling aspect ratio
	//	samplingAspectRatio = (c2.x - c1.x) / (c2.y - c1.y);

	//	//set sampling frustum
	//	sl = c1_obj.x;
	//	sb = c1_obj.y;// / samplingAspectRatio;
	//	sr = c2_obj.x;
	//	st = c2_obj.y;// / samplingAspectRatio;
	//}
	////end new



	////glm::vec3 blc, trc;
	////glm::vec3 n_blc, n_trc;
	////glm::ivec3 s_blc, s_trc;  //these are in screen space, to avoid roundoff errors, make them ints, since they represent screen pixel locaitons anyways
	////glm::vec3 pos;
	////glm::vec2 blc_tile_indx;
	////int img_width, img_height;
	////int num_tiles_in_h;
	////int tile_indx;
	////std::vector<int>::iterator it;

	//////first off, get top right corner and bottom left corner of visible tiles
	////Tiles.get_blc_and_trc_of_viible_tiles(blc, trc);
	////
	//////get the corners in screen space
	////s_blc = glm::project(blc, modelviewMat, projectionMat, viewportMat);
	////s_trc = glm::project(trc, modelviewMat, projectionMat, viewportMat);


	////for (int i = 0; i < lods.size(); i++)
	////{
	////	visible_tiles_per_level.push_back(std::vector<int>(0));
	////	//map blc and trc to the lod at hand.
	////	n_blc = glm::vec3(s_blc.x*std::pow(2, level_of_detail - lods[i]), s_blc.y*std::pow(2, level_of_detail - lods[i]), s_blc.z);
	////	n_trc = glm::vec3(s_trc.x*std::pow(2, level_of_detail - lods[i]), s_trc.y*std::pow(2, level_of_detail - lods[i]), s_trc.z);

	////	//get width and height of lod at hand
	////	LOD.get_lod_width_and_hight(lods[i], img_width, img_height);
	////	num_tiles_in_h = img_height / tile_h;
	////

	////	//get visible tiles in lods[i], that is, tiles within n_blc and n_trc
	////	for (float j = n_blc.x; j < n_trc.x; j = j + tile_w)
	////	{
	////		for (float k = n_blc.y; k < n_trc.y; k = k + tile_h)
	////		{
	////			pos = n_blc + glm::vec3(j, k, 0);

	////			blc_tile_indx = glm::vec2(int(n_blc.x) / img_width, int(ceil_blc.x) % img_height);

	////			tile_indx = (blc_tile_indx.x + j/tile_w)*num_tiles_in_h + (blc_tile_indx.y + k/tile_h);

	////			if (tile_indx < std::pow(num_tiles_in_h, 2))            //because of floating point errors, some tiles that are not visible, or even not there can get counted
	////			{
	////				//ensure that there are no duplicates
	////				it = std::find(visible_tiles_per_level[i].begin(), visible_tiles_per_level[i].end(), tile_indx);
	////				if (it==visible_tiles_per_level[i].end())
	////					visible_tiles_per_level[i].push_back(tile_indx);
	////			}
	////		}
	////	}
	////}
}
//bool found_in_page_texture(glm::vec3 pos, float lod)
//{
//	//check whether 'tile_indx' has an index in the 'page texture' which means that it is available in the 
//	//physical memory
//
//	//first, read the texture that the tile belongs to
//	GLubyte* temp_texture = new GLubyte[cur_w*cur_h*3];
//	glBindTexture(GL_TEXTURE_2D, pageTexture);
//
//	glGetTexImage(GL_TEXTURE_2D, std::floor(lod + 0.5), GL_RGB, GL_UNSIGNED_BYTE, temp_texture);
//
//	int MaxTextureSize;
//	glGetIntegerv(GL_MAX_TEXTURE_SIZE, &MaxTextureSize);
//
//	int phys_indx = temp_texture[int(cur_w * 3 * (pos.y - 0.5f*tile_h) + 3 * (pos.x - 0.5f*tile_w)+2)];
//
//    //now we check if the center of the tile is non-zero indicating that it is availalbe in memory
//
//	if (phys_indx/100.0 !=lod)
//	{
//		return false;
//	}
//	else
//	{
//		//a tile to be rendered is found in the physical texture, we retrive it, and put it in the render target
// 	//	glBindFramebuffer(GL_FRAMEBUFFER, PhysicalFbo);
//
//		//glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, rendertarget, 0);
//
//		//glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, physicalTexture, 0);
//		//glDrawBuffer(GL_COLOR_ATTACHMENT1);
//
//		////we need the position to copy from in the physical texture
//		//int indx_x = tile_w*temp_texture[int(cur_w * 3 * (pos.y - 0.5f*tile_h) + 3 * (pos.x - 0.5f*tile_w)    )];
//		//int indx_y = tile_h*temp_texture[int(cur_w * 3 * (pos.y - 0.5f*tile_h) + 3 * (pos.x - 0.5f*tile_w) + 1)];
//
//		//glBlitFramebuffer(             indx_x,              indx_y,       indx_x+tile_w,       indx_y+tile_h, 
//		//	              pos.x - 0.5f*tile_w, pos.y - 0.5f*tile_h, pos.x + 0.5f*tile_w, pos.y + 0.5f*tile_h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
//		//auto err = glGetError();
//
//
//
//		//new
//		int indx_x = tile_w*temp_texture[int(cur_w * 3 * (pos.y - 0.5f*tile_h) + 3 * (pos.x - 0.5f*tile_w)    )];
//		int indx_y = tile_h*temp_texture[int(cur_w * 3 * (pos.y - 0.5f*tile_h) + 3 * (pos.x - 0.5f*tile_w) + 1)];
//
//		glBindTexture(GL_TEXTURE_2D, physicalTexture);
//		glGetTexImage(GL_TEXTURE_2D, 0, GL_RG, GL_UNSIGNED_BYTE, physical_tex_data);
//
//		glBindTexture(GL_TEXTURE_2D, rendertarget);
//		glGetTexImage(GL_TEXTURE_2D, 0, GL_RG, GL_UNSIGNED_BYTE, screen_tex_data);
//
//		
//		for (int j = 0; j < tile_w; j++)
//		{
//			for (int k = 0; k < tile_h; k++)
//			{
//				screen_tex_data[int(windowSize.y * 2 * (pos.y - 0.5f*tile_h+k) + 2 * (pos.x - 0.5f*tile_w+j))]    = physical_tex_data[MaxTextureSize * 2 * (indx_y + k) + 2 * (indx_x + j)];
//				screen_tex_data[int(windowSize.y * 2 * (pos.y - 0.5f*tile_h+k) + 2 * (pos.x - 0.5f*tile_w+j) + 1)]= physical_tex_data[MaxTextureSize * 2 * (indx_y + k) + 2 * (indx_x + j) + 1];
//			}
//		}
//
//		glTexImage2D(GL_TEXTURE_2D, 0, GL_RG8, windowSize.x,windowSize.y, 0, GL_RG, GL_UNSIGNED_BYTE, screen_tex_data);
//		//end new
//
//
//		glBindFramebuffer(GL_FRAMEBUFFER, TiledFbo);
//	}
//	
//	return true;
//}
//void render_tile(std::vector<int> tiles_to_render, float lod)
//{
//	//render a tile, put it in 'physical texture', update 'page texture'
//
//	int MaxTextureSize;
//	glGetIntegerv(GL_MAX_TEXTURE_SIZE, &MaxTextureSize);
//
//	for (int t = 0; t < tiles_to_render.size(); t++)
//	{
//		//render 'tile'
//		int i = tiles_to_render[t];
//		glBindTexture(GL_TEXTURE_2D, TiledFbo_TextureBuffer[i]);
//		glTexImage2D(GL_TEXTURE_2D, 0, GL_RG8, windowSize.x, windowSize.y, 0, GL_RG, GL_UNSIGNED_BYTE, nullptr);
//		glGenerateMipmap(GL_TEXTURE_2D);
//
//		glBindFramebuffer(GL_FRAMEBUFFER, TiledFbo);// _DepthBuffer);
//
//		glActiveTexture(GL_TEXTURE0 + t);
//		glBindTexture(GL_TEXTURE_2D, TiledFbo_TextureBuffer[i]);
//
//		glBindFramebuffer(GL_FRAMEBUFFER, TiledFbo);
//		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0+t, GL_TEXTURE_2D, TiledFbo_TextureBuffer[i], 0);
//	}
//
//	glBindTexture(GL_TEXTURE_2D, 0);
//
//	delete[] TiledFbo_DrawBuffer;
//	TiledFbo_DrawBuffer = new GLenum[MaxFboTexCount];
//
//
//	for (int t = 0; t < MaxFboTexCount; t++)
//	{
//		TiledFbo_DrawBuffer[t] = GL_COLOR_ATTACHMENT0 + t;
//	}
//
//
//	glDrawBuffers(MaxFboTexCount, TiledFbo_DrawBuffer);
//
//	// Always check that our framebuffer is ok
//	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
//	{
//		std::cout << std::endl;
//		std::cout << "SOMETHING IS WRONG WITH THE TILED FBO" << std::endl;
//		std::cout << std::endl;
//	}
//	
//	//use FBO
//	glBindFramebuffer(GL_FRAMEBUFFER, TiledFbo);
//	glViewport(0, 0, windowSize.x, windowSize.y);// rayCastingSolutionRenderTarget.Width, rayCastingSolutionRenderTarget.Height);
//
//	// FIXME: viewport causes tears for some reason if sampling rate < 1024
//	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
//	// clear render target
//	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//
//	glEnable(GL_POINT_SPRITE);
//	glEnable(GL_PROGRAM_POINT_SIZE);
//	glEnable(GL_DEPTH_TEST);
//	glFrontFace(GL_CCW);
//
//
//	auto samplingProgramHandle = samplingShader->GetGlShaderProgramHandle();
//
//	assert(glUseProgram);
//	glUseProgram(samplingProgramHandle);
//
//	auto ttor = tiles_to_render.size();
//
//	glUniform1i(glGetUniformLocation(samplingProgramHandle, "ttor"), ttor);
//
//	glUniform1f(glGetUniformLocation(samplingProgramHandle, "tileW"), tile_w);
//	glUniform1f(glGetUniformLocation(samplingProgramHandle, "tileH"), tile_h);
//
//	{
//		int i = tiles_to_render[0];
//		glm::vec3 newpos = glm::project(Tiles.T[Tiles.visible[i]].c, modelviewMat, projectionMat, viewportMat);
//		glUniform3fv(glGetUniformLocation(samplingProgramHandle, "tc_0"), 1, &newpos[0]);
//	}
//	if (tiles_to_render.size() > 1)
//	{
//		int i = tiles_to_render[1];
//		glm::vec3 newpos = glm::project(Tiles.T[Tiles.visible[i]].c, modelviewMat, projectionMat, viewportMat);
//		glUniform3fv(glGetUniformLocation(samplingProgramHandle, "tc_1"), 1, &newpos[0]);
//	}
//	if (tiles_to_render.size() > 2)
//	{
//		int i = tiles_to_render[2];
//		glm::vec3 newpos = glm::project(Tiles.T[Tiles.visible[i]].c, modelviewMat, projectionMat, viewportMat);
//		glUniform3fv(glGetUniformLocation(samplingProgramHandle, "tc_2"), 1, &newpos[0]);
//	}
//	if (tiles_to_render.size() > 3)
//	{
//		int i = tiles_to_render[3];
//		glm::vec3 newpos = glm::project(Tiles.T[Tiles.visible[i]].c, modelviewMat, projectionMat, viewportMat);
//		glUniform3fv(glGetUniformLocation(samplingProgramHandle, "tc_3"), 1, &newpos[0]);
//	}
//	if (tiles_to_render.size() > 4)
//	{
//		int i = tiles_to_render[4];
//		glm::vec3 newpos = glm::project(Tiles.T[Tiles.visible[i]].c, modelviewMat, projectionMat, viewportMat);
//		glUniform3fv(glGetUniformLocation(samplingProgramHandle, "tc_4"), 1, &newpos[0]);
//	}
//	if (tiles_to_render.size() > 5)
//	{
//		int i = tiles_to_render[5];
//		glm::vec3 newpos = glm::project(Tiles.T[Tiles.visible[i]].c, modelviewMat, projectionMat, viewportMat);
//		glUniform3fv(glGetUniformLocation(samplingProgramHandle, "tc_5"), 1, &newpos[0]);
//	}
//	if (tiles_to_render.size() > 6)
//	{
//		int i = tiles_to_render[6];
//		glm::vec3 newpos = glm::project(Tiles.T[Tiles.visible[i]].c, modelviewMat, projectionMat, viewportMat);
//		glUniform3fv(glGetUniformLocation(samplingProgramHandle, "tc_6"), 1, &newpos[0]);
//	}
//	if (tiles_to_render.size() > 7)
//	{
//		int i = tiles_to_render[7];
//		glm::vec3 newpos = glm::project(Tiles.T[Tiles.visible[i]].c, modelviewMat, projectionMat, viewportMat);
//		glUniform3fv(glGetUniformLocation(samplingProgramHandle, "tc_7"), 1, &newpos[0]);
//	}
//
//	glUniform1f(glGetUniformLocation(samplingProgramHandle, "far"), farPlane);
//	glUniform1f(glGetUniformLocation(samplingProgramHandle, "near"), nearPlane);
//	glUniform1f(glGetUniformLocation(samplingProgramHandle, "particleSize"), particleRadius);
//	glUniform1f(glGetUniformLocation(samplingProgramHandle, "aspectRatio"), aspectRatio);
//
//	glUniform1i(glGetUniformLocation(samplingProgramHandle, "samplingRunIndex"), samplingRunIndex);
//	glUniform1i(glGetUniformLocation(samplingProgramHandle, "maxSamplingRuns"), maxSamplingRuns);
//
//	glUniform1f(glGetUniformLocation(samplingProgramHandle, "viewportWidth"), static_cast<float>(windowSize.x));
//
//	auto right = glm::normalize(glm::cross(normalize(camTarget - camPosi), camUp));
//	glUniform3fv(glGetUniformLocation(samplingProgramHandle, "right"), 1, &right[0]);
//	glUniform3fv(glGetUniformLocation(samplingProgramHandle, "up"), 1, &camUp[0]);
//
//	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "Model"), 1, GL_FALSE, &modelMat[0][0]);
//	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "ModelView"), 1, GL_FALSE, &modelviewMat[0][0]);
//	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "View"), 1, GL_FALSE, &viewMat[0][0]);
//	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "Projection"), 1, GL_FALSE, &projectionMat[0][0]);
//
//	glm::mat4 ViewAlignmentMatrix;
//	ViewAlignmentMatrix = glm::rotate(ViewAlignmentMatrix, cameraRotation.x, glm::vec3(0.0f, 1.0f, 0.0f));
//	ViewAlignmentMatrix = glm::rotate(ViewAlignmentMatrix, cameraRotation.y, glm::vec3(1.0f, 0.0f, 0.0f));
//
//	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "ViewAlignmentMatrix"), 1, GL_FALSE, &ViewAlignmentMatrix[0][0]);
//
//	glUniform3fv(glGetUniformLocation(samplingProgramHandle, "ViewPosition"), 1, &camPosi[0]);
//	glUniform2i(glGetUniformLocation(samplingProgramHandle, "viewSlice"), 0, 0);
//
//	// render particles using ray casting
//	// set uniforms
//	glUniformMatrix4fv(glGetUniformLocation(samplingProgramHandle, "ViewProjection"), 1, GL_FALSE, &viewprojectionMat[0][0]);
//
//	auto initialOffset = glm::vec3(0.0f, 0.0f, 0.0f);
//
//	// FIXME: fix magic numbers
//	if (particleInstances.x > 1) {
//		initialOffset.x = -0.125f * static_cast<float>(particleInstances.x) * modelExtent.x;
//	}
//	if (particleInstances.y > 1) {
//		initialOffset.y = -0.3f * static_cast<float>(particleInstances.y) * modelExtent.y;
//	}
//	if (particleInstances.z > 1) {
//		initialOffset.z = -0.125f * static_cast<float>(particleInstances.z) * modelExtent.z;
//	}
//
//	// render samples
//	for (int zInstance = 0; zInstance < particleInstances.z; ++zInstance)
//	{
//		for (int yInstance = 0; yInstance < particleInstances.y; ++yInstance)
//		{
//			for (int xInstance = 0; xInstance < particleInstances.x; ++xInstance)
//			{
//				auto modelOffset = initialOffset + glm::vec3(static_cast<float>(xInstance)* modelExtent.x, static_cast<float>(yInstance)* modelExtent.y, static_cast<float>(zInstance)* modelExtent.z);
//
//				glUniform3fv(glGetUniformLocation(samplingProgramHandle, "modelOffset"), 1, &modelOffset[0]);
//				particlesGlBuffer.Render(GL_POINTS);
//			}
//		}
//	}
//
//	glBindFramebuffer(GL_FRAMEBUFFER, glfinalfbo);
//	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
//	glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, rendertarget, 0);
//	
//
//	//glBindTexture(GL_TEXTURE_2D, TiledFbo_TextureBuffer[j]);
//	//glCopyTexSubImage2D(rendertarget, 0, 0, 0, x, y, tile_w, tile_h);
//
//	for (int t = 0; t < tiles_to_render.size(); t++)
//	{
//		int i = tiles_to_render[t];
//		glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, TiledFbo_TextureBuffer[i], 0);
//		glDrawBuffer(GL_COLOR_ATTACHMENT1);
//		glm::vec3 newpos = glm::project(Tiles.T[Tiles.visible[i]].c, modelviewMat, projectionMat, viewportMat);
//		glBlitFramebuffer(newpos.x - 0.5f*tile_w, newpos.y - 0.5f*tile_h, newpos.x + 0.5f*tile_w+1, newpos.y + 0.5f*tile_h+1,
//			              newpos.x - 0.5f*tile_w, newpos.y - 0.5f*tile_h, newpos.x + 0.5f*tile_w+1, newpos.y + 0.5f*tile_h+1, GL_COLOR_BUFFER_BIT, GL_LINEAR);
//	}
//	glBindFramebuffer(GL_FRAMEBUFFER, TiledFbo);
//	glDeleteBuffers(MaxFboTexCount, TiledFbo_DrawBuffer);
//
//
//
//
//
//
//	///////*********************************************** start of SVT **********************************************************
//
//
//	//unsigned char* temp_tex = new unsigned char[int(cur_w*cur_h) * 3];
//
//	//for (int t = 0; t < tiles_to_render.size(); t++)
//	//{
//	//	//put tile in 'physical texture'
//	//	int i = tiles_to_render[t];
//	//	glm::vec3 newpos = glm::project(Tiles.T[Tiles.visible[i]].c, modelviewMat, projectionMat, viewportMat);
//	//	
//	//	//old
//	//	//glBindFramebuffer(GL_FRAMEBUFFER, PhysicalFbo);
//	//	//glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, physicalTexture, 0);
//	//	//glFramebufferTexture2D(GL_READ_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, TiledFbo_TextureBuffer[i], 0);
//	//	//glDrawBuffer(GL_COLOR_ATTACHMENT1);
//	//	//glBlitFramebuffer(newpos.x - 0.5f*tile_w, newpos.y - 0.5f*tile_h, newpos.x + 0.5f*tile_w, newpos.y + 0.5f*tile_h,
//	//	//	                              phys_x,                 phys_y,          phys_x+tile_w,          phys_y+tile_h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
//	//	//end old
//
//
//	//	//new
//	//	glBindTexture(GL_TEXTURE_2D, TiledFbo_TextureBuffer[i]);
//	//	glGetTexImage(GL_TEXTURE_2D, 0, GL_RG, GL_UNSIGNED_BYTE, screen_tex_data);
//
//	//	glBindTexture(GL_TEXTURE_2D, physicalTexture);
//	//	glGetTexImage(GL_TEXTURE_2D, 0, GL_RG, GL_UNSIGNED_BYTE, physical_tex_data);
//
//	//	for (int j = 0; j < tile_w; j++)
//	//	{
//	//		for (int k = 0; k < tile_h; k++)
//	//		{
//	//			physical_tex_data[MaxTextureSize * 2 * (phys_y + k) + 2 * (phys_x + j)] = screen_tex_data[int(windowSize.y * 2 * (newpos.y - 0.5f*tile_h + k) + 2 * (newpos.x - 0.5f*tile_w + j))];
//	//			physical_tex_data[MaxTextureSize * 2 * (phys_y + k) + 2 * (phys_x + j) + 1] = screen_tex_data[int(windowSize.y * 2 * (newpos.y - 0.5f*tile_h + k) + 2 * (newpos.x - 0.5f*tile_w + j) + 1)];
//	//		}
//	//	}
//
//	//	glTexImage2D(GL_TEXTURE_2D, 0, GL_RG8, MaxTextureSize, MaxTextureSize, 0, GL_RG, GL_UNSIGNED_BYTE, physical_tex_data);
//	//	//end new
//
//
//	//	//-> update 'page' texture
//	//	glBindTexture(GL_TEXTURE_2D, pageTexture);
//
//	//	
//	//	glGetTexImage(GL_TEXTURE_2D, std::floor(lod + 0.5), GL_RGB, GL_UNSIGNED_BYTE, temp_tex);
//
//	//	//update pixels
//	//	temp_tex[int(cur_w * 3 * (newpos.y - 0.5f*tile_h) + 3 * (newpos.x - 0.5f*tile_w)    )] = phys_x / tile_w;
//	//	temp_tex[int(cur_w * 3 * (newpos.y - 0.5f*tile_h) + 3 * (newpos.x - 0.5f*tile_w) + 1)] = phys_y / tile_h;
//	//	temp_tex[int(cur_w * 3 * (newpos.y - 0.5f*tile_h) + 3 * (newpos.x - 0.5f*tile_w) + 2)] = lod*100;                   //meaning that the tile exists in the pysical texture
//	//	glTexImage2D(GL_TEXTURE_2D, std::floor(lod + 0.5), GL_RGB, cur_w, cur_h, 0, GL_RGB, GL_UNSIGNED_BYTE, temp_tex);
//
//	//	//MaxTextureSize = 1024;
//
//	//	phys_y += tile_h;
//
//	//	if (phys_y >= MaxTextureSize)
//	//	{
//	//		phys_y = 0;F
//	//		phys_x += tile_w;
//	//		if (phys_x >= MaxTextureSize)
//	//		{
//	//			phys_x = 0;
//	//		}
//	//	}
//	//}
//
//	//delete[]temp_tex;
//	//
//
//	//glBindFramebuffer(GL_FRAMEBUFFER, TiledFbo);
//} 
void resetSimLimits(bool flag)
{
	//reset limits
	//download limits
	std::vector<float> simLimits(2 + windowSize.x*windowSize.y, 0);
	{
		auto ssbo = simLimitsF_ssbo;
		{
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
			auto readMap = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
			memcpy(reinterpret_cast<char*>(&simLimits[0]), readMap, simLimits.size() * sizeof(*simLimits.begin()));

			glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
		}
	}

	{
		simLimits[0] = 1000000; //min
		simLimits[1] = 0;  //max

		for (int i = 0; i < windowSize.x*windowSize.y; i++)
		{
			if (flag)                      //if flag we reset all
			{
				simLimits[i + 2] = -1;
			}
			else if (simLimits[i + 2] != -2)  //otherwise we keep user selection
				simLimits[i + 2] = -1;
		}

		auto ssbo = simLimitsF_ssbo;
		{
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
			GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
			memcpy(p, reinterpret_cast<char*>(&simLimits[0]), simLimits.size() * sizeof(*simLimits.begin()));
			glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
		}
	}
}
void tile_based_culling(bool tile_flag)
{
	if (plainRayCasting || pointCloudRendering)
		return;



	//for now
	OrigL = -0.5f;
	OrigR = 0.5f;
	OrigB = -0.5f / aspectRatio;
	OrigT = 0.5f / aspectRatio;
	//remove later

	//get camera offset in object space, and pass that to the LOD class, and along with it, pass the lod we are at
	prev_lod = current_lod;
	current_lod = LOD.get_lod(cameraDistance);
	downsampling_lod = current_lod;
	LOD.get_lod_width_and_hight(current_lod, cur_w, cur_h);


	LOD.get_visible_in_ceil_and_floor(current_lod, windowSize.x, windowSize.y, cameraOffset, OrigR, OrigL, OrigT, OrigB, camPosi, obb);


	//set sampling parameters
	glm::vec3 c1, c2;
	glm::vec2 c1j, c2j;
	int lodw, lodh;
	LOD.get_lod_width_and_hight(std::floor(current_lod-lodDelta), lodw, lodh);
	LOD.myTiles[std::floor(current_lod-lodDelta)].get_blc_and_trc_of_viible_tiles(c1, c2, lodw, lodh);


	//set sampling texture parameters
	sw = c2.x - c1.x;
	sh = c2.y - c1.y;
	/*Helpers::Gl::DeleteRenderTarget(rayCastingSolutionRenderTarget);
	rayCastingSolutionRenderTarget = Helpers::Gl::CreateRenderTarget(c2.x-c1.x, c2.y-c1.y, GL_RG8, GL_RG, GL_UNSIGNED_BYTE);*/

	//set sampling projection matrix parameters
	c1j = LOD.pixel2obj(glm::vec2(c1), std::floor(current_lod-lodDelta));
	c2j = LOD.pixel2obj(glm::vec2(c2), std::floor(current_lod-lodDelta));

	float scale = initialCameraDistance / LOD.get_cam_dist(std::floor(current_lod-lodDelta));

	//get sampling frustum with respect to the new camera position in xy plane
	c1j -= glm::vec2(cameraOffset);
	c2j -= glm::vec2(cameraOffset);

	//debug
	//scale = OrigL / c1j.x;
	//end debug

	sl = c1j.x*scale;
	sr = c2j.x*scale;
	sb = c1j.y*scale;
	st = c2j.y*scale;

	//debug
	//sl = OrigL;
	//sr = OrigR;
	//sb = OrigB;
	//st = OrigT;
	//end debug




	//output lod
	//std::cout << "the current lod is " << current_lod <<", with camera distance equals to "<<cameraDistance<< std::endl;

	//debug

	//save_SamplingTexture = true;

	//display();
	//save_SamplingTexture = false;
	//end debug
}


void tile_based_panning(bool tile_flag)
{

	return;
	if (plainRayCasting || pointCloudRendering)
		return;

	glm::vec3 temp_camPosi = CameraPosition(cameraRotation, cameraDistance);
	glm::vec3 temp_camTarget = glm::vec3(0.0f, 0.0f, 0.0f);
	//update transformation matrices with new cam position
	glm::mat4x4 modelMatrix;
	const auto modelScale = initialCameraDistance / cameraDistance;
	// NOTE: turn scaling off for rendering the histograms
	//const auto modelScale = 1.0f;
	modelMatrix[0][0] = modelScale;
	modelMatrix[1][1] = modelScale;
	modelMatrix[2][2] = modelScale;
	auto viewMatrix = glm::lookAt(temp_camPosi, temp_camTarget, camUp);
	auto modeViewMatrix = modelMatrix * viewMatrix;
	modelviewMat = modeViewMatrix;
	float tw, th;

	glm::vec3 newpos = glm::project(camPosi, modelviewMat, projectionMat, viewportMat);
	//Tiles.intersect_with_camera(newpos,windowSize.x,windowSize.y);


	//glm::vec3 cameraOffset_pixels = glm::vec3((cameraOffset.x*tile_w) / tw_obj, (cameraOffset.y*tile_h) / th_obj, 0);

	glm::ivec2 lod_blc = glm::ivec2(Tiles_s.T[0].c.x - .5*tile_w, Tiles_s.T[0].c.y - .5*tile_h);
	glm::ivec2 cam_blc = glm::ivec2(newpos.x - .5*windowSize.x, newpos.y - .5*windowSize.y);
	//cam_blc -= glm::ivec2(cameraOffset_pixels.x, cameraOffset_pixels.y);

	Tiles.intersect_with_camera(lod_blc, cam_blc, windowSize.x, windowSize.y, tile_w, tile_h, cur_w, cur_h);

	//update visible tiles in Tiles_s
	Tiles_s.visible = Tiles.visible;

	//std::cout << "Number of visible tiles: " << Tiles.visible.size() << std::endl;

	reset();
}
void keyboardUp(unsigned char key, int mx, int my)
{
	//auto modifiers = glutGetModifiers();
	//std::cout << "in key up " << std::endl;
	//if (modifiers & GLUT_ACTIVE_ALT)
	//{
	//	std::cout << "alt released " << std::endl;
	//	computeAvgNDF(true);
	//}
}
void keyboard(unsigned char key, int mx, int my) {
	static const auto camOff = 0.025f;
	static const float powerStepSize = 0.5f;
	static const float particleRadiusStepSize = 0.000005f;
	static const int samplesPerRunStepSize = 2;
	float level_of_detail;
	float prev_lod;

	GLdouble posx, posy, posz;
	glm::vec3 nc, nh, nw, diff;
	glm::vec3 hvec, wvec;
	glm::vec3 newpos;
	glm::vec3 X, Y, Z;
	auto ssboSize = phys_tex_dim*phys_tex_dim * histogramResolution.x * histogramResolution.y * sizeof(float);
	//debug step 
	int tbts;
	std::vector<int> l;

	//std::vector<int> indicies;
	//std::vector<int> leaves;
	bool above;
	glm::vec3 vec;
	//std::vector<int> tint(particleCenters.size());
	int target_lod;

	//indicies.clear();
	//global_tree.get_leaves();

	const float ssboClearColor = 0.0f;

	float maxn = 0;
	float minn = 10000000000;

	//for (int i = 0; i < leaves.size(); i++)
	//{
	//	if (global_tree.nodes[leaves[i]].indicies.size()>maxn)
	//		maxn = global_tree.nodes[leaves[i]].indicies.size();
	//	else if (global_tree.nodes[leaves[i]].indicies.size() < minn)
	//		minn = global_tree.nodes[leaves[i]].indicies.size();
	//}

	clock_t tstart;
	std::vector<glm::vec3> dirs;

	dirs.push_back(glm::vec3(1, 1, 1));
	dirs.push_back(glm::vec3(1, 1, -1));
	dirs.push_back(glm::vec3(1, -1, -1));
	dirs.push_back(glm::vec3(-1, -1, -1));
	dirs.push_back(glm::vec3(-1, 1, 1));
	dirs.push_back(glm::vec3(-1, -1, 1));
	dirs.push_back(glm::vec3(-1, 1, -1));
	dirs.push_back(glm::vec3(1, -1, 1));

	bool rotation_test = true;

	//end debug step

	if (tweakbarInitialized) {
		if (TwEventKeyboardGLUT(key, mx, my)) {
			return;
		}
	}


	switch (key) {
		// escape key
	case 27:
		exit(0);
		break;
	case '8':
		binning_mode ++;
		binning_mode = binning_mode % 3;

		if (binning_mode == 0) {
			std::cout << "Binning using old method" << std::endl;
		}
		else if (binning_mode == 1) {
			std::cout << "Binning using Spherical Coordinates binning" << std::endl;
		}
		else if (binning_mode == 2) {
			std::cout << "Binning using Lambert Azimuthal Equal-Area projection binning" << std::endl;
		}
		//initialize();
		clear_NDF_Cache();
		tile_based_culling(false);
		tile_based_panning(false);
		update_page_texture();
		computeBinAreas();
		preIntegrateBins();
		reset();
		display();
		break;
	case '2':
		particleRadius -= .1*particleRadius;
		initialize();

		tile_based_culling(false);
		tile_based_panning(false);
		update_page_texture();
		reset();
		break;
	case '3':
		particleRadius += .1*particleRadius;
		initialize();

		tile_based_culling(false);
		tile_based_panning(false);
		update_page_texture();
		reset();
		break;
	case '#':
		showEmptyTiles = !showEmptyTiles;
		break;
	case '1':

		singleRay = !singleRay;

		if (singleRay)
		{
			if (plainRayCasting)
			{
				erase_progressive_raycasting_ssbo();
			}
			std::cout << "Shooting 1 ray per pixel " << std::endl;
		}
		else
		{
			std::cout << "Shooting the maximum number of rays per pixel " << std::endl;
		}
		mymaxsamplingruns = 1;

		//phys_x = 0;
		//phys_y = 0;

		initialize();

		tile_based_culling(false);
		tile_based_panning(false);
		update_page_texture();
		reset();
		break;
	case 'u':case 'U':
		cachedRayCasting = !cachedRayCasting;
		if (cachedRayCasting)
			noCaching = false;
		if (cachedRayCasting)
		{
			histogramResolution = glm::ivec2(2, 2);
			initialize();
			std::cout << "Rendering using cached Raycasting" << std::endl;
			//in both cases we cull and update tiles
			tile_based_culling(false);
			//tile_based_panning(false);
			update_page_texture();
			reset();
		}
		else
		{
			//clear_NDF_Cache();
			histogramResolution = origHistogramResolution;
			initialize();
			std::cout << "Rendering using ndfs" << std::endl;
			//in both cases we cull and update tiles
			tile_based_culling(false);
			//tile_based_panning(false);
			update_page_texture();
			reset();
			//computeBinAreas();
			preIntegrateBins();
		}

		break;
	case '6':
		samplingFlag = !samplingFlag;
		if (samplingFlag)
			std::cout << "sampling enabled" << std::endl;
		else
			std::cout << "sampling disabled" << std::endl;
		break;
	case '0':
		quantizeFlag = !quantizeFlag;
		if (quantizeFlag)
			std::cout << "quantization enabled" << std::endl;
		else
			std::cout << "quantiztion disabled" << std::endl;
		break;
	case 'b':case 'B':
		cachedRayCasting =! cachedRayCasting;
		if (cachedRayCasting)
			noCaching = true;
		//plainRayCasting = !plainRayCasting;
		if (cachedRayCasting&&noCaching)
		{
			histogramResolution = glm::ivec2(2, 2);
			initialize();
			std::cout << "Rendering using cached Raycasting" << std::endl;
			//in both cases we cull and update tiles
			tile_based_culling(false);
			//tile_based_panning(false);
			update_page_texture();
			reset();
		}
		else
		{
			//clear_NDF_Cache();
			histogramResolution = origHistogramResolution;
			initialize();
			std::cout << "Rendering using ndfs" << std::endl;
			//in both cases we cull and update tiles
			tile_based_culling(false);
			//tile_based_panning(false);
			update_page_texture();
			reset();
			//computeBinAreas();
			preIntegrateBins();
		}
		break;
	case '4':
		//pointCloudRendering = !pointCloudRendering;
		//if (pointCloudRendering)
		//{
		//	std::cout << "Rendering point cloud" << std::endl;
		//}
		//else
		//{
		//	std::cout << "Rendering using ndfs" << std::endl;
		//	tile_based_culling(false);
		//	tile_based_panning(false);
		//	update_page_texture();
		//}
		//reset();
		maxSamplingRunsFactor /= 2.0f;
		maxSamplingRunsFactor = (std::max<float>)(std::pow(2.f, -64), maxSamplingRunsFactor);
		maxSamplingRuns = std::max(1.0*maxSamplingRunsFactor, pow(4, std::floor(current_lod - lodDelta))*maxSamplingRunsFactor);
		std::cout << "max sampling runs factor decreased to: " << maxSamplingRunsFactor << std::endl;
		std::wcout << "max sampling runs decreased to: " << maxSamplingRuns << std::endl;
		break;

	case '5':
		cull = !cull;
		if (cull)
			std::cout << "sampling enabled" << std::endl;
		else
			std::cout << "sampling disabled" << std::endl;
		break;

		//case 'x': case 'X':
		//	
		//	//frustum/octree culling
		//	//intersect frustum with octree
		//	//traverse the octree in a dfs manner
		//	//remove from particlecenters anything not found in pindex
		//	//debug step 
		//	oclude_percentage_frustum -= 0.05f;
		//	if (oclude_percentage_frustum <= 0)
		//		oclude_percentage_frustum = 1;
		//	glutPostRedisplay(); 

		//	for (int i = 0; i < leaves.size(); i++)
		//	{
		//		if (leaves[i] == 486)
		//			i = i;


		//		//test if the node is partially inside/fully inside the frustum or not
		//		for (int k = 0; k < dirs.size(); k++)
		//		{
		//			
		//			//if a single vertex of the corners of the current node is above all planes, then the whole node
		//			//is regarded as inside the frustum
		//			above = true;
		//			for (int j = 0; j < test_frustum[frustum_indx].planes.size(); j++)
		//			{
		//				vec = global_tree.nodes[leaves[i]].center+global_tree.nodes[leaves[i]].extent*dirs[k]- test_frustum[frustum_indx].planes[j].p;
		//				if (glm::dot(vec, test_frustum[frustum_indx].planes[j].n) < 0)
		//				{
		//					above = false;
		//					break;
		//				}
		//			}
		//			if (above)
		//			{
		//				indicies.insert(indicies.end(), global_tree.nodes[leaves[i]].indicies.begin(), global_tree.nodes[leaves[i]].indicies.end());
		//				break;
		//			}
		//		}
		//		
		//	}
		//	//keep only unique
		//	std::sort(indicies.begin(), indicies.end());

		//	it = std::unique(indicies.begin(), indicies.end());
		//	indicies.resize(std::distance(indicies.begin(), it));
		//	//end debug step
		//
		//	tstart = clock();
		//	visible_indices.clear();
		//	global_tree.get_visible_indicies(0, test_frustum[frustum_indx], particleCenters, visible_indices);

		//	//keep only unique
		//	std::sort(visible_indices.begin(), visible_indices.end());
		//	
		//	it = std::unique(visible_indices.begin(), visible_indices.end());
		//	visible_indices.resize(std::distance(visible_indices.begin(), it));
		//	

		//	//debug
		//	//find a node in visible_indices that is not in visible
		//	it = std::set_difference(visible_indices.begin(), visible_indices.end(), indicies.begin(), indicies.end(), tint.begin());
		//	tint.resize(it - tint.begin());

		//	//pick first one, know which node it belongs to
		//	for (int i = 0; i < global_tree.nodes.size(); i++)
		//	{
		//		it = std::find(global_tree.nodes[i].indicies.begin(), global_tree.nodes[i].indicies.end(), tint[0]);
		//		if (it != global_tree.nodes[i].indicies.end())
		//		{
		//			i = i;
		//		}
		//	}
		//	//end debug


		//	std::cout<<"Time taken: "<< (double)(clock() - tstart) / CLOCKS_PER_SEC<<" seconds"<<std::endl;
		//	

		//	//end frustum/octree culling
		//	temp_particleCenters.clear();
		//	for (int i = 0; i < visible_indices.size();i++)
		//		temp_particleCenters.push_back(particleCenters[visible_indices[i]]);
		//	//end pass only visible particles to shader


		//	Helpers::Gl::CreateGlMeshFromBuffers(temp_particleCenters, particlesGlBuffer);
		//	reset();

		//	frustum_indx++;


		//	if (frustum_indx==test_frustum.size())
		//		frustum_indx = 0;


		//	break;
	case 'c':
		probeNDFsMode++;
		probeNDFsMode = probeNDFsMode % 6;
		if (probeNDFsMode == 0)
			std::cout << "Probe NDFs Mode Disabled" << std::endl;
		else
		{
			std::cout << "Probe NDFs Mode Enabled, WARNING: ONLY WORKS FOR 8*8 BINS" << std::endl;
			std::cout << "similarity metric used is: ";
			if (probeNDFsMode == 1)
			{
				std::cout << "User Selection Mode, Please select a region" << std::endl;
			}
			else if (probeNDFsMode == 2)
			{
				std::cout << "Minkowski-form distance, r=1 (L1 Norm)" << std::endl;
			}
			else if (probeNDFsMode == 3)
			{
				std::cout << "Minkowski-form distance, r=2 (L2 Norm)" << std::endl;
			}
			else if (0 == 4)
			{
				std::cout << "Histogram Intersection" << std::endl;
			}
			else if (probeNDFsMode == 5)
			{
				std::cout << "X^2 Statistics" << std::endl;
			}
			//else if (probeNDFsMode == 5)
			//{
			//	std::cout << "Jeffrey Divergence Distance" << std::endl;
			//}


			//reset limits
			resetSimLimits(false);
		}
		break;
	case 'C':
		resetSimLimits(true);
		computeAvgNDF(true);
		std::cout << "finished computing Average NDF for selected pixels" << std::endl;
		break;
	case '7':
		op0 = op1 = op2 = op3 = op4 = op5 = op6 = false;
		timings.clear();
		clear_NDF_Cache();
		tile_based_culling(false);
		tile_based_panning(false);
		update_page_texture();
		reset(); 
		display();
		break;
	case '9':
		maxSamplingRunsFactor*=2.0f;
		maxSamplingRuns = std::max(1.0*maxSamplingRunsFactor, pow(4, std::floor(current_lod - lodDelta))*maxSamplingRunsFactor);
		std::cout << "max sampling runs factor increased to: " << maxSamplingRunsFactor << std::endl;
		std::cout << "mas sampling runs is: " << maxSamplingRuns << std::endl;
		break;
	case 'q': case 'Q':

		percentage_of_cached_tiles.clear();
		if (plainRayCasting)
		{
			//reset progressive raycasting ssbo
			erase_progressive_raycasting_ssbo();
		}

		if (probeNDFsMode > 0)
		{
			resetSimLimits(true);
		}


		if (true)
		{
			apply_permenant_rotation(3, 0.0f);
		}
		else
		{
			cameraRotation.x -= camRotation;
		}

		//zoom_camPosi = camPosi;

		//phys_x = 0;
		//phys_y = 0;
		initialize();

		cameraDistance = std::max(LOD.min_cam_dist, std::min(cameraDistance, LOD.max_cam_dist - .3f));
		camPosi = CameraPosition(cameraRotation, cameraDistance);
		camPosi += cameraOffset;
		camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

		tile_based_culling(false);
		//tile_based_panning(false);
		update_page_texture();
		reset();
		display();
		//glutPostRedisplay();

		break;

	case 'e': case 'E':
		percentage_of_cached_tiles.clear();
		if (plainRayCasting)
		{
			//reset progressive raycasting ssbo
			erase_progressive_raycasting_ssbo();
		}

		if (probeNDFsMode > 0)
		{
			resetSimLimits(true);
		}



		if (true)
		{
			apply_permenant_rotation(-3, 0.0f);
		}
		else
		{
			cameraRotation.x -= camRotation;
		}

		//zoom_camPosi = camPosi;

		//phys_x = 0;
		//phys_y = 0;
		initialize();

		cameraDistance = std::max(LOD.min_cam_dist, std::min(cameraDistance, LOD.max_cam_dist - .3f));
		camPosi = CameraPosition(cameraRotation, cameraDistance);
		camPosi += cameraOffset;
		camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

		tile_based_culling(false);
		//tile_based_panning(false);
		update_page_texture();
		reset();
		display();
		//glutPostRedisplay();
		break;

	case '*':
		simPercentage += .01;
		simPercentage = std::min(simPercentage, 1.0f);
		std::cout << "similarity percentage is now " << simPercentage*100.0f << "%" << std::endl;

		break;
	case '/':
		simPercentage -= .01;
		simPercentage = std::max(simPercentage, 0.0f);
		std::cout << "similarity percentage is now " << simPercentage*100.0f << "%" << std::endl;

		break;

	case 't':
		visualize_tiles++;
		if (visualize_tiles > 2)
			visualize_tiles = -1;
		break; 
	case 'T':
		//text
		//1-> change light 1000 times and time
		{		
			//clear cache
			keyboard('7', 0, 0);
			//sample for 256 times
			for (int i = 0; i < 256; i++)
			{
				display();
			}

			//disable sampling
			timings.clear();
			cull = false;
			for (int i = 0; i < 1000; i++)
			{
				glFinish();
				w2.StartTimer();

				//mapToSphere(glm::vec2(rand()*windowSize.x, rand()*windowSize.y), LightDir);
				//LightDir.y = -LightDir.y;
				//LightDir.z = -LightDir.z;

				//preIntegrateBins();
				display();

				glFinish();
				w2.StopTimer();

				timings.push_back(w2.GetElapsedTime());
			}
			std::cout << "Time taken to relight: " << std::accumulate(timings.begin(), timings.end(), 0.0) / 1000.0 << std::endl;
			//enable sampling
			cull = true;
		}
		//2-zoom out by .2 1000 times (zoom in by .2 at end of each iteration)
		//{
		//	timings.clear();
		//	
		//	for (int i = 0; i < 1000; i++)
		//	{
		//		glFinish();
		//		w2.StartTimer();
		//		
		//		//zoom out
		//		//disable sampling
		//		cull = false;
		//		{
		//			prev_lod = current_lod;

		//			//debug
		//			//get cam distance that will take us to lower lod
		//			target_lod = std::min(10.0f, ceil(current_lod + 0.2f));
		//			cameraDistance = LOD.get_cam_dist(target_lod);
		//			//end debug

		//			cameraDistance = std::max(LOD.min_cam_dist, std::min(cameraDistance, LOD.max_cam_dist - .3f));

		//			camPosi = CameraPosition(cameraRotation, cameraDistance);
		//			camPosi += cameraOffset;
		//			camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

		//			zoom_camPosi = camPosi;

		//			tile_based_culling(false);
		//			tile_based_panning(false);

		//			update_page_texture();

		//			display();
		//		}
	
		//		glFinish();
		//		w2.StopTimer();

		//		timings.push_back(w2.GetElapsedTime());

		//		//zoom in
		//		//enable sampling
		//		cull = true;
		//		{
		//			prev_lod = current_lod;

		//			//debug
		//			//get cam distance that will take us to lower lod
		//			target_lod = std::max(0.0f, current_lod - 0.2f);
		//			cameraDistance = LOD.get_cam_dist(target_lod);
		//			//end debug

		//			cameraDistance = std::max(LOD.min_cam_dist, std::min(cameraDistance, LOD.max_cam_dist - .0001f));

		//			//cameraDistance = 2.0f;


		//			camPosi = CameraPosition(cameraRotation, cameraDistance);
		//			camPosi += cameraOffset;
		//			camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

		//			zoom_camPosi = camPosi;


		//			tile_based_culling(false);
		//			tile_based_panning(false);


		//			//debug
		//			//phys_x = phys_y = 0;
		//			//end debug

		//			update_page_texture();
		//			display();
		//		}
		//	}
		//	std::cout << "Time taken to zoom without crossing lod : " << std::accumulate(timings.begin(), timings.end(), 0.0) / 1000.0 << std::endl;
		//}

		//do the same tests for cached raycasting
		keyboard('u', 0, 0);
		//1-> change light 1000 times and time
		{
			timings.clear();
			for (int j = 0; j < 10; j++)
			{
				keyboard('7', 0, 0);
				glFinish();
				w2.StartTimer();
				for (int i = 0; i < 256; i++)
				{


					//preIntegrateBins();
					display();


				}
				glFinish();
				w2.StopTimer();

				timings.push_back(w2.GetElapsedTime());
			}

			std::cout << "Time taken to relight raycasting: " << std::accumulate(timings.begin(), timings.end(), 0.0) / 10.0 << std::endl;
			//enable sampling
			//cull = true;
		}
		//2-zoom out by .2 1000 times (zoom in by .2 at end of each iteration)
		//{
		//	timings.clear();

		//	for (int i = 0; i < 1000; i++)
		//	{
		//		glFinish();
		//		w2.StartTimer();

		//		//zoom out
		//		//disable sampling
		//		cull = false;
		//		{
		//			prev_lod = current_lod;

		//			//debug
		//			//get cam distance that will take us to lower lod
		//			target_lod = std::min(10.0f, ceil(current_lod + 0.2f));
		//			cameraDistance = LOD.get_cam_dist(target_lod);
		//			//end debug

		//			cameraDistance = std::max(LOD.min_cam_dist, std::min(cameraDistance, LOD.max_cam_dist - .3f));

		//			camPosi = CameraPosition(cameraRotation, cameraDistance);
		//			camPosi += cameraOffset;
		//			camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

		//			zoom_camPosi = camPosi;

		//			tile_based_culling(false);
		//			tile_based_panning(false);

		//			update_page_texture();

		//			display();
		//		}

		//		glFinish();
		//		w2.StopTimer();

		//		timings.push_back(w2.GetElapsedTime());

		//		//zoom in
		//		//enable sampling
		//		cull = true;
		//		{
		//			prev_lod = current_lod;

		//			//debug
		//			//get cam distance that will take us to lower lod
		//			target_lod = std::max(0.0f, current_lod - 0.2f);
		//			cameraDistance = LOD.get_cam_dist(target_lod);
		//			//end debug

		//			cameraDistance = std::max(LOD.min_cam_dist, std::min(cameraDistance, LOD.max_cam_dist - .0001f));

		//			//cameraDistance = 2.0f;


		//			camPosi = CameraPosition(cameraRotation, cameraDistance);
		//			camPosi += cameraOffset;
		//			camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

		//			zoom_camPosi = camPosi;


		//			tile_based_culling(false);
		//			tile_based_panning(false);


		//			//debug
		//			//phys_x = phys_y = 0;
		//			//end debug

		//			update_page_texture();
		//			display();
		//		}
		//	}
		//	std::cout << "Time taken to zoom without crossing lod cached raycasting: " << std::accumulate(timings.begin(), timings.end(), 0.0) / 1000.0 << std::endl;
		//}

		break;
	case 'z':

		tile_w /= 2.0f;
		tile_h /= 2.0f;
		std::cout << "tile size reduced to: " << tile_h << "^2" << std::endl;
		initialize();

		tile_based_culling(false);
		update_page_texture();
		reset();
		preIntegrateBins();
		//reshape(windowSize.x, windowSize.y);
		break;
	case 'Z':
		tile_w *= 2.0f;
		tile_h *= 2.0f;
		std::cout << "tile size increased to: " << tile_h << "^2" << std::endl;
		initialize();

		tile_based_culling(false);
		update_page_texture();
		reset();
		preIntegrateBins();
		//reshape(windowSize.x, windowSize.y);
		break;
	case '%':
		singleRay_NDF = !singleRay_NDF;
		if (singleRay_NDF)
		{
			std::cout << "Mode: single-sampled Ndf" << std::endl;
		}
		else
		{
			std::cout << "Mode: multi-sampled Ndf" << std::endl;
		}
		clear_NDF_Cache();
		reset();
		break;
	case '!':
		//saveSamplingTexture();
		save_SamplingTexture = true;
		break;
	case '(':
		binDiscretizations *= 2;
		binDiscretizations = std::min(int(binDiscretizations), maxSubBins);
		std::cout << "binnig discretizations increased to " << binDiscretizations << std::endl;
#if 0
		{
			//clear superPreintegratedbins ssb0
			float ssboClearColor = 0;
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, superPreIntegratedBins_ssbo);
			if (!glClearBufferData) {
				glClearBufferData = (PFNGLCLEARBUFFERDATAPROC)wglGetProcAddress("glClearBufferData");
			}
			assert(glClearBufferData);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, superPreIntegratedBins_ssbo);
			glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, &ssboClearColor);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

			//ssbo for super sampled pre-integrated bins
			{
				glGenBuffers(1, &superPreIntegratedBins_ssbo);

				const int MY_ARRAY_SIZE = binDiscretizations*binDiscretizations*histogramResolution.x*histogramResolution.y * 3;
				float* data = new float[MY_ARRAY_SIZE];

				for (int i = 0; i < MY_ARRAY_SIZE; i++)
					data[i] = 0;

				// Allocate storage for the UBO
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, superPreIntegratedBins_ssbo);
				glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof (GLfloat)* MY_ARRAY_SIZE, data, GL_DYNAMIC_COPY);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


				glBindBuffer(GL_SHADER_STORAGE_BUFFER, superPreIntegratedBins_ssbo);
				GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
				memcpy(p, &data, sizeof(data));
				glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);


				{
					unsigned int block_index = glGetProgramResourceIndex(preIntegrateBins_C->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "superPreIntegratedBins");
					GLuint binding_point_index = 0;
					glShaderStorageBlockBinding(preIntegrateBins_C->GetGlShaderProgramHandle(), block_index, binding_point_index);
					glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, superPreIntegratedBins_ssbo);
				}

				delete[]data;
			}
		}
#endif

		preIntegrateBins();
		break;
	case ')':
		binDiscretizations /= 2;
		binDiscretizations = std::max(1.0f, binDiscretizations);
		std::cout << "binnig discretizations decreased to " << binDiscretizations << std::endl;
#if 0
		{
			//clear superPreintegratedbins ssb0
			float ssboClearColor = 0;
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, superPreIntegratedBins_ssbo);
			if (!glClearBufferData) {
				glClearBufferData = (PFNGLCLEARBUFFERDATAPROC)wglGetProcAddress("glClearBufferData");
			}
			assert(glClearBufferData);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, superPreIntegratedBins_ssbo);
			glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, &ssboClearColor);
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

			//ssbo for super sampled pre-integrated bins
			{
				glGenBuffers(1, &superPreIntegratedBins_ssbo);

				const int MY_ARRAY_SIZE = binDiscretizations*binDiscretizations*histogramResolution.x*histogramResolution.y * 3;
				float* data = new float[MY_ARRAY_SIZE];

				for (int i = 0; i < MY_ARRAY_SIZE; i++)
					data[i] = 0;

				// Allocate storage for the UBO
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, superPreIntegratedBins_ssbo);
				glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof (GLfloat)* MY_ARRAY_SIZE, data, GL_DYNAMIC_COPY);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);


				glBindBuffer(GL_SHADER_STORAGE_BUFFER, superPreIntegratedBins_ssbo);
				GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
				memcpy(p, &data, sizeof(data));
				glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);


				{
					unsigned int block_index = glGetProgramResourceIndex(preIntegrateBins_C->GetGlShaderProgramHandle(), GL_SHADER_STORAGE_BLOCK, "superPreIntegratedBins");
					GLuint binding_point_index = 0;
					glShaderStorageBlockBinding(preIntegrateBins_C->GetGlShaderProgramHandle(), block_index, binding_point_index);
					glBindBufferBase(GL_SHADER_STORAGE_BUFFER, binding_point_index, superPreIntegratedBins_ssbo);
				}

				delete[]data;
			}
		}
#endif

		preIntegrateBins();
		break;
	case'^':
		specularExponent *= 2;
		specularExponent = std::min(2048.f, specularExponent);
		std::cout << "specular exponent increased to " << specularExponent << std::endl;
		if (cachedRayCasting || plainRayCasting)
		{
			initialize();
			//in both cases we cull and update tiles
			tile_based_culling(false);
			//tile_based_panning(false);
			update_page_texture();
			reset();
		}
		else
		{
			preIntegrateBins();
		}
		break;
	case '&':
		specularExponent /= 2;
		specularExponent = std::max(1.f, specularExponent);
		std::cout << "specular exponent decreased to " << specularExponent << std::endl;
		if (cachedRayCasting || plainRayCasting)
		{
			initialize();
			//in both cases we cull and update tiles
			tile_based_culling(false);
			//tile_based_panning(false);
			update_page_texture();
			reset();
		}
		else
		{
			preIntegrateBins();
		}
		break;
	case 'w': case 'W':
		//camPosi.z -= camOff;
		//cameraDistance -= camOff;
		//cameraDistance -= cameraDistance * 0.5f;//0.125f;
		cameraDistance -= 0.125f;

		prev_lod = current_lod;

		//debug
		//get cam distance that will take us to lower lod
		target_lod = std::max(0.0f, current_lod - 1);
		cameraDistance = LOD.get_cam_dist(target_lod);
		//end debug

		cameraDistance = std::max(LOD.min_cam_dist, std::min(cameraDistance, LOD.max_cam_dist - .0001f));

		//cameraDistance = 2.0f;


		camPosi = CameraPosition(cameraRotation, cameraDistance);
		camPosi += cameraOffset;
		camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

		zoom_camPosi = camPosi;


		tile_based_culling(false);
		tile_based_panning(false);


		//debug
		//phys_x = phys_y = 0;
		//end debug

		update_page_texture();

		//if (int(prev_lod)- int(current_lod)!=0)
		reset();
		glutPostRedisplay();
		break;

	case 'a': case 'A':
		percentage_of_cached_tiles.clear();
		if (plainRayCasting)
		{
			//reset progressive raycasting ssbo
			erase_progressive_raycasting_ssbo();
		}

		if (probeNDFsMode > 0)
		{
			resetSimLimits(true);
		}



		if (true)
		{
			apply_permenant_rotation(0, -3.0f);
		}
		else
		{
			cameraRotation.x -= camRotation;
		}

		//zoom_camPosi = camPosi;

		//phys_x = 0;
		//phys_y = 0;
		initialize();

		cameraDistance = std::max(LOD.min_cam_dist, std::min(cameraDistance, LOD.max_cam_dist - .3f));
		camPosi = CameraPosition(cameraRotation, cameraDistance);
		camPosi += cameraOffset;
		camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

		tile_based_culling(false);
		//tile_based_panning(false);
		update_page_texture();
		reset();
		display();
		//glutPostRedisplay();
		break;
	case 'S':
		streaming = !streaming;
		if (streaming)
		{
			std::cout << "streaming enabled!" << std::endl;
		}
		else
		{
			std::cout << "streaming disabled!" << std::endl;
		}
		keyboard('7', 0, 0);
		break;
	case 's': 
		//camPosi.z += camOff;
		//cameraDistance += camOff;
		//cameraDistance += cameraDistance * 0.5f;//0.125f;
		cameraDistance += 0.125f;

		prev_lod = current_lod;

		//debug
		//get cam distance that will take us to lower lod
		target_lod = std::min(10.0f, ceil(current_lod + 1));
		cameraDistance = LOD.get_cam_dist(target_lod);
		//end debug

		cameraDistance = std::max(LOD.min_cam_dist, std::min(cameraDistance, LOD.max_cam_dist - .3f));

		camPosi = CameraPosition(cameraRotation, cameraDistance);
		camPosi += cameraOffset;
		camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

		zoom_camPosi = camPosi;

		tile_based_culling(false);
		tile_based_panning(false);

		update_page_texture();
		//if (int(prev_lod) - int(current_lod) != 0)
		reset();
		glutPostRedisplay();
		break;

	case 'd': 
		percentage_of_cached_tiles.clear();
		if (plainRayCasting)
		{
			//reset progressive raycasting ssbo
			erase_progressive_raycasting_ssbo();
		}

		if (probeNDFsMode > 0)
		{
			resetSimLimits(true);
		}



		if (true)
		{
			apply_permenant_rotation(0, 3.0f);
		}
		else
		{
			cameraRotation.x -= camRotation;
		}

		//zoom_camPosi = camPosi;

		//phys_x = 0;
		//phys_y = 0;
		initialize();

		cameraDistance = std::max(LOD.min_cam_dist, std::min(cameraDistance, LOD.max_cam_dist - .3f));
		camPosi = CameraPosition(cameraRotation, cameraDistance);
		camPosi += cameraOffset;
		camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

		tile_based_culling(false);
		//tile_based_panning(false);
		update_page_texture();
		reset();
		display();
		//glutPostRedisplay();
		break;
	case 'D':
		lod_increment++;
		lod_increment = lod_increment % (int(LOD.get_lod(LOD.max_cam_dist)) - int(current_lod));
		
		downsampling_lod = current_lod + lod_increment;

		std::cout <<"current lod is"<<current_lod<<", downsampling lod is: "<<downsampling_lod << std::endl;
		break;
	case 'r': case 'R':
		// recompile shaders
		compileShaders(shaderPath);
		bindSSbos();
		break;

	case 'y': case 'Y':
		switchToRaycasting = !switchToRaycasting;
		if (switchToRaycasting)
		{
			std::cout << "Raycasting when zoomed in enabled" << std::endl;
			if (!cachedRayCasting)
				keyboard('u', 0, 0);
		}
		else
		{
			std::cout << "Raycasting when zoomed in disabled" << std::endl;
			if (cachedRayCasting)
				keyboard('u', 0, 0);
		}
		break;

	case 'h': case 'H':
		samplesPerRun = std::max(1, samplesPerRun - samplesPerRunStepSize);
		reset();
		break;


	case '+':
#ifdef DOWNSAMPLE_NDF
	{
				// stop sampling
				samplingRunIndex = maxSamplingRuns;

				// downsample current downsampled ssbo
				auto factor = static_cast<int>(std::pow(2, std::max(0, downsamplingFactor - 1)));
				auto currentResolution = glm::ivec2(windowSize.x / factor, windowSize.y / factor);
				auto ssboElements = size_t(currentResolution.x * currentResolution.y * histogramResolution.x * histogramResolution.y);
				auto ssboData = std::vector<float>(ssboElements);
				auto ssbo = ndfTree.GetLevels().front().GetShaderStorageBufferOject();
				{
					// download NTF
					glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
					auto readMap = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
					// FIXME: memcpy is unsafe
					memcpy(reinterpret_cast<char*>(&ssboData[0]), readMap, ssboData.size() * sizeof(*ssboData.begin()));
					glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
					glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
				}

				// divide rendering size by 2^downsamplingFactor but maintain window size
				++downsamplingFactor;

				auto downsampledElements = ssboElements / 4;
				auto downsampledData = std::vector<float>(downsampledElements);

				std::cout << "Downsampling " << ssboElements << " to " << downsampledElements << " factor of " << static_cast<float>(ssboElements) / static_cast<float>(downsampledElements) << std::endl;

				// downsample
				{
					auto targetResolution = currentResolution / 2;
					for (size_t y = 0; y < targetResolution.y; ++y) {
						for (size_t x = 0; x < targetResolution.x; ++x) {
							auto targetHistogramIndex = y * targetResolution.x + x;
							auto targetOffset = targetHistogramIndex * histogramResolution.x * histogramResolution.y;

							for (int histY = 0; histY < histogramResolution.y; ++histY) {
								for (int histX = 0; histX < histogramResolution.x; ++histX) {
									auto binIndex = size_t(histY * histogramResolution.x + histX);
									downsampledData[targetOffset + binIndex] = 0.0f;

									for (int multiY = 0; multiY < 2; ++multiY) {
										for (int multiX = 0; multiX < 2; ++multiX) {
											auto sourceHistogramIndex = (y * 2 + multiY) * currentResolution.x + x * 2 + multiX;
											auto sourceOffset = sourceHistogramIndex * histogramResolution.x * histogramResolution.y;

											downsampledData[targetOffset + binIndex] += ssboData[sourceOffset + binIndex];
										}
									}

									downsampledData[targetOffset + binIndex] *= 0.25f;

									// FIXME: fix energy scaling. Sampling is stopped so current energy is used. Should be scaled accordingly.
									if (downsamplingFactor == 2) {
										downsampledData[targetOffset + binIndex] *= 293.5f * 0.4f * 2.9f;
										// NOTE: takes many samples to be a good approximation
										ndfIntensityCorrection = 0.96f * 0.0375f;
									}
								}
							}
						}
					}
				}

				// upload downsampled version
				{
					auto newSize = downsampledData.size() * sizeof(*downsampledData.begin());
					std::cout << "New ssbo size: " << static_cast<float>(newSize) / (1024.0f * 1024.0f) << " mb" << std::endl;

					glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
					glBufferData(GL_SHADER_STORAGE_BUFFER, newSize, downsampledData.data(), GL_DYNAMIC_COPY);
					glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
				}
	}
#else
#ifdef READ_FROM_FILE
		if (timeDependent) {
			frameIntervalMs = std::max(1, frameIntervalMs - frameIntervalMsStep);
		}
		else {
			timeStep = std::min(timeStep + 1, timeStepUpperLimit);

			sparseENtfFilePath = fileFolder + fileSparseSubFolder + filePrefix + std::to_string(timeStep) + fileSuffix + ".entf";

			loadSparseENtf(sparseENtfFilePath);
		}
#else
		particleScale *= 1.1f;
		std::cout << "scale is " << particleScale << std::endl;
		clear_NDF_Cache();
		//initialize();

		tile_based_culling(false);
		tile_based_panning(false);
		update_page_texture();
		reset();
		display();
#endif // READ_FROM_FILE
#endif
		break;

	case '-':
#ifdef DOWNSAMPLE_NDF
		downsamplingFactor = std::max(1, downsamplingFactor - 1);
		reset();
#else
#ifdef READ_FROM_FILE
		if (timeDependent) {
			frameIntervalMs = std::max(1, frameIntervalMs + frameIntervalMsStep);
		}
		else {
			timeStep = std::max(timeStep - 1, timeStepLowerLimit);

			sparseENtfFilePath = fileFolder + fileSparseSubFolder + filePrefix + std::to_string(timeStep) + fileSuffix + ".entf";

			loadSparseENtf(sparseENtfFilePath);
		}
#else
		particleScale *= .9f;
		std::cout << "scale is " << particleScale << std::endl;
		clear_NDF_Cache();
		//initialize();

		tile_based_culling(false);
		tile_based_panning(false);
		update_page_texture();
		reset();
		display();
#endif // READ_FROM_FILE
#endif		
		break;

	case 'o':
		
		//maxSamplingRuns = std::max(1, maxSamplingRuns / 2);
		//std::cout << "Max sampling runs " << maxSamplingRuns << std::endl;
		//reset();
		GlobalRotationMatrix = glm::mat3();
		initialize();

		cameraDistance = std::min(LOD.get_cam_dist(LOD.initial_lod),LOD.max_cam_dist-.3f);//std::max(LOD.min_cam_dist, std::min(cameraDistance, LOD.max_cam_dist - .3f));
		camPosi = CameraPosition(cameraRotation, cameraDistance);
		camPosi += cameraOffset;
		camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

		tile_based_culling(false);
		//tile_based_panning(false);
		update_page_texture();
		reset();
		display();

		break;
	case'O':
		enableOcclusionCulluing = !enableOcclusionCulluing;
		if (enableOcclusionCulluing)
		{
			std::cout << "occlusion Culling enabled" << std::endl;
		}
		else
		{
			std::cout << "occlusion Culling disabled" << std::endl;
		}
			//buildHOM();

			//cellsTbe.clear();
			//cellsTbr.clear();
			//
			//for (int i = 0; i < cellsTbt.size(); i++)
			//	cellsTbt.pop();

			////start by testing the root node
			//cellsTbt.push(0);
			//tbts = cellsTbt.size();
			//
			//for (int i = 0; i < tbts; i++)
			//{
			//	testAgainstHOM(cellsTbt.front(), cellsTbe,cellsTbr);
			//	cellsTbt.pop();
			//}
			//	

			////cell is not visible and should be removed
			//for (int j = 0; j < cellsTbe.size(); j++)
			//{
			//	global_tree.getLeavesWithCommonAncestor(cellsTbe[j], l);
			//	for (int k = 0; k < l.size(); k++)
			//	{
			//		for (int i = 0; i < cellsToRender.size(); i++)
			//		{
			//			if (cellsToRender[i] == l[k])
			//			{
			//				//std::cout << "cell to be erased center is: " << global_tree.nodes[cellsTbe[j]].center.x << ", " << global_tree.nodes[cellsTbe[j]].center.y << ", " << global_tree.nodes[cellsTbe[j]].center.z << std::endl;
			//				cellsToRender.erase(cellsToRender.begin() + i);
			//				break;
			//			}
			//		}
			//	}
			//}

			//std::cout << "Cells reduced to: " << cellsToRender.size() << ", originally were: " << global_tree.leaves.size() << std::endl;
			//std::cout << "sample number: " << lowestSampleCount << std::endl;

		break;
	case 'p': 
		//maxSamplingRuns *= 2;
		//std::cout << "Max sampling runs " << maxSamplingRuns << std::endl;
		//reset();
		//for (int i = 0; i < 3; i++)
		//{
		//	binning_mode = i;
		//	clear_NDF_Cache();
		//	tile_based_culling(false);
		//	tile_based_panning(false);
		//	update_page_texture();
		//	computeBinAreas();
		//	preIntegrateBins();
		//	reset();
		//	display();

		//	for (int j = 0; j < 64; j++)
		//	{
		//		binLimit = j;
				printScreen = true;
				display();
				printScreen = false;
		//	}
		//}

		break;
	case 'P':
		brick++;
		brick = brick%30;
		keyboard('7',0,0);
		std::cout << "Now rendering brick number: " << brick << std::endl;
		break;
	case 'g': case 'G':
		cameraOffset.x += camOff;

		camPosi = CameraPosition(cameraRotation, cameraDistance);
		camPosi += cameraOffset;
		camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

		tile_based_culling(false);//tile_based_culling(false);
		update_page_texture();
		reset();
		glutPostRedisplay();
		break;

	case 'f': case 'F':
		cameraOffset.x -= camOff;

		camPosi = CameraPosition(cameraRotation, cameraDistance);
		camPosi += cameraOffset;
		camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

		tile_based_culling(false); //tile_based_culling(false);
		update_page_texture();
		reset();
		glutPostRedisplay();
		break;

	case 'm':
		renderMode = (renderMode + 1) % 3;
		if (cachedRayCasting || plainRayCasting)
		{
			initialize();
			//in both cases we cull and update tiles
			tile_based_culling(false);
			//tile_based_panning(false);
			update_page_texture();
			reset();
		}
		else
		{
			preIntegrateBins();
		}
		break;

	case'M':
		renderBarMode = !renderBarMode;
		break;

	case 'n': 
		
		if (renderTransfer) {
			renderTransfer = false;
		}
		else {
			renderTransfer = true;
		}
		break;

	case 'N':
		ndfOverlayMode = !ndfOverlayMode;

		if (ndfOverlayMode)
		{
			//downsampling_lod = std::min(current_lod + 4, LOD.get_lod(LOD.max_cam_dist));
			//clear ndf overlay
			{
				float ndfOverlayClearColor = 1;
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, ndfColors_ssbo);
				if (!glClearBufferData) {
					glClearBufferData = (PFNGLCLEARBUFFERDATAPROC)wglGetProcAddress("glClearBufferData");
				}
				assert(glClearBufferData);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, ndfColors_ssbo);
				glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, &ndfOverlayClearColor);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
			}
			overlayNDFs();
			std::cout << "finished computing ndf overlay!" << std::endl;
		}
		else
		{
			std::cout << "disabled computing ndf overlay!" << std::endl;
		}
		break;

	case 'v': 
		//if (!fullScreen) {
		//	glutFullScreen();
		//	fullScreen = true;
		//}
		//else {
		//	glutPositionWindow(0, 0);
		//	glutReshapeWindow(INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT);
		//	fullScreen = false;
		//}
		circularPattern = !circularPattern;
		if (circularPattern)
		{
			std::cout << "using circular pattern" << std::endl;
			//sample_count = circular_pattern_sample_count;
			circularPatternIterator = 0;
		}
		else
		{
			std::cout << "using square pattern" << std::endl;
			sample_count = square_pattern_sample_count;
		}

		clear_NDF_Cache();
		tile_based_culling(false);
		tile_based_panning(false);
		update_page_texture();
		reset();
		display();
		break;
	case 'V':
		ndfExplorerMode = !ndfExplorerMode;
		if (ndfExplorerMode)
		{
			std::cout << "Turned on NDF explorer mode" << std::endl;
			renderNdfExplorer();
			preIntegrateBins();
		}
		else
		{
			std::cout << "Turned off NDF explorer mode" << std::endl;
		}
		break;

	case 'l':
		/*if (!paused) {
		paused = true;
		}
		else {
		paused = false;
		}*/
		//binLimit++;
		//binLimit = int(binLimit) % (histogramResolution.x*histogramResolution.y);
		//std::cout << "bin limit is: " << binLimit << std::endl;
		//break;
		LightDir.x -= 1;
		glm::normalize(LightDir);
		preIntegrateBins();
		display();
		break;
	case 'L':
		LightDir.x += 1;
		glm::normalize(LightDir);
		preIntegrateBins();
		display();
		break;
	case 'j': case 'J':
		activeTransferFunctionIndex = (activeTransferFunctionIndex + 1) % transferFunctions.size();
		activeTransferFunction = &transferFunctions[activeTransferFunctionIndex];
		if (cachedRayCasting || plainRayCasting)
		{
			initialize();
			//in both cases we cull and update tiles
			tile_based_culling(false);
			//tile_based_panning(false);
			update_page_texture();
			reset();
		}
		else
		{
			preIntegrateBins();
		}
		break;

	case 'i': 
		drawNDF();
		break;
	case 'I':
		maxSamplingRunsFactor++;
		std::cout << "max sampling runs factor increased to: " << maxSamplingRunsFactor << std::endl;
		break;

	case 'k': 
		filterRadius *= .9;
		std::cout << "filter radius decreased to: " << filterRadius << std::endl;
		circularPatternIterator = 0;
		createFilter(filterRadius, gkStdv);
		clear_NDF_Cache();
		tile_based_culling(false);
		tile_based_panning(false);
		update_page_texture();
		reset();
		display();
		break;
	case 'K':
		filterRadius *= 1.1;
		std::cout << "filter radius increased to: " << filterRadius << std::endl;
		circularPatternIterator = 0;
		createFilter(filterRadius, gkStdv);
		clear_NDF_Cache();
		tile_based_culling(false);
		tile_based_panning(false);
		update_page_texture();
		reset();
		display();
		break;
	case'<':
		gkStdv *= 1.1;
		std::cout << "gaussian filter standard deviation increased to: " << gkStdv << std::endl;
		createFilter(filterRadius, gkStdv);
		circularPatternIterator = 0;

		clear_NDF_Cache();
		tile_based_culling(false);
		tile_based_panning(false);
		update_page_texture();
		reset();
		display();
		break;
	case'>':
		gkStdv *= .9;
		std::cout << "gaussian filter standard deviation decreased to: " << gkStdv << std::endl;
		createFilter(filterRadius, gkStdv);
		circularPatternIterator = 0;

		clear_NDF_Cache();
		tile_based_culling(false);
		tile_based_panning(false);
		update_page_texture();
		reset();
		display();
		break;
	default:
		//onceFlag = true;
		break;
	}
}
void clear_NDF_Cache()
{
	//clear NDF_Cache
	float ssboClearColor = 0;
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ndfTree.GetLevels().front().GetShaderStorageBufferOject());
	if (!glClearBufferData) {
		glClearBufferData = (PFNGLCLEARBUFFERDATAPROC)wglGetProcAddress("glClearBufferData");
	}
	assert(glClearBufferData);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, ndfTree.GetLevels().front().GetShaderStorageBufferOject());
	glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, &ssboClearColor);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	//clear sample count
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, SampleCount_ssbo);
	if (!glClearBufferData) {
		glClearBufferData = (PFNGLCLEARBUFFERDATAPROC)wglGetProcAddress("glClearBufferData");
	}
	assert(glClearBufferData);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, SampleCount_ssbo);
	glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, &ssboClearColor);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	//clear ndf overlay
	if (ndfOverlayMode)
	{
		float ndfOverlayClearColor = 1;
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ndfColors_ssbo);
		if (!glClearBufferData) {
			glClearBufferData = (PFNGLCLEARBUFFERDATAPROC)wglGetProcAddress("glClearBufferData");
		}
		assert(glClearBufferData);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ndfColors_ssbo);
		glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, &ndfOverlayClearColor);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

#ifdef CORE_FUNCTIONALITY_ONLY
#else
	//clear circular pattern sample count
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, circularPatternSampleCount_ssbo);
	if (!glClearBufferData) {
		glClearBufferData = (PFNGLCLEARBUFFERDATAPROC)wglGetProcAddress("glClearBufferData");
	}
	assert(glClearBufferData);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, circularPatternSampleCount_ssbo);
	glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, &ssboClearColor);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
#endif
	

	////clear color region
	//float regionSsboClearColor = 1;
	//glBindBuffer(GL_SHADER_STORAGE_BUFFER, region_ssbo);
	//if (!glClearBufferData) {
	//	glClearBufferData = (PFNGLCLEARBUFFERDATAPROC)wglGetProcAddress("glClearBufferData");
	//}
	//assert(glClearBufferData);
	//glBindBuffer(GL_SHADER_STORAGE_BUFFER, region_ssbo);
	//glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, &regionSsboClearColor);
	//glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);

	//4-> clear occupied array
	//
	occupied.clear();
	for (int i = 0; i < pow(phys_tex_dim / tile_w, 2); i++)
		occupied.push_back(std::make_pair(std::make_pair(false, glm::vec2(0, 0)), 0));
	//
	//5-> initialize page texture
	//
	glDeleteTextures(1, &Page_Texture);
	Page_Texture = 0;
	glGenTextures(1, &Page_Texture);
	glBindTexture(GL_TEXTURE_2D, Page_Texture);

	Page_Texture_Datatype* temp_page = new Page_Texture_Datatype[int(std::ceil(highest_w_res / float(tile_w))*std::ceil(highest_h_res / float(tile_h)) * 4)];

	for (int i = 0; i < std::ceil(highest_w_res / float(tile_w))*std::ceil(highest_h_res / float(tile_h)) * 4; i++)
		temp_page[i] = -1;




	glTexImage2D(GL_TEXTURE_2D, 0, Page_texture_internalFormat, std::ceil(highest_w_res / float(tile_w)), std::ceil(highest_h_res / float(tile_h)), 0, Page_Texture_format, Page_Texture_type, temp_page);
	glGenerateMipmap(GL_TEXTURE_2D);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);


	glBindTexture(GL_TEXTURE_2D, 0);
	delete[] temp_page;
}
void apply_permenant_rotation(float angle, bool axis)
{


	//if (axis)
	//{
	//	rotate_data(angle,0);
	//	return;
	//}
	//else
	//{
	//	rotate_data(0, angle);
	//	return;
	//}


	////here we'll transform the whole dataset permenately
	//glm::mat3x3 R;

	//angle = toRadiants*angle;


	//if (axis == 0)
	//{
	//	R = glm::mat3x3(1, 0, 0, 0, cos(angle), sin(angle), 0, -sin(angle), cos(angle));
	//}
	//else
	//{
	//	R = glm::mat3x3( cos(angle),0, -sin(angle), 0,1,0, sin(angle),0, cos(angle));
	//}


	//for (int i = 0; i < particleCenters.size(); i++)
	//{
	//	particleCenters[i] = R*particleCenters[i];
	//}

	//Helpers::Gl::CreateGlMeshFromBuffers(particleCenters, particlesGlBuffer);
}

void apply_permenant_rotation(float anglex, float angley)
{
	rotate_data(anglex, angley);
	//std::cout << anglex << ", " << angley << std::endl;
	return;

	//here we'll transform the whole dataset permenately

	//glm::mat4x4 Rx, Ry;


	//anglex = toRadiants*anglex;
	//angley = toRadiants*angley;



	//Rx = glm::mat4x4(1, 0, 0, 0, 0, cos(anglex), sin(anglex), 0, 0, -sin(anglex), cos(anglex), 0, 0, 0, 0, 1);
	//Ry = glm::mat4x4(cos(angley), 0, -sin(angley), 0, 0, 1, 0, 0, sin(angley), 0, cos(angley), 0, 0, 0, 0, 1);



	//for (int i = 0; i < particleCenters.size(); i++)
	//{
	//	particleCenters[i] = Ry*Rx*particleCenters[i];
	//}

    //   Helpers::Gl::MakeBuffer(particleCenters, particlesGlBuffer);
}

bool transferFunctionPressed(glm::ivec2 mousePosition, glm::vec2 &relativeMouse) 
{
	                                           //glViewport(windowSize.x - windowSize.y / 4, windowSize.y - windowSize.y / 4, windowSize.y / 6, windowSize.y / 6);
	relativeMouse.x = static_cast<float>(mousePosition.x - (windowSize.x - windowSize.y / 4)) / static_cast<float>(windowSize.y / 6);
	relativeMouse.y = 1.0f - static_cast<float>(windowSize.y - mousePosition.y - (windowSize.y - windowSize.y / 4)) / static_cast<float>(windowSize.y / 6);

	if (relativeMouse.x <= 0.0f || relativeMouse.x >= 1.0f) {
		return false;
	}
	if (relativeMouse.y <= 0.0f || relativeMouse.y >= 1.0f) {
		return false;
	}

	return true;
}
bool ndfPressed(glm::ivec2 mousePosition, glm::vec2 &relativeMouse) 
{
	if (probeNDFsMode > 0)
	{
		//glViewport(windowSize.x - windowSize.y / 4, windowSize.y - windowSize.y / 4                        , windowSize.y / 6, windowSize.y / 6);
		//glViewport(windowSize.x - windowSize.y / 4, windowSize.y - windowSize.y / 4 + -4 * windowSize.y / 6, windowSize.y / 6, windowSize.y / 6);
		relativeMouse.x = static_cast<float>(mousePosition.x - (windowSize.x - windowSize.y / 4)) / static_cast<float>(windowSize.y / 6);
		relativeMouse.y = 1.0f - static_cast<float>(windowSize.y - mousePosition.y - (windowSize.y - windowSize.y / 4-4*windowSize.y/6)) / static_cast<float>(windowSize.y / 6);

		if (relativeMouse.x <= 0.0f || relativeMouse.x >= 1.0f) 
		{
			return false;
		}
		if (relativeMouse.y <= 0.0f || relativeMouse.y >= 1.0f) 
		{
			return false;
		}

		std::cout << "ndf pressed" <<relativeMouse.x<<", "<<relativeMouse.y<< std::endl;
		return true;
	}
	return false;
}
glm::mat3 rotationMatrix(glm::vec3 axis, float angle)
{
	axis = glm::normalize(axis);
	float s = sin(angle);
	float c = cos(angle);
	float oc = 1.0f - c;

	return  glm::mat3(oc * axis.x * axis.x + c, oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s,
		oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c, oc * axis.y * axis.z - axis.x * s,
		oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c);
}
glm::vec3 blinnPhong(glm::vec3 normal, glm::vec3 light, glm::vec3 view, glm::vec3 diffuseColor, glm::vec3 specularColor, float specularExponent)
{
	glm::vec3 halfVector = glm::normalize(light + view);

	float diffuseIntensity = 1.0f * std::max(0.0f, -glm::dot(normal, -light));
	float diffuseWeight = 1.0f;

	//float diffuseIntensity = max(0.0f, abs(dot(normal, light)));
	//float diffuseIntensity = max(0.0f, dot(normal, -light));
	//float specularIntensity = max(0.0f, pow(dot(normal, halfVector), specularExponent));

	float nDotHalf = abs(glm::dot(normal, halfVector));
	float specularIntensity = 1.0f * std::max(0.0f, pow(nDotHalf, specularExponent));
	float specularWeight = 1.0f;
	//specularIntensity *= 0.0f;
	//diffuseIntensity *= 0.0f;

	const float ambientWeight = 0.0f;//0.25f;
	const glm::vec3 ambientColor = glm::vec3(1.0f, 0.0f, 0.0f);

	//return specularIntensity * specularColor;
	//return specularIntensity * vec3(0.0f, 0.5f, 0.0f) + min(1.0f, (1.0f - specularIntensity)) * vec3(0.5f, 0.0f, 0.0f);
	//return specularIntensity * specularColor + ambientIntensity * ambientColor;
	return (diffuseIntensity * diffuseColor * diffuseWeight + specularIntensity * specularColor * specularWeight + ambientWeight * ambientColor) / (diffuseWeight + specularWeight + ambientWeight);
	//return vec3(1.0f, 1.0f, 1.0f) - 0.25f * (diffuseIntensity * diffuseColor - specularIntensity * specularColor + ambientIntensity * ambientColor);
}
inline void preIntegrateBins_GPU()
{
	//call a compute shader with bindiscretization*bindiscretization*number of bins threads to fill a buffer of the same size.
	{
		GLuint ssboBindingPointIndex_superPreIntegratedBins = 0;

		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, preIntegrateBins_C->getSsboBiningIndex(superPreIntegratedBins_ssbo), superPreIntegratedBins_ssbo);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, preIntegrateBins_C->getSsboBiningIndex(simpleBinningAreas_ssbo), simpleBinningAreas_ssbo);

		//use shader
		auto Handle = preIntegrateBins_C->GetGlShaderProgramHandle();
		glUseProgram(Handle);

		glUniform2i(glGetUniformLocation(Handle, "histogramDiscretizations"), ndfTree.GetHistogramResolution().x, ndfTree.GetHistogramResolution().y);
		glUniform1i(glGetUniformLocation(Handle, "binDiscretizations"), binDiscretizations);
		glUniform1i(glGetUniformLocation(Handle, "binningMode"), binning_mode);
		glUniform1i(glGetUniformLocation(Handle, "renderMode"), renderMode);
		glUniform1f(glGetUniformLocation(Handle, "specularExp"), specularExponent);

		//for first binning mode, we need to send an index into the simple areas ssbo
		{
			int maxlevels = std::log2(maxSubBins);
			int curlevel = maxlevels - std::log2(binDiscretizations);
			int indx = 0;
			for (int i = 0; i < projectedNormalBinningAreas.size() && i < curlevel; i++)
			{
				indx += projectedNormalBinningAreas[i].size() * 3;
			}

			glUniform1i(glGetUniformLocation(Handle, "areaIndx"), indx);
		}

		auto viewMatrix = glm::lookAt(camPosi, camTarget, camUp);
		auto viewSpaceLightDir = viewMatrix * glm::vec4(LightDir, 0.0f);
		viewSpaceLightDir = glm::normalize(viewSpaceLightDir);

		glUniform3fv(glGetUniformLocation(Handle, "viewSpaceLightDir"), 1, &viewSpaceLightDir[0]);

		

		for (auto &transferFunction : transferFunctions)
		{
			glBindTexture(GL_TEXTURE_2D, activeTransferFunction->transferFunctionTexture.Texture);// activeTransferFunction->transferFunctionTexture.Texture);
			glUniform1i(glGetUniformLocation(Handle, "normalTransferSampler"), 0);
			break;
		}
		glBindImageTexture(0, ndfExplorer.RenderTexture, 0, false, 0, GL_READ_ONLY, GL_RGBA32F);
		glUniform1i(glGetUniformLocation(Handle, "tex"), 0);
		//glBindTexture(GL_TEXTURE_2D, activeChromeTexture->transferFunctionTexture.Texture);
		//glUniform1i(glGetUniformLocation(Handle, "chromeTexture"), 0);

		glDispatchCompute(histogramResolution.x *histogramResolution.y, binDiscretizations*binDiscretizations, 1);

		glUseProgram(0);
		{
			auto glErr = glGetError();
			if (glErr != GL_NO_ERROR) {
				std::cout << "glError: " << __FILE__ << " " << __LINE__ << std::endl;
			}
		}
	}

	//glFinish();

	//call a compute shader with number of bins threads to sum up the bin discretizations in each bin and put them in the bin.
#if 1
	{
	std::vector<float> data;
	auto dSize = size_t(histogramResolution.x * histogramResolution.y * 3);
	data.resize(dSize, 0.0f);

	std::vector<float> sdata;
	auto sdSize = size_t(histogramResolution.x * histogramResolution.y*binDiscretizations*binDiscretizations * 3);
	sdata.resize(sdSize, 0.0f);

	auto ssbo = superPreIntegratedBins_ssbo;
	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
		auto readMap = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_READ_ONLY);
		memcpy(reinterpret_cast<char*>(&sdata[0]), readMap, sdata.size() * sizeof(*sdata.begin()));

		glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}

	for (int i = 0; i < histogramResolution.x*histogramResolution.y; i++)
	{
		data[i * 3 + 0] = 0;
		data[i * 3 + 1] = 0;
		data[i * 3 + 2] = 0;

		for (int j = 0; j < binDiscretizations*binDiscretizations; j++)
		{
			data[i * 3 + 0] += sdata[(i*binDiscretizations*binDiscretizations * 3) + j * 3 + 0];
			data[i * 3 + 1] += sdata[(i*binDiscretizations*binDiscretizations * 3) + j * 3 + 1];
			data[i * 3 + 2] += sdata[(i*binDiscretizations*binDiscretizations * 3) + j * 3 + 2];
		}
	}

	ssbo = preIntegratedBins_ssbo;
	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, dSize* sizeof(float), data.data(), GL_DYNAMIC_COPY);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}
}
#endif
#if 0
	{
		glMemoryBarrier(GL_BUFFER_UPDATE_BARRIER_BIT);  //to make sure the previous shader finished writting in the ssbo we read from
		//GLuint ssboBindingPointIndex_superPreIntegratedBins = 0;

		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, sumPreIntegratedBins_C->getSsboBiningIndex(superPreIntegratedBins_ssbo), superPreIntegratedBins_ssbo);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, sumPreIntegratedBins_C->getSsboBiningIndex(preIntegratedBins_ssbo), preIntegratedBins_ssbo);

		//use shader
		auto Handle = sumPreIntegratedBins_C->GetGlShaderProgramHandle();
		glUseProgram(Handle);

		glUniform2i(glGetUniformLocation(Handle, "histogramDiscretizations"), ndfTree.GetHistogramResolution().x, ndfTree.GetHistogramResolution().y);
		glUniform1i(glGetUniformLocation(Handle, "binDiscretizations"), binDiscretizations);

		glDispatchCompute(histogramResolution.x*histogramResolution.y, 3, 1);

		glUseProgram(0);
		{
			auto glErr = glGetError();
			if (glErr != GL_NO_ERROR) {
				std::cout << "glError: " << __FILE__ << " " << __LINE__ << std::endl;
			}
		}
	}
#endif
}
inline void preIntegrateBins()
{

	preIntegrateBins_GPU();
#if 0

	//when the light changes, or when the binnig technique changes, we preintegrate the color for the bins.
	//std::cout << "entered integrate bins function" << std::endl;
	std::vector<float> data;
	auto Size = size_t(histogramResolution.x * histogramResolution.y * 3);
	data.resize(Size, 0.0f);
	glm::vec3 c, N;
	glm::vec3 diffuseColor = glm::vec3(1.0f, 1.0f, 1.0f);

	// NOTE: for this model to be physically based the BRDF would have to be perfectly specular
	float specularCorrection = 1.0f;
	glm::vec3 specularColor = specularCorrection * glm::vec3(1.0f, 1.0f, 1.0f);
	glm::vec2 transfer;

	float histogramX, histogramY;
	const float histogramScaleX = 1.0f / float(histogramResolution.x);
	const float histogramScaleY = 1.0f / float(histogramResolution.y);


	const float PI = 3.141592f;

	std::vector<glm::vec3> vec;
	float discretizationArea = 0;

	//calculate lighting parameters
	/*LightDir = CameraPosition({ lightRotationX, lightRotationY }, 1.0f);
	LightDir.y = -LightDir.y;*/

	auto viewMatrix = glm::lookAt(camPosi, camTarget, camUp);
	auto viewSpaceLightDir = viewMatrix * glm::vec4(LightDir, 0.0f);
	viewSpaceLightDir = glm::normalize(viewSpaceLightDir);

	glm::vec2 pos, diskCenter, v;
	diskCenter = glm::vec2(.5, .5);

	glm::vec2 pixel_dim = glm::vec2(1.0f / sw, 1.0f / sw);
	float totalarea = 0;

	std::vector<float> sBins;
	float vmax = 0;
	int idim = histogramResolution.x*histogramResolution.y*binDiscretizations*binDiscretizations;
	//uint8_t* img = new uint8_t[idim];
	//float* fimg = new float[idim];



	// debug
	{

		int maxlevels = std::log2(maxSubBins);
		int curlevel = maxlevels - std::log2(binDiscretizations);
		uint8_t* img = new uint8_t[projectedNormalBinningAreas[curlevel].size()];
		float* fimg = new float[projectedNormalBinningAreas[curlevel].size()];
		glm::vec2 pos;

		for (int i = 0; i < projectedNormalBinningAreas[curlevel].size(); i++)
		{
			pos = glm::vec2(projectedNormalBinningAreas[curlevel][i].y, projectedNormalBinningAreas[curlevel][i].z);
			N.x = pos.x*2.0f - 1.0f;
			N.y = pos.y*2.0f - 1.0f;
			N.z = sqrt(1.0f - N.x*N.x - N.y*N.y);

			glm::vec3 lightViewSpace = glm::vec3(viewSpaceLightDir.x, viewSpaceLightDir.y, viewSpaceLightDir.z);
			glm::vec3 c = blinnPhong(N, -lightViewSpace, glm::vec3(0.0f, 0.0f, 1.0f), diffuseColor, specularColor, specularExponent);

			fimg[i] = c.x*projectedNormalBinningAreas[curlevel][i].x;
			if (vmax < fimg[i])
				vmax = fimg[i];
		}

		for (int i = 0; i <projectedNormalBinningAreas[curlevel].size(); i++)
		{
			img[i] = fimg[i] * 255 / vmax;
		}

		PGMwrite("smallBins" + std::to_string(histogramResolution.x*histogramResolution.y) + "_" + std::to_string(binDiscretizations) + ".pgm", sqrt(projectedNormalBinningAreas[curlevel].size()), sqrt(projectedNormalBinningAreas[curlevel].size()), img);
	}
	//end debug
#endif

	return;



#if 0
	//pre integrate bins
	for (int i = 0; i < histogramResolution.x * histogramResolution.y; i++)
	{
		//initialize color
		data[i * 3 + 0] = 0;
		data[i * 3 + 1] = 0;
		data[i * 3 + 2] = 0;
		//c = glm::vec3(0, 0, 0);

		//get 2 index of bin
		histogramX = i % (histogramResolution.x);
		histogramY = int(i / histogramResolution.y);

		for (float j = 0; j < binDiscretizations; j++)
		{
			for (float k = 0; k < binDiscretizations; k++)
			{
				c = glm::vec3(0, 0, 0);
				if (binning_mode == 0)
				{
					discretizationArea = 0.0;

					//initialize position big bin's bottom left corner
					pos.x = histogramX*histogramScaleX;       //   ((histogramX + j / binDiscretizations)* histogramScaleX + 1.0f / (2.0f*binDiscretizations)*histogramScaleX) *2.0f - 1.0f;// -0.5f;
					pos.y = histogramY*histogramScaleY;      // ((histogramY + k / binDiscretizations)* histogramScaleY + 1.0f / (2.0f*binDiscretizations)*histogramScaleY) *2.0f - 1.0f;// -0.5f;

					//add to that the sub-bin discretization
					pos.x += j*(histogramScaleX / binDiscretizations);
					pos.y += k*(histogramScaleY / binDiscretizations);

					//now we are at the bottom left corner of the smaller bins (bins within big bin
					//we would like to be in the middle of that bin, so we add half the bin dimension to each coordiante
					pos.x += 0.5f*(histogramScaleX / binDiscretizations);
					pos.y += 0.5f*(histogramScaleY / binDiscretizations);

#if 1
					N.x = pos.x*2.0f - 1.0f;
					N.y = pos.y*2.0f - 1.0f;
					N.z = sqrt(1.0f - N.x*N.x - N.y*N.y);
					float l = glm::length(N);
					if (l <= 1.0f)
					{
						//float length = std::sqrt((N.x * N.x) + (N.y * N.y));
						//N.z = sqrt(1.0f - length);
						discretizationArea = (histogramScaleX / binDiscretizations)*(histogramScaleY / binDiscretizations) / (PI*0.5f*0.5f);  //add normalized area, so total area is = 1
					}
#endif

#if 0
					//'pos' must be within disk projection to be considered
					//our disk is of radius 0.5f;
					v = pos - diskCenter;

					if (glm::length(v) <= 0.5f)
					{
						//compute normal at that position
						N.x = pos.x*2.0f - 1.0f;
						N.y = pos.y*2.0f - 1.0f;

						float length = std::sqrt((N.x * N.x) + (N.y * N.y));
						N.z = sqrt(1.0f - length);
						if (N.z == N.z)
						{
							//now calculate bin area
							discretizationArea = (histogramScaleX / binDiscretizations)*(histogramScaleY / binDiscretizations);
							//normalize area so total area is 1
							discretizationArea /= PI*.5f*.5f;
						}
					}
#endif
				}
				else if (binning_mode == 1)
				{
					//float one_radian = PI / 180.0;
					//float s1 = 2 * PI / histogramResolution.y;
					//float s2 = (PI / 2) / histogramResolution.y;

					//float theta = histogramX*s1 + (1.0f / (2.0f*binDiscretizations))*s1;
					//float fi = histogramY*s2 + (1.0f / (2.0f*binDiscretizations))*s2;

					//theta -= PI;

					//N = glm::vec3(cos(theta)*sin(fi),
					//	sin(theta)*sin(fi),
					//	cos(fi));
					discretizationArea = 0.0;

					float s1 = 2 * PI;
					float s2 = (PI / 2);
					glm::vec2 ind = glm::vec2(histogramX, histogramY);
					float interval = 1.0 / histogramResolution.x;
					//get the position within bin, between (0,0) to (1,1)
					pos = ind*glm::vec2(interval, interval);

					//add sub-bin psotion
					pos += glm::vec2(interval, interval) * glm::vec2(j / float(binDiscretizations), k / float(binDiscretizations));

					//move to middle of sub-bin
					pos += glm::vec2(interval, interval) * glm::vec2(0.5*(1.0 / float(binDiscretizations)), 0.5*(1.0 / float(binDiscretizations)));


					float theta = pos.x*s1;
					float fi = pos.y*s2;


					N = glm::vec3(-cos(theta)*sin(fi),
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
				else if (binning_mode == 2)
				{
					//float s1 = (2.0f*sqrt(2.0f)) / histogramResolution.y;
					//float s2 = (2.0f*sqrt(2.0f)) / histogramResolution.y;

					//float X = histogramX*s1 + 1.0f / (2.0f*binDiscretizations)*s1;
					//float Y = histogramY*s2 + 1.0f / (2.0f*binDiscretizations)*s2;

					//X -= sqrt(2.0f);
					//Y -= sqrt(2.0f);


					//N = glm::vec3(sqrt(1.f - (X*X + Y*Y) / 4.0f)*X,
					//	sqrt(1.f - (X*X + Y*Y) / 4.0f)*Y,
					//	-1 * (-1.f + (X*X + Y*Y) / 2.0f));

					//N = glm::vec3(X / (2 * sqrt(1 / (-X*X + Y*Y + 4))),
					//	Y / (2 * sqrt(1 / (-X*X + Y*Y + 4))),
					//	-X*X / 2 + Y*Y / 2 + 1);

					//float s1 = 4.0f / histogramResolution.y;
					//float s2 = (2*PI) / histogramResolution.y;

					//float R = histogramX*s1 + (1.0f / (2.0f*binDiscretizations))*s1;
					//float theta = histogramY*s2 + (1.0f / (2.0f*binDiscretizations))*s2;


					//R = R - 2;
					//theta -= PI;

					//float fi = 2 * acos(R / 2.0);


					//N = glm::vec3(cos(theta)*sin(fi),
					//	sin(theta)*sin(fi),
					//	cos(fi));

					//newest
					{
						discretizationArea = 0.0;
						float s1 = (2.0f*sqrt(2.0f));
						float s2 = (2.0f*sqrt(2.0f));

						//get the position within bin, between (0,0) to (1,1)
						float interval = 1.0f / histogramResolution.x;;
						glm::vec2 ind = glm::vec2(histogramX, histogramY);
						pos = ind*glm::vec2(interval, interval);

						//add sub-bin psotion
						pos += glm::vec2(interval, interval) * glm::vec2(j / float(binDiscretizations), k / float(binDiscretizations));

						glm::vec2 s = pos;
						glm::vec2 e = s + glm::vec2(interval, interval) * glm::vec2((1.0 / float(binDiscretizations)), (1.0 / float(binDiscretizations)));;

						//move to middle of sub-bin
						pos += glm::vec2(interval, interval) * glm::vec2(0.5*(1.0 / float(binDiscretizations)), 0.5*(1.0 / float(binDiscretizations)));


						float X = pos.x*s1;
						float Y = pos.y*s2;

						X -= sqrt(2.0f);
						Y -= sqrt(2.0f);

						if (length(glm::vec2(X, Y)) <= sqrt(2.0f))//calculate the area
						{
							N = glm::vec3(sqrt(1.f - (X*X + Y*Y) / 4.0f)*X,
								sqrt(1.f - (X*X + Y*Y) / 4.0f)*Y,
								-1 * (-1.f + (X*X + Y*Y) / 2.0f));

							//area is dXdY
							//get X and Y of s
							glm::vec2 sXY = glm::vec2(s.x*s1, s.y*s2);
							glm::vec2 eXY = glm::vec2(e.x*s1, e.y*s2);

							float dX = eXY.x - sXY.x;
							float dY = eXY.y - sXY.y;

							discretizationArea = dX*dY / (PI*sqrt(2.0f)*sqrt(2.0f));
						}
					}
				}

				transfer = glm::vec2(N.x, N.y);

				if (renderMode <= 0)
				{
					glm::vec3 lightViewSpace = glm::vec3(viewSpaceLightDir.x, viewSpaceLightDir.y, viewSpaceLightDir.z);
					c = blinnPhong(N, -lightViewSpace, glm::vec3(0.0f, 0.0f, 1.0f), diffuseColor, specularColor, specularExponent);
				}
				else if (renderMode == 1)
				{
					const glm::vec3 leftColor = glm::vec3(0.35f, 0.65f, 0.8f);
					const glm::vec3 rightColor = glm::vec3(0.7f, 0.95f, 0.1f);
					const glm::vec3 bottomColor = glm::vec3(0.5f, 0.5f, 0.5f);
					const glm::vec3 topColor = glm::vec3(0.35f, 0.65f, 0.8f);

					glm::mat3 lightRotationZ = rotationMatrix(glm::vec3(0.0f, 0.0f, 1.0f), viewSpaceLightDir.x + PI);
					glm::mat3 lightRotationY = rotationMatrix(glm::vec3(0.0f, 1.0f, 0.0f), viewSpaceLightDir.y);

					transfer = glm::vec2((lightRotationZ * glm::vec3(transfer.x, transfer.y, 1.0f)).x, (lightRotationZ * glm::vec3(transfer.x, transfer.y, 1.0f)).y);
					transfer = glm::vec2((lightRotationY * glm::vec3(transfer.x, transfer.y, 1.0f)).x, (lightRotationY * glm::vec3(transfer.x, transfer.y, 1.0f)).y);

					diffuseColor = 0.5f * leftColor * (1.0f - transfer.x) + 0.5f * rightColor * transfer.x + 0.5f * bottomColor * (1.0f - transfer.y) + 0.5f * topColor * transfer.y;
					specularColor = diffuseColor;

					c = (diffuseColor + specularColor) * 0.5f;
				}

				data[i * 3 + 0] += c.x *discretizationArea;
				data[i * 3 + 1] += c.y *discretizationArea;
				data[i * 3 + 2] += c.z *discretizationArea;

				//glm::ivec2 twodind = glm::ivec2((i%histogramResolution.x)*binDiscretizations+j,(i/histogramResolution.y)*binDiscretizations+k);
				//fimg[int(twodind.y*binDiscretizations*histogramResolution.x+twodind.x)] = c.x*discretizationArea;
				//vmax = std::max(c.x*discretizationArea, vmax);
			}
		}
		//data[i * 3 + 0] /= binDiscretizations*binDiscretizations;
		//data[i * 3 + 1] /= binDiscretizations*binDiscretizations;
		//data[i * 3 + 2] /= binDiscretizations*binDiscretizations;

		//if (areaSwitch)
		//{
		//	//the NTF is essentially a CDF, to estimate the CDF using monte carlo integration, we should multiply by area of bin.
		//	data[i * 3 + 0] *= A[i];
		//	data[i * 3 + 1] *= A[i];
		//	data[i * 3 + 2] *= A[i];
		//}
	}


	auto ssbo = preIntegratedBins_ssbo;
	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, Size* sizeof(float), data.data(), GL_DYNAMIC_COPY);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}
#endif
}

inline void computeBinAreas()
{
	auto Size = size_t(histogramResolution.x * histogramResolution.y);
	if (binning_mode == 0 && A[0].size() == 0)
	{
		A[binning_mode].resize(Size, 0.0f);
		computeBinAreasSimple();
	}
	else if (binning_mode == 1 && A[1].size() == 0)
	{
		A[binning_mode].resize(Size, 0.0f);
		computeBinAreasSpherical();
	}
	else if (binning_mode == 2 && A[2].size() == 0)
	{
		A[binning_mode].resize(Size, 0.0f);
		computeBinAreasLambertAzimuthalEqualArea();
	}


	auto ssbo = binAreas_ssbo;
	{
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
		glBufferData(GL_SHADER_STORAGE_BUFFER, Size* sizeof(double), A[binning_mode].data(), GL_DYNAMIC_COPY);
		glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
	}
}

inline void computeBinAreasSimple()
{
	double interval, l;
	double PI = 3.14159265359;
	double samplesPerBin = 1024.0/(histogramResolution.x/8.0f);
	glm::ivec2 ind;

	//for the simple technique only, 
	//we would like to store an additional vector containing the a hierarchical representation of the area of each bin.
	//that is, here each bin is divided to 1024*1024 bins and that is how we calculate the area.
	//we wish to store the area for 1024x1024 sub division up to 1x1 sub-division in a tree fashion
	for (int i = samplesPerBin; i > 0; i = i / 2)
	{
		projectedNormalBinningAreas.push_back(std::vector<glm::vec3>());
	}

	for (int i = 0; i < samplesPerBin*samplesPerBin*histogramResolution.x*histogramResolution.y; i++)
		projectedNormalBinningAreas[0].push_back(glm::vec3());


	for (int i = 0; i < histogramResolution.x * histogramResolution.y; i++)
	{
		//initialize area
		A[0][i] = 0.0;

		areaHits.push_back(0);

		//get 2 index of bin
		ind = glm::ivec2(i % (histogramResolution.x), int(i / histogramResolution.y));

		for (int j = 0; j < samplesPerBin; j++)
		{
			for (int k = 0; k < samplesPerBin; k++)
			{
				interval = 1.0 / double(histogramResolution.y);
				glm::dvec2 p = glm::dvec2(ind.x*interval + j*(interval / samplesPerBin) + .5*(interval / samplesPerBin), ind.y*interval + k*(interval / samplesPerBin) + .5*(interval / samplesPerBin));
#if 0
				glm::vec2 N;
				N.x = p.x*2.0f - 1.0f;
				N.y = p.y*2.0f - 1.0f;
				//N.z = sqrt(1.0f - N.x*N.x - N.y*N.y);
				//if (N.z == N.z)
				//{
				l = glm::length(N);
				if (l <= 1.0f)
				{
					A[0][i] += ((interval / samplesPerBin)*(interval / samplesPerBin)) / (PI*0.5f*0.5f);  //add normalized area, so total area is = 1
				}
				//}
#endif
#if 1
				glm::dvec2 v = p - glm::dvec2(0.5, 0.5);
				l = glm::length(v);
				if (l <= .5)
				{
					A[0][i] += ((interval / samplesPerBin)*(interval / samplesPerBin)) / (PI*0.5*0.5);  //add normalized area, so total area is = 1
					areaHits[i]++;

					//store Normal vector and area of cells in projected binning Areas array
					{
						glm::ivec2 indx = glm::ivec2(ind.x*samplesPerBin + j, ind.y*samplesPerBin + k);
						projectedNormalBinningAreas[0][indx.y*samplesPerBin*histogramResolution.x + indx.x] = glm::vec3(((interval / samplesPerBin)*(interval / samplesPerBin)) / (PI*0.5*0.5), p);
					}

				}
				else
				{
					glm::ivec2 indx = glm::ivec2(ind.x*samplesPerBin + j, ind.y*samplesPerBin + k);
					projectedNormalBinningAreas[0][indx.y*samplesPerBin*histogramResolution.x + indx.x] = glm::vec3(0, p);
				}
#endif
			}
		}
	}


	//debug
	double sum;
	sum = std::accumulate(A[0].begin(), A[0].end(), 0.0);
	std::cout << "the sum of all areas for binning pattern 0 is: " << sum << std::endl;
	//end debug

	//now at level '0' in projectednormalbinningareas, we have samplesperbin^2 areas and normals
	//we store the areas and normals for the other levels as well
	{
		//adjust laybout of projectednormalbinning areas
		//std::vector<glm::vec3>tvec;
		//for (int i = 0; i < samplesPerBin*samplesPerBin*histogramResolution.x*histogramResolution.y; i++)
		//	tvec.push_back(glm::vec3(0,0,0));

		//for (int v = 0; v < projectedNormalBinningAreas[0].size(); v = v + 2 * sqrt(projectedNormalBinningAreas[0].size()))
		//{
		//	for (int u = 0; u < sqrt(projectedNormalBinningAreas[0].size()); u = u + 2)
		//	{
		//		tvec[v+u]=projectedNormalBinningAreas[0][v + u];
		//		tvec[v + u+1] = projectedNormalBinningAreas[0][v + u+1];
		//		tvec[v + sqrt(projectedNormalBinningAreas[0].size()) + u] = projectedNormalBinningAreas[0][v + sqrt(projectedNormalBinningAreas[0].size()) + u];
		//		tvec[v + sqrt(projectedNormalBinningAreas[0].size()) + u + 1] = projectedNormalBinningAreas[0][v + sqrt(projectedNormalBinningAreas[0].size()) + u+1];
		//	}
		//}

		//projectedNormalBinningAreas[0] = tvec;

		for (int i = 1; i < projectedNormalBinningAreas.size(); i++)
		{
			int mydim = std::sqrt(projectedNormalBinningAreas[i - 1].size()) / 2.0;

			for (int j = 0; j < mydim*mydim; j++)
				projectedNormalBinningAreas[i].push_back(glm::vec3());

			for (int u = 0; u < mydim; u++)
			{
				for (int v = 0; v < mydim; v++)
				{
					std::vector<glm::vec3> children;
					glm::ivec2 cIndx = glm::ivec2(2 * v, 2 * u);
					children.push_back(projectedNormalBinningAreas[i - 1][cIndx.y*std::sqrt(projectedNormalBinningAreas[i - 1].size()) + cIndx.x]);
					cIndx = glm::ivec2(2 * v, 2 * u + 1);
					children.push_back(projectedNormalBinningAreas[i - 1][cIndx.y*std::sqrt(projectedNormalBinningAreas[i - 1].size()) + cIndx.x]);
					cIndx = glm::ivec2(2 * v + 1, 2 * u);
					children.push_back(projectedNormalBinningAreas[i - 1][cIndx.y*std::sqrt(projectedNormalBinningAreas[i - 1].size()) + cIndx.x]);
					cIndx = glm::ivec2(2 * v + 1, 2 * u + 1);
					children.push_back(projectedNormalBinningAreas[i - 1][cIndx.y*std::sqrt(projectedNormalBinningAreas[i - 1].size()) + cIndx.x]);

					//if a child is inisde disk, add its position and area
					double pA = 0.0;
					glm::vec2 pPos = glm::vec2(0.0, 0.0);
					glm::vec2 aPos = glm::vec2(0.0, 0.0);
					double cCount = 0.0;
					for (int k = 0; k < children.size(); k++)
					{
						pPos += glm::vec2(children[k].y, children[k].z);
						if (children[k].x>0)  //if the area is bigger than 0
						{
							aPos += glm::vec2(children[k].y, children[k].z);
							pA += children[k].x;
							cCount++;
						}
					}

					if (pA > 0)
					{
						aPos /= cCount;
						projectedNormalBinningAreas[i][u*mydim + v] = glm::vec3(pA, aPos);
					}
					else
					{
						pPos /= 4.0f;
						projectedNormalBinningAreas[i][u*mydim + v] = glm::vec3(0, pPos);
					}

				}
			}

		}

		//even though we computed this area tree for 1-subbin up to samplesperbin^2 subbins, we know that we won't need more than bindiscritzations^2 subbmins, so to save memory,
		//we get rid of levels that we used to obtain an accurate area tree, but won't use anymore.
		int levels = std::log2(maxSubBins) + 1;
		int diff = projectedNormalBinningAreas.size() - levels;

		//////erase diff levels from projected normalbinareas
		projectedNormalBinningAreas.erase(projectedNormalBinningAreas.begin(), projectedNormalBinningAreas.begin() + diff);

		//now we copy the areas to gpu
		{

			std::vector<double> data;
			size_t Size = 0;

			for (int j = 0; j < projectedNormalBinningAreas.size(); j++)
			{
				for (int i = 0; i < projectedNormalBinningAreas[j].size(); i++)
				{
					data.push_back(projectedNormalBinningAreas[j][i].x);
					data.push_back(projectedNormalBinningAreas[j][i].y);
					data.push_back(projectedNormalBinningAreas[j][i].z);
				}
				Size += size_t(projectedNormalBinningAreas[j].size() * 3);
			}

			auto ssbo = simpleBinningAreas_ssbo;
			{
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
				glBufferData(GL_SHADER_STORAGE_BUFFER, Size* sizeof(double), data.data(), GL_DYNAMIC_COPY);
				glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
			}
		}

	}




	//draw some figures
#if 0
	{
		//draw images of these centers for each bin

		static int imgcount = 0;
		for (int i = 0; i < projectedNormalBinningAreas.size(); i++)
		{
			size_t w = 2 * sqrt(projectedNormalBinningAreas[0].size());
			size_t h = 2 * sqrt(projectedNormalBinningAreas[0].size());
			float* img = new float[w*h * 3];

			for (int j = 0; j < w*h * 3; j++)
				img[j] = 0;

			float maxa = 0.0f;
			for (int j = 0; j < projectedNormalBinningAreas[i].size(); j++)
			{
				if (maxa < projectedNormalBinningAreas[i][j].x)
					maxa = projectedNormalBinningAreas[i][j].x;
			}

			//draw bin color
#if 1
			int factor = w / sqrt(projectedNormalBinningAreas[i].size());

			for (int j = 0; j < h; j = j + factor)
			{
				for (int k = 0; k < w; k = k + factor)
				{
					for (int u = 0; u < factor; u++)
					{
						for (int v = 0; v < factor; v++)
						{
							glm::ivec2 img2d = glm::ivec2(k + v, j + u);
							glm::ivec2 array2d = glm::ivec2(img2d.x / factor, img2d.y / factor);
							img[img2d.y*w * 3 + img2d.x * 3] = 255 * (projectedNormalBinningAreas[i][array2d.y*sqrt(projectedNormalBinningAreas[i].size()) + array2d.x].x / maxa);
							img[img2d.y*w * 3 + img2d.x * 3 + 1] = 255 * (projectedNormalBinningAreas[i][array2d.y*sqrt(projectedNormalBinningAreas[i].size()) + array2d.x].x / maxa);
							img[img2d.y*w * 3 + img2d.x * 3 + 2] = 255 * (projectedNormalBinningAreas[i][array2d.y*sqrt(projectedNormalBinningAreas[i].size()) + array2d.x].x / maxa);
						}
					}
				}
			}
#endif
			//draw disk
#if 1
			float pointCount = 500000;
			for (float a = 0.0f; a < 360.0f; a += 360.0f / pointCount)
			{
				double heading = a * 3.1415926535897932384626433832795 / 180;
				glm::vec2 p = glm::vec2(cos(heading)*.5, sin(heading)*.5);
				p += glm::vec2(.5, .5);
				//p = glm::vec2(std::min(p.x,1.0f),std::min(p.y,1.0f));
				//p = glm::vec2(std::max(p.x, 0.0f), std::max(p.y, 0.0f));
				p *= w;
				p = glm::vec2(std::min(float(w - 1), p.x), std::min(float(w - 1), p.y));
				img[int(p.y)*w * 3 + int(p.x) * 3] = 0;
				img[int(p.y)*w * 3 + int(p.x) * 3 + 1] = 153;
				img[int(p.y)*w * 3 + int(p.x) * 3 + 2] = 153;
			}
#endif
			//draw bin centers
#if 1
			for (int j = 0; j < projectedNormalBinningAreas[i].size(); j++)
			{
				if (projectedNormalBinningAreas[i][j].x>0.0f)
				{
					glm::vec2 pos = glm::vec2(projectedNormalBinningAreas[i][j].y, projectedNormalBinningAreas[i][j].z);
					pos *= w;

					img[int(pos.y)*w * 3 + int(pos.x) * 3] = 153;// *(projectedNormalBinningAreas[i][j].x / maxa);
					img[int(pos.y)*w * 3 + int(pos.x) * 3 + 1] = 0;
					img[int(pos.y)*w * 3 + int(pos.x) * 3 + 2] = 153;
					//color some neighbours
					//{
					//	img[int((pos.y)*w + pos.x+1)] = 255;
					//	img[int((pos.y + 1)*w + pos.x + 1)] = 255;
					//	img[int((pos.y + 1)*w + pos.x)] = 255;
					//	img[int((pos.y + 1)*w + pos.x - 1)] = 255;
					//	img[int((pos.y )*w + pos.x - 1)] = 255;
					//	img[int((pos.y - 1)*w + pos.x - 1)] = 255;
					//	img[int((pos.y - 1)*w + pos.x)] = 255;
					//	img[int((pos.y - 1)*w + pos.x + 1)] = 255;						
					//}

					//img[int(pos.x*w + pos.y)] = 255 *(projectedNormalBinningAreas[i][j].x / maxa);
				}
			}
#endif
			saveBmpImage(w, h, img, "img" + std::to_string(projectedNormalBinningAreas[i].size()) + ".bmp");


			//PGMwrite("img" + std::to_string(projectedNormalBinningAreas[i].size()) + ".pgm", w, h, img);
			//delete[] img;
		}
	}
#endif
	//end debug

}
inline void computeBinAreasSpherical()
{

	//for each bin, divide it into a big number of small sub-bins, and sum up their projected area (projected solid angle)
	float interval, l;
	float PI = 3.14159265359;
	int samplesPerBin = 1024 / (histogramResolution.x / 8.0f);
	glm::vec2 ind, pos;

	float s1 = 2 * PI;
	float s2 = (PI / 2);

	glm::vec3 N;
	interval = 1.0f / histogramResolution.x;

	for (int i = 0; i < histogramResolution.x * histogramResolution.y; i++)
	{
		//initialize area
		A[1][i] = 0;

		//get 2 index of bin
		ind = glm::vec2(i % (histogramResolution.x), int(i / histogramResolution.y));

		for (int j = 0; j < samplesPerBin; j++)
		{
			for (int k = 0; k < samplesPerBin; k++)
			{
				//get the position within bin, between (0,0) to (1,1)
				pos = ind*glm::vec2(interval, interval);

				//add sub-bin psotion
				pos += glm::vec2(interval, interval) * glm::vec2(j / float(samplesPerBin), k / float(samplesPerBin));

				//move to middle of sub-bin
				pos += glm::vec2(interval, interval) * glm::vec2(0.5*(1.0 / float(samplesPerBin)), 0.5*(1.0 / float(samplesPerBin)));


				float theta = pos.x*s1;
				float fi = pos.y*s2;


				N = glm::vec3(cos(theta)*sin(fi),
					sin(theta)*sin(fi),
					cos(fi));


				if (length(N) <= 1)//calculate the area
				{
					//get small bin area
					float ba_onSphere = sin(fi)*(1.0f / samplesPerBin)*interval*s1*(1.0f / samplesPerBin)*interval*s2;      //solid angle (dw)
					float B = fi;                                                                                       //angle between projection plane normal and bin normal, acos(dot(N,(0,0,1)))=acos(N.z)=fi
					float ba_projected = N.z*ba_onSphere;                                                               //ba_projected=cos(B)*ba_onsphere=cos(fi)*ba_onsphere=N.z*ba_onsphere
					A[1][i] += ba_projected / PI;                                                                            //divide by PI to total area is '1'
				}
			}
		}
	}

	//debug
	float sum;
	sum = std::accumulate(A[1].begin(), A[1].end(), 0.0f);
	std::cout << "the sum of all areas for binning pattern 1 is: " << sum << std::endl;
	//end debug
}
inline void computeBinAreasLambertAzimuthalEqualArea()
{
	//for each bin, divide it into a big number of small sub-bins, and sum up their projected area (projected solid angle)
	float interval, l;
	float PI = 3.14159265359;
	int samplesPerBin = 1024/(histogramResolution.x/8.0f);
	glm::vec2 ind, pos, s, e;

	float s1 = (2.0f*sqrt(2.0f));
	float s2 = (2.0f*sqrt(2.0f));

	glm::vec3 N;
	interval = 1.0f / histogramResolution.x;

	//area is consistent


	for (int i = 0; i < histogramResolution.x * histogramResolution.y; i++)
	{
		//initialize area
		A[2][i] = 0;

		//get 2 index of bin
		ind = glm::vec2(i % (histogramResolution.x), int(i / histogramResolution.y));

		for (int j = 0; j < samplesPerBin; j++)
		{
			for (int k = 0; k < samplesPerBin; k++)
			{
				//get the position within bin, between (0,0) to (1,1)
				pos = ind*glm::vec2(interval, interval);

				//add sub-bin psotion
				pos += glm::vec2(interval, interval) * glm::vec2(j / float(samplesPerBin), k / float(samplesPerBin));

				s = pos;
				e = s + glm::vec2(interval, interval) * glm::vec2((1.0 / float(samplesPerBin)), (1.0 / float(samplesPerBin)));;

				//move to middle of sub-bin
				pos += glm::vec2(interval, interval) * glm::vec2(0.5*(1.0 / float(samplesPerBin)), 0.5*(1.0 / float(samplesPerBin)));


				float X = pos.x*s1;
				float Y = pos.y*s2;

				X -= sqrt(2.0f);
				Y -= sqrt(2.0f);


				N = glm::vec3(sqrt(1.f - (X*X + Y*Y) / 4.0f)*X,
					sqrt(1.f - (X*X + Y*Y) / 4.0f)*Y,
					-1 * (-1.f + (X*X + Y*Y) / 2.0f));

				//calculate the area
				if (length(glm::vec2(X, Y)) <= sqrt(2.0f))              //the projected disk has a radius of sqrt(2) if the 3d sphere is of radius 1 as per the projection (https://en.wikipedia.org/wiki/Lambert_azimuthal_equal-area_projection)
				{
					//area is dXdY
					//get X and Y of s
					glm::vec2 sXY = glm::vec2(s.x*s1, s.y*s2);
					glm::vec2 eXY = glm::vec2(e.x*s1, e.y*s2);

					float dX = eXY.x - sXY.x;
					float dY = eXY.y - sXY.y;

					A[2][i] += dX*dY / (PI*sqrt(2.0f)*sqrt(2.0f));
				}


			}
		}
	}

	//debug
	float sum;
	sum = std::accumulate(A[2].begin(), A[2].end(), 0.0f);
	std::cout << "the sum of all areas for binning pattern 2 is: " << sum << std::endl;
	//end debug
}

void mouse(int button, int state, int x, int y) {
	auto mouseDelta = lastMouse - glm::ivec2(x, y);
	lastMouse = glm::ivec2(x, y);

    if (tweakbarInitialized) {
        if (TwEventMouseButtonGLUT(button, state, x, y)) {
            return;
        }
    }

	if (button == GLUT_LEFT_BUTTON) {
		if (state == GLUT_DOWN) {
			leftHeld = true;
		}
		else {
			leftHeld = false;
		}
	}

	if (button == GLUT_RIGHT_BUTTON) {
		if (state == GLUT_DOWN) {
			rightHeld = true;
		}
		else {
			rightHeld = false;
		}
	}

	if (button == GLUT_MIDDLE_BUTTON) {
		if (state == GLUT_DOWN) {
			middleHeld = true;
		}
		else {
			middleHeld = false;
		}
	}

	/*if(button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
	if(zoomMode != 1) {
	zoomWindow = glm::vec2(static_cast<float>(windowSize.x - x) / static_cast<float>(windowSize.x), static_cast<float>(y) / static_cast<float>(windowSize.y));
	zoomMode = 1;
	zoomScale = 1.0f;
	} else {
	zoomScale *= 2.0f;
	}
	}*/

	/*if(button == GLUT_RIGHT_BUTTON && state == GLUT_DOWN) {
	zoomMode = 0;
	zoomScale = 0.0f;
	}*/

	glm::vec2 relativeMouse;
	if (transferFunctionPressed({ x, y }, relativeMouse) && leftHeld)
	{
		glm::vec3 color;
		NdfImposters::NormalTransferFunction<float> *fullChromeTexture = &chromeTexture[0];
		fullChromeTexture->FetchColor({ relativeMouse.x, 1.0f - relativeMouse.y }, color);             //always pick from full chrome texture
		setActiveChromeTexture(color);

		//transferFunction.Splat({ relMouseX, relMouseY }, { 1.0f, 0.0f, 0.0f }, NdfImposters::GaussianKernel<float>);
		activeTransferFunction->Splat({ relativeMouse.x, relativeMouse.y }, { 1.0f, 0.0f, 0.0f }, (button == GLUT_RIGHT_BUTTON), (button == GLUT_MIDDLE_BUTTON));
		preIntegrateBins();

		if (ndfExplorerMode)
		{
			//check if disk is picked
			glm::vec2 dir = glm::vec2(relativeMouse.x, 1.0f - relativeMouse.y) - glm::vec2(0.5, 0.5);
			if (glm::length(dir) < disks[0].radius)
			{
				diskPicked = true;
				std::cout << "disk picked" << std::endl;
			}
			else
			{
				diskPicked = false;
			}
			
			if (!diskPicked)
			{
				//Part about slices
				bool found = false;

				//check if a slice is picked
				{
					glm::vec2 dir = glm::vec2(relativeMouse.x, 1.0f - relativeMouse.y) - glm::vec2(0.5, 0.5);
					dir = dir / glm::length(dir);
					//get angle between dir and x axis
					float angle = acos(glm::dot(dir, glm::vec2(1, 0)));
					if (dir.y < 0)
						angle = (2 * 3.14f) - angle;

					//check if this angle falls in any of the angles
					for (int i = 0; i < slices.size(); i++)
					{
						if (angle >= (slices[i].angle - 0.5*slices[i].radius) && angle <= (slices[i].angle + 0.5*slices[i].radius))
						{
							found = true;
							sliceIndx = i;
							std::cout << "Picked slice " << i << std::endl;
							break;
						}
					}
				}
				//else add slice
				if (!found)
				{
					if (slices.size() < sliceColors.size())
					{
						slice s;
						//get angle of slice
						glm::vec2 dir = glm::vec2(relativeMouse.x, 1.0f - relativeMouse.y) - glm::vec2(0.5, 0.5);
						dir = dir / glm::length(dir);
						//get angle between dir and x axis
						s.angle = acos(glm::dot(dir, glm::vec2(1, 0)));
						if (dir.y < 0)
							s.angle = (2 * 3.14f) - s.angle;

						//debug
						{
							cout << "angle is " << s.angle*180.0f / 3.14 << ", and dir is " << dir.x << ", " << dir.y << std::endl;
						}

						s.radius = 30.0f*3.14 / 180.0f;  //radius is in angles, range of slice

						bool intersect = s.intersects(slices, 0, 0, -1);

						if (!intersect)
						{
							//get color
							for (int i = 0; i < sliceColors.size(); i++)
							{
								if (sliceColors[i].second == -1)
								{
									s.color = sliceColors[i].first;
									sliceColors[i].second = slices.size();
									break;
								}
							}

							slices.push_back(s);
							renderNdfExplorer();
							preIntegrateBins();
						}
						else
						{
							std::cout << "new slice would intersect with older slices" << std::endl;
						}
					}
					else
					{
						std::cout << "can't add more slices, maximum slices reached" << std::endl;
					}
				}
			}
		}
	}
	
	if (transferFunctionPressed({ x, y }, relativeMouse) && middleHeld)
	{
		if (sliceIndx >= 0)
		{
			slices.erase(slices.begin() + sliceIndx);
			
			for (int i = 0; i < sliceColors.size(); i++)
			{
				if (sliceColors[i].second == sliceIndx)
				{
					sliceColors[i].second = -1;
					break;
				}
			}

			renderNdfExplorer();
			preIntegrateBins();
		}
	}

	if (ndfPressed({ x, y }, relativeMouse))
	{
		adjustAvgNDF(relativeMouse,rightHeld);
	}
}

void setActiveChromeTexture(glm::vec3 color)
{
	//bit 0 -> purple
	//bit 1 -> blue
	//bit 2 -> orange
	//bit 3 -> green
	//bit 4 -> red
	
	//push colors
	std::vector<glm::vec3> colors;
	colors.push_back(glm::vec3(102, 45, 145));
	colors.push_back(glm::vec3(37, 170, 225));
	colors.push_back(glm::vec3(247, 148, 30));
	colors.push_back(glm::vec3(140, 198, 63));
	colors.push_back(glm::vec3(237, 28, 36));

	//see which color is color closest to
	int indx;
	float closest = 10000000;
	glm::vec3 diff;
	for (int i = 0; i < colors.size(); i++)
	{
		diff = color - colors[i];
		if (glm::length(diff) < closest)
		{
			closest = glm::length(diff);
			indx = i;
		}
	}

	//now we know which color the user clicked on, it's color of index 'indx'
	//so we invert the bit at index
	chromeTextureBits[indx] = !chromeTextureBits[indx];

	//now we set the active texture to the texture corresponding to chrometextureBits
	int activeTextureIndex = 0;
	bool found = false;
	for (int i = 1; i > -1 && !found; i--)
	{
		for (int j = 1; j > -1 && !found; j--)
		{
			for (int k = 1; k > -1 && !found; k--)
			{
				for (int l = 1; l > -1 && !found; l--)
				{
					for (int m = 1; m > -1 && !found; m--)
					{
						//if (i == 1)
						//	std::cout << "purple, ";
						//if (j == 1)
						//	std::cout << "blue, ";
						//if (k == 1)
						//	std::cout << "orange, ";
						//if (l == 1)
						//	std::cout << "green, ";
						//if (m == 1)
						//	std::cout << "red, ";

						//std::cout<<endl;

						if (chromeTextureBits[0] == i && chromeTextureBits[1] == j && chromeTextureBits[2] == k
							&& chromeTextureBits[3] == l && chromeTextureBits[4] == m)
						{
							activeChromeTexture = &chromeTexture[activeTextureIndex];
							found = true;
							break;
						}
						activeTextureIndex++;
					}
				}
			}
		}
	}
	
}

void mapToSphere(glm::vec2& p, glm::vec3& res)
{
	glm::vec2 TempPt;
	GLfloat length;

	//Copy paramter into temp point
	TempPt = p;

	//Adjust point coords and scale down to range of [-1 ... 1]
	//TempPt.x = (TempPt.x /((windowSize.x-1.0f)*0.5f)) - 1.0f;
	//TempPt.y = 1.0f - (TempPt.y /((windowSize.y-1.0f)*0.5f));

	TempPt.x = (TempPt.x - (0.5f*windowSize.x)) / (0.5f*windowSize.x);
	TempPt.y = (TempPt.y - (0.5f*windowSize.y)) / (0.5f*windowSize.y);

	//Compute the square of the length of the vector to the point from the center
	length = (TempPt.x * TempPt.x) + (TempPt.y * TempPt.y);

	//If the point is mapped outside of the sphere... (length > radius squared)
	if (length > 1.0f)
	{
		GLfloat norm;

		//Compute a normalizing factor (radius / sqrt(length))
		norm = 1.0f / sqrt(length);

		//Return the "normalized" vector, a point on the sphere
		res.x = TempPt.x * norm;
		res.y = TempPt.y * norm;
		res.z = 0.0f;
	}
	else    //Else it's on the inside
	{
		//Return a vector to a point mapped inside the sphere sqrt(radius squared - length)
		res.x = TempPt.x;
		res.y = TempPt.y;
		res.z = sqrt(1.0f - length);
	}
}

glm::quat RotationBetweenVectors(glm::vec3 start, glm::vec3 dest)
{
	start = glm::normalize(start);
	dest = glm::normalize(dest);

	float cosTheta = glm::dot(start, dest);
	glm::vec3 rotationAxis;

	if (cosTheta < -1 + 0.001f){
		// special case when vectors in opposite directions :
		// there is no "ideal" rotation axis
		// So guess one; any will do as long as it's perpendicular to start
		// This implementation favors a rotation around the Up axis,
		// since it's often what you want to do.
		rotationAxis = glm::cross(glm::vec3(0.0f, 0.0f, 1.0f), start);
		if (glm::length2(rotationAxis) < 0.01) // bad luck, they were parallel, try again!
			rotationAxis = glm::cross(glm::vec3(1.0f, 0.0f, 0.0f), start);

		rotationAxis = glm::normalize(rotationAxis);
		return glm::angleAxis(180.0f, rotationAxis);
	}

	// Implementation from Stan Melax's Game Programming Gems 1 article
	rotationAxis = glm::cross(start, dest);

	float s = sqrt((1 + cosTheta) * 2);
	float invs = 1 / s;

	return glm::quat(
		s * 0.5f,
		rotationAxis.x * invs,
		rotationAxis.y * invs,
		rotationAxis.z * invs
		);
}

glm::vec3 Map_to_trackball(float x, float y)
{
	glm::vec3 v;
	float d;

	v.x = ((2.0*x - windowSize.x) / windowSize.x);
	v.y = ((windowSize.y - 2.0*y) / windowSize.y);
	v.z = (0);

	d = glm::length(v);
	d = (d < 1.0) ? d : 1.0;
	v.z = sqrt(1.001 - d*d);

	//normalize v
	glm::normalize(v);

	return v;
}

void mouseMotionActive(int x, int y)
{
	auto mouseDelta = lastMouse - glm::ivec2(x, y);
	lastMouse = glm::ivec2(x, y);
	auto modifiers = glutGetModifiers();

	//std::cout << x << ", " << y << std::endl;

	//if (!leftHeld && !rightHeld && !middleHeld) {
	//	if (tweakbarInitialized) {
	//		if (TwMouseMotion(x, y)) {
	//			return;
	//		}
	//	}
	//}

	int r = 4;

	// no interaction if within transfer function
	glm::vec2 relativeMouse;
	if (transferFunctionPressed({ x, y }, relativeMouse))
	{
		return;
	}

	bool leftRight = leftHeld && rightHeld;

	if (!leftRight)
	{
		if (leftHeld)
		{

			if (ndfExplorerMode)
			{
				if (!diskPicked)
				{
					glm::ivec2 curLoc = glm::ivec2(x, y);
					glm::vec2 dir = curLoc - prevLoc;
					prevLoc = curLoc;
					float sign = dir.x / std::abs(dir.x);
					if (dir.x == 0)
						sign = 1;

					//first check if modification will cause intersection
#if 0
					bool intersect = false;
					float side1 = slices[sliceIndx].angle + sign*1.0f / (2 * 3.14f) - 0.5*slices[sliceIndx].radius;
					float side2 = slices[sliceIndx].angle + sign*1.0f / (2 * 3.14f) + 0.5*slices[sliceIndx].radius;
					side1 = std::fmod(side1, (2 * 3.14f));
					side2 = std::fmod(side2, (2 * 3.14f));

					if (side1 < 0)
						side1 = side1 + 2 * 3.14;
					if (side2 < 0)
						side2 = side2 + 2 * 3.14;

					float s1, s2;
					for (int i = 0; i < slices.size(); i++)
					{
						if (i != sliceIndx)
						{
							s1 = slices[i].angle - 0.5*slices[i].radius;
							s2 = slices[i].angle + 0.5*slices[i].radius;

							//check if side1 or side2 are between s1 and s2
							std::cout << s1 << ", " << s2 << ", " << side1 << ", " << side2 << std::endl;
							if ((side1 <= s2 && side1 >= s1) || (side2 <= s2 && side2 >= s1))
							{
								intersect = true;

								std::cout << "Intersect!" << std::endl;
								break;
							}
						}
					}
#else
					bool intersect=slices[sliceIndx].intersects(slices, sign*1.0f / (2 * 3.14f), 0, sliceIndx);
#endif

					if (!intersect)
					{
						slices[sliceIndx].angle += sign*1.0f / (2 * 3.14f);
						if (slices[sliceIndx].angle < 0)
							slices[sliceIndx].angle = slices[sliceIndx].angle + 2 * 3.14;
						slices[sliceIndx].angle = std::fmod(slices[sliceIndx].angle, 2 * 3.14f);
						//std::cout << "angle increased to " << slices[sliceIndx].angle << std::endl;
						renderNdfExplorer();
						preIntegrateBins();
					}
				}
				else
				{
					glm::ivec2 curLoc = glm::ivec2(x, y);
					glm::vec2 dir = curLoc - prevLoc;
					prevLoc = curLoc;
					float sign = dir.x / std::abs(dir.x);
					if (dir.x == 0)
						sign = 1;

					disks[0].radius += sign*.02;
					
					disks[0].radius = std::min(0.5f, disks[0].radius);
					disks[0].radius = std::max(0.0f, disks[0].radius);

					renderNdfExplorer();
					preIntegrateBins();
				}
			}
			else
			{
				if (plainRayCasting)
				{
					//reset progressive raycasting ssbo
					erase_progressive_raycasting_ssbo();
				}

				if (modifiers & GLUT_ACTIVE_CTRL)
				{
					//static const float angleRange = 180.0f;
					//// FIXME: breaks when dragging outside of the windows as the values get negtaive
					//lightRotationX = angleRange * std::asin(static_cast<float>(x) / static_cast<float>(windowSize.x)) + (angleRange * 0.5f);
					//lightRotationY = angleRange * std::asin(static_cast<float>(y) / static_cast<float>(windowSize.y)) + (angleRange * 0.5f);

					//LightDir = CameraPosition({ lightRotationX, lightRotationY }, 1.0f);
					//LightDir.y = -LightDir.y;

					//new
					mapToSphere(glm::vec2(x, y), LightDir);
					LightDir.y = -LightDir.y;
					LightDir.z = -LightDir.z;


					//std::cout << LightDir.x << ", " << LightDir.y << ", " << LightDir.z << std::endl;
					//end new

					if (cachedRayCasting)
					{
						initialize();
						tile_based_culling(false);
						update_page_texture();
						reset();

					}
					if (!cachedRayCasting&&!plainRayCasting)
					{
						//std::cout << "Rotating light " << lightRotationX << " " << lightRotationY << std::endl;
						preIntegrateBins();
					}

				}
				else if (modifiers & GLUT_ACTIVE_ALT)
				{
					//add a neighbourhood as well to the pixel
					for (int i = x - r; i < x + r; i++)
					{
						for (int j = y - r; j < y + r; j++)
						{
							bool found = false;
							if (i >= 0 && i < windowSize.x && j >= 0 && j < windowSize.y)
							{
								//first check if it's already added
								for (int k = 0; k < selectedPixels.size(); k++)
								{
									if (selectedPixels[k].x == i && selectedPixels[k].y == j)
									{
										found = true;
										break;
									}
								}
								if (!found)
								{
									selectedPixels.push_back(glm::vec2(i, j));
									drawSelection();
								}
							}
						}
					}
				}
				else
				{
					if (probeNDFsMode > 0)
					{
						//resetSimLimits(true);
						updateSimilarityLimits();
					}
					if (noCaching)
					{
						initialize();
					}
					//StopWatch w;

					//old
					const float zoomSpeed = -.075f; // -0.0125f;

					cameraDistance += static_cast<float>(mouseDelta.y) * zoomSpeed;
					//end old

					//new
					if (mouseDelta.y != 0)
					{
						cameraDistance += static_cast<float>(mouseDelta.y / (std::abs(mouseDelta.y)))*.001;
						//std::cout <<"camera distance 1 : "<< cameraDistance << std::endl;
					}
					else
					{
						return;
					}
					//end new



					float prev_lod, target_lod;

					prev_lod = current_lod;

					//debug
					//get cam distance that will take us to lower lod
					//target_lod = std::max(0.0f, current_lod - 1);
					//cameraDistance = LOD.get_cam_dist(target_lod);
					//end debug

					cameraDistance = std::max(LOD.min_cam_dist, std::min(cameraDistance, LOD.max_cam_dist - .3f));
					//std::cout << "camera distance 2: " << cameraDistance << std::endl;

					//cameraDistance = 2.0f;
					//std::cout << "camera distance: " << cameraDistance << std::endl;

					camPosi = CameraPosition(cameraRotation, cameraDistance);
					camPosi += cameraOffset;
					camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

					zoom_camPosi = camPosi;


					tile_based_culling(false);


					//w.StartTimer();
					//tile_based_panning(false);
					//w.StopTimer();
					//std::cout << "panning " << w.GetElapsedTime() << " ms" << std::endl;


					//debug
					//phys_x = phys_y = 0;
					//end debug
					//w.StartTimer();
					update_page_texture();
					//glFinish();
					//w.StopTimer();
					//std::cout << "update texture " << w.GetElapsedTime() << " ms" << std::endl;

					//if (int(prev_lod)- int(current_lod)!=0)
					//w.StartTimer();
					reset();
					//w.StopTimer();
					//std::cout << "reset" << w.GetElapsedTime() << " ms" << std::endl;

					//w.StartTimer();
					display();
					//glFinish();
					//w.StopTimer();
					//std::cout << "display " << w.GetElapsedTime() << " ms" << std::endl;
					////glutPostRedisplay();

					//w2.StopTimer();
					//std::cout << "The average time for mouse motion call back is: " << w2.GetElapsedTime() << "ms" << std::endl;
				}

			}

		}

		if (rightHeld)
		{
			if (ndfExplorerMode)
			{
				glm::ivec2 curLoc = glm::ivec2(x, y);
				glm::vec2 dir = curLoc - prevLoc;
				prevLoc = curLoc;
				float sign = dir.x / std::abs(dir.x);
				if (dir.x == 0)
					sign = 1;
#if 0
				//first check if modification will cause intersection
				bool intersect = false;
				float side1 = slices[sliceIndx].angle - 0.5*(slices[sliceIndx].radius + sign*1.0f / (2 * 3.14f));
				float side2 = slices[sliceIndx].angle + 0.5*(slices[sliceIndx].radius + sign*1.0f / (2 * 3.14f));
				side1 = std::fmod(side1, (2 * 3.14f));
				side2 = std::fmod(side2, (2 * 3.14f));

				if (side1 < 0)
					side1 = side1 + 2 * 3.14;
				if (side2 < 0)
					side2 = side2 + 2 * 3.14;

				float s1, s2;
				for (int i = 0; i < slices.size(); i++)
				{
					if (i != sliceIndx)
					{
						s1 = std::min(slices[i].angle - 0.5*slices[i].radius, slices[i].angle + 0.5*slices[i].radius);
						s2 = std::max(slices[i].angle - 0.5*slices[i].radius, slices[i].angle + 0.5*slices[i].radius);

						//check if side1 or side2 are between s1 and s2
						if ((side1 <= s2 && side1 >= s1) || (side2 <= s2 && side2 >= s1))
						{
							intersect = true;
							std::cout << "Intersect!" << std::endl;
							break;
						}
					}
				}
#else
				bool intersect = slices[sliceIndx].intersects(slices,0, sign*1.0f / (2 * 3.14f), sliceIndx);
#endif

				if (!intersect)
				{
					slices[sliceIndx].radius += sign*1.0f / (2 * 3.14f);
					/*if (slices[sliceIndx].radius < 0)
						slices[sliceIndx].radius = slices[sliceIndx].radius + 2 * 3.14;*/
					//slices[sliceIndx].radius = std::fmod(slices[sliceIndx].radius, 2*3.14f);

					slices[sliceIndx].radius = std::min(2 * 3.14f, slices[sliceIndx].radius);
					slices[sliceIndx].radius = std::max(0.0f, slices[sliceIndx].radius);

					//std::cout << "radius increased to " << slices[sliceIndx].radius << std::endl;
					renderNdfExplorer();
					preIntegrateBins();
				}
			}
			else
			{


				percentage_of_cached_tiles.clear();
				if (plainRayCasting)
				{
					//reset progressive raycasting ssbo
					erase_progressive_raycasting_ssbo();
				}

				if (probeNDFsMode > 0)
				{
					resetSimLimits(true);
				}

				static const float angleRange = 180.0f;
				// FIXME: breaks when dragging outside of the windows as the values get negtaive
				/*cameraRotation.x = 360.0f - angleRange * std::asin(static_cast<float>(x) / static_cast<float>(windowSize.x)) + (angleRange * 0.5f);
				cameraRotation.y = 360.0f - angleRange * std::asin(static_cast<float>(y) / static_cast<float>(windowSize.y)) + (angleRange * 0.5f);*/

				const float rotationSpeed = 0.1f;
				//cameraRotation.x += static_cast<float>(mouseDelta.x) * rotationSpeed;
				//cameraRotation.y += static_cast<float>(mouseDelta.y) * rotationSpeed;

				const float ssboClearColor = 0.0f;

				if (true)
				{
					//apply_permenant_rotation(static_cast<float>(mouseDelta.x) * rotationSpeed, static_cast<float>(mouseDelta.y) * rotationSpeed);
					//new
#if 0
					{
						glm::vec3 direction;

						static const float m_ROTSCALE = 90.0;
						float PI = 3.14159265359;
						glm::vec3 curpoint=Map_to_trackball(x,y);	
						direction=curpoint-lastpoint;
						glm::vec3 rotaxis=glm::cross(lastpoint,curpoint);
						glm::normalize(rotaxis);

						float velocity=glm::length(direction);
						float rot_angle=(velocity*m_ROTSCALE)*(PI/180.0f);
						lastpoint=curpoint;


						glm::quat q;
						q.x=rotaxis.x*sin(rot_angle/2.0f);
						q.y=rotaxis.y*sin(rot_angle/2.0f);
						q.z=rotaxis.z*sin(rot_angle/2.0f);
						q.w=cos(rot_angle/2.0f);
						GlobalRotationMatrix = glm::mat3_cast(q)*GlobalRotationMatrix;


					}
#else
					{
					glm::vec2 prevMouse = lastMouse + mouseDelta;
					glm::vec2 curMouse = lastMouse;

					glm::vec3 a, b;
					mapToSphere(prevMouse, a);
					mapToSphere(glm::vec2(lastMouse), b);

					a.y = -1.0f*a.y;
					a.z = -1.0f*a.z;

					b.y = -1.0f*b.y;
					b.z = -1.0f*b.z;

					glm::quat q = RotationBetweenVectors(a, b);

					//{
					//	glm::vec3 desiredUp = glm::vec3(0, 1, 0);
					//	glm::vec3 right = glm::cross(b, desiredUp);
					//	desiredUp = glm::cross(right, b);

					//	// Because of the 1rst rotation, the up is probably completely screwed up.
					//	// Find the rotation between the "up" of the rotated object, and the desired up
					//	glm::vec3 newUp = q * glm::vec3(0.0f, 1.0f, 0.0f);
					//	glm::quat rot2 = RotationBetweenVectors(newUp, desiredUp);

					//	q = rot2 * q; // remember, in reverse order.
					//}


					GlobalRotationMatrix = glm::mat3_cast(q)*GlobalRotationMatrix;
					//std::cout << "rotation matrix: " << std::endl;
					//std::cout << GlobalRotationMatrix[0].x << ", " << GlobalRotationMatrix[0].y << ", " << GlobalRotationMatrix[0].z << std::endl;
					//std::cout << GlobalRotationMatrix[1].x << ", " << GlobalRotationMatrix[1].y << ", " << GlobalRotationMatrix[1].z << std::endl;
					//std::cout << GlobalRotationMatrix[2].x << ", " << GlobalRotationMatrix[2].y << ", " << GlobalRotationMatrix[2].z << std::endl;
				}
#endif


				}
				else
				{
					cameraRotation.x -= camRotation;
				}

				//zoom_camPosi = camPosi;

				//phys_x = 0;
				//phys_y = 0;
				initialize();

				cameraDistance = std::max(LOD.min_cam_dist, std::min(cameraDistance, LOD.max_cam_dist - .3f));
				camPosi = CameraPosition(cameraRotation, cameraDistance);
				camPosi += cameraOffset;
				camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

				tile_based_culling(false);
				//tile_based_panning(false);
				update_page_texture();
				reset();

				//HOM related values
				{
					////build HOM
					//buildHOMFlag = true;

					////reset cells to render
					//cellsToRender = global_tree.leaves;
				}

				display();
				//glutPostRedisplay();
			}
		}
	}
	else
	{
		//const float zoomSpeed = -0.00125f;
		//	
		//cameraDistance += static_cast<float>(mouseDelta.y) * zoomSpeed;

		//reset();
	}

	if (middleHeld)
	{
		if (probeNDFsMode > 0)
		{
			resetSimLimits(true);
		}
		if (plainRayCasting)
		{
			//reset progressive raycasting ssbo
			erase_progressive_raycasting_ssbo();
		}

		if (noCaching)
		{
			initialize();
		}

		if (modifiers & GLUT_ACTIVE_CTRL)
		{


		}
		else
		{
			//StopWatch w;
			const float panningSpeed = 0.001f;// 0.00025f;
			cameraOffset.x += static_cast<float>(mouseDelta.x) * panningSpeed;
			cameraOffset.y -= static_cast<float>(mouseDelta.y) * panningSpeed;


			camPosi = CameraPosition(cameraRotation, cameraDistance);
			camPosi += cameraOffset;
			camTarget = glm::vec3(0.0f, 0.0f, 0.0f) + cameraOffset;

			//w.StartTimer();
			tile_based_culling(false);
			//tile_based_panning(false);
			//w.StopTimer();
			//std::cout << "panning " << w.GetElapsedTime() << " ms" << std::endl;

			//w.StartTimer();
			update_page_texture();
			//w.StopTimer();
			//std::cout << "update texture " << w.GetElapsedTime() << " ms" << std::endl;


			reset();
			display();
			//glutPostRedisplay();
		}

		//reset();
	}
}
void erase_progressive_raycasting_ssbo()
{
	//float* samples_data = new float[windowSize.x*windowSize.y*3];
	//for (int i = 0; i < windowSize.x*windowSize.y*3; i++)
	//{
	//	samples_data[i] = 0;
	//}
	//glBindBuffer(GL_SHADER_STORAGE_BUFFER, progressive_raycasting_ssbo);
	//GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
	//memcpy(p, &samples_data, sizeof(samples_data));
	//glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);


	float ssboClearColor = 0;
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, progressive_raycasting_ssbo);
	if (!glClearBufferData) {
		glClearBufferData = (PFNGLCLEARBUFFERDATAPROC)wglGetProcAddress("glClearBufferData");
	}
	assert(glClearBufferData);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, progressive_raycasting_ssbo);
	glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32F, GL_RED, GL_FLOAT, &ssboClearColor);
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
}

void adjustAvgNDF(glm::vec2 relativeMouse,bool clearFlag)
{

	//given a texture coordinate in 'relativemouse'
	//1-> map texture coordinate to 'bin' according to binning mode
	//2-> update a cpu avgndf that stores for each bin, the number of hits in that bin.
	//3-> compute cdf, from it compute ndf
	//4-> upload this ndf to avg ndf

	glm::vec2 quantizedRay;
	int histogramIndexCentral, histogramIndexR, histogramIndexB, histogramIndexBR;
	glm::dvec2 bilinearWeights;
	glm::vec2 newRay = relativeMouse;
	int HistogramWidth = histogramResolution.x;
	int HistogramHeight = histogramResolution.y;
	bool miss;
	const float PI = 3.141592f;
	const float toRadiants = PI / 180.0f;
	std::vector<float>avgNDF(histogramResolution.x*histogramResolution.y, 0.0f);
	float samples = 0.0f;
	//relativeMouse.y = 1.0 - relativeMouse.y;
	glm::vec2 transformedCoord;
	
	transformedCoord.x= relativeMouse.x * 2.0f - 1.0f;
	transformedCoord.y = relativeMouse.y * 2.0f - 1.0f;

	float l2 = glm::length(transformedCoord);

	if (l2 > 1)
		return;

	if (!clearFlag)
	{

		if (binning_mode == 0)
		{
			quantizedRay = glm::vec2(newRay.x * float(histogramResolution.x), newRay.y * float(histogramResolution.y));

			// NOTE: the total amount of energy will only be available if the max. number of samples is reached.
			// before that the renderer has to upscale the ndf samples according to the total energy.8
			histogramIndexCentral = int(quantizedRay.y) * HistogramWidth + int(quantizedRay.x);
			miss = (histogramIndexCentral == 0);
			if (histogramIndexCentral < 0 || histogramIndexCentral >= HistogramHeight * HistogramWidth)
			{
				return;
			}
		}
		else if (binning_mode == 1)
		{
			//spherical coordinates binning

			//get 3d normal
			glm::vec3 sourceNormal;
			sourceNormal.x = newRay.x*2.0f - 1.0f;
			sourceNormal.y = newRay.y*2.0f - 1.0f;
			//sourceNormal.z = 1.0f - sqrt(sourceNormal.x*sourceNormal.x + sourceNormal.y*sourceNormal.y);
			sourceNormal.z = sqrt(1.0f - sourceNormal.x*sourceNormal.x - sourceNormal.y*sourceNormal.y);

			quantizedRay = glm::vec2(newRay.x * float(HistogramWidth), newRay.y * float(HistogramHeight));

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
					return;


				float theta = atan2(sourceNormal.y, sourceNormal.x);
				float fi = acos(sourceNormal.z);


				//range of atan: -pi to pi
				//range of acos: 0 to pi

				//push all thetas by pi/2 to make the range from 0 to pi
				theta += PI;

				theta = fmod(theta, 2 * PI);
				fi = fmod(fi, 2 * PI);


				float s1 = 2 * PI / HistogramHeight;
				float s2 = (PI / 2) / HistogramHeight;


				glm::ivec2 binIndex = glm::ivec2(theta / s1, fi / s2);
				binIndex = glm::ivec2(std::min(binIndex.x, HistogramHeight - 1), std::min(binIndex.y, HistogramHeight - 1));
				histogramIndexCentral = binIndex.y  * HistogramWidth + binIndex.x;
			}

		}
		else if (binning_mode == 2)
		{
			//longitude/latitude binning
			//get 3d normal
			glm::vec3 sourceNormal;
			sourceNormal.x = newRay.x*2.0f - 1.0f;
			sourceNormal.y = newRay.y*2.0f - 1.0f;
			//sourceNormal.z = 1.0f - sqrt(sourceNormal.x*sourceNormal.x + sourceNormal.y*sourceNormal.y);
			sourceNormal.z = sqrt(1.0f - sourceNormal.x*sourceNormal.x - sourceNormal.y*sourceNormal.y);

			quantizedRay = glm::vec2(newRay.x * float((HistogramWidth)-1), newRay.y * float(HistogramHeight - 1));

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
					return;


				float X = sqrt(2.0f / (1.0 + sourceNormal.z))*sourceNormal.x;
				float Y = sqrt(2.0f / (1.0 + sourceNormal.z))*sourceNormal.y;

				//range of x and y: -sqrt(2) -> sqrt(2)

				X += sqrt(2.0f);
				Y += sqrt(2.0f);

				float s1 = (2.0f*sqrt(2.0f)) / HistogramHeight;    //X is in the range of -2 to 2
				float s2 = (2.0f*sqrt(2.0f)) / HistogramHeight;


				glm::ivec2 binIndex = glm::ivec2(X / s1, Y / s2);
				binIndex = glm::ivec2(std::min(binIndex.x, HistogramHeight - 1), std::min(binIndex.y, HistogramHeight - 1));
				histogramIndexCentral = binIndex.y  * HistogramWidth + binIndex.x;
			}
		}

		//update cpu avg ndf
		if (!miss)
		{
			//invert display of bin for clarity
			glm::ivec2 twodindx = glm::ivec2(histogramIndexCentral%HistogramWidth, histogramIndexCentral / HistogramWidth);
			//twodindx.x = HistogramHeight - twodindx.x;
			twodindx.y = HistogramHeight - twodindx.y;
			histogramIndexCentral = twodindx.y*HistogramWidth + twodindx.x;

			cpuAvgNDF[histogramIndexCentral]++;


			for (int i = 0; i < histogramResolution.x*histogramResolution.y; i++)
			{
				samples += cpuAvgNDF[i];
			}



			for (int i = 0; i < histogramResolution.x*histogramResolution.y; i++)
			{
				if (A[binning_mode][i]>0)
				{
					avgNDF[i] += (cpuAvgNDF[i] / samples);// / A[binning_mode][i];
				}
			}
			std::cout << "Bin number: " << histogramIndexCentral << " has been incremented" << std::endl;
			std::cout << "Total number of samples in avgNDF: " << samples << std::endl;
		}
	}
	else
	{
		cpuAvgNDF.clear();
		cpuAvgNDF.resize(histogramResolution.x*histogramResolution.y, 0.0f);
	}


	//upload avg ndf
	//upload avgNDF_ssbo
	{
		auto Size = size_t(histogramResolution.x*histogramResolution.y);
		auto ssboSize = Size*sizeof(float);
		auto ssbo = avgNDF_ssbo;
		{
			glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo);
			GLvoid* p = glMapBuffer(GL_SHADER_STORAGE_BUFFER, GL_WRITE_ONLY);
			memcpy(p, reinterpret_cast<char*>(&avgNDF[0]), avgNDF.size() * sizeof(*avgNDF.begin()));
			glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
		}
	}

	
}
