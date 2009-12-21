#ifndef ParticleInstancingRenderer_h
#define ParticleInstancingRenderer_h

#include <iostream>
#include <vector>
#include <DebugStream.h>

using namespace std;

#ifndef VTK_IMPLEMENT_MESA_CXX
  #if defined(__APPLE__) && (defined(VTK_USE_CARBON) || defined(VTK_USE_COCOA))
    #include <OpenGL/gl.h>
  #else
    #if defined(_WIN32)
       #include <windows.h>
    #endif
    #include <GL/glew.h>
  #endif
#else
   #include <GL/glew.h>
#endif

const size_t max_instanced_batch_size = 1048576; // maximal batch size, in spheres
const size_t size_per_instance_position_radius = 4 * sizeof(float);
const size_t size_per_instance_attribute =  4 * sizeof(unsigned char); // to hold the color's
const size_t size_per_instance = size_per_instance_position_radius + size_per_instance_attribute;
const size_t size_grid_vertex = 2 * sizeof(float);

enum  PostVertexShaderCacheSize
{
    PostVertexShaderCacheSizeModernGPU = 32,
};

struct QualityLevel
{
    QualityLevel(size_t slices, size_t stacks):
	Slices(slices),
	Stacks(stacks),
	IndicesCount(0),
	VertexCount(0),
	ListComplete(0),
	ListSphere(0),
	ListNested(0),
	VboGrid(0),
	VboIndices(0)
    {
    }

    ~QualityLevel()
    {
	if(0 != ListComplete)
	{
	    debug1 << "deleting ListComplete" << endl;
	    glDeleteLists(ListComplete,1);
	}

	if(0 != ListSphere)
	{
	    debug1 << "deleting ListSphere" << endl;
	    glDeleteLists(ListSphere,1);
	}
	if(0 != ListNested)
	{
	    debug1 << "deleting ListNested" << endl;
	    glDeleteLists(ListNested,1);
	}

	if(0 != VboGrid)
	{
	    debug1 << "deleting VboGrids" << endl;
	    glDeleteBuffers(1, &VboGrid);
	}

	if(0 != VboIndices)
	{
	    debug1 << "deleting VboIndices" << endl;
	    glDeleteBuffers(1, &VboIndices);
	}
    }

    size_t Stacks;
    size_t Slices;

    size_t IndicesCount;
    size_t VertexCount;

    GLuint ListComplete;
    GLuint ListSphere;
    GLuint ListNested;

    GLuint VboGrid;			
    GLuint VboIndices;			

    std::vector<float> Vertices;
    std::vector<unsigned short> Indices;
};

struct SphereData
{
    std::vector<float> Positions;
    std::vector<float> Radii;		
    std::vector<unsigned char> Attributes;

    void AddSphere(double* xyz , \
	           double radius, \
		   unsigned char* rgb)
    {
	Positions.push_back(xyz[0]);
	Positions.push_back(xyz[1]);
	Positions.push_back(xyz[2]);

	Radii.push_back(radius);
	
	Attributes.push_back(rgb[0]);
	Attributes.push_back(rgb[1]);
	Attributes.push_back(rgb[2]);
    }

    void CleanUp() {
	Positions.clear();
	Radii.clear();
	Attributes.clear();
    }

    size_t Size() const
    {
	return /*Attributes.size()*/ Radii.size();
    }

    float RadiusFloat(size_t which) const
    {
	return static_cast<float>(Radii[which]);
    }

    unsigned char AttributeRUChar(size_t which) const
    {
	return static_cast<unsigned char>(Attributes[3 * which + 0]);
    }
    
    unsigned char AttributeGUChar(size_t which) const
    {
	return static_cast<unsigned char>(Attributes[3 * which + 1]);
    }
    
    unsigned char AttributeBUChar(size_t which) const
    {
	return static_cast<unsigned char>(Attributes[3 * which + 2]);
    }

    float PositionXFloat(size_t which) const
    {
	return static_cast<float>(Positions[3 * which + 0]);
    }

    float PositionYFloat(size_t which) const
    {
	return static_cast<float>(Positions[3 * which + 1]);
    }

    float PositionZFloat(size_t which) const
    {
	return static_cast<float>(Positions[3 * which + 2]);
    }
};

class ParticleInstancingRenderer {
  public:
    ParticleInstancingRenderer() {
	live_quality_level = 0;
	max_texture_buffer_size = 0;
	program_instancing = 0;
	instanced_batch_size = 0;
	shuffle_indices = false;
    
	memset(vbo_per_instance_data_position_radius_batches, 0, 2 * sizeof(GLuint)); 
	memset(tbo_per_instance_data_position_radius_batches, 0, 2 * sizeof(GLuint)); 

        memset(vbo_per_instance_data_attributes_batches, 0, 2 * sizeof(GLuint)); 
        memset(tbo_per_instance_data_attributes_batches, 0, 2 * sizeof(GLuint));

	Initialize();
    }

    ~ParticleInstancingRenderer() {};

    void Initialize();
    void SetLiveQualityLevel(int level);
    void GenerateAndBuildTBO();
    void BuildShaders();
    void BuildQualityLevels() ;
    void AddInstancesData(double* xyz, double radius, unsigned char* rgb);
    void BuildVertexBuffer(const GLuint vbo, const std::vector<float>& vertices);
    void BuildIndexBuffer(const GLuint ibo, const std::vector<unsigned short>& indices);

    void BuildTextureBufferObject(const GLuint vbo, const GLuint tbo, size_t tbo_size, GLenum usage,  GLenum internal_format);

    void BuildSphereGrids();
    void BuildSphereGrid(std::vector<float>& vertices, std::vector<unsigned short>& indices, const size_t slices, const size_t stacks);

    GLboolean CopySphereDataToGpuBuffers(const SphereData& spheres,size_t start, size_t count, GLuint vbo_position_radius, GLuint vbo_attributes);

    void CleanUp();
    size_t Size();
    void RenderBatchedInstancing();
  
  private:
    int live_quality_level;
    int max_texture_buffer_size;
    GLuint program_instancing;
    size_t instanced_batch_size; 
    bool shuffle_indices; 
    
    GLuint vbo_per_instance_data_position_radius_batches[2]; 
    GLuint tbo_per_instance_data_position_radius_batches[2]; 

    GLuint vbo_per_instance_data_attributes_batches[2]; 
    GLuint tbo_per_instance_data_attributes_batches[2];
    
    SphereData sphere_data;
    vector<QualityLevel> quality_levels;
};

#endif
