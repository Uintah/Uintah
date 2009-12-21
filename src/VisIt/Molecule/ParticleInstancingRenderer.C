#include<cstring>
#include<algorithm>
#include "ParticleInstancingRenderer.h"

static GLboolean CheckExtension( char *extName )
{
    /*
     ** Search for extName in the extensions string.  Use of strstr()
     ** is not sufficient because extension names can be prefixes of
     ** other extension names.  Could use strtok() but the constant
     ** string returned by glGetString can be in read-only memory.
     */

    debug1 << "testing for extension: " << string(extName) << endl;

    char *p = (char *) glGetString(GL_EXTENSIONS);
    char *end;
    int extNameLen;

    extNameLen = strlen(extName);
    end = p + strlen(p);

    while (p < end) {
	int n = strcspn(p, " ");
	if ((extNameLen == n) && (strncmp(extName, p, n) == 0)) {
	    return GL_TRUE;
	}
	p += (n + 1);
    }
    return GL_FALSE;
}

static void gltutCheckErrors(const char* file, int line)
{
    bool errs = false;
    GLenum ret = glGetError();
    if(GL_NO_ERROR != ret)
    {
	debug1 << file << "(" << line << ") : " << /*gluErrorString(ret) <<*/ endl;
    }
}

// #ifdef _DEBUG
    #define CheckOpenGLError() {gltutCheckErrors(__FILE__, __LINE__);}
// #else
//     #define CheckOpenGLError() {}
// #endif

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

static string FormatBytes(size_t bytes)
{
    stringstream str;

    str << bytes << " B";
    if (bytes >= 1024 )
	str << " = " << bytes / 1024 << " KB";
    if (bytes >= 1024 * 1024 )
	str << " = " << bytes / (1024 * 1024) << " MB";
    if (bytes >= 1024 * 1024 * 1024 )
	str << " =  " << bytes / (1024 * 1024 * 1024) << " GB";

    return str.str();
}

static string LoadFile(const string& shader_file)
{
    debug1 << "loading file " << shader_file << endl;
    std::ifstream file(shader_file.c_str());
    std::stringstream str;

    if(file.good())
    {
	str << file.rdbuf();
	return str.str();
    }
    else
    {
	debug1 << "error opening " <<  shader_file << endl;
	return "error";
    }
}

static GLuint LoadShader(const std::string& shader_file, GLenum type)
{
    debug1 << "loading shader " << shader_file << endl;
    GLuint shader = glCreateShader(type);
    CheckOpenGLError();
    std::vector<const char*> source_strings;
    std::string main = LoadFile(shader_file);
    source_strings.push_back(main.c_str());
    glShaderSource(shader,GLsizei(source_strings.size()), &source_strings[0], 0);
    CheckOpenGLError();

    debug1 << "compiling" << endl;
    glCompileShader(shader);
    CheckOpenGLError();
    GLint compile_status;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_status);
    CheckOpenGLError();
    GLint info_log_length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_length);
    CheckOpenGLError();

    if(!compile_status )
    {
	debug1 << "error" << endl;
    }

    if(info_log_length > 0)
    {
	std::string info_log(size_t(info_log_length),' ');

	glGetShaderInfoLog(shader, info_log_length, 0, &info_log[0]);
	CheckOpenGLError();
	debug1 << info_log << endl;
    }

    if(!compile_status )
    {
	return 0;
    }

    return shader;
}

static GLuint LoadProgram(const std::string& vertex_shader_file, \
	const std::string& fragment_shader_file, \
	const std::string& geometry_shader_file = "", \
	GLint vertices_out = 1, \
	GLenum input_type = GL_POINTS, \
	GLenum output_type = GL_POINTS)
{
    GLuint program = glCreateProgram();
    CheckOpenGLError();

    GLuint vertex_shader = LoadShader(vertex_shader_file, GL_VERTEX_SHADER);
    CheckOpenGLError();
    GLuint fragment_shader = LoadShader(fragment_shader_file, GL_FRAGMENT_SHADER);
    CheckOpenGLError();

    glAttachShader(program, vertex_shader);
    CheckOpenGLError();
    glAttachShader(program, fragment_shader);
    CheckOpenGLError();

    if(!geometry_shader_file.empty())
    {
	GLuint geometry_shader = LoadShader(geometry_shader_file, GL_GEOMETRY_SHADER_EXT);
	CheckOpenGLError();
	glAttachShader(program, geometry_shader);
	CheckOpenGLError();

	glProgramParameteriEXT(program, GL_GEOMETRY_VERTICES_OUT_EXT, vertices_out);
	CheckOpenGLError();

	glProgramParameteriEXT(program, GL_GEOMETRY_INPUT_TYPE_EXT, input_type);
	CheckOpenGLError();

	glProgramParameteriEXT(program, GL_GEOMETRY_OUTPUT_TYPE_EXT, output_type);
	CheckOpenGLError();
    }

    debug1 << "linking" << endl;
    glLinkProgram(program);
    CheckOpenGLError();
    GLint link_status;
    glGetProgramiv(program, GL_LINK_STATUS, &link_status);
    CheckOpenGLError();

    GLint info_log_length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &info_log_length);
    CheckOpenGLError();

    if(!link_status)
    {
	debug1 << "error" << endl;
    }

    if(info_log_length > 0)
    {
	std::string info_log(size_t(info_log_length),' ');
	glGetProgramInfoLog(program, info_log_length, 0, &info_log[0]);
	CheckOpenGLError();
	debug1 << info_log << endl;
    }

    if(!link_status)
    {
	return 0;
    }

    return program;
}

void ParticleInstancingRenderer::BuildShaders() {
    debug1 << "building shaders" << endl;
    // program_instancing = LoadProgram("/home/collab/sshankar/visit_shigeru/src_nvd2/plots/Molecule/Instancing.Vertex.glsl", "/home/collab/sshankar/visit_shigeru/src_nvd2/plots/Molecule/Instancing.Fragment.glsl");
    program_instancing = LoadProgram("./Instancing.Vertex.glsl", "./Instancing.Fragment.glsl");
}

static void BuildGridIndices(const bool build_quads, size_t x0, size_t x1, size_t y0, size_t y1, \
	size_t width, size_t height, size_t cache_size, \
	std::vector<unsigned short>& indices)
{
    struct LocalFunctions
    {
	LocalFunctions(size_t width, size_t height, std::vector<unsigned short>& indices):
	    width(width),
	    height(height),
	    indices(indices)
	{
	}

	const size_t Index(const size_t x, const size_t y)
	{
	    // return x * height + y;
	    return (y * width + x);
	}

	void BuildTriangleStrip(const size_t x0, const size_t x1, const size_t y0, const size_t y1)
	{
	    for (size_t y = y0; y < y1 ; ++y)
		for (size_t x = x0; x < x1 ; ++x)
		{
		    EmitTriangle(Index(x + 0, y + 0), Index(x + 0, y + 1), Index(x + 1, y + 0));
		    EmitTriangle(Index(x + 1, y + 0), Index(x + 0, y + 1), Index(x + 1, y + 1));
		}
	}

	void EmitTriangle(size_t i0, size_t i1, size_t i2)
	{
	    indices.push_back(static_cast<unsigned short>(i0));
	    indices.push_back(static_cast<unsigned short>(i1));
	    indices.push_back(static_cast<unsigned short>(i2));
	}
	void PrefetchTriangle(const size_t x0, const size_t x1, const size_t y0)
	{
	    for (size_t x = x0; x < x1 ; ++x)
	    {
		EmitTriangle(Index(x + 0, y0 + 0), Index(x + 0, y0 + 0), Index(x + 1,y0 + 0));
	    }
	}

	void BuildQuadStrip(const size_t x0, const size_t x1, const size_t y0, const size_t y1)
	{
	    for (size_t y = y0; y < y1 ; ++y)
		for (size_t x = x0; x < x1 ; ++x)
		{
		    EmitQuad(Index(x + 0, y + 0), Index(x + 0, y + 1), Index(x + 1, y + 1), Index(x + 1, y + 0));
		}
	}

	void EmitQuad(size_t i0, size_t i1, size_t i2, size_t i3)
	{
	    indices.push_back(static_cast<unsigned short>(i0));
	    indices.push_back(static_cast<unsigned short>(i1));
	    indices.push_back(static_cast<unsigned short>(i2));
	    indices.push_back(static_cast<unsigned short>(i3));
	}
	void PrefetchQuad(const size_t x0, const size_t x1, const size_t y0)
	{
	    for (size_t x = x0; x < x1 ; ++x)
	    {
		EmitQuad(Index(x + 0, y0 + 0), Index(x + 0, y0 + 0), Index(x + 1, y0 + 0), Index(x + 1,y0 + 0));
	    }
	}


	const size_t width;
	const size_t height;
	std::vector<unsigned short>& indices;
    };

    LocalFunctions h(width, height,indices);

    const size_t strip_width = cache_size - 2;

    for(size_t x = 0; x <= x1; x+= strip_width)
    {
	const size_t strip_start = x;
	const size_t strip_end =  std::min((x + strip_width), width - 1);
	const bool prefetch = (2 * (strip_end - strip_start) + 1) > cache_size;

	if(prefetch)
	{
	    if(build_quads)
		h.PrefetchQuad(strip_start, strip_end, 0);
	    else
		h.PrefetchTriangle(strip_start, strip_end, 0);
	}

	if(build_quads)
	    h.BuildQuadStrip(strip_start, strip_end, 0, y1 - 1);
	else
	    h.BuildTriangleStrip(strip_start, strip_end, 0, y1 - 1);
    }
}

static void BuildGridVertices( std::vector<float> &vertices, size_t slices, size_t stacks ) 
{
    for(size_t i = 0; i < slices; ++i)
	for(size_t j = 0; j < stacks; ++j)
	{
	    const float u = (i) / float(slices - 1);
	    const float v = (j) / float(stacks - 1);

	    const float theta = u * M_PI;
	    const float phi = v * 2.0f * M_PI;

	    vertices.push_back(theta);
	    vertices.push_back(phi);
	}
}
    
void ParticleInstancingRenderer::Initialize() {
    // glewInit(); // not sure if this is required
    
    // debug1 << "OpenGL version: " << string((char *)glGetString(GL_VERSION)) << endl;

    // char extnName[128];
    // strcpy(extnName, "GL_EXT_gpu_shader4"); 

    // GLboolean chkExtn = CheckExtension(extnName);
    // if (chkExtn)
    	// debug1 << "extension GL_EXT_gpu_shader4 supported" << endl;
    // else
	// debug1 << "extension GL_EXT_gpu_shader4 not supported" << endl;

    glGetIntegerv(GL_MAX_TEXTURE_BUFFER_SIZE_EXT, &max_texture_buffer_size);
    debug1 << "maximal texture buffer size " << FormatBytes(max_texture_buffer_size) << endl;

    size_t instances_position_radius = size_t(max_texture_buffer_size) / (size_per_instance_position_radius);
    size_t instances_attribute  = size_t(max_texture_buffer_size) / (size_per_instance_attribute);
    size_t max_number_instances = std::min(instances_position_radius, instances_attribute);

    debug1 << "maximal number of instances  " << (max_number_instances) << " = " << max_number_instances / 1000000.0f << " millions";
    debug1 << " consuming " << FormatBytes(max_number_instances * size_per_instance) << " video memory" << endl;

    instanced_batch_size = std::min(max_number_instances / 2, max_instanced_batch_size);
    debug1 << "batch size " << instanced_batch_size << endl;

    GenerateAndBuildTBO();
    BuildQualityLevels();
    BuildSphereGrids();
    BuildShaders();
}

void ParticleInstancingRenderer::SetLiveQualityLevel(int level) {
    live_quality_level = level;
}

void ParticleInstancingRenderer::GenerateAndBuildTBO() {
    debug1 << "building buffers for batched instancing" << endl;

    glGenBuffers(2, vbo_per_instance_data_position_radius_batches);
    glGenTextures(2, tbo_per_instance_data_position_radius_batches);

    glGenBuffers(2, vbo_per_instance_data_attributes_batches);
    glGenTextures(2, tbo_per_instance_data_attributes_batches);

    for(size_t i = 0; i < 2; ++i)
    {
	BuildTextureBufferObject
	    (
	     vbo_per_instance_data_position_radius_batches[i], 
	     tbo_per_instance_data_position_radius_batches[i], 
	     instanced_batch_size * size_per_instance_position_radius, 
	     GL_STREAM_DRAW_ARB,
	     GL_RGBA32F_ARB
	    );

	BuildTextureBufferObject
	    (
	     vbo_per_instance_data_attributes_batches[i],
	     tbo_per_instance_data_attributes_batches[i],
	     instanced_batch_size * size_per_instance_attribute,
	     GL_STREAM_DRAW_ARB,
	     GL_RGBA8
	    );
    }
}

void ParticleInstancingRenderer::BuildQualityLevels() {
    quality_levels.clear();

    // in accordance with VisIt's quality level's
    quality_levels.push_back(QualityLevel( 6,  6));
    quality_levels.push_back(QualityLevel( 8,  8));
    quality_levels.push_back(QualityLevel(10, 10));
    quality_levels.push_back(QualityLevel(24, 24));
}

void ParticleInstancingRenderer::AddInstancesData(double* xyz, double radius, \
	unsigned char* rgb)  {
    sphere_data.AddSphere(xyz, radius, rgb);
}

void ParticleInstancingRenderer::BuildVertexBuffer(const GLuint vbo, const std::vector<float>& vertices)
{
    const size_t vbo_size = vertices.size() * sizeof(float);
    debug1 << "\tbuilding vertex buffer of size " << FormatBytes(vbo_size) << endl;

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    CheckOpenGLError();
    glBufferData(GL_ARRAY_BUFFER, vbo_size, &vertices[0], GL_STATIC_DRAW);
    CheckOpenGLError();
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    CheckOpenGLError();
}

void ParticleInstancingRenderer::BuildIndexBuffer(const GLuint ibo, const std::vector<unsigned short>& indices)
{
    const size_t ibo_size = indices.size() * sizeof(unsigned short);
    debug1 << "\tbuilding index buffer of size " << FormatBytes(ibo_size) << endl;

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
    CheckOpenGLError();
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, ibo_size, &indices[0], GL_STATIC_DRAW);
    CheckOpenGLError();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    CheckOpenGLError();
}

void ParticleInstancingRenderer::BuildTextureBufferObject(const GLuint vbo, const GLuint tbo, size_t tbo_size, \
	GLenum usage, GLenum internal_format)
{
    debug1 << "\tbuilding texture buffer for instance data of size " << FormatBytes(tbo_size) << endl;

    glBindBufferARB(GL_TEXTURE_BUFFER_EXT, vbo);
    CheckOpenGLError();

    glBufferDataARB(GL_TEXTURE_BUFFER_EXT, tbo_size, 0, usage);
    CheckOpenGLError();

    glBindTexture(GL_TEXTURE_BUFFER_EXT, tbo );
    CheckOpenGLError();
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1); // set 1-byte alignment
    CheckOpenGLError();

    glTexBufferEXT(GL_TEXTURE_BUFFER_EXT, internal_format, tbo);
    CheckOpenGLError();

    glBindBufferARB(GL_TEXTURE_BUFFER_EXT, 0);
    CheckOpenGLError();
    glBindTexture(GL_TEXTURE_BUFFER_EXT, 0);
    CheckOpenGLError();
}

void ParticleInstancingRenderer::BuildSphereGrids()
{
    std::vector<QualityLevel> tmp(quality_levels);

    if(shuffle_indices)
    {
	quality_levels.insert(quality_levels.end(),tmp.begin(), tmp.end());
    }

    for(size_t l = 0; l < quality_levels.size(); ++l)
    {
	QualityLevel& level = quality_levels[l];
	std::vector<float> vertices;
	std::vector<unsigned short > indices;
	BuildSphereGrid(vertices, indices, level.Slices, level.Stacks);

	if(shuffle_indices && (l > tmp.size()))
	{
	    std::vector<size_t> shuffle_mask(indices.size()/4);
	    for(size_t i = 0; i < shuffle_mask.size(); ++i)
	    {
		shuffle_mask[i] = i;
	    }
	    std::random_shuffle(shuffle_mask.begin(), shuffle_mask.end());

	    for(size_t i = 0; i < shuffle_mask.size(); ++i)
	    {
		for(size_t k = 0; k < 4; ++k)
		    indices[4 * i + k] = quality_levels[l - tmp.size()].Indices[4 * shuffle_mask[i] + k];
	    }
	}

	glGenBuffers(1, &level.VboGrid);
	BuildVertexBuffer(level.VboGrid, vertices);

	glGenBuffers(1, &level.VboIndices);
	BuildIndexBuffer(level.VboIndices, indices);

	level.IndicesCount = indices.size();
	level.VertexCount = vertices.size() / 2;
	level.Vertices = vertices;
	level.Indices = indices;
    }
}

void ParticleInstancingRenderer::BuildSphereGrid(std::vector<float>& vertices, std::vector<unsigned short>& indices, 
	const size_t _slices, const size_t _stacks)
{
    size_t slices =_slices;
    size_t stacks = _stacks;
    debug1 << "building sphere grid " << slices << " x " << stacks << endl;

    vertices.clear();
    vertices.reserve(2 * slices * stacks); // we need 2 floats per position
    indices.clear();
    indices.reserve(vertices.capacity()); // pre-allocate an estimate of the required space

    BuildGridVertices(vertices, slices, stacks);
    BuildGridIndices(true, 0, slices, 0, stacks, slices, stacks, PostVertexShaderCacheSizeModernGPU, indices);
    debug1 << "\tv " << vertices.size() << "/" <<  vertices.capacity() << endl;
    debug1 << "\ti " << indices.size() << "/" <<  indices.capacity() << endl;

    debug1 << "building sphere grid vertices " << vertices.size()/2 << " triangles " << indices.size()/3 << endl;
}

GLboolean ParticleInstancingRenderer::CopySphereDataToGpuBuffers(const SphereData& spheres, size_t start, size_t count, \
	GLuint vbo_position_radius, GLuint vbo_attributes )
{
    const size_t number_of_spheres = count;

    float* mapped_position_radius = 0;
    unsigned char* mapped_attributes = 0;

    // bind the buffers and get a mapped pointer to the elements
    glBindBufferARB ( GL_TEXTURE_BUFFER_EXT, vbo_position_radius );
    CheckOpenGLError();

    mapped_position_radius = reinterpret_cast<float*>(glMapBufferARB(GL_TEXTURE_BUFFER_EXT, GL_WRITE_ONLY_ARB));
    CheckOpenGLError();

    glBindBufferARB ( GL_TEXTURE_BUFFER_EXT, vbo_attributes);
    CheckOpenGLError();

    mapped_attributes = reinterpret_cast<unsigned char*>(glMapBufferARB(GL_TEXTURE_BUFFER_EXT, GL_WRITE_ONLY_ARB));
    CheckOpenGLError();

    if(!(mapped_position_radius &&  mapped_attributes))
	debug1 << "mapping failed: mapped_attributes " << mapped_attributes << " mapped_position_radius " <<  mapped_position_radius << endl;

    // now fill the buffer, this could be accelerated using OpenMP
    for(size_t i = 0; i < number_of_spheres; ++ i)
    {
	const size_t sphere = start + i;

	mapped_position_radius[4 * i + 0] = spheres.PositionXFloat(sphere);
	mapped_position_radius[4 * i + 1] = spheres.PositionYFloat(sphere);
	mapped_position_radius[4 * i + 2] = spheres.PositionZFloat(sphere);
	mapped_position_radius[4 * i + 3] = spheres.RadiusFloat(sphere);

	// debug1 << mapped_position_radius[4 * i + 0] << " " \
	     << mapped_position_radius[4 * i + 1] << " " \
	     << mapped_position_radius[4 * i + 2] << " " \
	     << mapped_position_radius[4 * i + 3] << endl; 

	mapped_attributes[4 * i + 0] = spheres.AttributeRUChar(sphere);
	mapped_attributes[4 * i + 1] = spheres.AttributeGUChar(sphere);
	mapped_attributes[4 * i + 2] = spheres.AttributeBUChar(sphere);
	mapped_attributes[4 * i + 3] = NULL;
	
	// debug1 << (int)mapped_attributes[4 * i + 0] << " " \
	     << (int)mapped_attributes[4 * i + 1] << " " \
	     << (int)mapped_attributes[4 * i + 2] << " " \
	     << (int)mapped_attributes[4 * i + 3] << endl; 
    }

    // unmap buffers
    const GLboolean unmapped_attributes = glUnmapBufferARB(GL_TEXTURE_BUFFER_EXT);
    CheckOpenGLError();

    glBindBufferARB ( GL_TEXTURE_BUFFER_EXT, vbo_position_radius);
    CheckOpenGLError();

    const GLboolean unmapped_position_radius = glUnmapBufferARB(GL_TEXTURE_BUFFER_EXT);
    CheckOpenGLError();

    glBindBufferARB ( GL_TEXTURE_BUFFER_EXT, 0 );
    CheckOpenGLError();

    if(!(unmapped_attributes &&  unmapped_position_radius))
	debug1 << "unmapping failed: unmapped_attributes" << static_cast<unsigned int>(unmapped_attributes) << " unmapped_position_radius" << static_cast<unsigned int>(unmapped_position_radius) << endl;

    return unmapped_attributes &&  unmapped_position_radius;
}

void ParticleInstancingRenderer::RenderBatchedInstancing()
{
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    CheckOpenGLError();

    // debug1 << "live_quality_level: " << live_quality_level << endl;

    const QualityLevel& level = quality_levels[live_quality_level];

    // debug1 << "program instancing: " << program_instancing << endl;
    // debug1 << "vertex count: " << level.VertexCount << endl;
    // debug1 << "index  count: " << level.IndicesCount << endl;

    glUseProgram(program_instancing);
    CheckOpenGLError();

    glValidateProgram(program_instancing);
    CheckOpenGLError();

    glUniform1i(glGetUniformLocation(program_instancing,"per_instance_data_position_radius"), 0);
    CheckOpenGLError();
    glUniform1i(glGetUniformLocation(program_instancing,"per_instance_data_attribute"), 1);
    CheckOpenGLError();

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, level.VboIndices);
    CheckOpenGLError();
    glBindBuffer(GL_ARRAY_BUFFER, level.VboGrid);
    CheckOpenGLError();
    glVertexPointer(2, GL_FLOAT, size_grid_vertex, reinterpret_cast<char*>(0) + 0);
    CheckOpenGLError();

    glEnableClientState(GL_VERTEX_ARRAY);
    CheckOpenGLError();
    
    const size_t total_number_of_instances = sphere_data.Size();
    const size_t number_of_batches = 1 + (total_number_of_instances - 1) / instanced_batch_size;
    
    // debug1 << "total_number_of_instances: " << total_number_of_instances << endl; 

    for(size_t batch = 0, instances_remaining = total_number_of_instances; batch < number_of_batches; ++batch, instances_remaining -= instanced_batch_size)
    {
	size_t current_buffer = batch % 2;
	const size_t start_instance = batch * instanced_batch_size;
	const size_t instance_count = std::min(instances_remaining,instanced_batch_size);

	// debug1 << "start_instance: " << start_instance << endl;
	// debug1 << "instance_count: " << instance_count << endl;

	CopySphereDataToGpuBuffers
	    (
	     sphere_data,
	     start_instance,
	     instance_count, 
	     vbo_per_instance_data_position_radius_batches[current_buffer],
	     vbo_per_instance_data_attributes_batches[current_buffer] 
	    );

	glActiveTexture(GL_TEXTURE0);
	CheckOpenGLError();
	glBindTexture(GL_TEXTURE_BUFFER_EXT, tbo_per_instance_data_position_radius_batches[current_buffer]);
	CheckOpenGLError();

	glActiveTexture(GL_TEXTURE1);
	CheckOpenGLError();
	glBindTexture(GL_TEXTURE_BUFFER_EXT, tbo_per_instance_data_attributes_batches[current_buffer]);
	CheckOpenGLError();

	glDrawElementsInstancedEXT(GL_QUADS, GLsizei(level.IndicesCount), GL_UNSIGNED_SHORT, reinterpret_cast<char*>(0), GLsizei(instance_count));

	CheckOpenGLError();
    }

    glDisableClientState(GL_VERTEX_ARRAY);
    CheckOpenGLError();

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    CheckOpenGLError();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    CheckOpenGLError();

    glActiveTexture(GL_TEXTURE1);
    CheckOpenGLError();
    glBindTexture(GL_TEXTURE_BUFFER_EXT, 0);
    CheckOpenGLError();

    glActiveTexture(GL_TEXTURE0);
    CheckOpenGLError();	
    glBindTexture(GL_TEXTURE_BUFFER_EXT, 0);
    CheckOpenGLError();

    glUseProgram(0);
    CheckOpenGLError();

    glPopAttrib();
    CheckOpenGLError();
}

void ParticleInstancingRenderer::CleanUp() {
    sphere_data.CleanUp();
}

size_t ParticleInstancingRenderer::Size() {
    return sphere_data.Size();
}

