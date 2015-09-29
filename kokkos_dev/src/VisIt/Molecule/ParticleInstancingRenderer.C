#include <cstring>
#include <algorithm>
#include <math.h>
#include "ParticleInstancingRenderer.h"



static GLboolean CheckExtension( const char *extName )
{
    /*
     ** Search for extName in the extensions string.  Use of strstr()
     ** is not sufficient because extension names can be prefixes of
     ** other extension names.  Could use strtok() but the constant
     ** string returned by glGetString can be in read-only memory.
     */

    char *p = (char *) glGetString(GL_EXTENSIONS);
    char *end;
    int extNameLen;

    extNameLen = strlen(extName);
    end = p + strlen(p);

    while (p < end) {
        int n = strcspn(p, " ");
        if ((extNameLen == n) && (strncmp(extName, p, n) == 0))
            return GL_TRUE;
        p += (n + 1);
    }
    return GL_FALSE;
}
 

static void gltutCheckErrors(const char* file, int line)
{
    bool errs = false;
    GLenum ret = glGetError();
    if(GL_NO_ERROR != ret)
        cerr << file << "(" << line << ") : " << /*gluErrorString(ret) << */endl;
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




class SphereGeometryVBO {
public:
    SphereGeometryVBO(size_t slices, size_t stacks);
    ~SphereGeometryVBO();

    void Initialize();

    size_t GetNumVertices() const;
    size_t GetNumIndices() const;
    GLuint GetVBOVertices() const;
    GLuint GetVBOIndices() const;

private:

    static void BuildSphereGrid(std::vector<float>& vertices, std::vector<unsigned short>& indices, const size_t slices, const size_t stacks);

    size_t stacks;
    size_t slices;

    size_t num_vertices;
    size_t num_indices;

    GLuint vbo_vertices;
    GLuint vbo_indices;
};




SphereGeometryVBO::SphereGeometryVBO(size_t pslices, size_t pstacks):
	slices(pslices),
	stacks(pstacks),
	num_vertices(0),
	num_indices(0),
	vbo_vertices(0),
	vbo_indices(0)
{
}

SphereGeometryVBO::~SphereGeometryVBO() {
	if(0 != vbo_vertices)
        glDeleteBuffers(1, &vbo_vertices);

	if(0 != vbo_indices)
	    glDeleteBuffers(1, &vbo_indices);
}


void SphereGeometryVBO::Initialize() {

	std::vector<float> vertices;
	std::vector<unsigned short > indices;
	BuildSphereGrid(vertices, indices, slices, stacks);

	glGenBuffers(1, &vbo_vertices);
    CheckOpenGLError();
    glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices);
    CheckOpenGLError();
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), &vertices[0], GL_STATIC_DRAW);
    CheckOpenGLError();
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    CheckOpenGLError();


	glGenBuffers(1, &vbo_indices);
    CheckOpenGLError();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vbo_indices);
    CheckOpenGLError();
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned short), &indices[0], GL_STATIC_DRAW);
    CheckOpenGLError();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    CheckOpenGLError();

	num_indices = indices.size();
	num_vertices = vertices.size() / 2;
}


size_t SphereGeometryVBO::GetNumVertices() const
{
    return num_vertices;
}

size_t SphereGeometryVBO::GetNumIndices() const
{
    return num_indices;
}

GLuint SphereGeometryVBO::GetVBOVertices() const
{
    return vbo_vertices;
}

GLuint SphereGeometryVBO::GetVBOIndices() const
{
    return vbo_indices;
}


void SphereGeometryVBO::BuildSphereGrid(std::vector<float>& vertices, std::vector<unsigned short>& indices, 
                                        const size_t slices, const size_t stacks)
{
    vertices.clear();
    vertices.reserve(2 * slices * (stacks+1)); // we need 2 floats per position
    indices.clear();
    indices.reserve(vertices.capacity()); // pre-allocate an estimate of the required space


    // create the grid of theta/phi vertices
    for(size_t j = 0; j <= stacks; ++j)
    {
        for(size_t i = 0; i < slices; ++i)
        {
            float theta = (i * 2.0f * M_PI) / slices;
            float phi = (j * M_PI) / stacks;
            vertices.push_back(theta);
            vertices.push_back(phi);
        }
    }

    // create quad strip connecting them
    for (int y=0; y<stacks; y++)
    {
        for (int x=0; x<=slices; x++)
        {
            indices.push_back((y+0)*slices+(x%slices));
            indices.push_back((y+1)*slices+(x%slices));
        }

        // add degenerate quad to move to next stack
        if (y!=stacks-1) {
            indices.push_back((y+0)*slices);
            indices.push_back((y+0)*slices);
            indices.push_back((y+1)*slices);
            indices.push_back((y+1)*slices);
        }
    }
}



ParticleInstancingRenderer::ParticleInstancingRenderer() {
    is_initialized = false;
    extensions_supported = false;
	quality_level = 0;
	program_instancing = 0;
	instanced_batch_size = 0;
    
	memset(tbo_position_radius_batches, 0, 2 * sizeof(GLuint)); 
	memset(tex_position_radius_batches, 0, 2 * sizeof(GLuint)); 

    memset(tbo_color_batches, 0, 2 * sizeof(GLuint)); 
    memset(tex_color_batches, 0, 2 * sizeof(GLuint));
}

ParticleInstancingRenderer::~ParticleInstancingRenderer() {
    glDeleteBuffers(2, tbo_position_radius_batches);
    glDeleteTextures(2, tex_position_radius_batches);

    glDeleteBuffers(2, tbo_color_batches);
    glDeleteTextures(2, tex_color_batches);

    for (size_t i=0; i<sphere_geometry_vbos.size(); ++i)
        delete sphere_geometry_vbos[i];
}


void ParticleInstancingRenderer::BuildShaders() {
    debug1 << "building shaders" << endl;
    // program_instancing = LoadProgram("/home/collab/sshankar/visit_shigeru/src_nvd2/plots/Molecule/Instancing.Vertex.glsl", "/home/collab/sshankar/visit_shigeru/src_nvd2/plots/Molecule/Instancing.Fragment.glsl");
    program_instancing = LoadProgram("./Instancing.Vertex.glsl", "./Instancing.Fragment.glsl");
}


bool ParticleInstancingRenderer::IsSupported() {
    Initialize();
    return extensions_supported;
}


void ParticleInstancingRenderer::Initialize() {

    // Initialize can be called at any time - don't actually initialize
    // more than once.
    if (is_initialized)
        return;

    is_initialized = true;
    extensions_supported = false;

    // check for the extensions that are required
    GLboolean shader4_supported = CheckExtension("GL_EXT_gpu_shader4");
    if (shader4_supported)
        debug2 << "ParticleInstancingRenderer: Extension GL_EXT_gpu_shader4 supported" << endl;
    else
        debug2 << "ParticleInstancingRenderer: Extension GL_EXT_gpu_shader4 not supported" << endl;

	
    GLboolean tbo_supported = CheckExtension("GL_EXT_texture_buffer_object");
    if (tbo_supported)
        debug2 << "ParticleInstancingRenderer: Extension GL_EXT_texture_buffer_object supported" << endl;
    else
        debug2 << "ParticleInstancingRenderer: Extension GL_EXT_texture_buffer_object not supported" << endl;


    extensions_supported = (shader4_supported && tbo_supported);
	if (extensions_supported) {
	    debug1 << "ParticleInstancingRenderer: Necessary extensions supported, "
               << "using the new Molecule plot implementation." << endl;
	}
	else
    {
	    debug1 << "ParticleInstancingRenderer: Necessary extensions not supported, "
               << "using the old Molecule plot implementation." << endl;
	}

    // don't do any more if the extensions aren't supported
    if (!extensions_supported)
        return;



    // 
    GLint max_texture_buffer_size;
    glGetIntegerv(GL_MAX_TEXTURE_BUFFER_SIZE_EXT, &max_texture_buffer_size);
    debug1 << "maximal texture buffer size " << FormatBytes(max_texture_buffer_size) << endl;

    size_t instances_position_radius = size_t(max_texture_buffer_size) / (4*sizeof(float));
    size_t instances_color  = size_t(max_texture_buffer_size) / (4*sizeof(unsigned char));
    instanced_batch_size = std::min(instances_position_radius, instances_color);


    debug1 << "ParticleInstancingRenderer: Max number of instances " 
           << instanced_batch_size << " = "
           << instanced_batch_size / 1000000.0f << " million" << endl;

    GenerateAndBuildTBO();
    BuildSphereGeometryVBOs();
    BuildShaders();
}
    

void ParticleInstancingRenderer::SetQualityLevel(int level) {
    quality_level = level;
}



void ParticleInstancingRenderer::GenerateAndBuildTBO() {
    debug1 << "building buffers for batched instancing" << endl;

    glGenBuffers(2, tbo_position_radius_batches);
    CheckOpenGLError();
    glGenTextures(2, tex_position_radius_batches);
    CheckOpenGLError();

    glGenBuffers(2, tbo_color_batches);
    CheckOpenGLError();
    glGenTextures(2, tex_color_batches);
    CheckOpenGLError();

    for(size_t i = 0; i < 2; ++i)
    {
        BuildTBO(tbo_position_radius_batches[i], 
                 tex_position_radius_batches[i], 
                 instanced_batch_size * 4 * sizeof(float), 
                 GL_DYNAMIC_DRAW_ARB,
                 GL_RGBA32F_ARB);

        BuildTBO(tbo_color_batches[i],
                 tex_color_batches[i],
                 instanced_batch_size * 4 * sizeof(unsigned char),
                 GL_DYNAMIC_DRAW_ARB,
                 GL_RGBA8);
    }
}

void ParticleInstancingRenderer::BuildSphereGeometryVBOs() {
    sphere_geometry_vbos.clear();

    // in accordance with VisIt's quality level's
    sphere_geometry_vbos.push_back(new SphereGeometryVBO( 6,  3));
    sphere_geometry_vbos.push_back(new SphereGeometryVBO(12,  6));
    sphere_geometry_vbos.push_back(new SphereGeometryVBO(24, 12));
    sphere_geometry_vbos.push_back(new SphereGeometryVBO(48, 24));

    for (int i=0; i<sphere_geometry_vbos.size(); ++i)
        sphere_geometry_vbos[i]->Initialize();
}


void ParticleInstancingRenderer::BuildTBO(const GLuint tbo, const GLuint tex, size_t tbo_size,  \
                                          GLenum usage, GLenum internal_format)
{
    debug1 << "\tbuilding texture buffer for instance data of size " << FormatBytes(tbo_size) << endl;

    glBindBufferARB(GL_TEXTURE_BUFFER_ARB, tbo);
    CheckOpenGLError();

    glBufferDataARB(GL_TEXTURE_BUFFER_ARB, tbo_size, 0, usage);
    CheckOpenGLError();

    glBindTexture(GL_TEXTURE_BUFFER_ARB, tex );
    CheckOpenGLError();

    glTexBufferARB(GL_TEXTURE_BUFFER_ARB, internal_format, tbo);
    CheckOpenGLError();

    // clear texture / bufer bindings
    glBindBufferARB(GL_TEXTURE_BUFFER_ARB, 0);
    CheckOpenGLError();
    glBindTexture(GL_TEXTURE_BUFFER_ARB, 0);
    CheckOpenGLError();
}



GLboolean ParticleInstancingRenderer::CopyParticleDataToGpuBuffers(size_t start, size_t count, \
                                                                   GLuint tbo_position_radius, GLuint tex_position_radius, GLuint tbo_color, GLuint tex_color )
{
    const size_t number_of_particles = count;

    float* mapped_position_radius = 0;
    unsigned char* mapped_color = 0;


    // bind the buffers and get a mapped pointer to the elements
    glBindBufferARB ( GL_TEXTURE_BUFFER_ARB, tbo_position_radius );
    CheckOpenGLError();

    // rebind the texture to the buffer
    // it's not clear why this is needed, but if we don't do it 
    // occasionally no objects get rendered.
    glBindTexture(GL_TEXTURE_BUFFER_ARB, tex_position_radius);
    CheckOpenGLError();
    glTexBufferARB(GL_TEXTURE_BUFFER_ARB, GL_RGBA32F_ARB, tbo_position_radius);
    CheckOpenGLError();
    glBindTexture(GL_TEXTURE_BUFFER_ARB, 0);
    CheckOpenGLError();

//#define GLMAPBUFFER
#ifdef GLMAPBUFFER
    mapped_position_radius = reinterpret_cast<float*>(glMapBufferARB(GL_TEXTURE_BUFFER_ARB, GL_WRITE_ONLY));
    CheckOpenGLError();

    if(!mapped_position_radius)
        cerr << " mapped_position_radius null " << endl;
#else
    glBufferData(GL_TEXTURE_BUFFER_ARB, count*4*sizeof(float), &particle_position_radius[start], GL_DYNAMIC_DRAW);
    CheckOpenGLError();
#endif


    // bind the buffers and get a mapped pointer to the elements
    glBindBufferARB ( GL_TEXTURE_BUFFER_ARB, tbo_color);
    CheckOpenGLError();

    // rebind the texture to the buffer
    // it's not clear why this is needed, but if we don't do it 
    // occasionally no objects get rendered.
    glBindTexture(GL_TEXTURE_BUFFER_ARB, tex_color);
    CheckOpenGLError();
    glTexBufferARB(GL_TEXTURE_BUFFER_ARB, GL_RGBA8, tbo_color);
    CheckOpenGLError();
    glBindTexture(GL_TEXTURE_BUFFER_ARB, 0);
    CheckOpenGLError();

#ifdef GLMAPBUFFER
    mapped_color = reinterpret_cast<unsigned char*>(glMapBufferARB(GL_TEXTURE_BUFFER_ARB, GL_WRITE_ONLY));
    CheckOpenGLError();

    if(!mapped_color)
        cerr << "mapping failed: mapped_color null" << endl;
#else
    glBufferData(GL_TEXTURE_BUFFER_ARB, count*4*sizeof(unsigned char), &particle_color[start], GL_DYNAMIC_DRAW);
    CheckOpenGLError();
#endif

#ifdef GLMAPBUFFER

    // now fill the buffer, this could be accelerated using OpenMP
    for(size_t i = 0; i < number_of_particles; ++ i)
    {
        const size_t p = start + i;

        mapped_position_radius[4 * i + 0] = particle_position_radius[p*4+0];
        mapped_position_radius[4 * i + 1] = particle_position_radius[p*4+1];
        mapped_position_radius[4 * i + 2] = particle_position_radius[p*4+2];
        mapped_position_radius[4 * i + 3] = particle_position_radius[p*4+3];

        mapped_color[4 * i + 0] = particle_color[p*4+0];
        mapped_color[4 * i + 1] = particle_color[p*4+1];
        mapped_color[4 * i + 2] = particle_color[p*4+2];
        mapped_color[4 * i + 3] = particle_color[p*4+3];
    }


    // unmap buffers
    const GLboolean unmapped_color = glUnmapBufferARB(GL_TEXTURE_BUFFER_ARB);
    CheckOpenGLError();

    glBindBufferARB(GL_TEXTURE_BUFFER_ARB, tbo_position_radius);
    CheckOpenGLError();

    const GLboolean unmapped_position_radius = glUnmapBufferARB(GL_TEXTURE_BUFFER_ARB);
    CheckOpenGLError();

#endif
    glBindBufferARB(GL_TEXTURE_BUFFER_ARB, 0);
    CheckOpenGLError();

    /*
    if(!(unmapped_color==GL_TRUE &&  unmapped_position_radius==GL_TRUE))
        cerr << "unmapping failed: unmapped_color" << static_cast<unsigned int>(unmapped_color) << " unmapped_position_radius" << static_cast<unsigned int>(unmapped_position_radius) << endl;

    return unmapped_color &&  unmapped_position_radius;
    */
    return true;
}



void ParticleInstancingRenderer::Render()
{

    glPushAttrib(GL_ALL_ATTRIB_BITS);
    CheckOpenGLError();

    const SphereGeometryVBO& level = *sphere_geometry_vbos[quality_level];

    glUseProgram(program_instancing);
    CheckOpenGLError();

    glValidateProgram(program_instancing);
    CheckOpenGLError();


    glUniform1i(glGetUniformLocation(program_instancing,"per_instance_data_position_radius"), 0);
    CheckOpenGLError();
    glUniform1i(glGetUniformLocation(program_instancing,"per_instance_data_attribute"), 1);
    CheckOpenGLError();


    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, level.GetVBOIndices());
    CheckOpenGLError();
    glBindBuffer(GL_ARRAY_BUFFER, level.GetVBOVertices());
    CheckOpenGLError();
    glVertexPointer(2, GL_FLOAT, 0, 0);
    CheckOpenGLError();

    glEnableClientState(GL_VERTEX_ARRAY);
    CheckOpenGLError();

    
    const size_t total_number_of_instances = NumParticles();
    const size_t number_of_batches = 1 + (total_number_of_instances - 1) / instanced_batch_size;

    for(size_t batch = 0, instances_remaining = total_number_of_instances; batch < number_of_batches; ++batch, instances_remaining -= instanced_batch_size)
    {
        size_t current_buffer = batch % 2;
        const size_t start_instance = batch * instanced_batch_size;
        const size_t instance_count = std::min(instances_remaining,instanced_batch_size);

        CopyParticleDataToGpuBuffers(start_instance,
                                     instance_count, 
                                     tbo_position_radius_batches[current_buffer],
                                     tex_position_radius_batches[current_buffer],
                                     tbo_color_batches[current_buffer], 
                                     tex_color_batches[current_buffer]);


        glActiveTexture(GL_TEXTURE0);
        CheckOpenGLError();
        glBindTexture(GL_TEXTURE_BUFFER_ARB, tex_position_radius_batches[current_buffer]);
        CheckOpenGLError();

        glActiveTexture(GL_TEXTURE1);
        CheckOpenGLError();
        glBindTexture(GL_TEXTURE_BUFFER_ARB, tex_color_batches[current_buffer]);
        CheckOpenGLError();

        glDrawElementsInstancedARB(GL_QUAD_STRIP, GLsizei(level.GetNumIndices()), GL_UNSIGNED_SHORT, 0, GLsizei(instance_count));
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
    glBindTexture(GL_TEXTURE_BUFFER_ARB, 0);
    CheckOpenGLError();

    glActiveTexture(GL_TEXTURE0);
    CheckOpenGLError();	
    glBindTexture(GL_TEXTURE_BUFFER_ARB, 0);
    CheckOpenGLError();

    glUseProgram(0);
    CheckOpenGLError();

    glPopAttrib();
    CheckOpenGLError();
}


void ParticleInstancingRenderer::ClearParticles() {

    particle_position_radius.clear();
    particle_color.clear();
}


size_t ParticleInstancingRenderer::NumParticles() const {
    // 4 entries for each particle
    return particle_position_radius.size()/4;
}


void ParticleInstancingRenderer::AddParticle(const double* xyz,
                                             const double radius,
                                             const unsigned char* rgb)
{
    particle_position_radius.push_back(xyz[0]);
    particle_position_radius.push_back(xyz[1]);
    particle_position_radius.push_back(xyz[2]);
    particle_position_radius.push_back(radius);

    particle_color.push_back(rgb[0]);
    particle_color.push_back(rgb[1]);
    particle_color.push_back(rgb[2]);
    particle_color.push_back(0);
}
