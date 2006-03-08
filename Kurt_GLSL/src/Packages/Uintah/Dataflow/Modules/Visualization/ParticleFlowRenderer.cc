//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : ParticleFlowRenderer.cc
//    Author : Kurt Zimmerman
//    Date   : March 1, 2006


#include <sci_glu.h>

#include <Packages/Uintah/Dataflow/Modules/Visualization/ParticleFlowRenderer.h>
#include <Core/Geom/DrawInfoOpenGL.h>
#include <Core/Util/NotFinished.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/Color.h>


#include <stdlib.h>
#include <stdio.h>
#include <fcntl.h>
#include <iostream>

using namespace Uintah;
using namespace SCIRun;
using std::cerr;

#if !defined(HAVE_GLEW)

#  if !defined(GLX_ARB_get_proc_address) || !defined(GLX_GLXEXT_PROTOTYPES)
      extern "C" void ( * glXGetProcAddressARB (const GLubyte *procName)) (void);
#  endif /* !defined(GLX_ARB_get_proc_address) || !defined(GLX_GLXEXT_PROTOTYPES) */
#  ifdef __APPLE__
#    include <mach-o/dyld.h>
#    include <stdlib.h>
#    include <string.h>

     static void *NSGLGetProcAddress (const GLubyte *name)
     {
       NSSymbol symbol;
       char *symbolName;
       /* prepend a '_' for the Unix C symbol mangling convention */
       symbolName = (char*)malloc(strlen((const char *)name) + 2);
       strcpy(symbolName+1, (const char *)name);
       symbolName[0] = '_';
       symbol = NULL;
       if (NSIsSymbolNameDefined(symbolName))
         symbol = NSLookupAndBindSymbol(symbolName);
       free(symbolName);
       return symbol ? NSAddressOfSymbol(symbol) : NULL;
     }
#    define getProcAddress(x) (NSGLGetProcAddress((const GLubyte*)x))
#  else
#    ifdef _WIN32
#      define getProcAddress(x) (wglGetProcAddress((LPCSTR) x))
#    else
#      define getProcAddress(x) ((*glXGetProcAddressARB)((const GLubyte*)x))
#    endif /* _WIN32 */
#  endif /* APPLE */

typedef void (GLAPIENTRY * PFNGLDISABLEVERTEXATTRIBARRAYPROC) (GLuint index);
typedef void (GLAPIENTRY * PFNGLENABLEVERTEXATTRIBARRAYPROC) (GLuint index);
typedef void (GLAPIENTRY * PFNGLVERTEXATTRIBPOINTERVPROC) (GLuint, GLint, GLenum, GLboolean, GLsizei, const GLvoid *);
typedef GLint (GLAPIENTRY * PFNGLGETUNIFORMLOCATIONPROC) (GLuint program, const GLchar *name);
typedef void (GLAPIENTRY * PFNGLGETSHADERIVPROC) (GLuint shader, GLenum pname, GLint *params);
typedef void (GLAPIENTRY * PFNGLGETSHADERINFOLOGPROC) (GLuint shader, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
typedef void (GLAPIENTRY * PFNGLGETPROGRAMIVPROC) (GLuint program, GLenum pname, GLint *params);
typedef void (GLAPIENTRY * PFNGLGETPROGRAMINFOLOGPROC) (GLuint program, GLsizei bufSize, GLsizei *length, GLchar *infoLog);
typedef void (GLAPIENTRY * PFNGLUNIFORM1FPROC) (GLint location, GLfloat v0);
typedef void (GLAPIENTRY * PFNGLUNIFORM4FPROC) (GLint location, GLfloat v0, GLfloat v1, GLfloat v2, GLfloat v3);
typedef void (GLAPIENTRY * PFNGLBINDATTRIBLOCATIONPROC) (GLuint program, GLuint index, const GLchar *name);
typedef void (GLAPIENTRY * PFNGLATTACHSHADERPROC) (GLuint program, GLuint shader);
typedef void (GLAPIENTRY * PFNGLCOMPILESHADERPROC) (GLuint shader);
typedef GLuint (GLAPIENTRY * PFNGLCREATESHADERPROC) (GLenum type);
typedef void (GLAPIENTRY * PFNGLSHADERSOURCEPROC) (GLuint shader, GLsizei count, const GLchar* *string, const GLint *length);
typedef GLuint (GLAPIENTRY * PFNGLCREATEPROGRAMPROC) (void);
typedef void (GLAPIENTRY * PFNGLLINKPROGRAMPROC) (GLuint program);
typedef void (GLAPIENTRY * PFNGLUSEPROGRAMPROC) (GLuint program);

static PFNGLENABLEVERTEXATTRIBARRAYPROC glEnableVertexAttribArray = 0;
static PFNGLDISABLEVERTEXATTRIBARRAYPROC glDisableVertexAttribArray = 0;
static PFNGLVERTEXATTRIBPOINTERVPROC glVertexAttribPointer = 0;
static PFNGLGETUNIFORMLOCATIONPROC glGetUniformLocation = 0;
static PFNGLGETSHADERIVPROC glGetShaderiv = 0;
static PFNGLGETSHADERINFOLOGPROC glGetShaderInfoLog = 0;
static PFNGLGETPROGRAMIVPROC glGetProgramiv = 0;
static PFNGLGETPROGRAMINFOLOGPROC glGetProgramInfoLog = 0;
static PFNGLUNIFORM4FPROC glUniform4f = 0;
static PFNGLUNIFORM1FPROC glUniform1f = 0;
static PFNGLBINDATTRIBLOCATIONPROC glBindAttribLocation = 0;
static PFNGLATTACHSHADERPROC glAttachShader = 0;
static PFNGLCOMPILESHADERPROC glCompileShader = 0;
static PFNGLCREATESHADERPROC glCreateShader = 0;
static PFNGLSHADERSOURCEPROC glShaderSource = 0;
static PFNGLLINKPROGRAMPROC glLinkProgram = 0;
static PFNGLUSEPROGRAMPROC glUseProgram = 0;
static PFNGLCREATEPROGRAMPROC glCreateProgram = 0;
#endif

int printOglError(char *file, int line);
#define printOpenGLError() printOglError(__FILE__, __LINE__)
int printOglError(char *file, int line)
{
    //
    // Returns 1 if an OpenGL error occurred, 0 otherwise.
    //
    GLenum glErr;
    int    retCode = 0;

    glErr = glGetError();
    while (glErr != GL_NO_ERROR)
    {
        printf("glError 0x%x file %s @ %d: %s\n", 
               glErr, file, line, gluErrorString(glErr));
        retCode = 1;
        glErr = glGetError();
    }
    return retCode;
}


float fXDiff = 206;
float fYDiff = 16;
float fZDiff = 10;
float fScale = 0.25;

// static members
bool ParticleFlowRenderer::functions_initialized_(false);
bool ParticleFlowRenderer::shader_functions_initialized_(false);

ParticleFlowRenderer::ParticleFlowRenderer() :
  GeomObj(),
  initialized_(false),
  cmap_dirty_(true),
  animating_(false),
  reset_(true),
  time_(0.0),
  time_increment_(0.002f),
  particle_time_(0.0),
  cmap_h_(0),
  vf_h_(0),
  di_(0),
  shader_(0),
  array_width_(1),
  array_height_(1),
  verts_(0),
  colors_(0),
  velocities_(0),
  start_times_(0)
{
}

ParticleFlowRenderer::ParticleFlowRenderer(const ParticleFlowRenderer& copy):
  initialized_(copy.initialized_),
  cmap_dirty_(copy.cmap_dirty_),
  animating_(copy.animating_),
  reset_(copy.reset_),
  time_( copy.time_),
  time_increment_(copy.time_increment_),
  particle_time_(copy.particle_time_),
  cmap_h_( copy.cmap_h_),
  vf_h_(copy.vf_h_),
  di_(copy.di_),
  shader_(copy.shader_),
  array_width_(copy.array_width_),
  array_height_(copy.array_height_),
  verts_(copy.verts_),
  colors_(copy.colors_),
  velocities_(copy.velocities_),
  start_times_(copy.start_times_)
{}

ParticleFlowRenderer::~ParticleFlowRenderer()
{
  cmap_h_ = 0;
  vf_h_ = 0;
  delete shader_;
}

GeomObj*
ParticleFlowRenderer::clone()
{
  return new ParticleFlowRenderer(*this);
}

void 
ParticleFlowRenderer::update_colormap( ColorMapHandle cmap )
{
  cmap_h_ = cmap;
  if(cmap_h_ != 0 && vf_h_ != 0 ){
    initialized_ = true;
  }
}

void 
ParticleFlowRenderer::update_vector_field( FieldHandle vfh )
{
  vf_h_ = vfh;
  if(cmap_h_ != 0 && vf_h_ != 0 ){
    initialized_ = true;
  }
  
}

void 
ParticleFlowRenderer::update_time( double time )
{
  time_ = time;
}
 
#ifdef SCI_OPENGL
void 
ParticleFlowRenderer::draw(DrawInfoOpenGL* di, Material* mat, double /* time */)
{
  if(!pre_draw(di, mat, 0)) return;
  di_ = di;
  drawPoints();
  di = 0;
}


void  
ParticleFlowRenderer::createPoints(GLint w, GLint h)
{
	GLfloat *vptr, *cptr, *velptr, *stptr;
	GLfloat i, j;

	if (verts_ != 0) 
          delete verts_;
        if (colors_ != 0)
          delete colors_;
        if (velocities_ != 0)
          delete velocities_;
        if (start_times_ != 0)
          delete start_times_;

	verts_  = scinew GLfloat[w * h * 3 * sizeof(float)];
	colors_ = scinew GLfloat[w * h * 3 * sizeof(float)];
	velocities_ = scinew GLfloat[w * h * 3 * sizeof(float)];
	start_times_ = scinew GLfloat[w * h * sizeof(float)];

	vptr = verts_;
	cptr = colors_;
	velptr = velocities_;
	stptr  = start_times_;

	for (i = 0.5 / w - 0.5; i < 0.5; i = i + 1.0/w)
		for (j = 0.5 / h - 0.5; j < 0.5; j = j + 1.0/h)
		{
			*vptr       = i;
			*(vptr + 1) = 0.0;
			*(vptr + 2) = j;
			vptr += 3;

			*cptr       = ((float) rand() / RAND_MAX) * 0.5 + 0.5;
			*(cptr + 1) = ((float) rand() / RAND_MAX) * 0.5 + 0.5;
			*(cptr + 2) = ((float) rand() / RAND_MAX) * 0.5 + 0.5;
			cptr += 3;

			*velptr       = (((float) rand() / RAND_MAX)) + 3.0;
			*(velptr + 1) =  ((float) rand() / RAND_MAX) * 10.0;
			*(velptr + 2) = (((float) rand() / RAND_MAX)) + 3.0;
			velptr += 3;

			*stptr = ((float) rand() / RAND_MAX) * 10.0;
			stptr++;
		}

	array_width_  = w;
	array_height_ = h;
}


void 
ParticleFlowRenderer::updateAnim()
{
  int location;
  location = shader_->getUniLoc(shader_->ProgramObject, "Time"); 

  particle_time_ += time_increment_;
  if (particle_time_ > 15.0)
    particle_time_ = 0.0;

  glUniform1f(location, particle_time_);

  printOpenGLError();  // Check for OpenGL errors
}

void  
ParticleFlowRenderer::drawPoints()
{	

  if( !functions_initialized_ ){
    if( ((glEnableVertexAttribArray = (PFNGLENABLEVERTEXATTRIBARRAYPROC)
           getProcAddress("glEnableVertexAttribArray")) &&
          (glDisableVertexAttribArray = (PFNGLDISABLEVERTEXATTRIBARRAYPROC)
           getProcAddress("glDisableVertexAttribArray")) &&
          (glVertexAttribPointer = (PFNGLVERTEXATTRIBPOINTERVPROC)
           getProcAddress("glVertexAttribPointer"))))
    {
      functions_initialized_ = true;
      createPoints(100,100);
    } else {
      cerr<<"procs undefined\n";
      return; //nothing drawn
    }
  }

  if( !shader_functions_initialized_){
    if( ParticleFlowShader::build_shader_functions() ){
      shader_functions_initialized_ = true;
    } else {
      cerr<<"shader functions not initialized\n";
      return; // do nothing
    }
  }

  GLchar *vertex_shader_source, *fragment_shader_source;

  if( shader_ == 0 ){
    shader_ = scinew ParticleFlowShader();
    shader_->readShaderSource("particle",&vertex_shader_source, 
                                         &fragment_shader_source);
    int success = shader_->installParticleShaders(vertex_shader_source,
                                                  fragment_shader_source);
    if( !success ){
      cerr<<"Shader installation failed\n";
      delete shader_;
      shader_ = 0;
      return;
    }
  }

//   glMatrixMode(GL_MODELVIEW);
//   glPushMatrix();
  glUseProgram(shader_->ProgramObject);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  if(animating_) updateAnim();

//   glLoadIdentity();
//   glTranslatef(0.0, 0.0, -5.0);
  
//   glRotatef(fYDiff, 1,0,0);
//   glRotatef(fXDiff, 0,1,0);
//   glRotatef(fZDiff, 0,0,1);
//   glScalef(fScale, fScale, fScale);


  GLboolean depth;
  glGetBooleanv(GL_DEPTH_TEST, &depth);
  glDepthFunc(GL_LESS);

  if( !depth ){
    glEnable(GL_DEPTH_TEST);
  }

  
  glPointSize(2.0);

  glVertexPointer(3, GL_FLOAT, 0, verts_);
  glColorPointer(3, GL_FLOAT, 0, colors_);
  glVertexAttribPointer(VELOCITY_ARRAY,  3, GL_FLOAT, GL_FALSE, 0, velocities_);
  glVertexAttribPointer(START_TIME_ARRAY, 1, GL_FLOAT, GL_FALSE, 0, start_times_);

  glEnableClientState(GL_VERTEX_ARRAY);
  glEnableClientState(GL_COLOR_ARRAY);
  glEnableVertexAttribArray(VELOCITY_ARRAY);
  glEnableVertexAttribArray(START_TIME_ARRAY);

  glDrawArrays(GL_POINTS, 0, array_width_ * array_height_);

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);
  glDisableVertexAttribArray(VELOCITY_ARRAY);
  glDisableVertexAttribArray(START_TIME_ARRAY);
  
  if( !depth ){
    glDisable(GL_DEPTH_TEST);
  }
  
  glUseProgram(0);
//   glPopMatrix();
}
#endif


// ************* Persistant stuff needs implementation ************** //

#define PARTICLEFLOWRENDERER_VERSION 1
void 
ParticleFlowRenderer::io(Piostream&)
{
  // nothing for now...
  NOT_FINISHED("ParticleFlowRenderer::io");
}

bool
ParticleFlowRenderer::saveobj(std::ostream&, const string&, GeomSave*)
{
  NOT_FINISHED("ParticleFlowRenderer::saveobj");
  return false;
}
// ****************************************************************** //

bool ParticleFlowShader::shader_functions_built_ = false;

bool
ParticleFlowShader::build_shader_functions()
{
  if(!shader_functions_built_){
    shader_functions_built_ = 
      ((glGetUniformLocation = (PFNGLGETUNIFORMLOCATIONPROC)
        getProcAddress("glGetUniformLocation")) &&
       (glGetShaderiv = (PFNGLGETSHADERIVPROC)
        getProcAddress("glGetShaderiv"))&&
       (glGetShaderInfoLog = (PFNGLGETSHADERINFOLOGPROC)
        getProcAddress("glGetShaderInfoLog"))&&
       (glGetProgramiv = (PFNGLGETPROGRAMIVPROC)
        getProcAddress("glGetProgramiv"))&&
       (glGetProgramInfoLog = ( PFNGLGETPROGRAMINFOLOGPROC)
        getProcAddress("glGetProgramInfoLog"))&&
       (glUniform4f = (PFNGLUNIFORM4FPROC)
        getProcAddress("glUniform4f"))&&
       (glUniform1f = (PFNGLUNIFORM1FPROC)
        getProcAddress("glUniform1f"))&&
       (glBindAttribLocation = (PFNGLBINDATTRIBLOCATIONPROC)
        getProcAddress("glBindAttribLocation"))&&
       (glAttachShader = (PFNGLATTACHSHADERPROC)
        getProcAddress("glAttachShader"))&&
       (glCompileShader = (PFNGLCOMPILESHADERPROC)
        getProcAddress("glCompileShader"))&&
       (glCreateShader = (PFNGLCREATESHADERPROC)
        getProcAddress("glCreateShader"))&&
       (glShaderSource = (PFNGLSHADERSOURCEPROC)
        getProcAddress("glShaderSource"))&&
       (glLinkProgram = (PFNGLLINKPROGRAMPROC)
        getProcAddress("glLinkProgram"))&&
       (glUseProgram = (PFNGLUSEPROGRAMPROC)
        getProcAddress("glUseProgram"))&&
       (glCreateProgram = (PFNGLCREATEPROGRAMPROC)
        getProcAddress("glCreateProgram")));
  }
  return shader_functions_built_;
}

GLint 
ParticleFlowShader::getUniLoc(GLuint program, const GLchar *name)
{
    GLint loc;

    loc = glGetUniformLocation(program, name);

    if (loc == -1)
        printf("No such uniform named \"%s\"\n", name);

    printOpenGLError();  // Check for OpenGL errors
    return loc;
}

void 
ParticleFlowShader::printShaderInfoLog(GLuint shader)
{
    int infologLength = 0;
    int charsWritten  = 0;
    GLchar *infoLog;

    printOpenGLError();  // Check for OpenGL errors

    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infologLength);

    printOpenGLError();  // Check for OpenGL errors

    if (infologLength > 0)
    {
        infoLog = new GLchar[infologLength];
        if (infoLog == NULL)
        {
            printf("ERROR: Could not allocate InfoLog buffer\n");
            exit(1);
        }
        glGetShaderInfoLog(shader, infologLength, &charsWritten, infoLog);
        printf("Shader InfoLog:\n%s\n\n", infoLog);
        delete infoLog;
    }
    printOpenGLError();  // Check for OpenGL errors
}

void 
ParticleFlowShader::printProgramInfoLog(GLuint program)
{
    int infologLength = 0;
    int charsWritten  = 0;
    GLchar *infoLog;

    printOpenGLError();  // Check for OpenGL errors

    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infologLength);

    printOpenGLError();  // Check for OpenGL errors

    if (infologLength > 0)
    {
        infoLog = new GLchar[infologLength];
        if (infoLog == NULL)
        {
            printf("ERROR: Could not allocate InfoLog buffer\n");
            exit(1);
        }
        glGetProgramInfoLog(program, infologLength, &charsWritten, infoLog);
        printf("Program InfoLog:\n%s\n\n", infoLog);
        delete infoLog;
    }
    printOpenGLError();  // Check for OpenGL errors
}

int 
ParticleFlowShader::shaderSize(char *fileName, EShaderType shaderType)
{
    //
    // Returns the size in bytes of the shader fileName.
    // If an error occurred, it returns -1.
    //
    // File name convention:
    //
    // <fileName>.vert
    // <fileName>.frag
    //
    int fd;
    char name[100];
    int count = -1;

    strcpy(name, fileName);

    switch (shaderType)
    {
        case EVertexShader:
            strcat(name, ".vert");
            break;
        case EFragmentShader:
            strcat(name, ".frag");
            break;
        default:
            printf("ERROR: unknown shader file type\n");
            exit(1);
            break;
    }
    //
    // Open the file, seek to the end to find its length
    //
#ifdef WIN32 /*[*/
    fd = _open(name, _O_RDONLY);
    if (fd != -1)
    {
        count = _lseek(fd, 0, SEEK_END) + 1;
        _close(fd);
    }
#else /*][*/
    fd = open(name, O_RDONLY);
    if (fd != -1)
    {
        count = lseek(fd, 0, SEEK_END) + 1;
        close(fd);
    }
#endif /*]*/

    return count;
}

int 
ParticleFlowShader::readShader(char *fileName, EShaderType shaderType,
                               char *shaderText, int size)
{
    //
    // Reads a shader from the supplied file and returns the shader in the
    // arrays passed in. Returns 1 if successful, 0 if an error occurred.
    // The parameter size is an upper limit of the amount of bytes to read.
    // It is ok for it to be too big.
    //
    FILE *fh;
    char name[100];
    int count;

    strcpy(name, fileName);

    switch (shaderType)
    {
        case EVertexShader:
            strcat(name, ".vert");
            break;
        case EFragmentShader:
            strcat(name, ".frag");
            break;
        default:
            printf("ERROR: unknown shader file type\n");
            exit(1);
            break;
    }

    //
    // Open the file
    //
    fh = fopen(name, "r");
    if (!fh)
        return -1;

    //
    // Get the shader from a file.
    //
    fseek(fh, 0, SEEK_SET);
    count = (int) fread(shaderText, 1, size, fh);
    shaderText[count] = '\0';

    if (ferror(fh))
        count = 0;

    fclose(fh);
    return count;
}

int
ParticleFlowShader::readShaderSource(char *fileName, GLchar **vertexShader,
                                     GLchar **fragmentShader)
{
    int vSize, fSize;

    //
    // Allocate memory to hold the source of our shaders.
    //
    vSize = shaderSize(fileName, EVertexShader);
    fSize = shaderSize(fileName, EFragmentShader);

    if ((vSize == -1) || (fSize == -1))
    {
        printf("Cannot determine size of the shader %s\n", fileName);
        return 0;
    }

    *vertexShader = (GLchar *) malloc(vSize);
    *fragmentShader = (GLchar *) malloc(fSize);

    //
    // Read the source code
    //
    if (!readShader(fileName, EVertexShader, *vertexShader, vSize))
    {
        printf("Cannot read the file %s.vert\n", fileName);
        return 0;
    }

    if (!readShader(fileName, EFragmentShader, *fragmentShader, fSize))
    {
        printf("Cannot read the file %s.frag\n", fileName);
        return 0;
    }

    return 1;
}

int 
ParticleFlowShader::installParticleShaders(const GLchar *particleVertex,
                                           const GLchar *particleFragment)
{

  GLint       vertCompiled, fragCompiled;    // status values
  GLint       linked;

  // Create a vertex shader object and a fragment shader object

  VertexShaderObject = glCreateShader(GL_VERTEX_SHADER);
  FragmentShaderObject = glCreateShader(GL_FRAGMENT_SHADER);

  // Load source code strings into shaders

  glShaderSource(VertexShaderObject, 1, &particleVertex, NULL);
  glShaderSource(FragmentShaderObject, 1, &particleFragment, NULL);

  // Compile the particle vertex shader, and print out
  // the compiler log file.

  glCompileShader(VertexShaderObject);
  printOpenGLError();  // Check for OpenGL errors
  glGetShaderiv(VertexShaderObject, GL_COMPILE_STATUS, &vertCompiled);
  printShaderInfoLog(VertexShaderObject);

  // Compile the particle vertex shader, and print out
  // the compiler log file.

  glCompileShader(FragmentShaderObject);
  printOpenGLError();  // Check for OpenGL errors
  glGetShaderiv(FragmentShaderObject, GL_COMPILE_STATUS, &fragCompiled);
  printShaderInfoLog(FragmentShaderObject);

  if (!vertCompiled || !fragCompiled)
    return 0;

  // Create a program object and attach the two compiled shaders

  ProgramObject = glCreateProgram();
  glAttachShader(ProgramObject, VertexShaderObject);
  glAttachShader(ProgramObject, FragmentShaderObject);

  // Bind generic attribute indices to attribute variable names

  glBindAttribLocation(ProgramObject, VELOCITY_ARRAY, "Velocity");
  glBindAttribLocation(ProgramObject, START_TIME_ARRAY, "StartTime");

  // Link the program object and print out the info log

  glLinkProgram(ProgramObject);
  printOpenGLError();  // Check for OpenGL errors
  glGetProgramiv(ProgramObject, GL_LINK_STATUS, &linked);
  printProgramInfoLog(ProgramObject);

  if (!linked)
    return 0;

  // Install program object as part of current state

  glUseProgram(ProgramObject);

  // Set up initial uniform values

  glUniform4f(getUniLoc(ProgramObject, "Background"), 0.0, 0.0, 0.0, 1.0);
  printOpenGLError();  // Check for OpenGL errors
  glUniform1f(getUniLoc(ProgramObject, "Time"), -5.0);
  printOpenGLError();  // Check for OpenGL errors

  glUseProgram(0);
  return 1;
}
