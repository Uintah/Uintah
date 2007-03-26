
#include <Core/Malloc/Allocator.h>
#include <Packages/Kurt/Dataflow/Modules/Visualization/GLSLShader.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sci_glu.h>

using namespace Kurt;
using namespace SCIRun;


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



GLint 
GLSLShader::getUniLoc(GLuint program, const GLchar *name)
{
    GLint loc;

    loc = glGetUniformLocation(program, name);

    if (loc == -1)
        printf("No such uniform named \"%s\"\n", name);

    printOpenGLError();  // Check for OpenGL errors
    return loc;
}

void 
GLSLShader::printShaderInfoLog(GLuint shader)
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
GLSLShader::printProgramInfoLog(GLuint program)
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
GLSLShader::shaderSize(char *fileName, EShaderType shaderType)
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
GLSLShader::readShader(char *fileName, EShaderType shaderType,
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
GLSLShader::read_shader_source(char *fileName, GLchar **vertexShader,
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

//     *vertexShader = (GLchar *) malloc(vSize);
//     *fragmentShader = (GLchar *) malloc(fSize);
    *vertexShader = scinew GLchar[vSize];
    *fragmentShader = scinew GLchar[fSize];

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

void
GLSLShader::install_shaders(const GLchar *particleVertex,
                                    const GLchar *particleFragment)
{
   // Create a vertex shader object and a fragment shader object

  vertex_shader_object_ = glCreateShader(GL_VERTEX_SHADER);
  fragment_shader_object_ = glCreateShader(GL_FRAGMENT_SHADER);

  // Load source code strings into shaders

  glShaderSource(vertex_shader_object_, 1, &particleVertex, NULL);
  glShaderSource(fragment_shader_object_, 1, &particleFragment, NULL);

  printOpenGLError();  // Check for OpenGL errors

} 
int
GLSLShader::compile_shaders()
{

  GLint       vertCompiled, fragCompiled;    // status values

  // Compile the particle vertex shader, and print out
  // the compiler log file.

  glCompileShader(vertex_shader_object_);
  printOpenGLError();  // Check for OpenGL errors
  glGetShaderiv(vertex_shader_object_, GL_COMPILE_STATUS, &vertCompiled);
  printShaderInfoLog(vertex_shader_object_);

  // Compile the particle vertex shader, and print out
  // the compiler log file.

  glCompileShader(fragment_shader_object_);
  printOpenGLError();  // Check for OpenGL errors
  glGetShaderiv(fragment_shader_object_, GL_COMPILE_STATUS, &fragCompiled);
  printShaderInfoLog(fragment_shader_object_);

  if (!vertCompiled || !fragCompiled)
    return 0;

  return 1;
}

void 
GLSLShader::create_program()
{
  // Create a program object and attach the two compiled shaders

  program_object_ = glCreateProgram();
}

void
GLSLShader::attach_objects()
{
  glAttachShader(program_object_, vertex_shader_object_);
  glAttachShader(program_object_, fragment_shader_object_);
}

void 
GLSLShader::bind_attribute_location(GLuint index, const GLchar *name)
{
  // Bind generic attribute indices to attribute variable names
  glBindAttribLocation(program_object_, index, name);
}

int
GLSLShader::link_program()
{
  GLint       linked;

  // Link the program object and print out the info log

  glLinkProgram(program_object_);
  printOpenGLError();  // Check for OpenGL errors
  glGetProgramiv(program_object_, GL_LINK_STATUS, &linked);
  printProgramInfoLog(program_object_);

  if (!linked)
    return 0;

  return 1;
}


//   // Install program object as part of current state

//   glUseProgram(program_object_);


void
GLSLShader::initialize_uniform( GLchar *name, GLuint count,
                                        GLboolean transpose, GLfloat *mat)
{
    // Install program object as part of current state

  glUseProgram(program_object_);

  // Set up initial uniform values
  glUniformMatrix4fv(getUniLoc(program_object_, name ), count, transpose, mat);
  printOpenGLError();  // Check for OpenGL errors
  glUseProgram(0);
}

void 
GLSLShader::initialize_uniform( GLchar *name , GLfloat u0, 
                                        GLfloat u1, GLfloat u2, GLfloat u3)
{
  // Install program object as part of current state

  glUseProgram(program_object_);

  // Set up initial uniform values
  glUniform4f(getUniLoc(program_object_, name ), u0, u1, u2, u3);
  printOpenGLError();  // Check for OpenGL errors
  glUseProgram(0);
}

void 
GLSLShader::initialize_uniform( GLchar *name , GLfloat u0, 
                                        GLfloat u1, GLfloat u2)
{
  // Install program object as part of current state

  glUseProgram(program_object_);

  // Set up initial uniform values
  glUniform3f(getUniLoc(program_object_, name ), u0, u1, u2);
  printOpenGLError();  // Check for OpenGL errors
  glUseProgram(0);
}

void 
GLSLShader::initialize_uniform( GLchar *name , GLfloat u0, GLfloat u1)
{
  // Install program object as part of current state

  glUseProgram(program_object_);

  // Set up initial uniform values
  glUniform2f(getUniLoc(program_object_, name ), u0, u1);
  printOpenGLError();  // Check for OpenGL errors
  glUseProgram(0);
}

void 
GLSLShader::initialize_uniform( GLchar *name , GLfloat u0 )
{
  // Install program object as part of current state

  glUseProgram(program_object_);

  // Set up initial uniform values
  glUniform1f(getUniLoc(program_object_, name ), u0);
  printOpenGLError();  // Check for OpenGL errors
  glUseProgram(0);
}
void 
GLSLShader::initialize_uniform( GLchar *name , GLint u0 )
{
  // Install program object as part of current state

  glUseProgram(program_object_);

  // Set up initial uniform values
  glUniform1i(getUniLoc(program_object_, name ), u0);
  printOpenGLError();  // Check for OpenGL errors
  glUseProgram(0);
}
