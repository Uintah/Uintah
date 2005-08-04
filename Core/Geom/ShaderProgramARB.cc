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
//    File   : ShaderProgramARB.cc
//    Author : Milan Ikits
//    Date   : Wed Jul  7 23:21:33 2004

#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>

#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Thread/Mutex.h>
#include <Core/Geom/TkOpenGLContext.h>
#include <Core/Util/Assert.h>
#include <Core/Util/Environment.h>

#include <iostream>
using std::cerr;
using std::endl;
using std::string;

#ifdef _WIN32
#define GL_PROXY_TEXTURE_3D 0x8070
typedef void (GLAPIENTRY * PFNGLTEXIMAGE3DPROC) (GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const GLvoid *pixels);
static PFNGLTEXIMAGE3DPROC glTexImage3D = 0;
#define GL_ARB_fragment_program 1
#define GL_FRAGMENT_PROGRAM_ARB 0x8804
typedef unsigned int uint;
#endif


#if defined(GL_ARB_fragment_program)
#  ifndef GL_ARB_vertex_program
#     define GL_VERTEX_PROGRAM_ARB 0x8620
#     define GL_PROGRAM_ERROR_POSITION_ARB 0x864B
#     define GL_PROGRAM_FORMAT_ASCII_ARB 0x8875
#  endif /* GL_ARB_vertex_program */
#  ifndef GL_ARB_fragment_program
#    define GL_FRAGMENT_PROGRAM_ARB 0x8804
#  endif /* GL_ARB_fragment_program */

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
#    endif // _WIN32
#  endif /* APPLE */


typedef void (GLAPIENTRY * SCIPFNGLGENPROGRAMSARBPROC) (GLsizei n, GLuint* programs);
typedef void (GLAPIENTRY * SCIPFNGLDELETEPROGRAMSARBPROC) (GLsizei n, const GLuint* programs);
typedef void (GLAPIENTRY * SCIPFNGLBINDPROGRAMARBPROC) (GLenum target, GLuint program);
typedef void (GLAPIENTRY * SCIPFNGLPROGRAMSTRINGARBPROC) (GLenum target, GLenum format, GLsizei len, const void* string);
typedef GLboolean (GLAPIENTRY * SCIPFNGLISPROGRAMARBPROC) (GLuint program);
typedef void (GLAPIENTRY * SCIPFNGLPROGRAMLOCALPARAMETER4FARBPROC) (GLenum target, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);

static SCIPFNGLGENPROGRAMSARBPROC glGenProgramsARB_SCI = 0;
static SCIPFNGLDELETEPROGRAMSARBPROC glDeleteProgramsARB_SCI = 0;
static SCIPFNGLBINDPROGRAMARBPROC glBindProgramARB_SCI = 0;
static SCIPFNGLPROGRAMSTRINGARBPROC glProgramStringARB_SCI = 0;
static SCIPFNGLISPROGRAMARBPROC glIsProgramARB_SCI = 0;
static SCIPFNGLPROGRAMLOCALPARAMETER4FARBPROC glProgramLocalParameter4fARB_SCI = 0;
#endif /* GL_ARB_fragment_program */


#ifdef _WIN32
/* ----------------------- WGL_ARB_extensions_string ----------------------- */

#ifndef WGL_ARB_extensions_string
#define WGL_ARB_extensions_string 1

typedef const char* (WINAPI * PFNWGLGETEXTENSIONSSTRINGARBPROC) (HDC hdc);


#endif /* WGL_ARB_extensions_string */

static PFNWGLGETEXTENSIONSSTRINGARBPROC wglGetExtensionsStringARB = 0;

bool WGLisExtensionSupported(const char *extension)
{
  const size_t extlen = strlen(extension);
  const char *supported = NULL;
  
  // Try To Use wglGetExtensionStringARB On Current DC, If Possible
  if (!wglGetExtensionsStringARB)
    wglGetExtensionsStringARB = 
      (PFNWGLGETEXTENSIONSSTRINGARBPROC)wglGetProcAddress("wglGetExtensionsStringARB");
  
  if (wglGetExtensionsStringARB)
    supported = wglGetExtensionsStringARB(wglGetCurrentDC());
  
  if (supported) {

    for (const char* p = supported; ; p++)
      {
	// Advance p Up To The Next Possible Match
	p = strstr(p, extension);
	
	if (p == NULL) {
	  break;
	}
	
	// Make Sure That Match Is At The Start Of The String Or That
	// The Previous Char Is A Space, Or Else We Could Accidentally
	// Match "wglFunkywglExtension" With "wglExtension"
	
	// Also, Make Sure That The Following Character Is Space Or NULL
	// Or Else "wglExtensionTwo" Might Match "wglExtension"
	if ((p==supported || p[-1]==' ') && (p[extlen]=='\0' || p[extlen]==' '))
	  {
	    return true;  // Match
	  }
      }
  }

  // If That Failed, Try Standard Opengl Extensions String
    supported = (char*)glGetString(GL_EXTENSIONS);
  
  // If That Failed Too, Must Be No Extensions Supported
  if (supported == NULL)
    return false;
  
  // Begin Examination At Start Of String, Increment By 1 On False Match
  for (const char* p = supported; ; p++)
    {
      // Advance p Up To The Next Possible Match
      p = strstr(p, extension);
      
      if (p == NULL) {
	return false;						// No Match
      }
      
      // Make Sure That Match Is At The Start Of The String Or That
      // The Previous Char Is A Space, Or Else We Could Accidentally
      // Match "wglFunkywglExtension" With "wglExtension"
      
      // Also, Make Sure That The Following Character Is Space Or NULL
      // Or Else "wglExtensionTwo" Might Match "wglExtension"
      if ((p==supported || p[-1]==' ') && (p[extlen]=='\0' || p[extlen]==' ')) {
	return true;						// Match
      }
    }
}
#endif


namespace SCIRun {

bool ShaderProgramARB::init_ = false;
bool ShaderProgramARB::supported_ = false;
bool ShaderProgramARB::non_2_textures_ = false;
int ShaderProgramARB::max_texture_size_1_ = 64;
int ShaderProgramARB::max_texture_size_4_ = 64;
static Mutex ShaderProgramARB_init_Mutex("ShaderProgramARB Init Lock");  

ShaderProgramARB::ShaderProgramARB(const string& program)
  : type_(0), id_(0), program_(program)
{}

ShaderProgramARB::~ShaderProgramARB ()
{}

bool
ShaderProgramARB::valid()
{
#if defined(GL_ARB_fragment_program)
  return shaders_supported() ? glIsProgramARB_SCI(id_) : false;
#else
  return false;
#endif
}


void
ShaderProgramARB::init_shaders_supported()
{
  if(!init_)
  {
    ShaderProgramARB_init_Mutex.lock();
    if (!init_)
    {
      if (sci_getenv_p("SCIRUN_DISABLE_SHADERS") ||
          sci_getenv_p("SCIRUN_NOGUI"))
      {
	supported_ = false;
      }
      else
      {
	// Create a test context.

	TkOpenGLContext *context =
	  new TkOpenGLContext(".testforshadersupport", 0, 0, 0);

	context->make_current();
	GLenum err = glGetError();
	if (err != GL_NO_ERROR)
	  fprintf(stderr,"GL error '%s'\n",gluErrorString(err));

#if defined(__sgi)
        max_texture_size_1_ = 256; // TODO: Just a guess, should verify this.
        max_texture_size_4_ = 256; // TODO: Just a guess, should verify this.
#else
        int i;

#ifdef _WIN32
	glTexImage3D = (PFNGLTEXIMAGE3DPROC)wglGetProcAddress("glTexImage3D");
#endif
	err = glGetError();
	if (err != GL_NO_ERROR)
	  fprintf(stderr,"GL error '%s'\n",gluErrorString(err));
        for (i = 128; i < 130000; i*=2)
        {
          glTexImage3D(GL_PROXY_TEXTURE_3D, 0, GL_LUMINANCE, i, i, i, 0,
                       GL_LUMINANCE, GL_UNSIGNED_BYTE, NULL);

          GLint width;
	  err = glGetError();
	  if (err != GL_NO_ERROR)
	    fprintf(stderr,"After Tex3D call GL error '%s'\n",gluErrorString(err));

          glGetTexLevelParameteriv(GL_PROXY_TEXTURE_3D, 0,
                                   GL_TEXTURE_WIDTH, &width);
	  err = glGetError();
	  if (err != GL_NO_ERROR)
	    fprintf(stderr,"After TexLevelParam GL error '%s'\n",gluErrorString(err));

          if (width == 0)
          {
            i /= 2;
            break;
          }
        }
	err = glGetError();
	if (err != GL_NO_ERROR)
	  fprintf(stderr,"GL error '%s'\n",gluErrorString(err));
        max_texture_size_1_ = i;

        for (i = 128; i < 130000; i*=2)
        {
          glTexImage3D(GL_PROXY_TEXTURE_3D, 0, GL_RGBA, i, i, i, 0,
                       GL_RGBA, GL_UNSIGNED_BYTE, NULL);

          GLint width;
          glGetTexLevelParameteriv(GL_PROXY_TEXTURE_3D, 0,
                                   GL_TEXTURE_WIDTH, &width);
          if (width == 0)
          {
            i /= 2;
            break;
          }
        }
        max_texture_size_4_ = i;
#endif

	// Check for non-power-of-two texture support.
#ifndef _WIN32
  non_2_textures_ = 
    gluCheckExtension((const GLubyte*)"GL_ARB_texture_non_power_of_two", 
			    glGetString(GL_EXTENSIONS));
#else
  non_2_textures_ =
    WGLisExtensionSupported("GL_ARB_texture_non_power_of_two");
#endif

#if defined(GL_ARB_fragment_program)

	supported_ =

#ifndef _WIN32
	  gluCheckExtension((const GLubyte*)"GL_ARB_vertex_program", 
			    glGetString(GL_EXTENSIONS)) &&
	  gluCheckExtension((const GLubyte*)"GL_ARB_fragment_program", 
			    glGetString(GL_EXTENSIONS)) &&
#else
	  WGLisExtensionSupported("GL_ARB_vertex_program") &&
	  WGLisExtensionSupported("GL_ARB_fragment_program") &&

#endif
	  (glGenProgramsARB_SCI = (SCIPFNGLGENPROGRAMSARBPROC)
	   getProcAddress("glGenProgramsARB")) &&
	  (glDeleteProgramsARB_SCI = (SCIPFNGLDELETEPROGRAMSARBPROC)
	   getProcAddress("glDeleteProgramsARB")) &&
	  (glBindProgramARB_SCI = (SCIPFNGLBINDPROGRAMARBPROC) 
	   getProcAddress("glBindProgramARB")) &&
	  (glProgramStringARB_SCI = (SCIPFNGLPROGRAMSTRINGARBPROC)
	   getProcAddress("glProgramStringARB")) &&
	  (glIsProgramARB_SCI = (SCIPFNGLISPROGRAMARBPROC)
	   getProcAddress("glIsProgramARB")) &&
	  (glProgramLocalParameter4fARB_SCI = 
	   (SCIPFNGLPROGRAMLOCALPARAMETER4FARBPROC)
	   getProcAddress("glProgramLocalParameter4fARB"));
#else
	supported_ = false;
#endif
	delete context;
      }
      init_ = true;
    }
    ShaderProgramARB_init_Mutex.unlock();
  }
}
  

bool
ShaderProgramARB::shaders_supported()
{
  ASSERTMSG(init_, "shaders_supported called before init_shaders_supported.");
  return supported_;
}


int
ShaderProgramARB::max_texture_size_1()
{
  return max_texture_size_1_;
}

int
ShaderProgramARB::max_texture_size_4()
{
  return max_texture_size_4_;
}

bool
ShaderProgramARB::texture_non_power_of_two()
{
  return non_2_textures_;
}

bool
ShaderProgramARB::create()
{
#if defined(GL_ARB_fragment_program)
  if(shaders_supported()) {
    glGenProgramsARB_SCI(1, &id_);
    glBindProgramARB_SCI(type_, id_);
    glProgramStringARB_SCI(type_, GL_PROGRAM_FORMAT_ASCII_ARB,
                           program_.length(), program_.c_str());
    if (glGetError() != GL_NO_ERROR)
    {
      int position;
      glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &position);
      int start = position;
      for (; start > 0 && program_[start] != '\n'; start--);
      if (program_[start] == '\n') start++;
      uint end = position;
      for (; end < program_.length()-1 && program_[end] != '\n'; end++);
      if (program_[end] == '\n') end--;
      int ss = start;
      int l = 1;
      for (; ss >= 0; ss--) { if (program_[ss] == '\n') l++; }
      string line((char*)(program_.c_str()+start), end-start+1);
      string underline = line;
      for (uint i=0; i<end-start+1; i++) underline[i] = '-';
      underline[position-start] = '#';
      glBindProgramARB_SCI(type_, 0);
      switch(type_) {
      case GL_VERTEX_PROGRAM_ARB:
        cerr << "Vertex ";
        break;
      case GL_FRAGMENT_PROGRAM_ARB:
        cerr << "Fragment ";
        break;
      default:
        break;
      }
      cerr << "Program error at line " << l << ", character "
           << position-start << ":" << endl << line << endl
           << underline << endl << endl
	   << "Entire Program Listing:\n" << program_ << endl;
      return true;
    }
    return false;
  }
#endif
  
  return true;
}


void
ShaderProgramARB::destroy ()
{
#if defined(GL_ARB_fragment_program)
  if(shaders_supported()) {
    glDeleteProgramsARB_SCI(1, &id_);
    id_ = 0;
  }
#endif
}

void
ShaderProgramARB::bind ()
{
#if defined(GL_ARB_fragment_program)
  if(shaders_supported()) {
    glEnable(type_);
    glBindProgramARB_SCI(type_, id_);
  }
#endif
}

void
ShaderProgramARB::release ()
{
#if defined(GL_ARB_fragment_program)
  if(shaders_supported()) {
    glBindProgramARB_SCI(type_, 0);
    glDisable(type_);
  }
#endif
}

void
ShaderProgramARB::enable ()
{
#if defined(GL_ARB_fragment_program)
  if(shaders_supported()) {
    glEnable(type_);
  }
#endif
}

void
ShaderProgramARB::disable ()
{
#if defined(GL_ARB_fragment_program)
  if(shaders_supported()) {
    glDisable(type_);
  }
#endif
}

void
ShaderProgramARB::makeCurrent ()
{
#if defined(GL_ARB_fragment_program)
  if(shaders_supported()) {
    glBindProgramARB_SCI(type_, id_);
  }
#endif
}

void
ShaderProgramARB::setLocalParam(int i, float x, float y, float z, float w)
{
#if defined(GL_ARB_fragment_program)
  if(shaders_supported()) {
    glProgramLocalParameter4fARB_SCI(type_, i, x, y, z, w);
  }
#endif
}

VertexProgramARB::VertexProgramARB(const string& program)
  : ShaderProgramARB(program)
{
#if defined(GL_ARB_fragment_program)
  type_ = GL_VERTEX_PROGRAM_ARB;
#endif
}

VertexProgramARB::~VertexProgramARB()
{}

FragmentProgramARB::FragmentProgramARB(const string& program)
  : ShaderProgramARB(program)
{
#if defined(GL_ARB_fragment_program)
  type_ = GL_FRAGMENT_PROGRAM_ARB;
#endif
}

FragmentProgramARB::~FragmentProgramARB()
{}

} // end namespace SCIRun
