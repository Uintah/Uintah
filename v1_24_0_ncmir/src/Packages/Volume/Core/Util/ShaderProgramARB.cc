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

#include <Packages/Volume/Core/Util/ShaderProgramARB.h>
//#include <Core/Exceptions/InternalError.h>
#include <Core/Util/DebugStream.h>

#include <iostream>
using std::cerr;
using std::endl;
using std::string;
//using SCIRun::InternalError;

#ifndef HAVE_GLEW

#ifndef GL_ARB_vertex_program

#define GL_VERTEX_PROGRAM_ARB 0x8620
#define GL_PROGRAM_ERROR_POSITION_ARB 0x864B
#define GL_PROGRAM_FORMAT_ASCII_ARB 0x8875

typedef void (GLAPIENTRY * PFNGLGENPROGRAMSARBPROC) (GLsizei n, GLuint* programs);
typedef void (GLAPIENTRY * PFNGLDELETEPROGRAMSARBPROC) (GLsizei n, const GLuint* programs);
typedef void (GLAPIENTRY * PFNGLBINDPROGRAMARBPROC) (GLenum target, GLuint program);
typedef void (GLAPIENTRY * PFNGLPROGRAMSTRINGARBPROC) (GLenum target, GLenum format, GLsizei len, const void* string);
typedef GLboolean (GLAPIENTRY * PFNGLISPROGRAMARBPROC) (GLuint program);
typedef void (GLAPIENTRY * PFNGLPROGRAMLOCALPARAMETER4FARBPROC) (GLenum target, GLuint index, GLfloat x, GLfloat y, GLfloat z, GLfloat w);

#endif /* GL_ARB_vertex_program */

#ifndef GL_ARB_fragment_program

#define GL_FRAGMENT_PROGRAM_ARB 0x8804

#endif /* GL_ARB_fragment_program */

#if !defined(GLX_ARB_get_proc_address) || !defined(GLX_GLXEXT_PROTOTYPES)

extern "C" void ( * glXGetProcAddressARB (const GLubyte *procName)) (void);

#endif /* GLX_ARB_get_proc_address */

#ifdef __APPLE__

#include <mach-o/dyld.h>
#include <stdlib.h>
#include <string.h>

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

#define getProcAddress(x) (NSGLGetProcAddress((const GLubyte*)x))

#else

#define getProcAddress(x) ((*glXGetProcAddressARB)((const GLubyte*)x))

#endif

static PFNGLGENPROGRAMSARBPROC glGenProgramsARB = 0;
static PFNGLDELETEPROGRAMSARBPROC glDeleteProgramsARB = 0;
static PFNGLBINDPROGRAMARBPROC glBindProgramARB = 0;
static PFNGLPROGRAMSTRINGARBPROC glProgramStringARB = 0;
static PFNGLISPROGRAMARBPROC glIsProgramARB = 0;
static PFNGLPROGRAMLOCALPARAMETER4FARBPROC glProgramLocalParameter4fARB = 0;

#endif /* HAVE_GLEW */

namespace Volume {

static SCIRun::DebugStream dbg("ShaderProgramARB", false);

static bool mInit = false;
static bool mSupported = false;

ShaderProgramARB::ShaderProgramARB(const string& program)
  : mType(0), mId(0), mProgram(program)
{}

ShaderProgramARB::~ShaderProgramARB ()
{}

bool
ShaderProgramARB::valid()
{
  return mSupported ? glIsProgramARB(mId) : false;
}

bool
ShaderProgramARB::create()
{
  if(!mInit) {
#ifdef HAVE_GLEW
    if (!GLEW_ARB_vertex_program || !GLEW_ARB_fragment_program) {
      mSupported = false;
      //SCI_THROW(InternalError("GL_ARB_fragment_program is not supported."));
    } else {
      mSupported = true;
    }
#else       
    if (!gluCheckExtension((const GLubyte*)"GL_ARB_vertex_program", glGetString(GL_EXTENSIONS)) ||
        !gluCheckExtension((const GLubyte*)"GL_ARB_fragment_program", glGetString(GL_EXTENSIONS))) {
      mSupported = false;
      //SCI_THROW(InternalError("GL_ARB_fragment_program is not supported."));
    } else {
      mSupported = true;
    }
    bool fail = !mSupported;
    fail = fail
      || (glGenProgramsARB = (PFNGLGENPROGRAMSARBPROC)getProcAddress("glGenProgramsARB")) == 0;
    fail = fail
      || (glDeleteProgramsARB = (PFNGLDELETEPROGRAMSARBPROC)getProcAddress("glDeleteProgramsARB")) == 0;
    fail = fail
      || (glBindProgramARB = (PFNGLBINDPROGRAMARBPROC)getProcAddress("glBindProgramARB")) == 0;
    fail = fail
      || (glProgramStringARB = (PFNGLPROGRAMSTRINGARBPROC)getProcAddress("glProgramStringARB")) == 0;
    fail = fail
      || (glIsProgramARB = (PFNGLISPROGRAMARBPROC)getProcAddress("glIsProgramARB")) == 0;
    fail = fail
      || (glProgramLocalParameter4fARB = (PFNGLPROGRAMLOCALPARAMETER4FARBPROC)getProcAddress("glProgramLocalParameter4fARB")) == 0;
    if(fail) {
      mSupported = false;
      cerr << "GL_ARB_fragment_program is not supported." << endl;
      //SCI_THROW(InternalError("GL_ARB_fragment_program is not supported."));
    } else {
      mSupported = true;
    }
#endif
    mInit = true;
  }

  if(mSupported) {
    dbg << mProgram << endl;
    glGenProgramsARB(1, &mId);
    glBindProgramARB(mType, mId);
    glProgramStringARB(mType, GL_PROGRAM_FORMAT_ASCII_ARB,
                       mProgram.length(), mProgram.c_str());
    if (glGetError() != GL_NO_ERROR)
    {
      int position;
      glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &position);
      int start = position;
      for (; start > 0 && mProgram[start] != '\n'; start--);
      if (mProgram[start] == '\n') start++;
      uint end = position;
      for (; end < mProgram.length()-1 && mProgram[end] != '\n'; end++);
      if (mProgram[end] == '\n') end--;
      int ss = start;
      int l = 1;
      for (; ss >= 0; ss--) { if (mProgram[ss] == '\n') l++; }
      string line((char*)(mProgram.c_str()+start), end-start+1);
      string underline = line;
      for (uint i=0; i<end-start+1; i++) underline[i] = '-';
      underline[position-start] = '#';
      glBindProgramARB(mType, 0);
      switch(mType) {
      case GL_VERTEX_PROGRAM_ARB:
        cerr << "Vertex" << endl;
        break;
      case GL_FRAGMENT_PROGRAM_ARB:
        cerr << "Fragment" << endl;
        break;
      default:
        break;
      }
      cerr << "program error at line " << l << ", character "
           << position-start << endl << endl << line << endl
           << underline << endl;
      return true;
    }
    return false;
  }
  
  return true;
}

void
ShaderProgramARB::destroy ()
{
  if(mSupported) {
    glDeleteProgramsARB(1, &mId);
    mId = 0;
  }
}

void
ShaderProgramARB::bind ()
{
  if(mSupported) {
    glEnable(mType);
    glBindProgramARB(mType, mId);
  }
}

void
ShaderProgramARB::release ()
{
  if(mSupported) {
    glBindProgramARB(mType, 0);
    glDisable(mType);
  }
}

void
ShaderProgramARB::enable ()
{
  if(mSupported) {
    glEnable(mType);
  }
}

void
ShaderProgramARB::disable ()
{
  if(mSupported) {
    glDisable(mType);
  }
}

void
ShaderProgramARB::makeCurrent ()
{
  if(mSupported) {
    glBindProgramARB(mType, mId);
  }
}

void
ShaderProgramARB::setLocalParam(int i, float x, float y, float z, float w)
{
  if(mSupported) {
    glProgramLocalParameter4fARB(mType, i, x, y, z, w);
  }    
}

VertexProgramARB::VertexProgramARB(const string& program)
  : ShaderProgramARB(program)
{
  mType = GL_VERTEX_PROGRAM_ARB;
}

VertexProgramARB::~VertexProgramARB()
{}

FragmentProgramARB::FragmentProgramARB(const string& program)
  : ShaderProgramARB(program)
{
  mType = GL_FRAGMENT_PROGRAM_ARB;
}

FragmentProgramARB::~FragmentProgramARB()
{}

} // end namespace Volume
