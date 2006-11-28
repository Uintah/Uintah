//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
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

#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Thread/Mutex.h>
#include <Core/Util/Assert.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/Environment.h>

#include <iostream>
using std::cerr;
using std::endl;
using std::string;


namespace SCIRun {

bool ShaderProgramARB::init_ = false;
bool ShaderProgramARB::supported_ = false;
bool ShaderProgramARB::non_2_textures_ = false;
int ShaderProgramARB::max_texture_size_1_ = 64;
int ShaderProgramARB::max_texture_size_4_ = 64;
static Mutex ShaderProgramARB_init_Mutex("ShaderProgramARB Init Lock");  


// GLEW SUPPORTED, we can check for shader support sanely.
#ifdef HAVE_GLEW


ShaderProgramARB::ShaderProgramARB(const string& program)
  : type_(0), id_(0), program_(program)
{}

ShaderProgramARB::~ShaderProgramARB ()
{}

bool
ShaderProgramARB::valid()
{
  return shaders_supported() ? glIsProgramARB(id_) : false;
}


void
ShaderProgramARB::init_shaders_supported()
{
  if (!init_)
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
        glewInit();
        glewExperimental = GL_TRUE;
	GLenum err = glGetError();
	if (err != GL_NO_ERROR)
	  fprintf(stderr,"GL error '%s'\n",gluErrorString(err));

#if defined(__sgi)
        max_texture_size_1_ = 256; // TODO: Just a guess, should verify this.
        max_texture_size_4_ = 256; // TODO: Just a guess, should verify this.
#else
        int i;
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
#endif // !sgi

	// Check for non-power-of-two texture support.
        non_2_textures_ = GLEW_ARB_texture_non_power_of_two;

	supported_ = GLEW_ARB_vertex_program && GLEW_ARB_fragment_program;
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
  if (shaders_supported())
  {
    glEnable(type_);
    glGenProgramsARB(1, (GLuint*)&id_);
    glBindProgramARB(type_, id_);
    glProgramStringARB(type_, GL_PROGRAM_FORMAT_ASCII_ARB,
                           program_.length(), program_.c_str());
    if (glGetError() != GL_NO_ERROR)
    {
      int position;
      glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, (GLint*)&position);
      int start = Abs(position);
      for (; start > 0 && program_[start] != '\n'; start--);
      if (program_[start] == '\n') start++;
      int end = position;
      for (; end < ((int)program_.length())-1 && program_[end] != '\n'; end++);
      if (program_[end] == '\n') end--;
      int ss = start;
      int l = 1;
      for (; ss >= 0; ss--) { if (program_[ss] == '\n') l++; }
      string line((char*)(program_.c_str()+start), end-start+1);
      string underline = line;
      for (int i=0; i<end-start+1; i++) underline[i] = '-';
      underline[position-start] = '#';
      glBindProgramARB(type_, 0);
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
  
  return true;
}


void
ShaderProgramARB::destroy ()
{
  if (shaders_supported())
  {
    glDeleteProgramsARB(1, (const GLuint*)&id_);
    id_ = 0;
  }
}

void
ShaderProgramARB::bind ()
{
  if (shaders_supported())
  {
    glEnable(type_);
    glBindProgramARB(type_, id_);
  }
}

void
ShaderProgramARB::release ()
{
  if (shaders_supported())
  {
    glBindProgramARB(type_, 0);
    glDisable(type_);
  }
}

void
ShaderProgramARB::setLocalParam(int i, float x, float y, float z, float w)
{
  if (shaders_supported())
  {
    glProgramLocalParameter4fARB(type_, i, x, y, z, w);
  }
}

VertexProgramARB::VertexProgramARB(const string& program)
  : ShaderProgramARB(program)
{
  type_ = GL_VERTEX_PROGRAM_ARB;
}

VertexProgramARB::~VertexProgramARB()
{
}

FragmentProgramARB::FragmentProgramARB(const string& program)
  : ShaderProgramARB(program)
{
  type_ = GL_FRAGMENT_PROGRAM_ARB;
}

FragmentProgramARB::~FragmentProgramARB()
{
}


#else  // NO GLEW, No shader support. (SGI)


ShaderProgramARB::ShaderProgramARB(const string& program)
  : type_(0), id_(0), program_(program)
{}

ShaderProgramARB::~ShaderProgramARB ()
{}

bool
ShaderProgramARB::valid()
{
  return false;
}


void
ShaderProgramARB::init_shaders_supported()
{
  if (!init_)
  {
    ShaderProgramARB_init_Mutex.lock();
    if (!init_)
    {
      supported_ = false;
      max_texture_size_1_ = 256; // TODO: Just a guess, should verify this.
      max_texture_size_4_ = 256; // TODO: Just a guess, should verify this.
      non_2_textures_ = false;
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
#if defined(__APPLE__) && (defined(__i386__) || defined(__x86_64__))
  return false;
#else
  return non_2_textures_;
#endif
}

bool
ShaderProgramARB::create()
{
  return true;
}


void
ShaderProgramARB::destroy ()
{
}

void
ShaderProgramARB::bind ()
{
}

void
ShaderProgramARB::release ()
{
}

void
ShaderProgramARB::setLocalParam(int i, float x, float y, float z, float w)
{
}

VertexProgramARB::VertexProgramARB(const string& program)
  : ShaderProgramARB(program)
{
}

VertexProgramARB::~VertexProgramARB()
{
}

FragmentProgramARB::FragmentProgramARB(const string& program)
  : ShaderProgramARB(program)
{
}

FragmentProgramARB::~FragmentProgramARB()
{
}

#endif // no shader support.


} // end namespace SCIRun
