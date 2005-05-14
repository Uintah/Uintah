/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



/*
 *  PBuffer.cc: Render geometry to a pbuffer using opengl
 *
 *  Written by:
 *   Kurt Zimmerman and Milan Ikits
 *   Department of Computer Science
 *   University of Utah
 *   December 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#include <sci_defs/chromium_defs.h>

#include <sci_gl.h>
#include <sci_glx.h>

#ifdef _WIN32
// get X stuff from tk
#include <tcl.h>
#include <tk.h>
#include <X11/Xlib.h>
#endif

#include <Dataflow/Modules/Render/PBuffer.h>
#include <Core/Util/Assert.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>
#include <Core/Util/Environment.h>

using std::cerr;

namespace SCIRun {

PBuffer::PBuffer( int doubleBuffer /* = GL_FALSE */ ):
  width_(0), height_(0), colorBits_(8),
  doubleBuffer_(doubleBuffer),
  depthBits_(8),
#ifndef _WIN32
  cx_(0),
#endif
#ifdef HAVE_PBUFFER
  fbc_(0),
  pbuffer_(0),
#endif
  dpy_(0)
{}

#ifdef HAVE_PBUFFER

bool
#ifndef _WIN32
PBuffer::create(Display* dpy, int screen, GLXContext sharedcontext,
		int width, int height, int colorBits, int depthBits)
#else
PBuffer::create(Display* dpy, int screen, /*GLXContext sharedcontext,*/
		int width, int height, int colorBits, int depthBits)
#endif
{
  dpy_ = dpy;
  screen_ = screen;
  width_ = width;
  height_ = height;
  colorBits_ = colorBits;

  if (sci_getenv_p("SCIRUN_DISABLE_PBUFFERS"))
  {
    return false;
  }
  
  // Set up a pbuffer associated with dpy
  int minor = 0 , major = 0;
  glXQueryVersion(dpy, &major, &minor);
  if( major >= 1 || (major == 1 &&  minor >1)) { // we can have a pbuffer
    //    cerr<<"We can have a pbuffer!\n";
    int attrib[32];
    int i = 0;
    attrib[i++] = GLX_RED_SIZE; attrib[i++] = colorBits;
    attrib[i++] = GLX_GREEN_SIZE; attrib[i++] = colorBits;
    attrib[i++] = GLX_BLUE_SIZE; attrib[i++] = colorBits;
    attrib[i++] = GLX_ALPHA_SIZE; attrib[i++] = colorBits;  
    attrib[i++] = GLX_DEPTH_SIZE; attrib[i++] = depthBits_;
    attrib[i++] = GLX_DRAWABLE_TYPE; attrib[i++] = GLX_PBUFFER_BIT;
    attrib[i++] = GLX_DOUBLEBUFFER; attrib[i++] = doubleBuffer_;
    attrib[i] = None;
    int nelements;
    fbc_ = glXChooseFBConfig( dpy, screen, attrib, &nelements );
    if( fbc_ == 0 ){
      //cerr<<"Can not configure for Pbuffer\n";
      return false;
    }

    int match = 0, a[8];
    for(int i = 0; i < nelements; i++) {
      glXGetFBConfigAttrib(dpy, fbc_[i],
			   GLX_RED_SIZE, &a[0]);
      glXGetFBConfigAttrib(dpy, fbc_[i],
			   GLX_GREEN_SIZE, &a[1]);
      glXGetFBConfigAttrib(dpy, fbc_[i],
			   GLX_BLUE_SIZE, &a[2]);
      glXGetFBConfigAttrib(dpy, fbc_[i],
			   GLX_ALPHA_SIZE, &a[3]);
      glXGetFBConfigAttrib(dpy, fbc_[i],
			   GLX_DEPTH_SIZE, &a[4]);
      glXGetFBConfigAttrib(dpy, fbc_[i],
			   GLX_ACCUM_RED_SIZE, &a[5]);
      glXGetFBConfigAttrib(dpy, fbc_[i],
			   GLX_ACCUM_GREEN_SIZE, &a[6]);
      glXGetFBConfigAttrib(dpy, fbc_[i],
			   GLX_ACCUM_BLUE_SIZE, &a[7]);
      // printf("r = %d, b = %d, g = %d, a = %d, z = %d, ar = %d, ag = %d, ab = %d\n",
      //	     a[0],a[1],a[2],a[3],a[4], a[5], a[6], a[7]);

      if((a[0] >= 8) && (a[1] >= 8) &&
	 (a[2] >= 8) && (a[3] >= 8) && 
	 (a[4] >= 8) && (a[5] == 0) && (a[6] == 0) && (a[7] == 0) )
      {
	match = i;
// 	printf("fbConfigList[%d] matches the selected attribList\n", i);
	break;
      }
    }


    i = 0;
    attrib[i++] = GLX_PBUFFER_WIDTH; attrib[i++] = width_;
    attrib[i++] = GLX_PBUFFER_HEIGHT; attrib[i++] = height_;
    attrib[i] = None;
    pbuffer_ = glXCreatePbuffer( dpy, fbc_[match], attrib );
    if( pbuffer_ == 0 ) {
      //cerr<<"Cannot create Pbuffer\n";
      return false;
    }

#ifndef _WIN32
    cx_ = glXCreateNewContext( dpy, *fbc_, GLX_RGBA_TYPE, sharedcontext, True);
    if( !cx_ ){
      //cerr<<"Cannot create Pbuffer context\n";
      return false;
    }
// else cerr<<"Pbuffer successfully created\n";
  } else {
    //cerr<<"GLXVersion = "<<major<<"."<<minor<<"\n";
    cx_ = 0;
    return false;
  }
#else
    return false;
  }
#endif
  return true;
} // end create()

void
PBuffer::destroy()
{
#ifndef _WIN32
  if( cx_ ) {
    glXDestroyContext( dpy_, cx_ );
    cx_ = 0;
  }

  if( pbuffer_ ) {
    glXDestroyPbuffer( dpy_, pbuffer_);
    pbuffer_ = 0;
  }
#endif
}

void
PBuffer::makeCurrent()
{
#ifndef _WIN32
  glXMakeCurrent( dpy_, pbuffer_, cx_ );
#endif
}

bool
PBuffer::is_current()
{
#ifndef _WIN32
  return (cx_ == glXGetCurrentContext());
#endif
}

#else // ifdef HAVE_PBUFFER

bool
#ifndef _WIN32
PBuffer::create(Display* /*dpy*/, int /*screen*/, GLXContext /*shared*/,
#else
PBuffer::create(Display* /*dpy*/, int /*screen*/, //GLXContext /*shared*/,
#endif
		int /*width*/, int /*height*/,
		int /*colorBits*/, int /*depthBits*/)
{
  return false;
}

void
PBuffer::destroy()
{
}

void
PBuffer::makeCurrent()
{
  // This better not be called, because create returned false.
  ASSERTFAIL("PBuffer::makeCurrent: HAVE_PBUFFER is not defined");
}

bool
PBuffer::is_current()
{
  return false;
}

#endif // ifdef HAVE_PBUFFER

} // end namespace SCIRun

