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

#include <sci_defs.h>

#include <Dataflow/Modules/Render/PBuffer.h>
#include <Core/Util/Assert.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

using std::cerr;

namespace SCIRun {

PBuffer::PBuffer( int doubleBuffer /* = GL_FALSE */ ):
  width_(0), height_(0), colorBits_(8),
  doubleBuffer_(doubleBuffer),
  depthBits_(8),
  valid_(false),
  cx_(0),
#ifdef HAVE_PBUFFER
  fbc_(0),
  pbuffer_(0),
#endif
  dpy_(0)
{}

#ifdef HAVE_PBUFFER

bool
PBuffer::create(Display* dpy, int screen,
		int width, int height, int colorBits, int depthBits)
{
  dpy_ = dpy;
  screen_ = screen;
  width_ = width;
  height_ = height;
  colorBits_ = colorBits;
  
  // Set up a pbuffer associated with dpy
  int minor = 0 , major = 0;
#ifndef HAVE_CHROMIUM
  glXQueryVersion(dpy, &major, &minor);
#endif
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
#ifndef HAVE_CHROMIUM
    fbc_ = glXChooseFBConfig( dpy, screen, attrib, &nelements );
#endif
    if( fbc_ == 0 ){
      //cerr<<"Can not configure for Pbuffer\n";
      return false;
    }

    int match = 0, a[8];
    for(int i = 0; i < nelements; i++) {
#ifndef HAVE_CHROMIUM
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
#endif
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
#ifndef HAVE_CHROMIUM
    pbuffer_ = glXCreatePbuffer( dpy, fbc_[match], attrib );
#endif
    if( pbuffer_ == 0 ) {
      //cerr<<"Cannot create Pbuffer\n";
      return false;
    }

#ifndef HAVE_CHROMIUM
    cx_ = glXCreateNewContext( dpy, *fbc_, GLX_RGBA_TYPE, NULL, True);
#endif
    if( !cx_ ){
      //cerr<<"Cannot create Pbuffer context\n";
      return false;
    }
// else cerr<<"Pbuffer successfully created\n";
    valid_ = true;
  } else {
    //cerr<<"GLXVersion = "<<major<<"."<<minor<<"\n";
    cx_ = 0;
    return false;
  }

  return true;
} // end create()

void
PBuffer::destroy()
{
  if( cx_ ) {
#ifndef HAVE_CHROMIUM
    glXDestroyContext( dpy_, cx_ );
#endif
    cx_ = 0;
  }

  if( pbuffer_ ) {
#ifndef HAVE_CHROMIUM
    glXDestroyPbuffer( dpy_, pbuffer_);
#endif
    pbuffer_ = 0;
  }
  valid_ = false;
}

void
PBuffer::makeCurrent()
{
  if( valid_ ) {
#ifndef HAVE_CHROMIUM
    glXMakeCurrent( dpy_, pbuffer_, cx_ );
#endif
  }
}

bool
PBuffer::is_current()
{
#ifdef HAVE_CHROMIUM
  return false;
#else
  return (cx_ == glXGetCurrentContext());
#endif
}

#else // ifdef HAVE_PBUFFER

bool
PBuffer::create(Display* /*dpy*/, int /*screen*/, int /*width*/,
                int /*height*/, int /*colorBits*/, int /*depthBits*/)
{
  return false;
}

void
PBuffer::destroy()
{
  valid_ = false;
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

