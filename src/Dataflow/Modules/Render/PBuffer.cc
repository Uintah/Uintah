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
#include <Dataflow/Modules/Render/PBuffer.h>

#ifdef _WIN32
// get X stuff from tk
#include <tcl.h>
#include <tk.h>
#include <X11/Xlib.h>
#include <windows.h>

#ifndef WGL_ARB_pbuffer
#define WGL_ARB_pbuffer 1

#define WGL_DRAW_TO_PBUFFER_ARB 0x202D
#define WGL_MAX_PBUFFER_PIXELS_ARB 0x202E
#define WGL_MAX_PBUFFER_WIDTH_ARB 0x202F
#define WGL_MAX_PBUFFER_HEIGHT_ARB 0x2030
#define WGL_PBUFFER_LARGEST_ARB 0x2033
#define WGL_PBUFFER_WIDTH_ARB 0x2034
#define WGL_PBUFFER_HEIGHT_ARB 0x2035
#define WGL_PBUFFER_LOST_ARB 0x2036


typedef HPBUFFERARB (WINAPI * PFNWGLCREATEPBUFFERARBPROC) (HDC hDC, int iPixelFormat, int iWidth, int iHeight, const int* piAttribList);
typedef BOOL (WINAPI * PFNWGLDESTROYPBUFFERARBPROC) (HPBUFFERARB hPbuffer);
typedef HDC (WINAPI * PFNWGLGETPBUFFERDCARBPROC) (HPBUFFERARB hPbuffer);
typedef BOOL (WINAPI * PFNWGLQUERYPBUFFERARBPROC) (HPBUFFERARB hPbuffer, int iAttribute, int* piValue);
typedef int (WINAPI * PFNWGLRELEASEPBUFFERDCARBPROC) (HPBUFFERARB hPbuffer, HDC hDC);


#endif /* WGL_ARB_pbuffer */

/* -------------------------- WGL_ARB_pixel_format ------------------------- */

#ifndef WGL_ARB_pixel_format
#define WGL_ARB_pixel_format 1

#define WGL_NUMBER_PIXEL_FORMATS_ARB 0x2000
#define WGL_DRAW_TO_WINDOW_ARB 0x2001
#define WGL_DRAW_TO_BITMAP_ARB 0x2002
#define WGL_ACCELERATION_ARB 0x2003
#define WGL_NEED_PALETTE_ARB 0x2004
#define WGL_NEED_SYSTEM_PALETTE_ARB 0x2005
#define WGL_SWAP_LAYER_BUFFERS_ARB 0x2006
#define WGL_SWAP_METHOD_ARB 0x2007
#define WGL_NUMBER_OVERLAYS_ARB 0x2008
#define WGL_NUMBER_UNDERLAYS_ARB 0x2009
#define WGL_TRANSPARENT_ARB 0x200A
#define WGL_SHARE_DEPTH_ARB 0x200C
#define WGL_SHARE_STENCIL_ARB 0x200D
#define WGL_SHARE_ACCUM_ARB 0x200E
#define WGL_SUPPORT_GDI_ARB 0x200F
#define WGL_SUPPORT_OPENGL_ARB 0x2010
#define WGL_DOUBLE_BUFFER_ARB 0x2011
#define WGL_STEREO_ARB 0x2012
#define WGL_PIXEL_TYPE_ARB 0x2013
#define WGL_COLOR_BITS_ARB 0x2014
#define WGL_RED_BITS_ARB 0x2015
#define WGL_RED_SHIFT_ARB 0x2016
#define WGL_GREEN_BITS_ARB 0x2017
#define WGL_GREEN_SHIFT_ARB 0x2018
#define WGL_BLUE_BITS_ARB 0x2019
#define WGL_BLUE_SHIFT_ARB 0x201A
#define WGL_ALPHA_BITS_ARB 0x201B
#define WGL_ALPHA_SHIFT_ARB 0x201C
#define WGL_ACCUM_BITS_ARB 0x201D
#define WGL_ACCUM_RED_BITS_ARB 0x201E
#define WGL_ACCUM_GREEN_BITS_ARB 0x201F
#define WGL_ACCUM_BLUE_BITS_ARB 0x2020
#define WGL_ACCUM_ALPHA_BITS_ARB 0x2021
#define WGL_DEPTH_BITS_ARB 0x2022
#define WGL_STENCIL_BITS_ARB 0x2023
#define WGL_AUX_BUFFERS_ARB 0x2024
#define WGL_NO_ACCELERATION_ARB 0x2025
#define WGL_GENERIC_ACCELERATION_ARB 0x2026
#define WGL_FULL_ACCELERATION_ARB 0x2027
#define WGL_SWAP_EXCHANGE_ARB 0x2028
#define WGL_SWAP_COPY_ARB 0x2029
#define WGL_SWAP_UNDEFINED_ARB 0x202A
#define WGL_TYPE_RGBA_ARB 0x202B
#define WGL_TYPE_COLORINDEX_ARB 0x202C
#define WGL_TRANSPARENT_RED_VALUE_ARB 0x2037
#define WGL_TRANSPARENT_GREEN_VALUE_ARB 0x2038
#define WGL_TRANSPARENT_BLUE_VALUE_ARB 0x2039
#define WGL_TRANSPARENT_ALPHA_VALUE_ARB 0x203A
#define WGL_TRANSPARENT_INDEX_VALUE_ARB 0x203B

typedef BOOL (WINAPI * PFNWGLCHOOSEPIXELFORMATARBPROC) (HDC hdc, const int* piAttribIList, const FLOAT *pfAttribFList, UINT nMaxFormats, int *piFormats, UINT *nNumFormats);
typedef BOOL (WINAPI * PFNWGLGETPIXELFORMATATTRIBFVARBPROC) (HDC hdc, int iPixelFormat, int iLayerPlane, UINT nAttributes, const int* piAttributes, FLOAT *pfValues);
typedef BOOL (WINAPI * PFNWGLGETPIXELFORMATATTRIBIVARBPROC) (HDC hdc, int iPixelFormat, int iLayerPlane, UINT nAttributes, const int* piAttributes, int *piValues);

#endif /* WGL_ARB_pixel_format */


static PFNWGLCHOOSEPIXELFORMATARBPROC wglChoosePixelFormatARB = 0;
static PFNWGLCREATEPBUFFERARBPROC wglCreatePbufferARB = 0;
static PFNWGLGETPBUFFERDCARBPROC wglGetPbufferDCARB = 0;
static PFNWGLDESTROYPBUFFERARBPROC wglDestroyPbufferARB = 0;


#endif // _WIN32

#include <Core/Util/Assert.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>
#include <Core/Util/Environment.h>


using std::cerr;

namespace SCIRun {

PBuffer::PBuffer( int doubleBuffer /* = GL_FALSE */ ) :
#ifndef _WIN32
  cx_(0),
#ifdef HAVE_PBUFFER
  fbc_(0),
  pbuffer_(0),
#endif  
  dpy_(0),
  screen_(0),
#else // _WIN32
  dc_(0),
  rc_(0),
#ifdef HAVE_PBUFFER
  pbuffer_(0),
#endif
#endif
  width_(0),
  height_(0),
  colorBits_(8),
  doubleBuffer_(doubleBuffer),
  depthBits_(8)
{}

#ifdef HAVE_PBUFFER

bool
#ifndef _WIN32
PBuffer::create(Display* dpy, int screen, GLXContext sharedcontext,
		int width, int height, int colorBits, int depthBits)
{
  dpy_ = dpy;
  screen_ = screen;

#else // WIN32
  PBuffer::create(HDC dc, HGLRC sharedRc,
		int width, int height, int colorBits, int depthBits)
{
  dc_ = dc;
  
#endif

  width_ = width;
  height_ = height;
  colorBits_ = colorBits;

  if (sci_getenv_p("SCIRUN_DISABLE_PBUFFERS"))
  {
    return false;
  }
  
  // Set up a pbuffer associated with dpy
  int minor = 0 , major = 0;

#ifndef _WIN32
  glXQueryVersion(dpy, &major, &minor);
#else
    const char* version = (char *)glGetString(GL_VERSION);

  sscanf(version, "%d.%d", &major, &minor);
#endif

  if ( major >= 1 || (major == 1 &&  minor >1))
  { // we can have a pbuffer
    int attrib[32];
    int i = 0;

#ifndef _WIN32

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
    if ( fbc_ == 0 )
    {
      return false;
    }

    int match = 0, a[8];
    for (int i = 0; i < nelements; i++)
    {
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

      if((a[0] >= 8) && (a[1] >= 8) &&
	 (a[2] >= 8) && (a[3] >= 8) && 
	 (a[4] >= 8) && (a[5] == 0) && (a[6] == 0) && (a[7] == 0) )
      {
	match = i;
	break;
      }
    }

    i = 0;
    attrib[i++] = GLX_PBUFFER_WIDTH; attrib[i++] = width_;
    attrib[i++] = GLX_PBUFFER_HEIGHT; attrib[i++] = height_;
    attrib[i] = None;
    pbuffer_ = glXCreatePbuffer( dpy, fbc_[match], attrib );
    if( pbuffer_ == 0 )
    {
      return false;
    }
    cx_ = glXCreateNewContext( dpy, *fbc_, GLX_RGBA_TYPE, sharedcontext, True);
    if( !cx_ )
    {
      return false;
    }
  }
  else
  {
    cx_ = 0;
    return false;
  }
  return true;

#else //_WIN32

    attrib[i++] = WGL_RED_BITS_ARB;   attrib[i++] = colorBits;
    attrib[i++] = WGL_GREEN_BITS_ARB; attrib[i++] = colorBits;
    attrib[i++] = WGL_BLUE_BITS_ARB;  attrib[i++] = colorBits;
    attrib[i++] = WGL_ALPHA_BITS_ARB; attrib[i++] = colorBits;  
    attrib[i++] = WGL_DEPTH_BITS_ARB; attrib[i++] = depthBits_;
    attrib[i++] = WGL_DRAW_TO_PBUFFER_ARB; attrib[i++] = true;
    attrib[i++] = WGL_DOUBLE_BUFFER_ARB; attrib[i++] = doubleBuffer_;
    attrib[i] = 0;

    int pf;
    unsigned int nformats;

    if (wglChoosePixelFormatARB(dc_,
				attrib,
				0,
				1,
				&pf,
				&nformats) == false || nformats == 0)
      {
	return false;
      }

    i = 0;
    attrib[i++] = WGL_PBUFFER_LARGEST_ARB;
    attrib[i++] = GL_FALSE;
    attrib[i] = 0;
    pbuffer_ = wglCreatePbufferARB( dc, pf, width_, height_, attrib );
    if ( pbuffer_ == 0 )
    {
      return false;
    }

    dc_ = wglGetPbufferDCARB(pbuffer_);

    if (!dc_) {
      return false;
    }

    rc_ = wglCreateContext( dc );

    if( !rc_ )
    {
      return false;
    }

    if (wglShareLists(rc_, sharedRc) == 0)
    {
      return false;
    }
  }
  else
  {
    rc_ = 0;
    return false;
  }
  return true;
				

#endif // _WIN32

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
#else // WIN32
  if ( rc_ ) {
    wglDeleteContext(rc_);
    rc_ = 0;
  }
  if ( pbuffer_ ) {
    wglDestroyPbufferARB(pbuffer_);
    pbuffer_ = 0;
  }
#endif
}

void
PBuffer::makeCurrent()
{
#ifndef _WIN32
  glXMakeCurrent( dpy_, pbuffer_, cx_ );
#else // WIN32
  wglMakeCurrent( dc_, rc_ );
#endif
}

bool
PBuffer::is_current()
{
#ifndef _WIN32
  return (cx_ == glXGetCurrentContext());
#else // WIN32
  return (rc_ == wglGetCurrentContext());
#endif
}

#else // ifdef HAVE_PBUFFER

bool
#ifndef _WIN32
PBuffer::create(Display* /*dpy*/, int /*screen*/, GLXContext /*shared*/,
#else
PBuffer::create(HDC, HGLRC,
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

