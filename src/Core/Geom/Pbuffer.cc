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
//    File   : Pbuffer.cc
//    Author : Milan Ikits
//    Date   : Sun Jun 27 17:49:45 2004

#include <Core/Geom/Pbuffer.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Util/Environment.h>

#include <iostream>
#include <string>

#include <stdio.h>

#include <sci_glu.h>
#include <sci_glx.h>

#ifdef _WIN32
#include <windows.h>
#endif

using std::cerr;
using std::endl;
using std::string;

#ifndef _WIN32

#ifndef GLX_ATI_pixel_format_float

#define GLX_RGBA_FLOAT_ATI_BIT 0x00000100

#endif /* GLX_ATI_pixel_format_float */

/* ------------------------- GLX_ATI_render_texture ------------------------ */

#ifndef GLX_ATI_render_texture

#define GLX_BIND_TO_TEXTURE_RGB_ATI 0x9800
#define GLX_BIND_TO_TEXTURE_RGBA_ATI 0x9801
#define GLX_TEXTURE_FORMAT_ATI 0x9802
#define GLX_TEXTURE_TARGET_ATI 0x9803
#define GLX_MIPMAP_TEXTURE_ATI 0x9804
#define GLX_TEXTURE_RGB_ATI 0x9805
#define GLX_TEXTURE_RGBA_ATI 0x9806
#define GLX_NO_TEXTURE_ATI 0x9807
#define GLX_TEXTURE_CUBE_MAP_ATI 0x9808
#define GLX_TEXTURE_1D_ATI 0x9809
#define GLX_TEXTURE_2D_ATI 0x980A
#define GLX_MIPMAP_LEVEL_ATI 0x980B
#define GLX_CUBE_MAP_FACE_ATI 0x980C
#define GLX_TEXTURE_CUBE_MAP_POSITIVE_X_ATI 0x980D
#define GLX_TEXTURE_CUBE_MAP_NEGATIVE_X_ATI 0x980E
#define GLX_TEXTURE_CUBE_MAP_POSITIVE_Y_ATI 0x980F
#define GLX_TEXTURE_CUBE_MAP_NEGATIVE_Y_ATI 0x9810
#define GLX_TEXTURE_CUBE_MAP_POSITIVE_Z_ATI 0x9811
#define GLX_TEXTURE_CUBE_MAP_NEGATIVE_Z_ATI 0x9812
#define GLX_FRONT_LEFT_ATI 0x9813
#define GLX_FRONT_RIGHT_ATI 0x9814
#define GLX_BACK_LEFT_ATI 0x9815
#define GLX_BACK_RIGHT_ATI 0x9816
#define GLX_AUX0_ATI 0x9817
#define GLX_AUX1_ATI 0x9818
#define GLX_AUX2_ATI 0x9819
#define GLX_AUX3_ATI 0x981A
#define GLX_AUX4_ATI 0x981B
#define GLX_AUX5_ATI 0x981C
#define GLX_AUX6_ATI 0x981D
#define GLX_AUX7_ATI 0x981E
#define GLX_AUX8_ATI 0x981F
#define GLX_AUX9_ATI 0x9820
#define GLX_BIND_TO_TEXTURE_LUMINANCE_ATI 0x9821
#define GLX_BIND_TO_TEXTURE_INTENSITY_ATI 0x9822

typedef void ( * PFNGLXBINDTEXIMAGEATIPROC) (Display *dpy, GLXPbuffer pbuf, int buffer);
typedef void ( * PFNGLXRELEASETEXIMAGEATIPROC) (Display *dpy, GLXPbuffer pbuf, int buffer);

#endif /* GLX_ATI_render_texture */

#ifndef GLX_NV_float_buffer

#define GLX_FLOAT_COMPONENTS_NV 0x20B0

#endif /* GLX_NV_float_buffer */

#else // WIN32

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

DECLARE_HANDLE(HPBUFFERARB);

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

/* ------------------------- WGL_ARB_render_texture ------------------------ */

#ifndef WGL_ARB_render_texture
#define WGL_ARB_render_texture 1

#define WGL_BIND_TO_TEXTURE_RGB_ARB 0x2070
#define WGL_BIND_TO_TEXTURE_RGBA_ARB 0x2071
#define WGL_TEXTURE_FORMAT_ARB 0x2072
#define WGL_TEXTURE_TARGET_ARB 0x2073
#define WGL_MIPMAP_TEXTURE_ARB 0x2074
#define WGL_TEXTURE_RGB_ARB 0x2075
#define WGL_TEXTURE_RGBA_ARB 0x2076
#define WGL_NO_TEXTURE_ARB 0x2077
#define WGL_TEXTURE_CUBE_MAP_ARB 0x2078
#define WGL_TEXTURE_1D_ARB 0x2079
#define WGL_TEXTURE_2D_ARB 0x207A
#define WGL_MIPMAP_LEVEL_ARB 0x207B
#define WGL_CUBE_MAP_FACE_ARB 0x207C
#define WGL_TEXTURE_CUBE_MAP_POSITIVE_X_ARB 0x207D
#define WGL_TEXTURE_CUBE_MAP_NEGATIVE_X_ARB 0x207E
#define WGL_TEXTURE_CUBE_MAP_POSITIVE_Y_ARB 0x207F
#define WGL_TEXTURE_CUBE_MAP_NEGATIVE_Y_ARB 0x2080
#define WGL_TEXTURE_CUBE_MAP_POSITIVE_Z_ARB 0x2081
#define WGL_TEXTURE_CUBE_MAP_NEGATIVE_Z_ARB 0x2082
#define WGL_FRONT_LEFT_ARB 0x2083
#define WGL_FRONT_RIGHT_ARB 0x2084
#define WGL_BACK_LEFT_ARB 0x2085
#define WGL_BACK_RIGHT_ARB 0x2086
#define WGL_AUX0_ARB 0x2087
#define WGL_AUX1_ARB 0x2088
#define WGL_AUX2_ARB 0x2089
#define WGL_AUX3_ARB 0x208A
#define WGL_AUX4_ARB 0x208B
#define WGL_AUX5_ARB 0x208C
#define WGL_AUX6_ARB 0x208D
#define WGL_AUX7_ARB 0x208E
#define WGL_AUX8_ARB 0x208F
#define WGL_AUX9_ARB 0x2090

typedef BOOL (WINAPI * PFNWGLBINDTEXIMAGEARBPROC) (HPBUFFERARB hPbuffer, int iBuffer);
typedef BOOL (WINAPI * PFNWGLRELEASETEXIMAGEARBPROC) (HPBUFFERARB hPbuffer, int iBuffer);
typedef BOOL (WINAPI * PFNWGLSETPBUFFERATTRIBARBPROC) (HPBUFFERARB hPbuffer, const int* piAttribList);


#endif /* WGL_ARB_render_texture */

/* ----------------------- WGL_ARB_extensions_string ----------------------- */

#ifndef WGL_ARB_extensions_string
#define WGL_ARB_extensions_string 1

typedef const char* (WINAPI * PFNWGLGETEXTENSIONSSTRINGARBPROC) (HDC hdc);


#endif /* WGL_ARB_extensions_string */

/* -------------------- WGL_NV_render_texture_rectangle -------------------- */

#ifndef WGL_NV_render_texture_rectangle
#define WGL_NV_render_texture_rectangle 1

#define WGL_BIND_TO_TEXTURE_RECTANGLE_RGB_NV 0x20A0
#define WGL_BIND_TO_TEXTURE_RECTANGLE_RGBA_NV 0x20A1
#define WGL_TEXTURE_RECTANGLE_NV 0x20A2


#endif /* WGL_NV_render_texture_rectangle */

/* ----------------------- WGL_ATI_pixel_format_float ---------------------- */

#ifndef WGL_ATI_pixel_format_float
#define WGL_ATI_pixel_format_float 1

#define WGL_TYPE_RGBA_FLOAT_ATI 0x21A0
#define GL_RGBA_FLOAT_MODE_ATI 0x8820
#define GL_COLOR_CLEAR_UNCLAMPED_VALUE_ATI 0x8835


#endif /* WGL_ATI_pixel_format_float */
/* -------------------------- WGL_NV_float_buffer -------------------------- */

#ifndef WGL_NV_float_buffer
#define WGL_NV_float_buffer 1

#define WGL_FLOAT_COMPONENTS_NV 0x20B0
#define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_R_NV 0x20B1
#define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RG_NV 0x20B2
#define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGB_NV 0x20B3
#define WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGBA_NV 0x20B4
#define WGL_TEXTURE_FLOAT_R_NV 0x20B5
#define WGL_TEXTURE_FLOAT_RG_NV 0x20B6
#define WGL_TEXTURE_FLOAT_RGB_NV 0x20B7
#define WGL_TEXTURE_FLOAT_RGBA_NV 0x20B8

#define WGLEW_NV_float_buffer WGLEW_GET_VAR(__WGLEW_NV_float_buffer)

#endif /* WGL_NV_float_buffer */

#endif //  WIN32

#ifndef GL_NV_float_buffer

#define GL_FLOAT_R_NV 0x8880
#define GL_FLOAT_RG_NV 0x8881
#define GL_FLOAT_RGB_NV 0x8882
#define GL_FLOAT_RGBA_NV 0x8883
#define GL_FLOAT_R16_NV 0x8884
#define GL_FLOAT_R32_NV 0x8885
#define GL_FLOAT_RG16_NV 0x8886
#define GL_FLOAT_RG32_NV 0x8887
#define GL_FLOAT_RGB16_NV 0x8888
#define GL_FLOAT_RGB32_NV 0x8889
#define GL_FLOAT_RGBA16_NV 0x888A
#define GL_FLOAT_RGBA32_NV 0x888B
#define GL_TEXTURE_FLOAT_COMPONENTS_NV 0x888C
#define GL_FLOAT_CLEAR_COLOR_VALUE_NV 0x888D
#define GL_FLOAT_RGBA_MODE_NV 0x888E

#endif /* GL_NV_float_buffer */

#ifndef GL_NV_texture_rectangle

#define GL_TEXTURE_RECTANGLE_NV 0x84F5
#define GL_TEXTURE_BINDING_RECTANGLE_NV 0x84F6
#define GL_PROXY_TEXTURE_RECTANGLE_NV 0x84F7
#define GL_MAX_RECTANGLE_TEXTURE_SIZE_NV 0x84F8

#endif /* GL_NV_texture_rectangle */

#ifndef _WIN32
#if !defined(GLX_ARB_get_proc_address) || !defined(GLX_GLXEXT_PROTOTYPES)

extern "C" void ( * glXGetProcAddressARB (const GLubyte *procName)) (void);

#endif /* GLX_ARB_get_proc_address */
#endif


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

#elif defined(_WIN32)

#define getProcAddress(x) (wglGetProcAddress((LPCSTR)x))

#else

#define getProcAddress(x) ((*glXGetProcAddressARB)((const GLubyte*)x))

#endif

#ifndef _WIN32
static PFNGLXBINDTEXIMAGEATIPROC glXBindTexImageATI = 0;
static PFNGLXRELEASETEXIMAGEATIPROC glXReleaseTexImageATI = 0;
#else
static PFNWGLGETEXTENSIONSSTRINGARBPROC wglGetExtensionsStringARB = 0;
static PFNWGLCHOOSEPIXELFORMATARBPROC wglChoosePixelFormatARB = 0;
static PFNWGLBINDTEXIMAGEARBPROC wglBindTexImageARB = 0;
static PFNWGLRELEASETEXIMAGEARBPROC wglReleaseTexImageARB = 0;
static PFNWGLCREATEPBUFFERARBPROC wglCreatePbufferARB = 0;
static PFNWGLGETPBUFFERDCARBPROC wglGetPbufferDCARB = 0;
static PFNWGLDESTROYPBUFFERARBPROC wglDestroyPbufferARB = 0;
static PFNWGLQUERYPBUFFERARBPROC wglQueryPbufferARB = 0;
#endif

static bool mInit = false;
static bool mSupported = false;

static bool mATI_render_texture = false;
static bool mATI_pixel_format_float = false;
static bool mNV_float_buffer = false;
static bool mNV_texture_rectangle = false;

const string program =
"!!ARBfp1.0 \n"
"TEX result.color, fragment.texcoord[0], texture[0], RECT; \n"
"END";

static SCIRun::FragmentProgramARB* mShader = 0;

namespace SCIRun {

struct PbufferImpl
{
#ifndef _WIN32
  PbufferImpl () : display_(0), pbuffer_(0), context_(0) {}
  Display* display_;
  GLXPbuffer pbuffer_; 
  GLXContext context_;

  Display* saved_display_;
  GLXDrawable saved_drawable_;
  GLXContext saved_context_;
#else // _WIN32
  PbufferImpl() : pbuffer_(0), dc_(0), rc_(0) {}
  HPBUFFERARB pbuffer_; 
  HDC   dc_;
  HGLRC rc_;
  HDC   saved_dc_;
  HGLRC saved_rc_;
#endif
};


#ifdef _WIN32

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
    
  // If That Failed, Try Standard Opengl Extensions String
  if (supported == NULL)
    supported = (char*)glGetString(GL_EXTENSIONS);
    
  // If That Failed Too, Must Be No Extensions Supported
  if (supported == NULL)
    return false;
    
  // Begin Examination At Start Of String, Increment By 1 On False Match
  for (const char* p = supported; ; p++)
  {
    // Advance p Up To The Next Possible Match
    p = strstr(p, extension);
	
    if (p == NULL)
      return false;						// No Match
	
    // Make Sure That Match Is At The Start Of The String Or That
    // The Previous Char Is A Space, Or Else We Could Accidentally
    // Match "wglFunkywglExtension" With "wglExtension"
	
    // Also, Make Sure That The Following Character Is Space Or NULL
    // Or Else "wglExtensionTwo" Might Match "wglExtension"
    if ((p==supported || p[-1]==' ') && (p[extlen]=='\0' || p[extlen]==' '))
      return true;						// Match
  }
}

#endif



bool
Pbuffer::create ()
{

  if (sci_getenv_p("SCIRUN_DISABLE_PBUFFERS"))
  {
    mSupported = false;
    return false;
  }

#if defined(__ECC)
  // For now no Pbuffer support on the Altix system
  mSupported = false;
  return false;
#elif defined (_WIN32)
  if(!mInit) {
    /* query GL version */

    int major, minor;
    const char* version = (char *)glGetString(GL_VERSION);

    sscanf(version, "%d.%d", &major, &minor);

    // get the procedure address for checking for the wgl extensions.
    mATI_render_texture =
      WGLisExtensionSupported("WGL_ARB_render_texture");

    mATI_pixel_format_float = 
      WGLisExtensionSupported("WGL_ATI_pixel_format_float");
    
    mNV_float_buffer = 
      WGLisExtensionSupported("WGL_NV_float_buffer") &&
      WGLisExtensionSupported("GL_NV_float_buffer") &&
      WGLisExtensionSupported("GL_ARB_fragment_program");

    mNV_texture_rectangle = 
      WGLisExtensionSupported("GL_NV_texture_rectangle");
    
    mSupported = WGLisExtensionSupported("WGL_ARB_pixel_format");

    if (mSupported)
      wglChoosePixelFormatARB = (PFNWGLCHOOSEPIXELFORMATARBPROC)wglGetProcAddress("wglChoosePixelFormatARB");

    mSupported = 
      mSupported &&
      WGLisExtensionSupported("WGL_ARB_render_texture");

    if (mSupported)
    {
      wglBindTexImageARB = (PFNWGLBINDTEXIMAGEARBPROC)wglGetProcAddress("wglBindTexImageARB");
      wglReleaseTexImageARB = (PFNWGLRELEASETEXIMAGEARBPROC)wglGetProcAddress("wglReleaseTexImageARB");
    }

    // Check for version.
    if(minor < 3 || (mFormat == GL_FLOAT && 
		     !(mATI_pixel_format_float || mNV_float_buffer)))
    {
      mSupported = false;
    }
    else
    {
      mSupported = true;
    }
    mInit = true;
  }

    if (mSupported)
    {
      mImpl->dc_ =  wglGetCurrentDC();
      if (mImpl->dc_ == 0)
      {
	cerr << "[Pbuffer::create] Failed to obtain current device context" << endl;
	return true;
      }

      // Get current context.
      HGLRC rc = wglGetCurrentContext();
      if (rc == 0)
      {
	cerr << "[Pbuffer::create] Failed to obtain current GL context" << endl;
	return true;
      }
      int attrib[64];
      int i;
      i = 0;

      // Accelerated OpenGL support.
      attrib[i++] = WGL_SUPPORT_OPENGL_ARB;
      attrib[i++] = GL_TRUE;

      // Pbuffer capable.
      attrib[i++] = WGL_DRAW_TO_PBUFFER_ARB;
      attrib[i++] = GL_TRUE;

      // Format
      if (mFormat == GL_FLOAT)
	{
	  if (mATI_pixel_format_float)
	    {
	      attrib[i++] = WGL_PIXEL_TYPE_ARB;
	      attrib[i++] = WGL_TYPE_RGBA_FLOAT_ATI;
	    }
	  else if (mNV_float_buffer)
	    {
	      attrib[i++] = WGL_PIXEL_TYPE_ARB;
	      attrib[i++] = WGL_TYPE_RGBA_ARB;
	      attrib[i++] = WGL_FLOAT_COMPONENTS_NV;
	      attrib[i++] = GL_TRUE;
	    }
	}
      else // GL_INT
	{
	  attrib[i++] = WGL_PIXEL_TYPE_ARB;
	  attrib[i++] = WGL_TYPE_RGBA_ARB;
	}
      // color buffer spec
      if (mNumColorBits != GL_DONT_CARE)
	{
	  attrib[i++] = WGL_RED_BITS_ARB;
	  attrib[i++] = mNumColorBits;
	  attrib[i++] = WGL_GREEN_BITS_ARB;
	  attrib[i++] = mNumColorBits;
	  attrib[i++] = WGL_BLUE_BITS_ARB;
	  attrib[i++] = mNumColorBits;
	  attrib[i++] = WGL_ALPHA_BITS_ARB;
	  attrib[i++] = mNumColorBits;
	}
      // double buffer spec
      if (mDoubleBuffer != GL_DONT_CARE)
	{
	  attrib[i++] = WGL_DOUBLE_BUFFER_ARB;
	  attrib[i++] = mDoubleBuffer ? GL_TRUE : GL_FALSE;;
	}
      // aux buffer spec
      if (mNumAuxBuffers != GL_DONT_CARE)
	{
	  attrib[i++] = WGL_AUX_BUFFERS_ARB;
	  attrib[i++] = mNumAuxBuffers;
	}
      // depth buffer spec
      if (mNumDepthBits != GL_DONT_CARE)
	{
	  attrib[i++] = WGL_DEPTH_BITS_ARB;
	  attrib[i++] = mNumDepthBits;
	}
      // stencil buffer spec
      if (mNumStencilBits != GL_DONT_CARE)
	{
	  attrib[i++] = WGL_STENCIL_BITS_ARB;
	  attrib[i++] = mNumStencilBits;
	}
      // accum buffer spec
      if (mNumAccumBits != GL_DONT_CARE)
	{
	  attrib[i++] = WGL_ACCUM_RED_BITS_ARB;
	  attrib[i++] = mNumAccumBits;
	  attrib[i++] = WGL_ACCUM_GREEN_BITS_ARB;
	  attrib[i++] = mNumAccumBits;
	  attrib[i++] = WGL_ACCUM_BLUE_BITS_ARB;
	  attrib[i++] = mNumAccumBits;
	  attrib[i++] = WGL_ACCUM_ALPHA_BITS_ARB;
	  attrib[i++] = mNumAccumBits;
	}
      // render to texture
      if (mRenderTex)
	{
	  if (mFormat == GL_FLOAT &&  mNV_float_buffer)
	    {
	      attrib[i++] = WGL_BIND_TO_TEXTURE_RECTANGLE_FLOAT_RGBA_NV;
	      attrib[i++] = GL_TRUE;
	    }
	  else
	    {
	      attrib[i++] = WGL_BIND_TO_TEXTURE_RGBA_ARB;
	      attrib[i++] = GL_TRUE;
	    }
	}
      attrib[i] = 0;
      unsigned int c = 0;
      int pf;
      if (wglChoosePixelFormatARB(mImpl->dc_, attrib, 0, 1, &pf, &c) == 0 || c == 0)
	{
	  cerr << "[Pbuffer::Pbuffer] Failed to find suitable pixel format\n";
	  return true;
	}
 
      // allocate the buffer
      i = 0;
      if (mRenderTex)
	{
	  // format and target
	  if (mFormat == GL_FLOAT && mNV_float_buffer)
	    {
	      attrib[i++] = WGL_TEXTURE_FORMAT_ARB;
	      attrib[i++] = WGL_TEXTURE_FLOAT_RGBA_NV;
	      attrib[i++] = WGL_TEXTURE_TARGET_ARB;
	      attrib[i++] = WGL_TEXTURE_RECTANGLE_NV;
	    }
	  else
	    {
	      attrib[i++] = WGL_TEXTURE_FORMAT_ARB;
	      attrib[i++] = WGL_TEXTURE_RGBA_ARB;
	      attrib[i++] = WGL_TEXTURE_TARGET_ARB;
	      attrib[i++] = WGL_TEXTURE_2D_ARB;
	    }
	  // no mipmap
	  attrib[i++] = WGL_MIPMAP_TEXTURE_ARB;
	  attrib[i++] = GL_FALSE;
	}
      // fail if can't allocate
      attrib[i++] = WGL_PBUFFER_LARGEST_ARB;
      attrib[i++] = GL_FALSE;
      attrib[i++] = 0;
      // create pbuffer
      mImpl->pbuffer_ = wglCreatePbufferARB(mImpl->dc_, pf, mWidth, mHeight, attrib);
      if (mImpl->pbuffer_ == 0)
	{
	  cerr << "[Pbuffer::Pbuffer] Failed to create pbuffer\n";
	  return true;
	}
      // create device context
      mImpl->dc_ = wglGetPbufferDCARB(mImpl->pbuffer_);
      if (mImpl->dc_ == 0)
	{
	  cerr << "[Pbuffer::Pbuffer] Failed to create device context\n";
	  return true;
	}
      // create rendering context
      mImpl->rc_ = wglCreateContext(mImpl->dc_);
      if (mImpl->rc_ == 0)
	{
	  cerr << "[Pbuffer::Pbuffer] Failed to create rendering context\n";
	  return true;
	}
      if (wglShareLists(rc, mImpl->rc_) == 0)
	{
	  cerr << "[Pbuffer::create] Failed to set context sharing\n";
	  return true;
	}
      
      // get actual size
      wglQueryPbufferARB(mImpl->pbuffer_, WGL_PBUFFER_WIDTH_ARB, &mWidth);
      wglQueryPbufferARB(mImpl->pbuffer_, WGL_PBUFFER_HEIGHT_ARB, &mHeight);
      if (mRenderTex) {
	// create pbuffer texture object
	glGenTextures(1, &mTex);
	if(mFormat == GL_FLOAT) {
	  if(mNV_float_buffer) {
	    mTexTarget = GL_TEXTURE_RECTANGLE_NV;
	    if(mNumColorBits == 16)
	      mTexFormat = GL_FLOAT_RGBA16_NV;
	    else
	      mTexFormat = GL_FLOAT_RGBA32_NV;
	  } else {
	    mTexTarget = GL_TEXTURE_2D;
	  }
	} else {
	  mTexTarget = GL_TEXTURE_2D;
	  mTexFormat = GL_RGBA;
	}
	glBindTexture(mTexTarget, mTex);
#ifdef GL_CLAMP_TO_EDGE
	glTexParameteri(mTexTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(mTexTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
#else
	glTexParameteri(mTexTarget, GL_TEXTURE_WRAP_S, GL_CLAMP);
	glTexParameteri(mTexTarget, GL_TEXTURE_WRAP_T, GL_CLAMP);
#endif
	glTexParameteri(mTexTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(mTexTarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	if(!mATI_render_texture) {
	  unsigned char* data = new unsigned char[mWidth*mHeight*4];
	  glTexImage2D(mTexTarget, 0, mTexFormat, mWidth, mHeight, 0,
		       GL_RGBA, GL_UNSIGNED_BYTE, data);
	  delete [] data;
	}
	if(!mShader) {
	  mShader = new FragmentProgramARB(program);
	  if(mShader->create()) return true;
	}
      }
      return false;
    }
    return true;

#else // GLX
  if(!mInit) {
    /* query GLX version */
    int major, minor;
    const char* version = glXGetClientString(glXGetCurrentDisplay(), GLX_VERSION);
    sscanf(version, "%d.%d", &major, &minor);

    mATI_render_texture = 
      gluCheckExtension((GLubyte*)"GLX_ATI_render_texture",
			(GLubyte*)glXGetClientString(glXGetCurrentDisplay(), 
						     GLX_EXTENSIONS));

    mATI_pixel_format_float = 
      gluCheckExtension((GLubyte*)"GLX_ATI_pixel_format_float", 
			(GLubyte*)glXGetClientString(glXGetCurrentDisplay(), 
						     GLX_EXTENSIONS));

    mNV_float_buffer = 
      gluCheckExtension((GLubyte*)"GLX_NV_float_buffer", 
			(GLubyte*)glXGetClientString(glXGetCurrentDisplay(), 
						     GLX_EXTENSIONS))
      && gluCheckExtension((GLubyte*)"GL_NV_float_buffer", 
			   (GLubyte*)glGetString(GL_EXTENSIONS))
      && gluCheckExtension((GLubyte*)"GL_ARB_fragment_program", 
			   (GLubyte*)glGetString(GL_EXTENSIONS));

    mNV_texture_rectangle = 
      gluCheckExtension((GLubyte*)"GL_NV_texture_rectangle", 
			(GLubyte*)glGetString(GL_EXTENSIONS));

    if(minor < 3 || (mFormat == GL_FLOAT && 
		     !(mATI_pixel_format_float || mNV_float_buffer))) {
      mSupported = false;
    } else {
      mSupported = true;
    }
    //       || (mRenderTex && !mATI_render_texture)

    if(mSupported && mATI_render_texture) {
      bool fail = false;
      fail = fail || (glXBindTexImageATI = (PFNGLXBINDTEXIMAGEATIPROC)
		      getProcAddress("glXBindTexImageATI")) == 0;
      fail = fail || (glXReleaseTexImageATI = (PFNGLXRELEASETEXIMAGEATIPROC)
		      getProcAddress("glXReleaseTexImageATI")) == 0;
      if(fail) {
        mSupported = false;
        cerr << "GL_ATI_render_texture is not supported." << endl;
      }
    }

    mInit = true;
  }
  
  if(mSupported) {
    // get current display
    mImpl->display_ = glXGetCurrentDisplay();
    if (mImpl->display_ == 0)
    {
      cerr << "[Pbuffer::create] Failed to obtain current display" << endl;
      return false;
    }
    // get current context
    GLXContext ctx = glXGetCurrentContext();
    if (ctx == 0)
    {
      cerr << "[Pbuffer::create] Failed to obtain current context" << endl;
      return false;
    }
    // find suitable visual for the pbuffer
    int attrib[64];
    GLXFBConfig* fbc;
    int n_fbc;
    int i;
    if (mSeparate)
    {
      i = 0;
      // pbuffer capable
      attrib[i++] = GLX_DRAWABLE_TYPE;
      attrib[i++] = mNumColorBits > 8 ? GLX_PBUFFER_BIT :
        GLX_PBUFFER_BIT | GLX_WINDOW_BIT;
      // format
      if (mFormat == GL_FLOAT)
      {
        if (mATI_pixel_format_float)
        {
          attrib[i++] = GLX_RENDER_TYPE;
          attrib[i++] = GLX_RGBA_FLOAT_ATI_BIT;
        }
        else if (mNV_float_buffer)
        {
          attrib[i++] = GLX_RENDER_TYPE;
          attrib[i++] = GLX_RGBA_BIT;
          attrib[i++] = GLX_FLOAT_COMPONENTS_NV;
          attrib[i++] = GL_TRUE;
        }
      }
      else // GL_INT
      {
        attrib[i++] = GLX_RENDER_TYPE;
        attrib[i++] = GLX_RGBA_BIT;
      }
      // color buffer spec
      if (mNumColorBits != GL_DONT_CARE)
      {
        attrib[i++] = GLX_RED_SIZE;
        attrib[i++] = mNumColorBits;
        attrib[i++] = GLX_GREEN_SIZE;
        attrib[i++] = mNumColorBits;
        attrib[i++] = GLX_BLUE_SIZE;
        attrib[i++] = mNumColorBits;
        attrib[i++] = GLX_ALPHA_SIZE;
        attrib[i++] = mNumColorBits;
      }
      // double buffer spec
      if (mDoubleBuffer != GL_DONT_CARE)
      {
        attrib[i++] = GLX_DOUBLEBUFFER;
        attrib[i++] = mDoubleBuffer ? GL_TRUE : GL_FALSE;
      }
      // aux buffer spec
      if (mNumAuxBuffers != GL_DONT_CARE)
      {
        attrib[i++] = GLX_AUX_BUFFERS;
        attrib[i++] = mNumAuxBuffers;
      }
      // depth buffer spec
      if (mNumDepthBits != GL_DONT_CARE)
      {
        attrib[i++] = GLX_DEPTH_SIZE;
        attrib[i++] = mNumDepthBits;
      }
      // stencil buffer spec
      if (mNumStencilBits != GL_DONT_CARE)
      {
        attrib[i++] = GLX_STENCIL_SIZE;
        attrib[i++] = mNumStencilBits;
      }
      // accum buffer spec
      if (mNumAccumBits != GL_DONT_CARE)
      {
        attrib[i++] = GLX_ACCUM_RED_SIZE;
        attrib[i++] = mNumAccumBits;
        attrib[i++] = GLX_ACCUM_GREEN_SIZE;
        attrib[i++] = mNumAccumBits;
        attrib[i++] = GLX_ACCUM_BLUE_SIZE;
        attrib[i++] = mNumAccumBits;
        attrib[i++] = GLX_ACCUM_ALPHA_SIZE;
        attrib[i++] = mNumAccumBits;
      }
      // render to texture
      if (mRenderTex)
      {
        if (mATI_render_texture)
        {
          attrib[i++] = GLX_BIND_TO_TEXTURE_RGBA_ATI;
          attrib[i++] = GL_TRUE;
        }
      }
      attrib[i] = None;
    }
    else
    {
      // get fb config id for current context
      int id = 0;
      if (glXQueryContext(mImpl->display_, ctx, GLX_FBCONFIG_ID, &id) != Success)
      {
        cerr << "[Pbuffer::create] Failed to query fbconfig id from context"
             << endl;
        return false;
      }
      // choose fb config with given id
      attrib[0] = GLX_FBCONFIG_ID;
      attrib[1] = id;
      attrib[2] = None;
    }
    // choose fb config
    fbc = glXChooseFBConfig(mImpl->display_, DefaultScreen(mImpl->display_),
                            attrib, &n_fbc);
    if (fbc == 0 || n_fbc == 0)
    {
      cerr << "[Pbuffer::create] Failed to obtain fb config" << endl;
      return false;
    }
    glXGetFBConfigAttrib(mImpl->display_, *fbc, GLX_FBCONFIG_ID, &mVisualId);
    glXGetFBConfigAttrib(mImpl->display_, *fbc, GLX_RED_SIZE, &mNumColorBits);
    // create pbuffer
    i = 0;
    attrib[i++] = GLX_PBUFFER_WIDTH;
    attrib[i++] = mWidth;
    attrib[i++] = GLX_PBUFFER_HEIGHT;
    attrib[i++] = mHeight;
    attrib[i++] = GLX_LARGEST_PBUFFER; // we need exact size or fail
    attrib[i++] = GL_FALSE;
    attrib[i++] = GLX_PRESERVED_CONTENTS; // we don't want to lose the buffer
    attrib[i++] = GL_TRUE;
    if (mRenderTex && mATI_render_texture)
    {
      attrib[i++] = GLX_TEXTURE_FORMAT_ATI;
      attrib[i++] = GLX_TEXTURE_RGBA_ATI;
      attrib[i++] = GLX_TEXTURE_TARGET_ATI;
      attrib[i++] = GLX_TEXTURE_2D_ATI;
      attrib[i++] = GLX_MIPMAP_TEXTURE_ATI;
      attrib[i++] = GL_FALSE;
    }
    attrib[i] = None;
    mImpl->pbuffer_ = glXCreatePbuffer(mImpl->display_, *fbc, attrib);
    if (mImpl->pbuffer_ == 0)
    {
      cerr << "[Pbuffer::create] Failed to create pbuffer" << endl;
      return false;
    }
    // create context
    if (mSeparate)
    {
      mImpl->context_ = glXCreateNewContext(mImpl->display_, *fbc, GLX_RGBA_TYPE,
                                            ctx, True);
      if (mImpl->context_ == 0)
      {
        cerr << "[Pbuffer::create] Failed to create context" << endl;
        return false;
      }
    }
    else
    {
      mImpl->context_ = ctx;
    }
    // query attributes
    glXQueryDrawable(mImpl->display_, mImpl->pbuffer_, GLX_WIDTH,
                     (unsigned int*)&mWidth);
    glXQueryDrawable(mImpl->display_, mImpl->pbuffer_, GLX_HEIGHT,
                     (unsigned int*)&mHeight);
    // ...
    if (mRenderTex) {
      // create pbuffer texture object
      glGenTextures(1, &mTex);
      if(mFormat == GL_FLOAT) {
        if(mNV_float_buffer) {
          mTexTarget = GL_TEXTURE_RECTANGLE_NV;
          if(mNumColorBits == 16)
            mTexFormat = GL_FLOAT_RGBA16_NV;
          else
            mTexFormat = GL_FLOAT_RGBA32_NV;
        } else {
          mTexTarget = GL_TEXTURE_2D;
        }
      } else {
        mTexTarget = GL_TEXTURE_2D;
        mTexFormat = GL_RGBA;
      }
      glBindTexture(mTexTarget, mTex);
      glTexParameteri(mTexTarget, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
      glTexParameteri(mTexTarget, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
      glTexParameteri(mTexTarget, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(mTexTarget, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      if(!mATI_render_texture) {
        unsigned char* data = new unsigned char[mWidth*mHeight*4];
        glTexImage2D(mTexTarget, 0, mTexFormat, mWidth, mHeight, 0,
                     GL_RGBA, GL_UNSIGNED_BYTE, data);
        delete [] data;
      }
      if(!mShader) {
        mShader = new FragmentProgramARB(program);
        if(mShader->create()) return false;
      }
    }
    return true;
  }
  return false;
#endif
}

void
Pbuffer::destroy ()
{
#ifndef _WIN32
  if (mSeparate && mImpl->context_ != 0)
  {
    //glXMakeCurrent(mImpl->display_, mImpl->pbuffer_, 0);
    glXDestroyContext(mImpl->display_, mImpl->context_);
  }
  if (mImpl->pbuffer_ != 0)
  {
    glXDestroyPbuffer(mImpl->display_, mImpl->pbuffer_);
  }
  if(mShader)
  {
    mShader->destroy();
  }
#else // WIN32
  if (/*mSeparate && */mImpl->rc_ != 0)
  {
    wglDeleteContext(mImpl->rc_);
  }
  if (mImpl->pbuffer_ != 0)
  {
    wglDestroyPbufferARB(mImpl->pbuffer_);
  }
  if(mShader)
  {
    mShader->destroy();
  }
#endif
}


void
Pbuffer::makeCurrent ()
{
#ifndef _WIN32
  glXMakeCurrent(mImpl->display_, mImpl->pbuffer_, mImpl->context_);
#else
  wglMakeCurrent(mImpl->dc_, mImpl->rc_);
#endif
}

bool
Pbuffer::is_current ()
{
#ifndef _WIN32
  return (mImpl->context_ == glXGetCurrentContext());
#else // WIN32
  return (mImpl->rc_ == wglGetCurrentContext());
#endif
}


void
Pbuffer::swapBuffers ()
{
  if (mRenderTex && !mATI_render_texture)
  {
    GLint buffer;
    glGetIntegerv(GL_DRAW_BUFFER, &buffer);
    glReadBuffer(buffer);
    glBindTexture(mTexTarget, mTex);
    glCopyTexSubImage2D(mTexTarget, 0, 0, 0, 0, 0, mWidth, mHeight);
    glBindTexture(mTexTarget, 0);
  }
  if (mDoubleBuffer)
  {
#ifndef _WIN32 
   glXSwapBuffers(mImpl->display_, mImpl->pbuffer_);
#else
   wglSwapLayerBuffers(mImpl->dc_, WGL_SWAP_MAIN_PLANE);
#endif
  }
  else
  {
    glFinish();
  }
}


void
Pbuffer::bind (unsigned int buffer)
{
  if(mRenderTex)
  {
    glEnable(mTexTarget);
    glBindTexture(mTexTarget, mTex);
    if(mATI_render_texture)
    {
#ifndef _WIN32
      glXBindTexImageATI(mImpl->display_, mImpl->pbuffer_, 
			 buffer == GL_FRONT ? 
                         GLX_FRONT_LEFT_ATI : GLX_BACK_LEFT_ATI);
#else
      wglBindTexImageARB(mImpl->pbuffer_,
			 buffer == GL_FRONT ? WGL_FRONT_LEFT_ARB : WGL_BACK_LEFT_ARB);
#endif
    }
    if (mFormat == GL_FLOAT && mNV_float_buffer)
    {
      if (mUseDefaultShader)
      {
        mShader->bind();
      }
      if (mUseTextureMatrix)
      {
        glMatrixMode(GL_TEXTURE);
        glPushMatrix();
        glLoadIdentity();
        glScalef(mWidth, mHeight, 1.0);
        glMatrixMode(GL_MODELVIEW);
      }
    }
  }
}


void
Pbuffer::release (unsigned int buffer)
{
  if(mRenderTex)
  {
    if(mATI_render_texture)
    {
#ifndef _WIN32

      glXReleaseTexImageATI(mImpl->display_, mImpl->pbuffer_,
			    buffer == GL_FRONT ? GLX_FRONT_LEFT_ATI : GLX_BACK_LEFT_ATI);
#else
      wglReleaseTexImageARB(mImpl->pbuffer_,
			    buffer == GL_FRONT ? WGL_FRONT_LEFT_ARB : WGL_BACK_LEFT_ARB);

#endif
    }
    glBindTexture(mTexTarget, 0);
    glDisable(mTexTarget);
    if (mFormat == GL_FLOAT && mNV_float_buffer)
    {
      if (mUseTextureMatrix)
      {
        glMatrixMode(GL_TEXTURE);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
      }
      if (mUseDefaultShader)
      {
        mShader->release();
      }
    }
  }
}


void
Pbuffer::activate ()
{
#ifndef _WIN32
  // save context state
  mImpl->saved_display_ = glXGetCurrentDisplay();
  mImpl->saved_drawable_ = glXGetCurrentDrawable();
  mImpl->saved_context_ = glXGetCurrentContext();
  // set read/write context to pbuffer
  glXMakeCurrent(mImpl->display_, mImpl->pbuffer_, mImpl->context_);
#else
  mImpl->saved_dc_ = wglGetCurrentDC();
  mImpl->saved_rc_ = wglGetCurrentContext();
  wglMakeCurrent(mImpl->dc_, mImpl->rc_);
#endif
}


void
Pbuffer::deactivate ()
{
#ifndef _WIN32
  glXMakeCurrent(mImpl->saved_display_, mImpl->saved_drawable_,
		 mImpl->saved_context_);
#else
  wglMakeCurrent(mImpl->saved_dc_, mImpl->saved_rc_);
#endif
}


bool
Pbuffer::need_shader()
{
  return mFormat == GL_FLOAT && mNV_float_buffer;
}


void
Pbuffer::set_use_default_shader(bool b)
{
  mUseDefaultShader = b;
}


void
Pbuffer::set_use_texture_matrix(bool b)
{
  mUseTextureMatrix = b;
}


Pbuffer::Pbuffer (int width, int height, int format, int numColorBits,
		  /* int numChannels, */ bool isRenderTex, int isDoubleBuffer,
		  int numAuxBuffers, int numDepthBits, int numStencilBits,
		  int numAccumBits)
  : mWidth(width),
    mHeight(height),
    mFormat(format),
    mNumColorBits(numColorBits),
    /* mNumChannels(numChannels), */
    mRenderTex(isRenderTex),
    mDoubleBuffer(isDoubleBuffer),
    mNumAuxBuffers(numAuxBuffers),
    mNumDepthBits(numDepthBits),
    mNumStencilBits(numStencilBits),
    mNumAccumBits(numAccumBits),
    mSeparate(true),
    mTex(0),
    mTexTarget(GL_TEXTURE_2D),
    mUseDefaultShader(true),
    mImpl(new PbufferImpl)
{
}


Pbuffer::~Pbuffer ()
{
  delete mImpl;
}


} // end namespace SCIRun
