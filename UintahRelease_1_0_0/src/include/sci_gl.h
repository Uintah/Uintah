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

#if !defined(SCI_GL_H)
#define SCI_GL_H

#include <sci_defs/ogl_defs.h>

#if defined(_WIN32)

/*
 * GLEW does not include <windows.h> to avoid name space pollution.
 * GL needs GLAPI and GLAPIENTRY, GLU needs APIENTRY, CALLBACK, and wchar_t
 * defined properly.
 */
/* <windef.h> */
#ifndef APIENTRY
#define GLEW_APIENTRY_DEFINED
#  if defined(__CYGWIN__) || defined(__MINGW32__)
#    define APIENTRY __stdcall
#  elif (_MSC_VER >= 800) || defined(_STDCALL_SUPPORTED)
#    define APIENTRY __stdcall
#  else
#    define APIENTRY
#  endif
#endif

#ifndef GLAPI
#  if defined(__CYGWIN__) || defined(__MINGW32__)
#    define GLAPI extern
#  endif
#endif

/* <winnt.h> */
#ifndef CALLBACK
#define GLEW_CALLBACK_DEFINED
#  if defined(__CYGWIN__) || defined(__MINGW32__)
#    define CALLBACK __attribute__ ((__stdcall__))
#  elif (defined(_M_MRX000) || defined(_M_IX86) || defined(_M_ALPHA) || defined(_M_PPC)) && !defined(MIDL_PASS)
#    define CALLBACK __stdcall
#  else
#    define CALLBACK
#  endif
#endif

/* <wingdi.h> and <winnt.h> */
#ifndef WINGDIAPI
#define GLEW_WINGDIAPI_DEFINED
#define WINGDIAPI __declspec(dllimport)
#endif

/* <ctype.h> */
#if defined(_MSC_VER) && !defined(_WCHAR_T_DEFINED)
typedef unsigned short wchar_t;
#  define _WCHAR_T_DEFINED
#endif

/* <stddef.h> */
#if !defined(_W64)
#  if !defined(__midl) && (defined(_X86_) || defined(_M_IX86)) && _MSC_VER >= 1300
#    define _W64 __w64
#  else
#    define _W64
#  endif
#endif

#ifndef GLAPI
#  if defined(__CYGWIN__) || defined(__MINGW32__)
#    define GLAPI extern
#  else
#    define GLAPI WINGDIAPI
#  endif
#endif

#ifndef GLAPIENTRY
#define GLAPIENTRY APIENTRY
#endif

#else // ! _WIN32

#ifndef GLAPIENTRY
  #define GLAPIENTRY
#endif

#ifndef GLAPI
  #define GLAPI
#endif

#endif // _WIN32

#if defined(HAVE_GLEW)

#include <GL/glew.h>

#ifdef __cplusplus
extern "C" {
#endif

extern int sci_glew_init();

#ifdef __cplusplus
}
#endif

#else /* !HAVE_GLEW */
 #if defined(CORRECT_OGLEXT_HDRS)
  #define GL_GLEXT_PROTOTYPES
  #include <GL/gl.h>
  #include <GL/glext.h>
 #else
  #include <GL/gl.h>
 #endif
#endif

#ifdef _WIN32
// gl extensions - declare and do them once (the ones that can be used in many places)
//   leave the pbuffer ones where they are.
// these will be initialized in TkOpenGLContext.cc
#  define GL_TEXTURE_3D 0x806F
#  define GL_TEXTURE0 0x84C0
#  define GL_TEXTURE1 0x84C1
#  define GL_TEXTURE2 0x84C2
#  define GL_TEXTURE3 0x84C3

#  define GL_TEXTURE0_ARB 0x84C0
#  define GL_TEXTURE1_ARB 0x84C1
#  define GL_TEXTURE2_ARB 0x84C2
#  define GL_MAX_TEXTURE_UNITS_ARB 0x84E2

#  define GL_TEXTURE_3D 0x806F
#  define GL_CLAMP_TO_EDGE 0x812F
#  define GL_TEXTURE_WRAP_R 0x8072
#  define GL_UNPACK_IMAGE_HEIGHT 0x806E
#  define GL_SHARED_TEXTURE_PALETTE_EXT 0x81FB
#  define GL_PROXY_TEXTURE_3D 0x8070

#  define GL_FRAGMENT_PROGRAM_ARB 0x8804

#  define GL_ARB_fragment_program 1
#  define GL_EXT_shared_texture_palette 1

#  define GL_FUNC_ADD 0x8006
#  define GL_MAX 0x8008

  typedef unsigned int uint;

  typedef void (GLAPIENTRY * PFNGLACTIVETEXTUREPROC) (GLenum texture);
  typedef void (GLAPIENTRY * PFNGLBLENDEQUATIONPROC) (GLenum mode);
  typedef void (GLAPIENTRY * PFNGLTEXIMAGE3DPROC) (GLenum target, GLint level, GLint internalFormat, GLsizei width, GLsizei height, GLsizei depth, GLint border, GLenum format, GLenum type, const GLvoid *pixels);
  typedef void (GLAPIENTRY * PFNGLTEXSUBIMAGE3DPROC) (GLenum target, GLint level, GLint xoffset, GLint yoffset, GLint zoffset, GLsizei width, GLsizei height, GLsizei depth, GLenum format, GLenum type, const GLvoid *pixels);
  typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD1FPROC) (GLenum target, GLfloat s);
  typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD2FVPROC) (GLenum target, const GLfloat *v);
  typedef void (GLAPIENTRY * PFNGLMULTITEXCOORD3FPROC) (GLenum target, GLfloat s, GLfloat t, GLfloat r);
  typedef void (GLAPIENTRY * PFNGLCOLORTABLEPROC) (GLenum target, GLenum internalformat, GLsizei width, GLenum format, GLenum type, const GLvoid *table);

  __declspec(dllimport) PFNGLACTIVETEXTUREPROC glActiveTexture;
  __declspec(dllimport) PFNGLBLENDEQUATIONPROC glBlendEquation;
  __declspec(dllimport) PFNGLTEXIMAGE3DPROC glTexImage3D;
  __declspec(dllimport) PFNGLTEXSUBIMAGE3DPROC glTexSubImage3D;
  __declspec(dllimport) PFNGLMULTITEXCOORD1FPROC glMultiTexCoord1f;
  __declspec(dllimport) PFNGLMULTITEXCOORD2FVPROC glMultiTexCoord2fv;
  __declspec(dllimport) PFNGLMULTITEXCOORD3FPROC glMultiTexCoord3f;
  __declspec(dllimport) PFNGLCOLORTABLEPROC glColorTable;

#endif

#endif  /* #define SCI_GL_H */
