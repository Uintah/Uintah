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

#include <Packages/Volume/Core/Util/Pbuffer.h>
#include <iostream>

#ifdef _WIN32
#include <GL/wglew.h>
#elif defined(__APPLE__)
#include <AGL/agl.h>
#else
#include <GL/glxew.h>
#endif

using std::cerr;
using std::endl;

namespace Volume {

#ifdef _WIN32

struct PbufferImpl
{
  PbufferImpl ()
    : mPbuffer(0), mDc(0), mRc(0) {}
  HPBUFFERARB mPbuffer; 
  HDC mDc;
  HGLRC mRc;
};

bool
Pbuffer::create ()
{
  // extension check
  if (!WGLEW_ARB_pbuffer
      || (mSeparate && !WGLEW_ARB_pixel_format)
      || (mRenderTex && !WGLEW_ARB_render_texture)
      || (mFormat == GL_FLOAT
	  && (!WGLEW_ATI_pixel_format_float && !WGLEW_NV_float_buffer)))
  {
    cerr << "[Pbuffer::create] The required extensions are not "
      "supported on this platform" << endl;
    return true;
  }
  // get the current opengl device and rendering context
  HDC dc = wglGetCurrentDC();
  HGLRC rc = wglGetCurrentContext();
  if (dc == 0 || rc == 0)
  {
    cerr << "[Pbuffer::create] Failed to obtain current device"
      " and rendering contexts" << endl;
    return true;
  }
  // find suitable visual for the pbuffer
  if (mSeparate)
  {
    // choose a pixel format that meets the specification
    int attrib[64];
    int i = 0;
    // accelerated OpenGL support
    attrib[i++] = WGL_SUPPORT_OPENGL_ARB;
    attrib[i++] = GL_TRUE;
    // pbuffer capable
    attrib[i++] = WGL_DRAW_TO_PBUFFER_ARB;
    attrib[i++] = GL_TRUE;
    // format
    if (mFormat == GL_FLOAT)
    {
      if (WGLEW_ATI_pixel_format_float)
      {
	attrib[i++] = WGL_PIXEL_TYPE_ARB;
	attrib[i++] = WGL_TYPE_RGBA_FLOAT_ATI;
      }
      else if (WGLEW_NV_float_buffer)
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
//       attrib[i++] = WGL_ALPHA_BITS_ARB;
//       attrib[i++] = mNumColorBits;
//       attrib[i++] = WGL_COLOR_BITS_ARB;
//       attrib[i++] = 3*mNumColorBits;
    }
    // double buffer spec
    if (mDoubleBuffer != GL_DONT_CARE)
    {
      attrib[i++] = WGL_DOUBLE_BUFFER_ARB;
      attrib[i++] = mDoubleBuffer ? GL_TRUE : GL_FALSE;
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
      if (mFormat == GL_FLOAT && WGLEW_NV_float_buffer)
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
    // find the pixel format supported by the device context that is the
    // best match to a given pixel format specification
    unsigned int c = 0;
    if (wglChoosePixelFormatARB(dc, attrib, 0, 1, &mVisualId, &c) == 0 || c == 0)
    {
      cerr << "[Pbuffer::create] Failed to find a suitable pixel format" << endl;
      return true;
    }
  }
  else
  {
    // to share a rendering context, use the same pixel format
    mVisualId = GetPixelFormat(dc);
  }
  // query visual attributes
  {
    int attrib[16], value[16];
    // color buffer info
    attrib[0] = WGL_BLUE_BITS_ARB;
    attrib[1] = WGL_ALPHA_BITS_ARB;
    // accum buffer info
    attrib[2] = WGL_ACCUM_BLUE_BITS_ARB;
    // double, depth, stencil, and aux buffer info
    attrib[3] = WGL_DOUBLE_BUFFER_ARB;
    attrib[4] = WGL_DEPTH_BITS_ARB;
    attrib[5] = WGL_STENCIL_BITS_ARB;
    attrib[6] = WGL_AUX_BUFFERS_ARB;  
    // query attributes
    wglGetPixelFormatAttribivARB(dc, mVisualId, 0, 7, attrib, value);
    mNumColorBits = value[0];
    mNumAccumBits = value[2];
    mDoubleBuffer = value[3];
    mNumDepthBits = value[4];
    mNumStencilBits = value[5];
    mNumAuxBuffers = value[6];
  }
  // allocate the buffer
  // it turns out that there is a huge penalty for copying from
  // a ren2tex enabled pbuffer on nvidia boards and it's not possible
  // to do it on ati cards
  // so just make sure you disable ren2tex if you don't need it
  {
    int attrib[16];
    int i = 0;
    if (mRenderTex)
    {
      // format and target
      if (mFormat == GL_FLOAT && WGLEW_NV_float_buffer)
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
      attrib[i++] = GL_FALSE; //mMipMap ? GL_TRUE : GL_FALSE;
    }
    attrib[i++] = WGL_PBUFFER_LARGEST_ARB;
    attrib[i++] = GL_FALSE;
    attrib[i++] = 0;
    mImpl->mPbuffer = wglCreatePbufferARB(dc, mVisualId, mWidth, mHeight, attrib);
    // create device context
    mImpl->mDc = wglGetPbufferDCARB(mImpl->mPbuffer);
    if (mImpl->mDc == 0)
    {
      cerr << "[Pbuffer::create] Failed to get device context" << endl;
      return true;
    }
    // create rendering context
    if (mSeparate)
    {
      mImpl->mRc = wglCreateContext(mImpl->mDc);
      if (mImpl->mRc == 0)
      {
	cerr << "[Pbuffer::create] Failed to create rendering context" << endl;
	return true;
      }
      if (wglShareLists(rc, mImpl->mRc) == 0)
      {
	cerr << "[Pbuffer::create] Failed to set context sharing" << endl;
	return true;
      }
    }
    else
    {
      mImpl->mRc = rc;
    }
    wglQueryPbufferARB(mImpl->mPbuffer, WGL_PBUFFER_WIDTH_ARB, &mWidth);
    wglQueryPbufferARB(mImpl->mPbuffer, WGL_PBUFFER_HEIGHT_ARB, &mHeight);
  }
  if (mRenderTex)
  {
    // create pbuffer texture object
    glGenTextures(1, &mTex);
    if (mFormat == GL_FLOAT && WGLEW_NV_float_buffer)
      mTexFormat = GL_TEXTURE_RECTANGLE_NV;
    else
      mTexFormat = GL_TEXTURE_2D;
    glBindTexture(mTexFormat, mTex);
    glTexParameteri(mTexFormat, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(mTexFormat, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(mTexFormat, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(mTexFormat, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  }
  return false;
}

void
Pbuffer::destroy ()
{
  if (mImpl->mDc)
    wglMakeCurrent(mImpl->mDc, 0);
  if (mSeparate && mImpl->mRc)
    wglDeleteContext(mImpl->mRc);
  if (mImpl->mPbuffer && mImpl->mDc)
    wglReleasePbufferDCARB(mImpl->mPbuffer, mImpl->mDc);
  if (mImpl->mPbuffer)
    wglDestroyPbufferARB(mImpl->mPbuffer);
  if (glIsTexture(mTex))
    glDeleteTextures(1, &mTex);
}

void
Pbuffer::makeCurrent ()
{
  // set read/write context to pbuffer
  if (mImpl->mDc != wglGetCurrentDC()
      || mImpl->mRc != wglGetCurrentContext())
  {
    cerr << "Pbuffer::makeCurrent" << endl;
    wglMakeCurrent(mImpl->mDc, mImpl->mRc);
  }
}

void
Pbuffer::swapBuffers ()
{
  if (mDoubleBuffer)
    wglSwapLayerBuffers(mImpl->mDc, WGL_SWAP_MAIN_PLANE);
  else
    glFlush();
}

void
Pbuffer::bind (unsigned int buffer)
{
  if (mRenderTex)
  {
    glBindTexture(mTexFormat, mTex);
    unsigned int wb;
    if (buffer >= GL_AUX0 && buffer <= GL_AUX3)
      wb = WGL_AUX0_ARB + (buffer - GL_AUX0);
    else if (buffer >= GL_FRONT_LEFT && buffer <= GL_BACK_RIGHT)
      wb = WGL_FRONT_LEFT_ARB + (buffer - GL_FRONT_LEFT);
    else if (buffer == GL_FRONT)
      wb = WGL_FRONT_LEFT_ARB;
    else if (buffer == GL_BACK)
      wb = WGL_BACK_LEFT_ARB;
    wglBindTexImageARB(mImpl->mPbuffer, wb);
  }
}

void
Pbuffer::release (unsigned int buffer)
{
  if (mRenderTex)
  {
    unsigned int wb;
    if (buffer >= GL_AUX0 && buffer <= GL_AUX3)
      wb = WGL_AUX0_ARB + (buffer - GL_AUX0);
    else if (buffer >= GL_FRONT_LEFT && buffer <= GL_BACK_RIGHT)
      wb = WGL_FRONT_LEFT_ARB + (buffer - GL_FRONT_LEFT);
    else if (buffer == GL_FRONT)
      wb = WGL_FRONT_LEFT_ARB;
    else if (buffer == GL_BACK)
      wb = WGL_BACK_LEFT_ARB;
    wglReleaseTexImageARB(mImpl->mPbuffer, wb);
    glBindTexture(mTexFormat, 0);
  }
}

#elif defined(__APPLE__) && !defined(PBUFFER_APPLE_GLX)

//------------------------------------------------------------------------------

struct PbufferImpl
{
  PbufferImpl()
    : mContext(0), mPbuffer(0) {}
  AGLContext mContext;
  AGLPbuffer mPbuffer;
};

bool
Pbuffer::create ()
{
  //
  // extension check
  //
  
  // get current context
  AGLContext ctx = aglGetCurrentContext();
  if (0 == ctx)
  {
    cerr << "[Pbuffer::create] Failed to obtain current context" << endl;
    return true;
  }
  //
  // TODO: Need to check if PBuffers are actually supported
  //

  // find suitable visual for the pbuffer
  AGLPixelFormat pf;
  int attrib[64];
  if (mSeparate)
  {
    // choose a pixel format that meets the specification
    int i = 0;
    // accelerated
    attrib[i++] = AGL_ACCELERATED;
    // offscreen capable
    //attrib[i++] = AGL_OFFSCREEN;
    attrib[i++] = AGL_CLOSEST_POLICY; //implied by AGL_OFFSCREEN
    attrib[i++] = AGL_NO_RECOVERY;
    // TODO: Is there a way to set this flag?
    if (mFormat == GL_FLOAT)
    {
      // TODO: Need to figure out how to support FLOAT pbuffers
      attrib[i++] = AGL_RGBA;
      cerr << "[PBuffer::create(): FLOAT pbuffers not supported yet!" << endl;
      assert(0);
    }
    else // GL_INT
    {
      attrib[i++] = AGL_RGBA;
    }
    // color buffer spec
    if (mNumColorBits != GL_DONT_CARE)
    {
      attrib[i++] = AGL_RED_SIZE;
      attrib[i++] = mNumColorBits;
      attrib[i++] = AGL_GREEN_SIZE;
      attrib[i++] = mNumColorBits;
      attrib[i++] = AGL_BLUE_SIZE;
      attrib[i++] = mNumColorBits;
      attrib[i++] = AGL_ALPHA_SIZE;
      attrib[i++] = mNumColorBits;
      attrib[i++] = AGL_BUFFER_SIZE;
      attrib[i++] = mNumColorBits*4;
    }
    // double buffer spec
    if (mDoubleBuffer != GL_DONT_CARE)
    {
      attrib[i++] = AGL_DOUBLEBUFFER;
      attrib[i++] = mDoubleBuffer ? GL_TRUE : GL_FALSE;
    }
    // aux buffer spec
    if (mNumAuxBuffers != GL_DONT_CARE)
    {
      attrib[i++] = AGL_AUX_BUFFERS;
      attrib[i++] = mNumAuxBuffers;
    }
    // depth buffer spec
    //
    if (mNumDepthBits != GL_DONT_CARE)
    {
      attrib[i++] = AGL_DEPTH_SIZE;
      attrib[i++] = mNumDepthBits;
    }
    // stencil buffer spec
    if (mNumStencilBits != GL_DONT_CARE)
    {
      attrib[i++] = AGL_STENCIL_SIZE;
      attrib[i++] = mNumStencilBits;
    }
    // accum buffer spec
    if (mNumAccumBits != GL_DONT_CARE)
    {
      attrib[i++] = AGL_ACCUM_RED_SIZE;
      attrib[i++] = mNumAccumBits;
      attrib[i++] = AGL_ACCUM_GREEN_SIZE;
      attrib[i++] = mNumAccumBits;
      attrib[i++] = AGL_ACCUM_BLUE_SIZE;
      attrib[i++] = mNumAccumBits;
      attrib[i++] = AGL_ACCUM_ALPHA_SIZE;
      attrib[i++] = mNumAccumBits;
    }
    // render to texture
    if (mRenderTex)
    {
      if (mFormat == GL_FLOAT && GLEW_NV_float_buffer)
      {
	cerr << "[PBuffer::create(): FLOAT pbuffers not supported yet!" << endl;
	assert( 0 );
      }
      else
      {
	// NOTHING FOR NOW
      }
    }
    //
    attrib[i] = AGL_NONE;
    // find the pixel format supported by the device context that is the
    // best match to a given pixel format specification
    pf = aglChoosePixelFormat(NULL, 0, attrib);
    if (pf == NULL)
    {
      GLenum error = aglGetError();
      cerr << "[Pbuffer::create] Failed to find a suitable pixel format: "
	   << (char*)aglErrorString(error) << endl;
      return true;
    }
  }
  else
  {
    // to share a rendering context, use the same pixel format
    // TODO: implement sharing context
    cerr << "Shared rendering context not supported yet!!!" << endl;
    return true;
  }
  // query visual attributes
  {
    // AGL_PIXEL_SIZE <= AGL_BUFFER_SIZE
    aglDescribePixelFormat(pf, AGL_PIXEL_SIZE, &mNumColorBits);
    mNumColorBits /= 4;
    aglDescribePixelFormat(pf, AGL_ALPHA_SIZE, &mNumAccumBits); 
    aglDescribePixelFormat(pf, AGL_DOUBLEBUFFER, &mDoubleBuffer); 
    aglDescribePixelFormat(pf, AGL_DEPTH_SIZE, &mNumDepthBits); 
    aglDescribePixelFormat(pf, AGL_STENCIL_SIZE, &mNumStencilBits); 
    aglDescribePixelFormat(pf, AGL_AUX_BUFFERS, &mNumAuxBuffers); 
    aglDescribePixelFormat(pf, AGL_VIRTUAL_SCREEN, &mVisualId); 
  }
  // create rendering context
  mImpl->mContext = aglCreateContext(pf, ctx);
  if (0 == mImpl->mContext)
  {
    cerr << "[PBuffer::create] Failed to create context: "
	 << (char*)aglErrorString(aglGetError()) << endl;
    return true;
  }
  // create Pbuffer
  if (!aglCreatePBuffer(mWidth, mHeight, GL_TEXTURE_2D,
			GL_RGBA, 0, &(mImpl->mPbuffer)))
  {
    cerr << "[PBuffer::create] Failed to create pbuffer: "
	 << (char*)aglErrorString(aglGetError()) << endl;
    return true;
  }
  // free pixel format
  aglDestroyPixelFormat(pf);

//   GLenum target, internalFormat;
//   GLint max_level;
//   /*
//   GLboolean ok = aglDescribePBuffer( mImpl>aglPbuffer, &mWidth, &mHeight
// 				     &target, &internalFormat, &max_level );
//   */
//   if( error != AGL_NO_ERROR ) {
//     cerr << "[PBuffer::create():] Can't read AGL Pbuffer description: " << ( ( char* ) aglErrorString( error ) ) << endl;
//   }
  
  // create texture
  if (mRenderTex)
  {
    // if no texture name create one and texture from the pbuffer
    glGenTextures(1, &mTex);
    glBindTexture(mTexFormat, mTex);
    //aglTexImagePBuffer(mImpl->mContext, mImpl->mPbuffer, GL_FRONT);
    glTexParameteri(mTexFormat, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(mTexFormat, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(mTexFormat, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(mTexFormat, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  }
  return false;
}

void
Pbuffer::destroy ()
{
  if (glIsTexture(mTex))
    glDeleteTextures(1, &mTex);
  if (mImpl->mPbuffer)
  {
    aglDestroyPBuffer(mImpl->mPbuffer);
    mImpl->mPbuffer = 0;
  }
  if (mSeparate && 0 != mImpl->mContext)
  {
    aglSetCurrentContext(mImpl->mContext);
    aglSetDrawable(mImpl->mContext, 0);
    aglSetCurrentContext(0);
    aglDestroyContext(mImpl->mContext);
    mImpl->mContext = 0;
  }
}

void
Pbuffer::makeCurrent ()
{
  if (aglGetCurrentContext() != mImpl->mContext)
	//      || aglGetDrawable(aglGetCurrentContext()) != mImpl->mPbuffer)
  {
    aglSetCurrentContext(mImpl->mContext);
    aglSetPBuffer(mImpl->mContext, mImpl->mPbuffer, 0, 0,
		  aglGetVirtualScreen(mImpl->mContext)); 
  }
}

void
Pbuffer::swapBuffers ()
{
  if (mDoubleBuffer)
    aglSwapBuffers(mImpl->mContext);
  else
    glFlush();
}

void
Pbuffer::bind (unsigned int buffer)
{
  if (mRenderTex)
  {
    cerr << "HI" << endl;
    glBindTexture(mTexFormat, mTex);
    if (!aglTexImagePBuffer(aglGetCurrentContext(), mImpl->mPbuffer, buffer))
    {
      cerr << "HEY" << endl;
    }
  }
}

void
Pbuffer::release (unsigned int buffer)
{
  if (mRenderTex)
  {
    glBindTexture(mTexFormat, 0);
  }
}

#else // UNIX

//------------------------------------------------------------------------------

struct PbufferImpl
{
  PbufferImpl ()
   : mDisplay(0), mPbuffer(0), mContext(0) {}
  Display* mDisplay;
  GLXPbuffer mPbuffer; 
  GLXContext mContext;

  Display* mSaveDisplay;
  GLXDrawable mSaveDrawable;
  GLXContext mSaveContext;
};

bool
Pbuffer::create ()
{
  // extension check
  if (!GLXEW_VERSION_1_3
      || (mRenderTex && !GLXEW_ATI_render_texture) // render texture extensions
      || (mFormat == GL_FLOAT // float buffer extensions
	  && !(GLXEW_ATI_pixel_format_float || GLXEW_NV_float_buffer)))
  {
    cerr << "[Pbuffer::create] The required extensions are not "
      "supported on this platform" << endl;
    //return true;
  }
  // get current display
  mImpl->mDisplay = glXGetCurrentDisplay();
  if (mImpl->mDisplay == 0)
  {
    cerr << "[Pbuffer::create] Failed to obtain current display" << endl;
    return true;
  }
  // get current context
  GLXContext ctx = glXGetCurrentContext();
  if (ctx == 0)
  {
    cerr << "[Pbuffer::create] Failed to obtain current context" << endl;
    return true;
  }
  // find suitable visual for the pbuffer
  int attrib[64];
  GLXFBConfig* fbc;
  int n_fbc;
  int i;
  if (mSeparate)
  {
    i = 0;
    // accelerated OpenGL support
    //attrib[i++] = WGL_SUPPORT_OPENGL_ARB;
    //attrib[i++] = GL_TRUE;
    // pbuffer capable
    attrib[i++] = GLX_DRAWABLE_TYPE;
    attrib[i++] = mNumColorBits > 8 ? GLX_PBUFFER_BIT :
      GLX_PBUFFER_BIT | GLX_WINDOW_BIT;
    // format
    if (mFormat == GL_FLOAT)
    {
      //if (GLXEW_ATI_pixel_format_float)
      {
	attrib[i++] = GLX_RENDER_TYPE;
	attrib[i++] = GLX_RGBA_FLOAT_ATI_BIT;
      }
//       else if (GLXEW_NV_float_buffer)
//       {
// 	attrib[i++] = GLX_RENDER_TYPE;
// 	attrib[i++] = GLX_RGBA_BIT;
// 	attrib[i++] = GLX_FLOAT_COMPONENTS_NV;
// 	attrib[i++] = GL_TRUE;
//       }
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
//       attrib[i++] = GLX_ALPHA_SIZE;
//       attrib[i++] = mNumColorBits;
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
      //if (GLXEW_ATI_render_texture)
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
    if (glXQueryContext(mImpl->mDisplay, ctx, GLX_FBCONFIG_ID, &id) != Success)
    {
      cerr << "[Pbuffer::create] Failed to query fbconfig id from context"
	   << endl;
      return true;
    }
    // choose fb config with given id
    attrib[0] = GLX_FBCONFIG_ID;
    attrib[1] = id;
    attrib[2] = None;
  }
  // choose fb config
  fbc = glXChooseFBConfig(mImpl->mDisplay, DefaultScreen(mImpl->mDisplay),
			  attrib, &n_fbc);
  if (fbc == 0 || n_fbc == 0)
  {
    cerr << "[Pbuffer::create] Failed to obtain fb config" << endl;
    return true;
  }
  glXGetFBConfigAttrib(mImpl->mDisplay, *fbc, GLX_FBCONFIG_ID, &mVisualId);
  glXGetFBConfigAttrib(mImpl->mDisplay, *fbc, GLX_RED_SIZE, &mNumColorBits);
#if 0
  // get visual id from fb config
  XVisualInfo* vi = glXGetVisualFromFBConfig(mImpl->mDisplay, *fbc);
  if (vi == 0)
  {
    cerr << "[Pbuffer::create] Failed to get visual from fb config" << endl;
    return true;
  }
  mVisualId = XVisualIDFromVisual(vi->visual);
#endif
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
  if (mRenderTex && GLXEW_ATI_render_texture)
  {
    attrib[i++] = GLX_TEXTURE_FORMAT_ATI;
    attrib[i++] = GLX_TEXTURE_RGBA_ATI;
    attrib[i++] = GLX_TEXTURE_TARGET_ATI;
    attrib[i++] = GLX_TEXTURE_2D_ATI;
    attrib[i++] = GLX_MIPMAP_TEXTURE_ATI;
    attrib[i++] = GL_FALSE;
  }
  attrib[i] = None;
  mImpl->mPbuffer = glXCreatePbuffer(mImpl->mDisplay, *fbc, attrib);
  if (mImpl->mPbuffer == 0)
  {
    cerr << "[Pbuffer::create] Failed to create pbuffer" << endl;
    return true;
  }
  // create context
  if (mSeparate)
  {
    mImpl->mContext = glXCreateNewContext(mImpl->mDisplay, *fbc, GLX_RGBA_TYPE,
					  ctx, True);
    if (mImpl->mContext == 0)
    {
      cerr << "[Pbuffer::create] Failed to create context" << endl;
      return true;
    }
  }
  else
  {
    mImpl->mContext = ctx;
  }
  // query attributes
  glXQueryDrawable(mImpl->mDisplay, mImpl->mPbuffer, GLX_WIDTH,
		   (unsigned int*)&mWidth);
  glXQueryDrawable(mImpl->mDisplay, mImpl->mPbuffer, GLX_HEIGHT,
		   (unsigned int*)&mHeight);
  cerr << mWidth << " x " << mHeight << endl;
  // ...
  if (mRenderTex)
  {
    // create pbuffer texture object
    glGenTextures(1, &mTex);
    if (mFormat == GL_FLOAT && GLXEW_NV_float_buffer)
      mTexFormat = GL_TEXTURE_RECTANGLE_NV;
    else
      mTexFormat = GL_TEXTURE_2D;
    glBindTexture(mTexFormat, mTex);
    glTexParameteri(mTexFormat, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(mTexFormat, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexParameteri(mTexFormat, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(mTexFormat, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  }
  return false;
}

void
Pbuffer::destroy ()
{
  if (mSeparate && mImpl->mContext != 0)
  {
    //glXMakeCurrent(mImpl->mDisplay, mImpl->mPbuffer, 0);
    glXDestroyContext(mImpl->mDisplay, mImpl->mContext);
  }
  if (mImpl->mPbuffer != 0)
    glXDestroyPbuffer(mImpl->mDisplay, mImpl->mPbuffer);
}

void
Pbuffer::makeCurrent ()
{
  // set read/write context to pbuffer
  //if (mImpl->mPbuffer != glXGetCurrentDrawable()
  //    || mImpl->mContext != glXGetCurrentContext())
  {
    //glXMakeContextCurrent(mImpl->mDisplay, mImpl->mPbuffer,
    //			  mImpl->mPbuffer, mImpl->mContext);
    glXMakeCurrent(mImpl->mDisplay, mImpl->mPbuffer, mImpl->mContext);
  }
}

void
Pbuffer::swapBuffers ()
{
  if (mDoubleBuffer)
    glXSwapBuffers(mImpl->mDisplay, mImpl->mPbuffer);
  else
    glFlush();
}

void
Pbuffer::bind (unsigned int buffer)
{
  if (mRenderTex)
  {
    glEnable(mTexFormat);
    glBindTexture(mTexFormat, mTex);
    glXBindTexImageATI(mImpl->mDisplay, mImpl->mPbuffer, buffer == GL_FRONT ? 
		       GLX_FRONT_LEFT_ATI : GLX_BACK_LEFT_ATI);
  }
}

void
Pbuffer::release (unsigned int buffer)
{
  if (mRenderTex)
  {
    glXReleaseTexImageATI(mImpl->mDisplay, mImpl->mPbuffer, buffer == GL_FRONT ?
			  GLX_FRONT_LEFT_ATI : GLX_BACK_LEFT_ATI);
    glBindTexture(mTexFormat, 0);
    glDisable(mTexFormat);
  }
}

void
Pbuffer::enable ()
{
  // save context state
  mImpl->mSaveDisplay = glXGetCurrentDisplay();
  mImpl->mSaveDrawable = glXGetCurrentDrawable();
  mImpl->mSaveContext = glXGetCurrentContext();
  // set read/write context to pbuffer
  glXMakeCurrent(mImpl->mDisplay, mImpl->mPbuffer, mImpl->mContext);
}

void
Pbuffer::disable ()
{
  glXMakeCurrent(mImpl->mSaveDisplay, mImpl->mSaveDrawable, mImpl->mSaveContext);
}

#endif // _WIN32

Pbuffer::Pbuffer (int width, int height, bool isRenderTex)
  : mWidth(width), mHeight(height), mRenderTex(isRenderTex), mSeparate(false),
    mTex(0), mTexFormat(GL_TEXTURE_2D), mImpl(new PbufferImpl)
{}

Pbuffer::Pbuffer (int width, int height, int format, int numColorBits,
		  /* int numChannels, */ bool isRenderTex, int isDoubleBuffer,
		  int numAuxBuffers, int numDepthBits, int numStencilBits,
		  int numAccumBits)
  : mWidth(width), mHeight(height), mFormat(format), mNumColorBits(numColorBits),
    /* mNumChannels(numChannels), */ mRenderTex(isRenderTex),
    mDoubleBuffer(isDoubleBuffer), mNumAuxBuffers(numAuxBuffers),
    mNumDepthBits(numDepthBits), mNumStencilBits(numStencilBits),
    mNumAccumBits(numAccumBits), mSeparate(true), mTex(0),
    mTexFormat(GL_TEXTURE_2D), mImpl(new PbufferImpl)
{}

Pbuffer::~Pbuffer ()
{
  delete mImpl;
}

} // end namespace Volume
