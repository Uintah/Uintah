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

#include <Core/Volume/Pbuffer.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Util/Environment.h>

#include <iostream>
#include <string>

#include <stdio.h>

#include <sci_glu.h>
#include <sci_glx.h>

using std::cerr;
using std::endl;
using std::string;

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

#ifndef _WIN32
typedef void ( * PFNGLXBINDTEXIMAGEATIPROC) (Display *dpy, GLXPbuffer pbuf, int buffer);
typedef void ( * PFNGLXRELEASETEXIMAGEATIPROC) (Display *dpy, GLXPbuffer pbuf, int buffer);
#endif

#endif /* GLX_ATI_render_texture */

#ifndef GLX_NV_float_buffer

#define GLX_FLOAT_COMPONENTS_NV 0x20B0

#endif /* GLX_NV_float_buffer */

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

#ifndef _WIN32
static PFNGLXBINDTEXIMAGEATIPROC glXBindTexImageATI = 0;
static PFNGLXRELEASETEXIMAGEATIPROC glXReleaseTexImageATI = 0;
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
  PbufferImpl ()
   : mDisplay(0), mPbuffer(0), mContext(0) {}
  Display* mDisplay;
  GLXPbuffer mPbuffer; 
  GLXContext mContext;

  Display* mSaveDisplay;
  GLXDrawable mSaveDrawable;
  GLXContext mSaveContext;
#endif
};

bool
Pbuffer::create ()
{
  if (sci_getenv_p("SCIRUN_DISABLE_PBUFFERS"))
  {
    mSupported = false;
    return true;
  }

#if defined(__ECC) || defined(_WIN32)
  // For now no Pbuffer support on the Altix or windows system
  mSupported = false;
  return true;
#else
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
        if(mShader->create()) return true;
      }
    }
    return false;
  }
  return true;
#endif
}

void
Pbuffer::destroy ()
{
#ifndef _WIN32
  if (mSeparate && mImpl->mContext != 0)
  {
    //glXMakeCurrent(mImpl->mDisplay, mImpl->mPbuffer, 0);
    glXDestroyContext(mImpl->mDisplay, mImpl->mContext);
  }
  if (mImpl->mPbuffer != 0)
    glXDestroyPbuffer(mImpl->mDisplay, mImpl->mPbuffer);
  if(mShader)
    mShader->destroy();
#endif
}

void
Pbuffer::makeCurrent ()
{
#ifndef _WIN32
  // set read/write context to pbuffer
  //if (mImpl->mPbuffer != glXGetCurrentDrawable()
  //    || mImpl->mContext != glXGetCurrentContext())
  {
    //glXMakeContextCurrent(mImpl->mDisplay, mImpl->mPbuffer,
    //			  mImpl->mPbuffer, mImpl->mContext);
    glXMakeCurrent(mImpl->mDisplay, mImpl->mPbuffer, mImpl->mContext);
  }
#endif
}

void
Pbuffer::swapBuffers ()
{
#ifndef _WIN32
  if(mRenderTex && !mATI_render_texture) {
    GLint buffer;
    glGetIntegerv(GL_DRAW_BUFFER, &buffer);
    glReadBuffer(buffer);
    glBindTexture(mTexTarget, mTex);
    glCopyTexSubImage2D(mTexTarget, 0, 0, 0, 0, 0, mWidth, mHeight);
    glBindTexture(mTexTarget, 0);
  }
  if (mDoubleBuffer)
    glXSwapBuffers(mImpl->mDisplay, mImpl->mPbuffer);
  else
    glFinish();
#endif
}

void
Pbuffer::bind (unsigned int buffer)
{
#ifndef _WIN32
  if(mRenderTex)
  {
    glEnable(mTexTarget);
    glBindTexture(mTexTarget, mTex);
    if(mATI_render_texture) {
      glXBindTexImageATI(mImpl->mDisplay, mImpl->mPbuffer, buffer == GL_FRONT ? 
                         GLX_FRONT_LEFT_ATI : GLX_BACK_LEFT_ATI);
    }
    if(mFormat == GL_FLOAT && mNV_float_buffer) {
      if(mUseDefaultShader) {
        mShader->bind();
      }
      if(mUseTextureMatrix) {
        glMatrixMode(GL_TEXTURE);
        glPushMatrix();
        glLoadIdentity();
        glScalef(mWidth, mHeight, 1.0);
        glMatrixMode(GL_MODELVIEW);
      }
    }
  }
#endif
}

void
Pbuffer::release (unsigned int buffer)
{
#ifndef _WIN32
  if(mRenderTex)
  {
    if(mATI_render_texture) {
      glXReleaseTexImageATI(mImpl->mDisplay, mImpl->mPbuffer, buffer == GL_FRONT ?
                            GLX_FRONT_LEFT_ATI : GLX_BACK_LEFT_ATI);
    }
    glBindTexture(mTexTarget, 0);
    glDisable(mTexTarget);
    if(mFormat == GL_FLOAT && mNV_float_buffer) {
      if(mUseTextureMatrix) {
        glMatrixMode(GL_TEXTURE);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
      }
      if(mUseDefaultShader) {
        mShader->release();
      }
    }
  }
#endif
}

void
Pbuffer::activate ()
{
#ifndef _WIN32
  // save context state
  mImpl->mSaveDisplay = glXGetCurrentDisplay();
  mImpl->mSaveDrawable = glXGetCurrentDrawable();
  mImpl->mSaveContext = glXGetCurrentContext();
  // set read/write context to pbuffer
  glXMakeCurrent(mImpl->mDisplay, mImpl->mPbuffer, mImpl->mContext);
#endif
}

void
Pbuffer::deactivate ()
{
#ifndef _WIN32
  glXMakeCurrent(mImpl->mSaveDisplay, mImpl->mSaveDrawable, mImpl->mSaveContext);
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
    mSeparate(true), mTex(0),
    mTexTarget(GL_TEXTURE_2D),
    mUseDefaultShader(true),
    mImpl(new PbufferImpl)
{}

Pbuffer::~Pbuffer ()
{
  delete mImpl;
}

} // end namespace SCIRun
