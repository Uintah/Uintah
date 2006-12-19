//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
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
//    File   : OSXOpenGLContext.cc
//    Author : McKay Davis
//    Date   : Thu Sep  7 09:09:47 2006
#if defined (__APPLE__)

#include <Core/Geom/OSXOpenGLContext.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/Color.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h> // for SWAP
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/Assert.h>

// General gui lock, not really x11 specific
#include <Core/Geom/X11Lock.h>

#include <iostream>
#include <set>

using namespace SCIRun;
using namespace std;
  
namespace SCIRun {

AGLContext OSXOpenGLContext::first_context_ = NULL;

OSXOpenGLContext::OSXOpenGLContext(int x, 
                                   int y,
                                   unsigned int width, 
                                   unsigned int height,
                                   bool border) : 
  OpenGLContext(),
  mutex_("GL lock"),
  width_(width),
  height_(height)
{
  // Sepeate functions so we can set gdb breakpoints in constructor
  create_context(x, y, width, height, border);
}



void
OSXOpenGLContext::create_context(int x,
                                 int y,
                                 unsigned int width, 
                                 unsigned int height,
                                 bool border)
{



  Rect win_rect;
  win_rect.left = x;
  win_rect.right = x + width;
  win_rect.top = y;
  win_rect.bottom = y + height;

  //  SetRect(&win_rect, x, y, width, height);
  
  int win_flags = (kWindowStandardDocumentAttributes | 
                   //kWindowStandardHandlerAttribute | 
                   kWindowLiveResizeAttribute);

  win_flags &= GetAvailableWindowAttributes(kDocumentWindowClass);

  CreateNewWindow(kDocumentWindowClass, win_flags, &win_rect, &window_);
  SetWindowTitleWithCFString(window_, CFSTR("Seg3D"));

  ProcessSerialNumber id;
  GetCurrentProcess(&id);
  SetFrontProcess(&id);

  ShowWindow(window_);
  ActivateWindow(window_, true);

  GetWindowPortBounds(window_, &win_rect);
  width_ = win_rect.right - win_rect.left;
  height_ = win_rect.bottom - win_rect.top;

  GLint pix_attr[] = {
    AGL_RGBA,
    AGL_GREEN_SIZE, 1,
    AGL_DOUBLEBUFFER,
    AGL_DEPTH_SIZE, 16,
    AGL_NONE
  };

  AGLPixelFormat apf = aglChoosePixelFormat (NULL, 0, pix_attr);
  context_ = aglCreateContext(apf, first_context_);
  aglDestroyPixelFormat(apf);

  ASSERT(context_);

  aglSetDrawable(context_, GetWindowPort(window_));
  if (!first_context_) {
    first_context_ = context_;
    make_current();
    ShaderProgramARB::init_shaders_supported();
    release();
  }

}



OSXOpenGLContext::~OSXOpenGLContext()
{
  release();
  X11Lock::lock();
  aglDestroyContext(context_);
  DisposeWindow(window_);
  X11Lock::unlock();
}


bool
OSXOpenGLContext::make_current()
{
  ASSERT(context_);
  bool result = true;

  //  OSXLock::lock();
  result = aglSetCurrentContext(context_);
  // OSXLock::unlock();

  if (!result)
  {
    std::cerr << "OSX GL context failed make current.\n";
  }

  return result;
}


void
OSXOpenGLContext::release()
{
  //  X11Lock::lock();
  //  glXMakeCurrent(display_, None, NULL);
  //  X11Lock::unlock();
}


int
OSXOpenGLContext::width()
{
  Rect win_rect;
  GetWindowPortBounds(window_, &win_rect);
  width_ = win_rect.right - win_rect.left;
  height_ = win_rect.bottom - win_rect.top;

  return width_;
}


int
OSXOpenGLContext::height()
{
  Rect win_rect;
  GetWindowPortBounds(window_, &win_rect);
  width_ = win_rect.right - win_rect.left;
  height_ = win_rect.bottom - win_rect.top;

  return height_;
}


void
OSXOpenGLContext::swap()
{  
  X11Lock::lock();
  aglSwapBuffers(context_);
  aglUpdateContext(context_);
  DrawGrowIcon(window_);
  X11Lock::unlock();
}



}

#endif
