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
//    File   : X11OpenGLContext.cc
//    Author : McKay Davis
//    Date   : May 30 2006

#include <Core/Geom/X11OpenGLContext.h>
#include <Core/Geom/X11Lock.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/Color.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h> // for SWAP
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/Assert.h>
#include <sci_glx.h>
#include <iostream>
#include <set>

// X11 includes
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xmu/StdCmap.h>


using namespace SCIRun;
using namespace std;
  
vector<int> X11OpenGLContext::valid_visuals_ = vector<int>();
GLXContext X11OpenGLContext::first_context_ = NULL;

X11OpenGLContext::X11OpenGLContext(int visual, 
                                   int x, 
                                   int y,
                                   unsigned int width, 
                                   unsigned int height) : 
  OpenGLContext(),
  mutex_("GL lock")
{
  // Sepeate functions so we can set gdb breakpoints in constructor
  create_context(visual, x, y, width, height);
}



void
X11OpenGLContext::create_context(int visual, int x, int y,
                                 unsigned int width, 
                                 unsigned int height)
{
  X11Lock::lock();
  display_ = XOpenDisplay((char *)0);
  XSync(display_, False);

  screen_ = DefaultScreen(display_);
  root_window_ = DefaultRootWindow(display_);

  if (valid_visuals_.empty())
    listvisuals();
  ASSERT(!valid_visuals_.empty());

  visual = Clamp(visual, 0, (int)valid_visuals_.size()-1);
  visualid_ = valid_visuals_[visual];
  ASSERT(visualid_);

  int n;
  XVisualInfo temp_vi;
  temp_vi.visualid = visualid_;
  vi_ = XGetVisualInfo(display_, VisualIDMask, &temp_vi, &n);
  if(!vi_ || n != 1) {
    throw ("Cannot find Visual ID #" + to_string(visualid_) + 
           string(__FILE__)+to_string(__LINE__));
    X11Lock::unlock();
  }
    

  attributes_.colormap = XCreateColormap(display_, root_window_, 
                                         vi_->visual, AllocNone);
  attributes_.override_redirect = false;
  //  unsigned int valuemask = (CWX | CWY | CWWidth | CWHeight | 
  //                            CWBorderWidth | CWSibling | CWStackMode);

  window_ = XCreateWindow(display_, 
                          root_window_,
                          x, y, 
                          width, height,
                          0,
                          vi_->depth,
                          InputOutput,
                          vi_->visual,
                          //                          0, //valuemask,
                          CWColormap | CWOverrideRedirect,
                          &attributes_);
  
  if (!window_) {
    throw ("Cannot create X11 Window " + 
           string(__FILE__)+to_string(__LINE__));
    X11Lock::unlock();
  }

  XMapRaised(display_, window_);
  XMoveResizeWindow(display_, window_, x, y, width, height);
  XSync(display_, False);

  context_ = glXCreateContext(display_, vi_, first_context_, 1);
  if (!context_) {
    throw ("Cannot create GLX Context" + 
           string(__FILE__)+to_string(__LINE__));
    X11Lock::unlock();
  }


  X11Lock::unlock();

  if (!first_context_)
    first_context_ = context_;

  width_ = width;
  height_ = height;

}



X11OpenGLContext::~X11OpenGLContext()
{
  release();
  X11Lock::lock();
  XSync(display_, False);

  glXDestroyContext(display_, context_);

  XSync(display_, False);

  XDestroyWindow(display_,window_);

  XSync(display_, False);

  XCloseDisplay(display_);

  X11Lock::unlock();

}


bool
X11OpenGLContext::make_current()
{
  ASSERT(context_);
  bool result = true;
  X11Lock::lock();
  result = glXMakeCurrent(display_, window_, context_);
  X11Lock::unlock();
  if (!result)
  {
    std::cerr << "X11 GL context failed make current.\n";
  }

  return result;
}


void
X11OpenGLContext::release()
{
  X11Lock::lock();
  glXMakeCurrent(display_, None, NULL);
  X11Lock::unlock();
}


int
X11OpenGLContext::width()
{
  X11Lock::lock();

  // TODO: optimize out to configure events
  XWindowAttributes attr;
  XGetWindowAttributes(display_, window_, &attr);
  width_ = attr.width;
  height_ = attr.height;
  X11Lock::unlock();

  return width_;
}


int
X11OpenGLContext::height()
{
  return height_;
}


void
X11OpenGLContext::swap()
{  
  X11Lock::lock();
  glXSwapBuffers(display_, window_);
  X11Lock::unlock();
}



#define GETCONFIG(attrib) \
if(glXGetConfig(display, &vinfo[i], attrib, &value) != 0){\
  cerr << "Error getting attribute: " << #attrib << std::endl; \
  return; \
}


void
X11OpenGLContext::listvisuals()
{
  valid_visuals_.clear();
  vector<string> visualtags;
  vector<int> scores;
  int nvis;
  Display *display;
  display = XOpenDisplay((char *)0);
  int screen = DefaultScreen(display);

  XVisualInfo* vinfo = XGetVisualInfo(display, 0, NULL, &nvis);
  if(!vinfo)
  {
    cerr << "XGetVisualInfo failed";
    return;
  }
  for(int i=0;i<nvis;i++)
  {
    int score=0;
    int value;
    GETCONFIG(GLX_USE_GL);
    if(!value)
      continue;
    GETCONFIG(GLX_RGBA);
    if(!value)
      continue;
    GETCONFIG(GLX_LEVEL);
    if(value != 0)
      continue;
    if(vinfo[i].screen != screen)
      continue;
    char buf[20];
    sprintf(buf, "id=%02x, ", (unsigned int)(vinfo[i].visualid));
    valid_visuals_.push_back(vinfo[i].visualid);
    string tag(buf);
    GETCONFIG(GLX_DOUBLEBUFFER);
    if(value)
    {
      score+=200;
      tag += "double, ";
    }
    else
    {
      tag += "single, ";
    }
    GETCONFIG(GLX_STEREO);
    if(value)
    {
      score+=1;
      tag += "stereo, ";
    }
    tag += "rgba=";
    GETCONFIG(GLX_RED_SIZE);
    tag+=to_string(value)+":";
    score+=value;
    GETCONFIG(GLX_GREEN_SIZE);
    tag+=to_string(value)+":";
    score+=value;
    GETCONFIG(GLX_BLUE_SIZE);
    tag+=to_string(value)+":";
    score+=value;
    GETCONFIG(GLX_ALPHA_SIZE);
    tag+=to_string(value);
    score+=value;
    GETCONFIG(GLX_DEPTH_SIZE);
    tag += ", depth=" + to_string(value);
    score+=value*5;
    GETCONFIG(GLX_STENCIL_SIZE);
    score += value * 2;
    tag += ", stencil="+to_string(value);
    tag += ", accum=";
    GETCONFIG(GLX_ACCUM_RED_SIZE);
    tag += to_string(value) + ":";
    GETCONFIG(GLX_ACCUM_GREEN_SIZE);
    tag += to_string(value) + ":";
    GETCONFIG(GLX_ACCUM_BLUE_SIZE);
    tag += to_string(value) + ":";
    GETCONFIG(GLX_ACCUM_ALPHA_SIZE);
    tag += to_string(value);
#ifdef __sgi
    tag += ", samples=";
    GETCONFIG(GLX_SAMPLES_SGIS);
    if(value)
      score+=50;
    tag += to_string(value);
#endif

    tag += ", score=" + to_string(score);
    
    visualtags.push_back(tag);
    scores.push_back(score);
  }
  for(int i=0;i < int(scores.size())-1;i++)
  {
    for(int j=i+1;j< int(scores.size());j++)
    {
      if(scores[i] < scores[j])
      {
	SWAP(scores[i], scores[j]);
	SWAP(visualtags[i], visualtags[j]);
	SWAP(valid_visuals_[i], valid_visuals_[j]);
      }
    }
  }
}

