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
 *  TkOpenGLContext.cc:
 *
 *  Written by:
 *   McKay Davis
 *   December 2004
 *
 */

#include <Core/Geom/TkOpenGLContext.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/Color.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/GuiInterface/GuiInterface.h>
#include <Core/GuiInterface/TclObj.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h> // for SWAP
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/Assert.h>
#include <sci_gl.h>
#include <sci_glu.h>
#include <sci_glx.h>
#include <iostream>
#include <set>

using namespace SCIRun;
using namespace std;
  
extern "C" Tcl_Interp* the_interp;

#ifndef _WIN32
static GLXContext first_context = NULL;
vector<int> TkOpenGLContext::valid_visuals_ = vector<int>();
#endif

TkOpenGLContext::TkOpenGLContext(const string &id, int visualid, 
				 int width, int height)
  : visualid_(visualid),
    id_(id)
    
{
#ifndef _WIN32
  mainwin_ = Tk_MainWindow(the_interp);
  display_ = Tk_Display(mainwin_);
  release();
  screen_number_ = Tk_ScreenNumber(mainwin_);
  if (!mainwin_) throw scinew InternalError("Cannot find main Tk window");
  if (!display_) throw scinew InternalError("Cannot find X Display");
    
  geometry_ = 0;
  cursor_ = 0;
  x11_win_ = 0;
  context_ = 0;
  vi_ = 0;

  if (valid_visuals_.empty())
    listvisuals();
  if (visualid < 0 || visualid >= (int)valid_visuals_.size())
  {
    cerr << "Bad visual id, does not exist.\n";
    visualid = 0;
  }
  visualid_ = valid_visuals_[visualid];

  if (visualid_) {
    int n;
    XVisualInfo temp_vi;
    temp_vi.visualid = visualid_;
    vi_ = XGetVisualInfo(display_, VisualIDMask, &temp_vi, &n);
    if(!vi_ || n!=1) {
      throw scinew InternalError("Cannot find Visual ID #"+to_string(visualid_));
    }
  } else {
    /* Pick the right visual... */
    int idx = 0;
    int attributes[50];
    attributes[idx++] = GLX_BUFFER_SIZE;
    attributes[idx++] = buffersize_;
    attributes[idx++] = GLX_LEVEL;
    attributes[idx++] = level_;
    if (rgba_)
      attributes[idx++] = GLX_RGBA;
    if (doublebuffer_)
      attributes[idx++] = GLX_DOUBLEBUFFER;
    if (stereo_)
      attributes[idx++] = GLX_STEREO;
    attributes[idx++] = GLX_AUX_BUFFERS;
    attributes[idx++] = auxbuffers_;
    attributes[idx++] = GLX_RED_SIZE;
    attributes[idx++] = redsize_;
    attributes[idx++] = GLX_GREEN_SIZE;
    attributes[idx++] = greensize_;
    attributes[idx++] = GLX_BLUE_SIZE;
    attributes[idx++] = bluesize_;
    attributes[idx++] = GLX_ALPHA_SIZE;
    attributes[idx++] = alphasize_;
    attributes[idx++] = GLX_DEPTH_SIZE;
    attributes[idx++] = depthsize_;
    attributes[idx++] = GLX_STENCIL_SIZE;
    attributes[idx++] = stencilsize_;
    attributes[idx++] = GLX_ACCUM_RED_SIZE;
    attributes[idx++] = accumredsize_;
    attributes[idx++] = GLX_ACCUM_GREEN_SIZE;
    attributes[idx++] = accumgreensize_;
    attributes[idx++] = GLX_ACCUM_BLUE_SIZE;
    attributes[idx++] = accumbluesize_;
    attributes[idx++] = GLX_ACCUM_ALPHA_SIZE;
    attributes[idx++] = accumalphasize_;
#if 0
    attributes[idx++]=GLX_SAMPLES_SGIS;
    attributes[idx++]=4;
#endif
    attributes[idx++]=None;

    vi_ = glXChooseVisual(display_, screen_number_, attributes);
  }

  if (!vi_) throw scinew InternalError("Cannot find Visual");
  colormap_ = XCreateColormap(display_, Tk_WindowId(mainwin_), 
			      vi_->visual, AllocNone);

  tkwin_ = Tk_CreateWindowFromPath(the_interp, mainwin_, 
				   ccast_unsafe(id),
				   (char *) NULL);
  if (!tkwin_) throw scinew InternalError("Cannot create Tk Window");
  Tk_GeometryRequest(tkwin_, width, height);


  int result = Tk_SetWindowVisual(tkwin_, vi_->visual, vi_->depth, colormap_);
  if (result != 1) throw scinew InternalError("Cannot set Tk Window Visual");

  Tk_MakeWindowExist(tkwin_);

  x11_win_ = Tk_WindowId(tkwin_);
  if (!x11_win_) throw scinew InternalError("Cannot get Tk X11 window ID");

  XSync(display_, False);
  if (!first_context) {
    first_context = glXCreateContext(display_, vi_, 0, 1);
  }
  context_ = glXCreateContext(display_, vi_, first_context, 1);
  if (!context_) throw scinew InternalError("Cannot create GLX Context");
#endif
}


TkOpenGLContext::~TkOpenGLContext()
{
#ifndef _WIN32
  TCLTask::lock();
  release();
  glXDestroyContext(display_, context_);
  XSync(display_, False);
  TCLTask::unlock();
#endif
}


bool
TkOpenGLContext::make_current()
{
#ifndef _WIN32
  ASSERT(context_);
  const bool result = glXMakeCurrent(display_, x11_win_, context_);

  if (!result)
  {
    std::cerr << "GL context: " << id_ << " failed make current.\n";
  }

  return result;
#else
  return false;
#endif
}


void
TkOpenGLContext::release()
{
#ifndef _WIN32
  glXMakeCurrent(display_, None, NULL);
#endif
}


int
TkOpenGLContext::width()
{
  return Tk_Width(tkwin_);
}


int
TkOpenGLContext::height()
{
  return Tk_Height(tkwin_);
}


void
TkOpenGLContext::swap()
{
#ifndef _WIN32
  glXSwapBuffers(display_, x11_win_);
#endif
}



#define GETCONFIG(attrib) \
if(glXGetConfig(display, &vinfo[i], attrib, &value) != 0){\
  cerr << "Error getting attribute: " << #attrib << std::endl; \
  TCLTask::unlock(); \
  return string(""); \
}


string
TkOpenGLContext::listvisuals()
{
#ifndef _WIN32
  TCLTask::lock();
  Tk_Window topwin=Tk_MainWindow(the_interp);
  if(!topwin)
  {
    cerr << "Unable to locate main window!\n";
    TCLTask::unlock();
    return string("");
  }
  Display *display =Tk_Display(topwin);
  int screen=Tk_ScreenNumber(topwin);
  valid_visuals_.clear();
  vector<string> visualtags;
  vector<int> scores;
  int nvis;
  XVisualInfo* vinfo=XGetVisualInfo(display, 0, NULL, &nvis);
  if(!vinfo)
  {
    cerr << "XGetVisualInfo failed";
    TCLTask::unlock();
    return string("");
  }
  int i;
  for(i=0;i<nvis;i++)
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
  for(i=0;(unsigned int)i<scores.size()-1;i++)
  {
    for(unsigned int j=i+1;j<scores.size();j++)
    {
      if(scores[i] < scores[j])
      {
	SWAP(scores[i], scores[j]);
	SWAP(visualtags[i], visualtags[j]);
	SWAP(valid_visuals_[i], valid_visuals_[j]);
      }
    }
  }
  string ret_val;
  for (unsigned int k = 0; k < visualtags.size(); ++k)
    ret_val = ret_val + "{" + visualtags[k] +"} ";
  TCLTask::unlock();
  return ret_val;
#else
  string val;
  return val;
#endif
}
