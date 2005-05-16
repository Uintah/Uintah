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

#ifdef _WIN32
#include <tkWinInt.h>
#include <tkIntPlatDecls.h>
#include <windows.h>
#include <strstream>
#endif

using namespace SCIRun;
using namespace std;
  
extern "C" Tcl_Interp* the_interp;

vector<int> TkOpenGLContext::valid_visuals_ = vector<int>();

#ifndef _WIN32
static GLXContext first_context = NULL;
#else
static HGLRC first_context = NULL;
#endif

#ifdef _WIN32

void
PrintErr(char* func_name)
{
  LPVOID lpMsgBuf;
  DWORD dw = GetLastError(); 
  
  if (dw) {
    FormatMessage(
		  FORMAT_MESSAGE_ALLOCATE_BUFFER | 
		  FORMAT_MESSAGE_FROM_SYSTEM,
		  NULL,
		  dw,
		  MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
		  (LPTSTR) &lpMsgBuf,
		  0, NULL );
    
    fprintf(stderr, 
	    "%s failed with error %ld: %s", 
	    func_name, dw, (char*)lpMsgBuf); 
    LocalFree(lpMsgBuf);
  }
}

const char* TkOpenGLContext::ReportCapabilities()
{
  make_current();

  if (!this->hDC_)
    {
      return "no device context";
    }

  int pixelFormat = GetPixelFormat(this->hDC_);
  PIXELFORMATDESCRIPTOR pfd;

  DescribePixelFormat(this->hDC_, pixelFormat, sizeof(PIXELFORMATDESCRIPTOR), &pfd);

  const char *glVendor = (const char *) glGetString(GL_VENDOR);
  const char *glRenderer = (const char *) glGetString(GL_RENDERER);
  const char *glVersion = (const char *) glGetString(GL_VERSION);
  const char *glExtensions = (const char *) glGetString(GL_EXTENSIONS);

  ostrstream strm;
  strm << "OpenGL vendor string:  " << glVendor << endl;
  strm << "OpenGL renderer string:  " << glRenderer << endl;
  strm << "OpenGL version string:  " << glVersion << endl;
  strm << "OpenGL extensions:  " << glExtensions << endl;
  strm << "PixelFormat Descriptor:" << endl;
  strm << "depth:  " << static_cast<int>(pfd.cDepthBits) << endl;
  if (pfd.cColorBits <= 8)
    {
      strm << "class:  PseudoColor" << endl;
    } 
  else
    {
      strm << "class:  TrueColor" << endl;
    }
  strm << "buffer size:  " << static_cast<int>(pfd.cColorBits) << endl;
  strm << "level:  " << static_cast<int>(pfd.bReserved) << endl;
  if (pfd.iPixelType == PFD_TYPE_RGBA)
    {
    strm << "renderType:  rgba" << endl;
    }
  else
    {
    strm <<"renderType:  ci" << endl;
    }
  if (pfd.dwFlags & PFD_DOUBLEBUFFER) {
    strm << "double buffer:  True" << endl;
  } else {
    strm << "double buffer:  False" << endl;
  }
  if (pfd.dwFlags & PFD_STEREO) {
    strm << "stereo:  True" << endl;  
  } else {
    strm << "stereo:  False" << endl;
  }
  if (pfd.dwFlags & PFD_GENERIC_FORMAT) {
    strm << "hardware acceleration:  False" << endl; 
  } else {
    strm << "hardware acceleration:  True" << endl; 
  }
  strm << "rgba:  redSize=" << static_cast<int>(pfd.cRedBits) << " greenSize=" << static_cast<int>(pfd.cGreenBits) << "blueSize=" << static_cast<int>(pfd.cBlueBits) << "alphaSize=" << static_cast<int>(pfd.cAlphaBits) << endl;
  strm << "aux buffers:  " << static_cast<int>(pfd.cAuxBuffers)<< endl;
  strm << "depth size:  " << static_cast<int>(pfd.cDepthBits) << endl;
  strm << "stencil size:  " << static_cast<int>(pfd.cStencilBits) << endl;
  strm << "accum:  redSize=" << static_cast<int>(pfd.cAccumRedBits) << " greenSize=" << static_cast<int>(pfd.cAccumGreenBits) << "blueSize=" << static_cast<int>(pfd.cAccumBlueBits) << "alphaSize=" << static_cast<int>(pfd.cAccumAlphaBits) << endl;

  strm << ends;
  
  return strm.str();
}
#endif


TkOpenGLContext::TkOpenGLContext(const string &id, int visualid, 
				 int width, int height)
  : visualid_(visualid),
    id_(id)
    
{
#ifdef _WIN32
  XVisualInfo visinfo;        // somewhat equivalent to pfd
  PIXELFORMATDESCRIPTOR pfd;  
#endif

  mainwin_ = Tk_MainWindow(the_interp);
  display_ = Tk_Display(mainwin_);

#ifdef _WIN32
  PrintErr("TkOpenGLContext::TKCopenGLContext");
#endif

  release();

#ifdef _WIN32
  PrintErr("TkOpenGLContext::TKCopenGLContext");
#endif

  screen_number_ = Tk_ScreenNumber(mainwin_);
  if (!mainwin_) throw scinew InternalError("Cannot find main Tk window");
  if (!display_) throw scinew InternalError("Cannot find X Display");
    
  geometry_ = 0;
  cursor_ = 0;
  x11_win_ = 0;
  context_ = 0;
  vi_ = 0;

#ifdef _WIN32
  PrintErr("TkOpenGLContext::TKCopenGLContext");
#endif

  if (valid_visuals_.empty())
    listvisuals();
  if (visualid < 0 || visualid >= (int)valid_visuals_.size())
    {
      cerr << "Bad visual id, does not exist.\n";
      visualid_ = 0;
    } else {
      visualid_ = valid_visuals_[visualid];
    }

#ifdef _WIN32
  PrintErr("TkOpenGLContext::TKCopenGLContext");
#endif

#ifdef _WIN32
      visualid_ = 0;
#endif

  if (visualid_) {

    int n;
    XVisualInfo temp_vi;
    temp_vi.visualid = visualid_;
    vi_ = XGetVisualInfo(display_, VisualIDMask, &temp_vi, &n);
    if(!vi_ || n!=1) {
      throw scinew InternalError("Cannot find Visual ID #"+to_string(visualid_));
    }
  } else {
#ifndef _WIN32
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
#else //_WIN32
      // I am using the *PixelFormat commands from win32 because according
      // to the Windows page, we should prefer this to wgl*PixelFormatARB.
      // Unfortunately, this means that the Windows code will differ
      // substantially from that of other platforms.  However, it has the
      // advantage that we don't have to use the wglGetProc to get
      // the procedure address, or test to see if the applicable extension
      // is supported.  WM:VI

    PrintErr("TkOpenGLContext::TKCopenGLContext");

    DWORD dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL;
//       if (doublebuffer_)
    dwFlags |= PFD_DOUBLEBUFFER;
//       if (stereo_)
// 	dwFlags |= PFD_STEREO;

      PIXELFORMATDESCRIPTOR npfd = { 
	sizeof(PIXELFORMATDESCRIPTOR),  
	1,                     // version number 
	dwFlags,
	PFD_TYPE_RGBA,         // RGBA type 
	32, // color depth
	8, 0, 
	8, 0, 
	8, 0,  // color bits  
	8, 0,  // alpha buffer 
	0+
	0+
	0,// accumulation buffer 
	0, 
	0, 
	0, 
	0,// accum bits 
	32,  // 32-bit z-buffer 
	0,// no stencil buffer 
	0, // no auxiliary buffer 
	PFD_MAIN_PLANE,        // main layer 
	0,                     // reserved 
	0, 0, 0                // layer masks ignored 
      }; 

      pfd = npfd;
//       PIXELFORMATDESCRIPTOR pfd = { 
// 	sizeof(PIXELFORMATDESCRIPTOR),  
// 	1,                     // version number 
// 	dwFlags,
// 	PFD_TYPE_RGBA,         // RGBA type 
// 	buffersize_, // color depth
// 	redsize_, 0, 
// 	greensize_, 0, 
// 	bluesize_, 0,  // color bits  
// 	alphasize_,0,  // alpha buffer 
// 	accumredsize_+
// 	accumgreensize_+
// 	accumbluesize_,// accumulation buffer 
// 	accumredsize_, 
// 	accumgreensize_, 
// 	accumbluesize_, 
// 	accumalphasize_,// accum bits 
// 	depthsize_,  // 32-bit z-buffer 
// 	stencilsize_,// no stencil buffer 
// 	auxbuffers_, // no auxiliary buffer 
// 	PFD_MAIN_PLANE,        // main layer 
// 	0,                     // reserved 
// 	0, 0, 0                // layer masks ignored 
//       }; 

      HWND hWND = Tk_GetHWND( Tk_WindowId(mainwin_) );
      hDC_ = GetDC(hWND);
      
      int iPixelFormat;
      if ((iPixelFormat = ChoosePixelFormat(hDC_, &pfd)) == 0)
	{
	  fprintf(stderr,"TkOpenGLContext:: ChoosePixelFormat failed!\n");
	}

    if (!first_context && SetPixelFormat(hDC_, iPixelFormat, &pfd) == FALSE)
      {
	fprintf(stderr,"TkOpenGLContext:: SetPixelFormat failed!\n");
      }

    PrintErr("TkOpenGLContext::TKCopenGLContext");

    visualid_ = iPixelFormat;

//       XVisualInfo xvi;
//       xvi.screen = screen_number_;
//       int n_ret=0;
//       vi_ = XGetVisualInfo(display_,
// 			   VisualScreenMask,
// 			   &xvi,
// 			   &n_ret
// 			   );

    vi_ = &visinfo;
    vi_->visual = DefaultVisual(display_,DefaultScreen(display_));
    vi_->depth = vi_->visual->bits_per_rgb;

#endif

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

//   {
//     fprintf(stderr,"Before TkWinGet...\n");
//     HWND hWND = Tk_GetHWND(Tk_WindowId(tkwin_));
//     PrintErr("Tk_GetHWND");
    
//     fprintf(stderr,"After TkWinGet...\n");
//     hDC_ = GetDC(hWND);
//     PrintErr("GetDC");
//   }

  x11_win_ = Tk_WindowId(tkwin_);
  if (!x11_win_) throw scinew InternalError("Cannot get Tk X11 window ID");

  XSync(display_, False);


#ifndef _WIN32
  if (!first_context) {
    first_context = glXCreateContext(display_, vi_, 0, 1);
  }
  context_ = glXCreateContext(display_, vi_, first_context, 1);
  if (!context_) throw scinew InternalError("Cannot create GLX Context");
#else // _WIN32

  PrintErr("TkOpenGLContext::TKCopenGLContext");

  context_ = wglCreateContext(hDC_);
  PrintErr("TkOpenGLContext::TKOpenGLContext");

  if (first_context == NULL) {
    first_context = context_;
  } else {
    wglShareLists(first_context,context_);
  }

  PrintErr("TkOpenGLContext::TKCopenGLContext");

  if (!context_) throw scinew InternalError("Cannot create WGL Context");

  PrintErr("TkOpenGLContext::TKCopenGLContext");

  fprintf(stderr,"%s\n",ReportCapabilities());

#endif
}


TkOpenGLContext::~TkOpenGLContext()
{

  TCLTask::lock();
  release();
#ifndef _WIN32
  glXDestroyContext(display_, context_);
#else
  if (context_ != first_context)
    wglDeleteContext(context_);
#endif
  XSync(display_, False);
  TCLTask::unlock();
}


bool
TkOpenGLContext::make_current()
{
  ASSERT(context_);

  bool result = true;

#ifndef _WIN32
  result = glXMakeCurrent(display_, x11_win_, context_);
#else  // _WIN32
  HGLRC current = wglGetCurrentContext();

  if (current != context_) {
    
    result = wglMakeCurrent(hDC_,context_);
    
    PrintErr("TkOpenGLContext::make_current");
  }

#endif


  if (!result)
  {
    std::cerr << "GL context: " << id_ << " failed make current.\n";
  }

  return result;
}


void
TkOpenGLContext::release()
{

#ifndef _WIN32
  glXMakeCurrent(display_, None, NULL);
#else // WIN32
  if (wglGetCurrentContext() != NULL)
    wglMakeCurrent(NULL,NULL);
  PrintErr("TkOpenGLContext::release()");
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
#else //_WIN32
  SwapBuffers(hDC_);
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
#ifdef _WIN32
  valid_visuals_.clear();
  return string("");
#endif

  TCLTask::lock();
  Tk_Window topwin=Tk_MainWindow(the_interp);
  if(!topwin)
  {
    cerr << "Unable to locate main window!\n";
    TCLTask::unlock();
    return string("");
  }
#ifndef _WIN32
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
#else // _WIN32

  // I am using the *PixelFormat commands from win32 because according
  // to the Windows page, we should prefer this to wgl*PixelFormatARB.
  // Unfortunately, this means that the Windows code will differ
  // substantially from that of other platforms.  However, it has the
  // advantage that we don't have to use the wglGetProc to get
  // the procedure address, or test to see if the applicable extension
  // is supported.  WM:VI

  PrintErr("TkOpenGLContext::listvisuals");

  HWND hWND = TkWinGetWrapperWindow(topwin);
  HDC dc;
  if ((dc = GetDC(hWND)) == 0)
    {
      fprintf(stderr,"Bad DC returned by GetDC\n");
      
    }

  PrintErr("TkOpenGLContext::listvisuals");

  valid_visuals_.clear();
  vector<string> visualtags;
  vector<int> scores;

  int  id, level, db, stereo, r,g,b,a, depth, stencil, ar, ag, ab, aa;

  PrintErr("TkOpenGLContext::listvisuals");

  int iPixelFormat;
  if ((iPixelFormat = GetPixelFormat(dc)) == 0)
    {
      fprintf(stderr,"Error: Bad Pixel Format Retrieved\n");
      return string("");
      
    }

  PrintErr("TkOpenGLContext::listvisuals");

  PIXELFORMATDESCRIPTOR pfd;
  
  DescribePixelFormat(dc,iPixelFormat,sizeof(PIXELFORMATDESCRIPTOR),&pfd);

  PrintErr("TkOpenGLContext::listvisuals");

   int i;
   int nvis = 1;
  for(i=0;i<nvis;i++)
  {
    int score=0;
    int value;
    value = ((pfd.dwFlags & PFD_SUPPORT_OPENGL) == PFD_SUPPORT_OPENGL);
    if(!value)
      continue;
    fprintf(stderr,"Got GL support\n");

    value = (pfd.iPixelType == PFD_TYPE_RGBA);
    if(!value)
      continue;
    fprintf(stderr,"Got RGBA support\n");
//     if(vinfo[i].screen != screen)
//       continue;
    char buf[20];
    sprintf(buf, "id=%02x, ", (unsigned int)0);
    fprintf(stderr,"Adding 0 to visuals\n");
    valid_visuals_.push_back(0);
    string tag(buf);
    value = ((pfd.dwFlags & PFD_DOUBLEBUFFER) == PFD_DOUBLEBUFFER);
    if(value)
    {
      score+=200;
      tag += "double, ";
    }
    else
    {
      tag += "single, ";
    }
    value = ((pfd.dwFlags & PFD_STEREO) == PFD_STEREO);
    if(value)
    {
      score+=1;
      tag += "stereo, ";
    }
    tag += "rgba=";
    value = pfd.cRedBits;
    tag+=to_string(value)+":";
    score+=value;
    value = pfd.cGreenBits;
    tag+=to_string(value)+":";
    score+=value;
    value = pfd.cBlueBits;
    tag+=to_string(value)+":";
    score+=value;
    value = pfd.cAlphaBits;
    tag+=to_string(value);
    score+=value;
    value = pfd.cDepthBits;
    tag += ", depth=" + to_string(value);
    score+=value*5;
    value = pfd.cStencilBits;
    score += value * 2;
    tag += ", stencil="+to_string(value);
    tag += ", accum=";
    value = pfd.cAccumRedBits;;
    tag += to_string(value) + ":";
    value = pfd.cAccumGreenBits;
    tag += to_string(value) + ":";
    value = pfd.cAccumBlueBits;
    tag += to_string(value) + ":";
    value = pfd.cAccumAlphaBits;
    tag += to_string(value);

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

#endif
}
