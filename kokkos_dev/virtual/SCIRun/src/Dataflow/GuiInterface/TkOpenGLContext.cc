/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

#include <Dataflow/GuiInterface/TkOpenGLContext.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/Color.h>
#include <Core/Exceptions/InternalError.h>
#include <Dataflow/GuiInterface/GuiInterface.h>
#include <Dataflow/GuiInterface/TCLInterface.h>
#include <Dataflow/GuiInterface/TclObj.h>
#include <Dataflow/GuiInterface/TCLTask.h>
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
#include <sstream>
#include <process.h>
#endif

using namespace SCIRun;
using namespace std;
  
extern "C" Tcl_Interp* the_interp;

vector<int> TkOpenGLContext::valid_visuals_ = vector<int>();

#if (TK_MAJOR_VERSION>=8 && TK_MINOR_VERSION>=4)
#  define HAVE_TK_SETCLASSPROCS
/* pointer to Tk_SetClassProcs function in the stub table */
#endif

#ifndef _WIN32
static GLXContext first_context = NULL;
#else
static HGLRC first_context = NULL;
#endif

#ifdef _WIN32
HINSTANCE dllHINSTANCE=0;

#ifndef HAVE_GLEW
void initGLextensions();
#endif

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
  SetLastError(0);
}

const char* TkOpenGLContext::ReportCapabilities()
{
  if (!this->hDC_)
    {
      return "no device context";
    }

  const char *glVendor = (const char *) glGetString(GL_VENDOR);
  const char *glRenderer = (const char *) glGetString(GL_RENDERER);
  const char *glVersion = (const char *) glGetString(GL_VERSION);
  const char *glExtensions = (const char *) glGetString(GL_EXTENSIONS);

  ostringstream strm;
  if (glVendor) {
    strm << "OpenGL vendor string:  " << glVendor << endl;
    strm << "OpenGL renderer string:  " << glRenderer << endl;
    strm << "OpenGL version string:  " << glVersion << endl;
#if 0
    strm << "OpenGL extensions:  " << glExtensions << endl;
#endif
  }
  else {
    strm << "    Invalid OpenGL context\n";
  }

#if 0
  int pixelFormat = GetPixelFormat(this->hDC_);
  PIXELFORMATDESCRIPTOR pfd;

  DescribePixelFormat(this->hDC_, pixelFormat, sizeof(PIXELFORMATDESCRIPTOR), &pfd);


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
#endif
  strm << ends;
  
  cerr << strm.str().c_str();
  return 0;
}

char *tkGlClassName = "TkGL";
bool tkGlClassInitialized = false;
WNDCLASS tkGlClass;
WNDPROC tkWinChildProc=NULL;


// win32 event loop
LRESULT CALLBACK 
WndProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam)
{

  LONG result;
  TkOpenGLContext *tkCxt = (TkOpenGLContext*) GetWindowLong(hWnd, 0);
  WNDCLASS childClass;

  switch (msg) {
  case WM_WINDOWPOSCHANGED:
    /* Should be processed by DefWindowProc, otherwise a double buffered
       context is not properly resized when the corresponding window is resized.*/
    result = TRUE;
    break;
    
  case WM_NCCREATE:
    result = TRUE;
    break;

  case WM_DESTROY:
    result = TRUE;
    if (tkCxt)
      delete tkCxt;
    break;
    
  default:
    {
      if (tkWinChildProc == NULL) {
	GetClassInfo(Tk_GetHINSTANCE(),TK_WIN_CHILD_CLASS_NAME,
		     &childClass);
	tkWinChildProc = childClass.lpfnWndProc;
      }
      result = tkWinChildProc(hWnd, msg, wParam, lParam);
    }
  }
  Tcl_ServiceAll();
  return result;
}

static Window
TkGLMakeWindow(Tk_Window tkwin, Window parent, ClientData data)
{
  HWND parentWin;
  int style;
  HINSTANCE hInstance;
  
  TkOpenGLContext *tkCxt = (TkOpenGLContext*) data;
  tkCxt->vi_ = new XVisualInfo();
  tkCxt->display_ = Tk_Display(tkCxt->tkwin_);

  hInstance = Tk_GetHINSTANCE();

  // next register our own window class.... 
  
  if (!tkGlClassInitialized) {
    tkGlClassInitialized = true;
    tkGlClass.style = CS_HREDRAW | CS_VREDRAW;// | CS_OWNDC;
    tkGlClass.cbClsExtra = 0;
    tkGlClass.cbWndExtra = sizeof(long); /* To save TkCxt */
    tkGlClass.hInstance = dllHINSTANCE;
    tkGlClass.hbrBackground = NULL;
    tkGlClass.lpszMenuName = NULL;
    //tkGlClass.lpszClassName = TK_WIN_CHILD_CLASS_NAME;
    //tkGlClass.lpfnWndProc = TkWinChildProc;
    tkGlClass.lpszClassName = tkGlClassName;
    tkGlClass.lpfnWndProc = WndProc;
    tkGlClass.hIcon = NULL;
    tkGlClass.hCursor = NULL;

    RegisterClass(&tkGlClass);
    PrintErr("MakeWindow RegisterClass");
  }

  /*
   * Create the window, then ensure that it is at the top of the
   * stacking order.
   */

  int x = Tk_X(tkwin), y = Tk_Y(tkwin), width = Tk_Width(tkwin), height = Tk_Height(tkwin);
  if (width == 0 || height == 0) {
    style = WS_POPUP;
    parentWin = 0;
  }
  else {
    style = WS_CHILD | WS_CLIPCHILDREN | WS_CLIPSIBLINGS;
    parentWin = Tk_GetHWND(parent);
  }

  tkCxt->hWND_ = CreateWindow(tkGlClassName, "SCIRun GL Viewer Screen",
			      style, x, y, width, height,
			      parentWin, NULL, dllHINSTANCE, 
			      NULL);
  PrintErr("CreateWindow");

  SetWindowLong(tkCxt->hWND_, 0, (LONG) tkCxt);

  if (width != 0 && height != 0)
    SetWindowPos(tkCxt->hWND_, HWND_TOP, 0, 0, 0, 0, SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE);

  tkCxt->hDC_ = GetDC(tkCxt->hWND_);

  // Set up the pixel format for the display....

  DWORD dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL; //| PFD_GENERIC_ACCELERATED;
  if (tkCxt->doublebuffer_)
    dwFlags |= PFD_DOUBLEBUFFER;

  PIXELFORMATDESCRIPTOR pfd = { 
    sizeof(PIXELFORMATDESCRIPTOR),  
    1,                     // version number 
    dwFlags,
    PFD_TYPE_RGBA,         // RGBA type 
    24, // color depth
    8, 0, 8, 0, 8, 0,  // color bits  
    8, 0,  // alpha buffer 
    0+0+0,// accumulation buffer 
    0, 0, 0, 0,// accum bits 
    32,  // 32-bit z-buffer 
    0,// no stencil buffer 
    0, // no auxiliary buffer 
    PFD_MAIN_PLANE,        // main layer 
    0,                     // reserved 
    0, 0, 0                // layer masks ignored 
  }; 


  int iPixelFormat = ChoosePixelFormat(tkCxt->hDC_, &pfd);
  PrintErr("ChoosePixelFormat");
  
  SetPixelFormat(tkCxt->hDC_, iPixelFormat, &pfd);
  PrintErr("SetPixelFormat");

   /* Get the actual pixel format */
  DescribePixelFormat(tkCxt->hDC_, iPixelFormat, sizeof(pfd), &pfd);

  tkCxt->context_ = wglCreateContext(tkCxt->hDC_);
  PrintErr("wglCreateContext(context_):");
#if 1
   if (!first_context) {
     if ((first_context = wglCreateContext(tkCxt->hDC_)) == NULL)
       PrintErr("wglCreateContext(first_context):");
   }
  if (wglShareLists(first_context,tkCxt->context_) == FALSE)
    PrintErr("wglShareLists");
#endif

  tkCxt->make_current();
#ifndef HAVE_GLEW
  initGLextensions();
#endif
  tkCxt->ReportCapabilities();

  if (!tkCxt->context_) throw scinew InternalError("Cannot create WGL Context", __FILE__, __LINE__);
  
  /* Just for portability, define the simplest visinfo */
  tkCxt->vi_->visual = DefaultVisual(tkCxt->display_, DefaultScreen(tkCxt->display_));   
  tkCxt->vi_->depth = tkCxt->vi_->visual->bits_per_rgb;
  /*
  * find a colormap
  */

  tkCxt->screen_number_ = Tk_ScreenNumber(tkCxt->tkwin_);

  tkCxt->colormap_ = DefaultColormap(tkCxt->display_, tkCxt->screen_number_);

  int result = Tk_SetWindowVisual(tkCxt->tkwin_, tkCxt->vi_->visual, 
				  tkCxt->vi_->depth,tkCxt->colormap_ );
  if (result != 1) throw scinew InternalError("Cannot set Tk Window Visual", 
					      __FILE__, __LINE__);

  SelectPalette(tkCxt->hDC_, ((TkWinColormap *)tkCxt->colormap_)->palette, TRUE);
  RealizePalette(tkCxt->hDC_);
  tkCxt->visualid_ = iPixelFormat;

//   cerr << "after create window\n";
//   PrintErr("MakeWindow setwindowpos");
//   cerr << "after set window pos\n";
  Window win = Tk_AttachHWND(tkCxt->tkwin_, tkCxt->hWND_);
  XMapWindow(tkCxt->display_, win);
  return win;

}

#endif // _WIN32


TkOpenGLContext::TkOpenGLContext(const string &id, int visualid, 
				 int width, int height) : 
  OpenGLContext(),
  mutex_("GL lock")
{
#ifdef _WIN32
  make_win32_gl_context(id, visualid, width, height);
#else
  make_x11_gl_context(id, visualid, width, height);
#endif
}


void
TkOpenGLContext::make_x11_gl_context(const string &id, int visualid, 
                                     int width, int height)
{
#ifndef _WIN32
  visualid_ = visualid;
  id_ = id;

  mainwin_ = Tk_MainWindow(the_interp);
  if (!mainwin_) 
    throw scinew InternalError("Cannot find main Tk window",__FILE__,__LINE__);

  display_ = Tk_Display(mainwin_);
  if (!display_) 
    throw scinew InternalError("Cannot find X Display", __FILE__, __LINE__);

  screen_number_ = Tk_ScreenNumber(mainwin_);
  geometry_ = 0;
  cursor_   = 0;
  x11_win_  = 0;
  context_  = 0;
  vi_       = 0;

  if (valid_visuals_.empty())
    listvisuals();
  if (visualid < 0 || visualid >= (int)valid_visuals_.size())
    {
      cerr << "Bad visual id, does not exist.\n";
      visualid_ = 0;
    } else {
      visualid_ = valid_visuals_[visualid];
    }

  if (visualid_) {

    int n;
    XVisualInfo temp_vi;
    temp_vi.visualid = visualid_;
    vi_ = XGetVisualInfo(display_, VisualIDMask, &temp_vi, &n);
    if(!vi_ || n!=1) {
      throw 
        scinew InternalError("Cannot find Visual ID #"+to_string(visualid_), 
                             __FILE__, __LINE__);
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

  
  if (!vi_) 
    throw scinew InternalError("Cannot find Visual", __FILE__, __LINE__);

  colormap_ = XCreateColormap(display_, Tk_WindowId(mainwin_), 
			      vi_->visual, AllocNone);


  tkwin_ = Tk_CreateWindowFromPath(the_interp, mainwin_, 
				   ccast_unsafe(id),
				   (char *) NULL);

  if (!tkwin_) 
    throw scinew InternalError("Cannot create Tk Window", __FILE__, __LINE__);



  Tk_GeometryRequest(tkwin_, width, height);
  int result = Tk_SetWindowVisual(tkwin_, vi_->visual, vi_->depth, colormap_);
  if (result != 1) 
    throw scinew InternalError("Tk_SetWindowVisual failed",__FILE__,__LINE__);

  Tk_MakeWindowExist(tkwin_);
  if (Tk_WindowId(tkwin_) == 0) {
    throw InternalError("Tk_MakeWindowExist failed", __FILE__, __LINE__);
  }

  x11_win_ = Tk_WindowId(tkwin_);
  if (!x11_win_) 
    throw scinew InternalError("Cannot get Tk X11 window ID",__FILE__,__LINE__);

  XSync(display_, False);


  if (!first_context) {
    first_context = glXCreateContext(display_, vi_, 0, 1);
  }
  context_ = glXCreateContext(display_, vi_, first_context, 1);
  if (!context_) 
    throw scinew InternalError("Cannot create GLX Context",__FILE__,__LINE__);
#endif
}



void
TkOpenGLContext::make_win32_gl_context(const string &id, int visualid, 
                                       int width, int height)
{
#ifdef _WIN32
  visualid_ = visualid;
  id_ = id;

  mainwin_ = Tk_MainWindow(the_interp);
  if (!mainwin_) 
    throw scinew InternalError("Cannot find main Tk window",__FILE__,__LINE__);

  display_ = Tk_Display(mainwin_);
  if (!display_) 
    throw scinew InternalError("Cannot find X Display", __FILE__, __LINE__);

  screen_number_ = Tk_ScreenNumber(mainwin_);
  geometry_ = 0;
  cursor_   = 0;
  x11_win_  = 0;
  context_  = 0;
  vi_       = 0;

  if (valid_visuals_.empty())
    listvisuals();
  if (visualid < 0 || visualid >= (int)valid_visuals_.size())
    {
      cerr << "Bad visual id, does not exist.\n";
      visualid_ = 0;
    } else {
      visualid_ = valid_visuals_[visualid];
    }

  if (visualid_) {

    int n;
    XVisualInfo temp_vi;
    temp_vi.visualid = visualid_;
    vi_ = XGetVisualInfo(display_, VisualIDMask, &temp_vi, &n);
    if(!vi_ || n!=1) {
      throw 
        scinew InternalError("Cannot find Visual ID #"+to_string(visualid_), 
                             __FILE__, __LINE__);
    }
  }

  tkwin_ = Tk_CreateWindowFromPath(the_interp, mainwin_, 
				   ccast_unsafe(id),
				   (char *) NULL);

  if (!tkwin_) 
    throw scinew InternalError("Cannot create Tk Window", __FILE__, __LINE__);

#  ifdef HAVE_TK_SETCLASSPROCS
  Tk_ClassProcs *procsPtr;
  procsPtr = (Tk_ClassProcs*) Tcl_Alloc(sizeof(Tk_ClassProcs));
  procsPtr->size             = sizeof(Tk_ClassProcs);
#ifdef _WIN32
  procsPtr->createProc       = TkGLMakeWindow;
#endif
  procsPtr->worldChangedProc = NULL;
  procsPtr->modalProc        = NULL;
  Tk_SetClassProcs(tkwin_,procsPtr,(ClientData)this); 
#  else
//   TkClassProcs *procsPtr;
//   Tk_FakeWin *winPtr = (Tk_FakeWin*)(tkwin_);
    
//   procsPtr = (TkClassProcs*)Tcl_Alloc(sizeof(TkClassProcs));
//   procsPtr->createProc     = TkGLMakeWindow;
//   procsPtr->geometryProc   = NULL;
//   procsPtr->modalProc      = NULL;
//   winPtr->dummy17 = (char*)procsPtr;
//   winPtr->dummy18 = (ClientData)this;
#  endif
  Tk_GeometryRequest(tkwin_, width, height);
  Tk_ResizeWindow(tkwin_, width, height);

  Tk_MakeWindowExist(tkwin_);
  if (Tk_WindowId(tkwin_) == 0) {
    throw InternalError("Tk_MakeWindowExist failed", __FILE__, __LINE__);
  }

  x11_win_ = Tk_WindowId(tkwin_);
  if (!x11_win_) 
    throw scinew InternalError("Cannot get Tk X11 window ID", __FILE__, __LINE__);

  XSync(display_, False);

  release();
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

  // remove the TkGL Context from the window, so it won't delete it in the callback
  SetWindowLong(hWND_, 0, (LONG) 0);
  DestroyWindow(hWND_);
#endif
  XSync(display_, False);
  TCLTask::unlock();
}


bool
TkOpenGLContext::make_current()
{
  ASSERT(context_);

  bool result = true;
  //  cerr << "Make current " << id_ << "\n";
#ifndef _WIN32
  GuiInterface::getSingleton()->pause();
  result = glXMakeCurrent(display_, x11_win_, context_);
  GuiInterface::getSingleton()->unpause();

#else  // _WIN32
  HGLRC current = wglGetCurrentContext();
  PrintErr("wglGetCurrentContext");

  if (current != context_) {
    result = wglMakeCurrent(hDC_,context_);
    PrintErr("wglMakeCurrent");
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
  GuiInterface::getSingleton()->pause();
  glXMakeCurrent(display_, None, NULL);
  GuiInterface::getSingleton()->unpause();
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

#ifndef EXPERIMENTAL_TCL_THREAD
  TCLTask::lock();
#endif
  Tk_Window topwin=Tk_MainWindow(the_interp);
  if(!topwin)
  {
    cerr << "Unable to locate main window!\n";
#ifndef EXPERIMENTAL_TCL_THREAD
    TCLTask::unlock();
#endif
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
#ifndef EXPERIMENTAL_TCL_THREAD
    TCLTask::unlock();
#endif
    return string("");
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
  string ret_val;
  for (unsigned int k = 0; k < visualtags.size(); ++k)
    ret_val = ret_val + "{" + visualtags[k] +"} ";
#ifndef EXPERIMENTAL_TCL_THREAD
    TCLTask::unlock();
#endif
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

  //int  id, level, db, stereo, r,g,b,a, depth, stencil, ar, ag, ab, aa;

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

#ifdef _WIN32
#ifndef HAVE_GLEW
// initialize > gl 1.1 stuff here
__declspec(dllexport) PFNGLACTIVETEXTUREPROC glActiveTexture = 0;
__declspec(dllexport) PFNGLBLENDEQUATIONPROC glBlendEquation = 0;
__declspec(dllexport) PFNGLTEXIMAGE3DPROC glTexImage3D = 0;
__declspec(dllexport) PFNGLTEXSUBIMAGE3DPROC glTexSubImage3D = 0;
__declspec(dllexport) PFNGLMULTITEXCOORD1FPROC glMultiTexCoord1f;
__declspec(dllexport) PFNGLMULTITEXCOORD2FVPROC glMultiTexCoord2fv = 0;
__declspec(dllexport) PFNGLMULTITEXCOORD3FPROC glMultiTexCoord3f = 0;
__declspec(dllexport) PFNGLCOLORTABLEPROC glColorTable = 0;

void initGLextensions()
{
  static bool initialized = false;
  if (!initialized) {
    initialized = true;
    glActiveTexture = (PFNGLACTIVETEXTUREPROC)wglGetProcAddress("glActiveTexture");
    glBlendEquation = (PFNGLBLENDEQUATIONPROC)wglGetProcAddress("glBlendEquation");
    glTexImage3D = (PFNGLTEXIMAGE3DPROC)wglGetProcAddress("glTexImage3D");
    glTexSubImage3D = (PFNGLTEXSUBIMAGE3DPROC)wglGetProcAddress("glTexSubImage3D");
    glMultiTexCoord1f = (PFNGLMULTITEXCOORD1FPROC)wglGetProcAddress("glMultiTexCoord1fARB");
    glMultiTexCoord2fv = (PFNGLMULTITEXCOORD2FVPROC)wglGetProcAddress("glMultiTexCoord2fv");
    glMultiTexCoord3f = (PFNGLMULTITEXCOORD3FPROC)wglGetProcAddress("glMultiTexCoord3f");
    glColorTable = (PFNGLCOLORTABLEPROC)wglGetProcAddress("glColorTable");
  }
}
#endif
BOOL WINAPI DllMain(HINSTANCE hinstance, DWORD reason, LPVOID reserved)
{
  switch (reason) {
  case DLL_PROCESS_ATTACH:
    dllHINSTANCE = hinstance; break;
  default: break;
  }
  return TRUE;
}
#endif
