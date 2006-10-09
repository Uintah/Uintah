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
//    File   : Win32OpenGLContext.cc
//    Author : McKay Davis
//    Date   : May 30 2006

#include <Core/Geom/Win32OpenGLContext.h>
#include <Core/Geom/X11Lock.h>
#include <Core/Events/keysyms.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/Color.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h> // for SWAP
#include <Core/Thread/Mailbox.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/Assert.h>
#include <sci_glx.h>
#include <iostream>
#include <set>


using namespace SCIRun;
using namespace std;

HINSTANCE dllHINSTANCE=0;

void initGLextensions();


vector<int> Win32OpenGLContext::valid_visuals_ = vector<int>();
HGLRC Win32OpenGLContext::first_context_ = NULL;

char *GlClassName = "GLarea";
bool GlClassInitialized = false;
WNDCLASS GlClass;
extern HINSTANCE dllHINSTANCE;
LRESULT CALLBACK WindowEventProc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam);
void initGLextensions();

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

Win32OpenGLContext::Win32OpenGLContext(int visual, 
                                   int x, 
                                   int y,
                                   unsigned int width, 
                                   unsigned int height,
                                   bool border) : 
  OpenGLContext(),
  mutex_("GL lock")
{
  width_ = width;
  height_ = height;
  // Seperate functions so we can set gdb breakpoints in constructor
  // callback to WindowEventProc will override width and height
  create_context(visual, x, y, width, height, border);
}



void
Win32OpenGLContext::create_context(int visual, int x, int y,
                                 unsigned int width, 
                                 unsigned int height,
                                 bool border)
{
  // register class
  if (!GlClassInitialized) {
    GlClassInitialized = true;
    GlClass.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
    GlClass.cbClsExtra = 0;
    GlClass.cbWndExtra = sizeof(long); // to save context pointer
    GlClass.hInstance = dllHINSTANCE;
    GlClass.hbrBackground = NULL;
    GlClass.lpszMenuName = NULL;
    GlClass.lpszClassName = GlClassName;
    GlClass.lpfnWndProc = WindowEventProc;
    GlClass.hIcon = NULL;
    GlClass.hCursor = NULL;

    if (!RegisterClass(&GlClass))
      PrintErr("RegisterClass");
  }

  int style = WS_OVERLAPPED;

#if 0
  // if parent window...
  style = WS_CHILD | WS_CLIPCHILDREN | WS_CLIPSIBLINGS;
#endif

  // create window (without parent for now)
  window_ = CreateWindow(GlClassName, GlClassName,
	                 style, x, y, width, height,
			 NULL, NULL, dllHINSTANCE, 
			 NULL);

  if (!window_)
    PrintErr("CreateWindow");

  SetWindowLong(window_, 0, (LONG) this);

  // create context
  DWORD dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER | PFD_GENERIC_ACCELERATED;

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


  if ((hDC_ = GetDC(window_)) == 0)
  {
    fprintf(stderr,"Bad DC returned by GetDC\n");
      
  }
  PrintErr("GetDC");

  int iPixelFormat = ChoosePixelFormat(hDC_, &pfd);
  if (iPixelFormat == 0)
    PrintErr("ChoosePixelFormat");
  
  SetPixelFormat(hDC_, iPixelFormat, &pfd);
  PrintErr("SetPixelFormat");

   /* Get the actual pixel format */
  DescribePixelFormat(hDC_, iPixelFormat, sizeof(pfd), &pfd);
  PrintErr("DescribePixelFormat");

  if ((context_ = wglCreateContext(hDC_)) == 0)
    PrintErr("wglCreateContext");

  if (!first_context_) {
    if ((first_context_ = wglCreateContext(hDC_)) == NULL) {
        PrintErr("wglCreateContext (first)");
    }

    make_current();
    ShaderProgramARB::init_shaders_supported();
    release();
  }


  }
  if (wglShareLists(first_context_,context_) == FALSE) {
    PrintErr("wglShareLists");
  }

  ShowWindow(window_, SW_SHOW);
  UpdateWindow(window_);

  initGLextensions();
}



Win32OpenGLContext::~Win32OpenGLContext()
{
  release();
  X11Lock::lock();

  if (context_ != first_context_)
    wglDeleteContext(context_);

  ReleaseDC(window_, hDC_);
  DestroyWindow(window_);

  X11Lock::unlock();

}


bool
Win32OpenGLContext::make_current()
{
  ASSERT(context_);
  bool result = true;
  X11Lock::lock();
  HGLRC current = wglGetCurrentContext();

  if (current != context_) {
    if ((result = wglMakeCurrent(hDC_,context_)) == 0)
      PrintErr("wglMakeCurrent");
  }
  X11Lock::unlock();
  if (!result)
  {
    std::cerr << "Win32 GL context failed make current.\n";
  }

  return result;
}


void
Win32OpenGLContext::release()
{
  X11Lock::lock();
  if (wglGetCurrentContext() != NULL)
    wglMakeCurrent(NULL,NULL);
  X11Lock::unlock();
}


int
Win32OpenGLContext::width()
{
  return width_;
}


int
Win32OpenGLContext::height()
{
  return height_;
}


void
Win32OpenGLContext::swap()
{  
  X11Lock::lock();
  SwapBuffers(hDC_);
  X11Lock::unlock();
}



#define GETCONFIG(attrib) \
if(glXGetConfig(display, &vinfo[i], attrib, &value) != 0){\
  cerr << "Error getting attribute: " << #attrib << std::endl; \
  return; \
}


void
Win32OpenGLContext::listvisuals()
{
  valid_visuals_.clear();
  HDC dc;
  if ((dc = GetDC(NULL)) == 0) // get DC For entire screen
  {
    fprintf(stderr,"Bad DC returned by GetDC\n");
      
  }

  vector<string> visualtags;
  vector<int> scores;

  //int  id, level, db, stereo, r,g,b,a, depth, stencil, ar, ag, ab, aa;

  int iPixelFormat;
  if ((iPixelFormat = GetPixelFormat(dc)) == 0)
  {
    fprintf(stderr,"Error: Bad Pixel Format Retrieved\n");
    ReleaseDC(NULL, dc);
    return;  
  }

  PIXELFORMATDESCRIPTOR pfd;
  DescribePixelFormat(dc,iPixelFormat,sizeof(PIXELFORMATDESCRIPTOR),&pfd);

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
  ReleaseDC(NULL, dc);
}

// win32 event loop
LRESULT CALLBACK 
WindowEventProc(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam)
{
  LONG result;
  Win32OpenGLContext *cxt = (Win32OpenGLContext*) GetWindowLong(hwnd, 0);
  switch (msg) {
  case WM_SIZE:
    {
      cxt->width_ = LOWORD(lParam);
      cxt->height_ = HIWORD(lParam);
      break;
    }
  case WM_MOVE: break;
  default: result = DefWindowProc(hwnd, msg, wParam, lParam);
  }
  
  return result;
}

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

BOOL WINAPI DllMain(HINSTANCE hinstance, DWORD reason, LPVOID reserved)
{
  switch (reason) {
  case DLL_PROCESS_ATTACH:
    dllHINSTANCE = hinstance; break;
  default: break;
  }
  return TRUE;
}
