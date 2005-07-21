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
 *  TkOpenGLContext.h:
 *
 *  Written by:
 *   McKay Davis
 *   December 2004
 */


#ifndef SCIRun_Core_2d_TkOpenGLContext_h
#define SCIRun_Core_2d_TkOpenGLContext_h

#include <sci_glx.h>
#include <stdio.h>
#include <tk.h>

#ifdef _WIN32
#  include <tkWinInt.h>
#  include <tkWinPort.h>
#  include <X11\XUtil.h>
#endif

#ifdef __sgi
#  include <X11/extensions/SGIStereo.h>
#endif

#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <sgi_stl_warnings_on.h>

using std::string;
using std::vector;

namespace SCIRun {

class TkOpenGLContext {
public:
  TkOpenGLContext(const string &, int visualid=0, 
		  int width=640, int height = 480);
  virtual ~TkOpenGLContext();
  
  static string		listvisuals();
  bool			make_current();
  void			release();
  int			width();
  int			height();
  void			swap();

  #ifdef _WIN32
  const char*           ReportCapabilities();
  #endif

  static vector<int>	valid_visuals_;
  int			direct_;
  int			buffersize_;
  int			level_;
  int			rgba_;
  int			doublebuffer_;
  int			stereo_;
  int			auxbuffers_;
  int			redsize_;
  int			greensize_;
  int			bluesize_;
  int			alphasize_;
  int			depthsize_;
  int			stencilsize_;
  int			accumredsize_;
  int			accumgreensize_;
  int			accumbluesize_;
  int			accumalphasize_;
  int			visualid_;
  int			screen_number_;
  char*			geometry_;
  Tcl_Interp *		interp_;  /* Interpreter associated with widget. */
  Display *		display_; /* X's token for the window's display. */
  Window		x11_win_;
  Tk_Window		tkwin_;
  Tk_Window		mainwin_;
#ifndef _WIN32
  GLXContext		context_;
#else
  HDC                   hDC_;
  HGLRC                 context_;
  HWND                  hWND_;
#endif
  XVisualInfo*		vi_;
  Colormap		colormap_;
  Tk_Cursor		cursor_;
  string		id_;
};

} // End namespace SCIRun

#endif // SCIRun_Core_2d_OpenGLContext_h
