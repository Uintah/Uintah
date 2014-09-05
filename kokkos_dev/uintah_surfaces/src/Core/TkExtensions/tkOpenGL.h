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

#ifndef SCIRun_Core_TkExtensions_tkOpenGL
#define SCIRun_Core_TkExtensions_tkOpenGL

/* tkOpenGL.h --
 *
 * ClientData struct for the Tk 'opengl' widget command
 *
 */

#include <sci_glx.h>
#include <stdio.h>
#include <tk.h>

#ifdef _WIN32
#  include <windows.h>
#  include <tkWinInt.h>
#  include <tkWinPort.h>
#  include <X11\XUtil.h>
#endif

#ifdef __sgi
#  include <X11/extensions/SGIStereo.h>
#endif


/*
 * A data structure of the following type is kept for each OpenGL
 * widget managed by this file:
 */

typedef struct {
    int direct;
    int buffersize;
    int level;
    int rgba;
    int doublebuffer;
    int stereo;
    int auxbuffers;
    int redsize;
    int greensize;
    int bluesize;
    int alphasize;
    int depthsize;
    int stencilsize;
    int accumredsize;
    int accumgreensize;
    int accumbluesize;
    int accumalphasize;
    int visualid;
    char* geometry;
    Tk_Cursor cursor;

    int num_fb;

    Tcl_Interp *interp;		/* Interpreter associated with widget. */
    Tk_Window tkwin;
    Display *display;		/* X's token for the window's display. */
    Window x11_win;
    int screen_number;
#ifndef _WIN32
    GLXWindow glx_win;
    GLXContext cx;
    XVisualInfo* vi;
    GLXFBConfig *fb_configs;
#endif
} OpenGLClientData;



#endif
