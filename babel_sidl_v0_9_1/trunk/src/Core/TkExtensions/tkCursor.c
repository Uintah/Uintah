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


#include "tk.h"
#include <string.h>
#if (TCL_MINOR_VERSION >= 4)
#define TCLCONST const
#else
#define TCLCONST
#endif

int
Tk_CursorCmd(clientData, interp, argc, argv)
     ClientData clientData;
     Tcl_Interp *interp;                 /* Current interpreter. */
     int argc;                           /* Number of arguments. */
     TCLCONST char **argv;               /* Argument strings. */
{
  int length;
  char c;
  Tk_Window tkwin = (Tk_Window) clientData;
  Tk_Window winPtr;
 
  if (argc < 2) {
    Tcl_AppendResult(interp, "wrong # args:  should be \"",
                     argv[0], " ?options?\"", (char *) NULL);
    return TCL_ERROR;
  }
 
  length = strlen(argv[1]);
  c = argv[1][0];
  if ( c == 'w' && strncmp(argv[1],"warp",length) == 0 ) {
    int x,y;
    Window win;
 
    if (argc != 5) {
      Tcl_AppendResult(interp, "wrong # args:  should be \"",
                       argv[0], " warp <window> <x> <y>\"", (char *) NULL);
      return TCL_ERROR;
    }
    if ( argv[2][0] == '.' ) {
      winPtr = (Tk_Window ) Tk_NameToWindow(interp, argv[2], tkwin);
      if (winPtr == NULL) {
        return TCL_ERROR;
      }
      win = Tk_WindowId(winPtr);
    } else {
      winPtr = (Tk_Window )clientData;
      win = None;
    }
 
    if  ( Tk_GetPixels(interp,winPtr,argv[3],&x) != TCL_OK )return TCL_ERROR;
    if  ( Tk_GetPixels(interp,winPtr,argv[4],&y) != TCL_OK )return TCL_ERROR;

#ifndef _WIN32
    XWarpPointer(Tk_Display(winPtr),None,win,0,0,0,0,x,y);
#endif
  }
 
  return TCL_OK;
}
