/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

#include "tk.h"
#include <string.h>

int
Tk_CursorCmd(clientData, interp, argc, argv)
     ClientData clientData;
     Tcl_Interp *interp;                 /* Current interpreter. */
     int argc;                           /* Number of arguments. */
     char **argv;                        /* Argument strings. */
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
