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
 
    XWarpPointer(Tk_Display(winPtr),None,win,0,0,0,0,x,y);
  }
 
  return TCL_OK;
}
