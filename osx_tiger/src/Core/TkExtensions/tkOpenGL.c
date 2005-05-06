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
 * tkOpenGL.c --
 *
 *	This module implements "OpenGL" widgets.
 *
 */

#include <Core/TkExtensions/tkOpenGL.h>
#include <string.h>

#if (TCL_MINOR_VERSION >= 4)
#define TCLCONST const
#else
#define TCLCONST
#endif

#ifdef HAVE_GLEW
int sci_glew_init()
{
  static int glew_init = 0;
  if(!glew_init) {
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (GLEW_OK != err )
    {
	/* problem: glewInit failed, something is seriously wrong */
	fprintf(stderr, "Error: %s\n", glewGetErrorString(err)); 
	return 1;
    }
    fprintf(stdout, "Status: Using GLEW %s\n", glewGetString(GLEW_VERSION));
    glew_init = 1;
  }
  return 0;
}
#endif

/*
 * Information used for argv parsing.
 */

static Tk_ConfigSpec configSpecs[] = {
    {TK_CONFIG_BOOLEAN, "-direct", "direct", "direct",
     "1", Tk_Offset(OpenGLClientData, direct), 0, 0},
    {TK_CONFIG_INT, "-buffersize", "bufferSize", "BufferSize",
     "8", Tk_Offset(OpenGLClientData, buffersize), 0, 0},
    {TK_CONFIG_INT, "-level", "level", "level",
     "0", Tk_Offset(OpenGLClientData, level), 0, 0},
    {TK_CONFIG_BOOLEAN, "-rgba", "rgba", "rgba",
     "1", Tk_Offset(OpenGLClientData, rgba), 0, 0},
    {TK_CONFIG_BOOLEAN, "-doublebuffer", "doublebuffer", "doublebuffer",
     "0", Tk_Offset(OpenGLClientData, doublebuffer), 0, 0},
    {TK_CONFIG_BOOLEAN, "-stereo", "glsize", "glsize",
     "0", Tk_Offset(OpenGLClientData, stereo), 0, 0},
    {TK_CONFIG_INT, "-auxbuffers", "glsize", "glsize",
     "0", Tk_Offset(OpenGLClientData, auxbuffers), 0, 0},
    {TK_CONFIG_INT, "-redsize", "glsize", "glsize",
     "2", Tk_Offset(OpenGLClientData, redsize), 0, 0},
    {TK_CONFIG_INT, "-greensize", "glsize", "glsize",
     "2", Tk_Offset(OpenGLClientData, greensize), 0, 0},
    {TK_CONFIG_INT, "-bluesize", "glsize", "glsize",
     "2", Tk_Offset(OpenGLClientData, bluesize), 0, 0},
    {TK_CONFIG_INT, "-alphasize", "glsize", "glsize",
     "0", Tk_Offset(OpenGLClientData, alphasize), 0, 0},
    {TK_CONFIG_INT, "-depthsize", "glsize", "glsize",
     "2", Tk_Offset(OpenGLClientData, depthsize), 0, 0},
    {TK_CONFIG_INT, "-stencilsize", "glsize", "glsize",
     "0", Tk_Offset(OpenGLClientData, stencilsize), 0, 0},
    {TK_CONFIG_INT, "-accumredsize", "glsize", "glsize",
     "0", Tk_Offset(OpenGLClientData, accumredsize), 0, 0},
    {TK_CONFIG_INT, "-accumgreensize", "glsize", "glsize",
     "0", Tk_Offset(OpenGLClientData, accumgreensize), 0, 0},
    {TK_CONFIG_INT, "-accumbluesize", "glsize", "glsize",
     "0", Tk_Offset(OpenGLClientData, accumbluesize), 0, 0},
    {TK_CONFIG_INT, "-accumalphasize", "glsize", "glsize",
     "0", Tk_Offset(OpenGLClientData, accumalphasize), 0, 0},
    {TK_CONFIG_INT, "-visualid", "visualId", "VisualId",
     "0", Tk_Offset(OpenGLClientData, visualid), 0, 0},
    {TK_CONFIG_STRING, "-geometry", "geometry", "Geometry",
     "100x100", Tk_Offset(OpenGLClientData, geometry), 0, 0},
    {TK_CONFIG_CURSOR, "-cursor", "cursor", "Cursor",
     "left_ptr", Tk_Offset(OpenGLClientData, cursor), 0, 0},
    {TK_CONFIG_END, (char *) NULL, (char *) NULL, (char *) NULL,
	(char *) NULL, 0, 0, 0}
    };

/*
 * Forward declarations for procedures defined later in this file:
 */

static int		OpenGLConfigure _ANSI_ARGS_((Tcl_Interp *interp,
						     OpenGLClientData *OpenGLPtr,
						     int argc, char **argv,
						     int flags));
static void		OpenGLDestroy _ANSI_ARGS_((ClientData clientData));
static void		OpenGLEventProc _ANSI_ARGS_((ClientData clientData,
						     XEvent *eventPtr));
static int		OpenGLWidgetCmd _ANSI_ARGS_((ClientData clientData,
						     Tcl_Interp *, 
						     int argc, 
						     char **argv));
static int		OpenGLListVisuals _ANSI_ARGS_((Tcl_Interp *interp, 
						       OpenGLClientData *OpenGLPtr));


/*
 *--------------------------------------------------------------
 *
 * OpenGLCmd --
 *
 *	This procedure is invoked to process the "opengl" Tcl
 *	command.  It creates a new "opengl" widget.
 *
 * Results:
 *	A standard Tcl result.
 *
 * Side effects:
 *	A new widget is created and configured.
 *
 *--------------------------------------------------------------
 */

int
OpenGLCmd(clientData, interp, argc, argv)
    ClientData clientData;	/* Main window associated with interpreter. */
    Tcl_Interp *interp;		/* Current interpreter. */
    int argc;			/* Number of arguments. */
    TCLCONST char **argv;	/* Argument strings. */
{
#ifndef _WIN32
    Tk_Window mainwin = (Tk_Window) clientData;
    OpenGLClientData *OpenGLPtr;
    Colormap cmap;
    Tk_Window tkwin;
    int attributes[50];
    XVisualInfo temp_vi;
    int tempid;
    int n, i;
    int idx = 0;



    if (argc < 2) {
	Tcl_AppendResult(interp, "wrong # args:  should be \"",
			 argv[0], " pathName ?options?\"", (char *) NULL);
	return TCL_ERROR;
    }

    tkwin = Tk_CreateWindowFromPath(interp, mainwin, argv[1], (char *) NULL);
    if (tkwin == NULL) {
	return TCL_ERROR;
    }
    Tk_SetClass(tkwin, "OpenGL");

    /* Allocate and initialize the widget record. */
    OpenGLPtr = (OpenGLClientData *) ckalloc(sizeof(OpenGLClientData));
    OpenGLPtr->geometry = 0;
    OpenGLPtr->cursor = 0;

    OpenGLPtr->interp = interp;
    OpenGLPtr->tkwin = tkwin;
    OpenGLPtr->display = Tk_Display(tkwin);
    OpenGLPtr->x11_win=0;
    OpenGLPtr->screen_number = Tk_ScreenNumber(tkwin);
    OpenGLPtr->glx_win=0;
    OpenGLPtr->cx=0;
    OpenGLPtr->vi=0;
    OpenGLPtr->fb_configs=glXGetFBConfigs(OpenGLPtr->display, 
					  OpenGLPtr->screen_number, 
					  &(OpenGLPtr->num_fb));
    
    Tk_CreateEventHandler(OpenGLPtr->tkwin, 
			  StructureNotifyMask,
			  OpenGLEventProc, 
			  (ClientData) OpenGLPtr);
    
    Tcl_CreateCommand(interp, 
		      Tk_PathName(OpenGLPtr->tkwin), 
		      OpenGLWidgetCmd,
		      (ClientData) OpenGLPtr, 
		      (Tcl_CmdDeleteProc *)0);
    if (OpenGLConfigure(interp, OpenGLPtr, argc-2, argv+2, 0) != TCL_OK) {
	return TCL_ERROR;
    }

#if 0
    OpenGLPtr->vi = 
      glXGetVisualFromFBConfig(OpenGLPtr->display,
			       OpenGLPtr->fb_configs[OpenGLPtr->visualid]);
    
    cmap = XCreateColormap(OpenGLPtr->display,
			   Tk_WindowId(Tk_MainWindow(OpenGLPtr->interp)),
			   OpenGLPtr->vi->visual,
			   AllocNone);
    
    if (Tk_SetWindowVisual(OpenGLPtr->tkwin, 
			   OpenGLPtr->vi->visual,
			   OpenGLPtr->vi->depth, 
			   cmap) != 1) {
      Tcl_AppendResult(interp, "Error setting colormap for window", (char*)NULL);
      return TCL_ERROR;
    }
    XSync(OpenGLPtr->display, False);
#endif


    tempid = OpenGLPtr->visualid;

#ifdef _WIN32
    OpenGLPtr->visualid = 0;
#endif

    if (OpenGLPtr->visualid) {
      temp_vi.visualid = OpenGLPtr->visualid;
      OpenGLPtr->vi = 
	XGetVisualInfo(OpenGLPtr->display, VisualIDMask, &temp_vi, &n);
      if(!OpenGLPtr->vi || n!=1){
	    Tcl_AppendResult(interp, "Error finding visual", NULL);
	    return TCL_ERROR;
      }
    } else {
      /*
       * Pick the right visual...
       */
      attributes[idx++]=GLX_BUFFER_SIZE;
      attributes[idx++]=OpenGLPtr->buffersize;
      attributes[idx++]=GLX_LEVEL;
      attributes[idx++]=OpenGLPtr->level;
      if(OpenGLPtr->rgba)
	attributes[idx++]=GLX_RGBA;
      if(OpenGLPtr->doublebuffer)
	attributes[idx++]=GLX_DOUBLEBUFFER;
      if(OpenGLPtr->stereo)
	attributes[idx++]=GLX_STEREO;
      attributes[idx++]=GLX_AUX_BUFFERS;
      attributes[idx++]=OpenGLPtr->auxbuffers;
      attributes[idx++]=GLX_RED_SIZE;
      attributes[idx++]=OpenGLPtr->redsize;
      attributes[idx++]=GLX_GREEN_SIZE;
      attributes[idx++]=OpenGLPtr->greensize;
      attributes[idx++]=GLX_BLUE_SIZE;
      attributes[idx++]=OpenGLPtr->bluesize;
      attributes[idx++]=GLX_ALPHA_SIZE;
      attributes[idx++]=OpenGLPtr->alphasize;
      attributes[idx++]=GLX_DEPTH_SIZE;
      attributes[idx++]=OpenGLPtr->depthsize;
      attributes[idx++]=GLX_STENCIL_SIZE;
      attributes[idx++]=OpenGLPtr->stencilsize;
      attributes[idx++]=GLX_ACCUM_RED_SIZE;
      attributes[idx++]=OpenGLPtr->accumredsize;
      attributes[idx++]=GLX_ACCUM_GREEN_SIZE;
      attributes[idx++]=OpenGLPtr->accumgreensize;
      attributes[idx++]=GLX_ACCUM_BLUE_SIZE;
      attributes[idx++]=OpenGLPtr->accumbluesize;
      attributes[idx++]=GLX_ACCUM_ALPHA_SIZE;
      attributes[idx++]=OpenGLPtr->accumalphasize;
#if 0
      attributes[idx++]=GLX_SAMPLES_SGIS;
      attributes[idx++]=4;
#endif
      attributes[idx++]=None;
	  
#ifndef _WIN32
      OpenGLPtr->vi = 
	glXChooseVisual(OpenGLPtr->display, 
			OpenGLPtr->screen_number,
			attributes);
#else
      vi = (XVisualInfo*)malloc(sizeof(XVisxcreatecolormapualInfo));
      vi->visualid=tempid;
#endif
    }

    if (!OpenGLPtr->vi) {
      Tcl_AppendResult(interp, "Error selecting visual", (char*)NULL);
      return TCL_ERROR;
    }
    
    OpenGLPtr->visualid=tempid;
    
#ifdef _WIN32
    cmap = XCreateColormap(Tk_Display(tkwin),
			   RootWindow(Tk_Display(tkwin), 0/*vi->screen*/),
			   0/*vi->visual*/, AllocNone);

    //if( Tk_SetWindowVisual(tkwin, 0/*vi->visual*/, vi->depth, cmap) != 1){
#else
      
    cmap = XCreateColormap(OpenGLPtr->display,
			   Tk_WindowId(Tk_MainWindow(OpenGLPtr->interp)),
			   OpenGLPtr->vi->visual, 
			   AllocNone);

    if( Tk_SetWindowVisual(OpenGLPtr->tkwin, 
			   OpenGLPtr->vi->visual, 
			   OpenGLPtr->vi->depth, cmap) != 1){
#endif
      Tcl_AppendResult(interp, "Error setting visual for window", (char*)NULL);
	return TCL_ERROR;
    }
    XSync(OpenGLPtr->display, False);

    interp->result = Tk_PathName(OpenGLPtr->tkwin);
#endif
    return TCL_OK;
}

/*
 *--------------------------------------------------------------
 *
 * OpenGLWidgetCmd --
 *
 *	This procedure is invoked to process the Tcl command
 *	that corresponds to a widget managed by this module.
 *	See the user documentation for details on what it does.
 *
 * Results:
 *	A standard Tcl result.
 *
 * Side effects:
 *	See the user documentation.
 *
 *--------------------------------------------------------------
 */

static int
OpenGLWidgetCmd(clientData, interp, argc, argv)
    ClientData clientData;		/* Information about OpenGL widget. */
    Tcl_Interp *interp;			/* Current interpreter. */
    int argc;				/* Number of arguments. */
    char **argv;			/* Argument strings. */
{
  OpenGLClientData *OpenGLPtr = (OpenGLClientData *) clientData;
  int result = TCL_OK;
  int length;
  //  int id;
  char c;
  
  if (argc < 2) {
    Tcl_AppendResult(interp, "wrong # args: should be \"",
		     argv[0], " option ?arg arg ...?\"", (char *) NULL);
    return TCL_ERROR;
  }
  
  Tk_Preserve((ClientData) OpenGLPtr);
  c = argv[1][0];
  length = strlen(argv[1]);
  if ((c == 'c') && (strncmp(argv[1], "configure", length) == 0)) {
    if (argc == 2) {
      result = Tk_ConfigureInfo(interp, OpenGLPtr->tkwin, configSpecs,
				(char *) OpenGLPtr, (char *) NULL, 0);
    } else if (argc == 3) {
      result = Tk_ConfigureInfo(interp, OpenGLPtr->tkwin, configSpecs,
				(char *) OpenGLPtr, argv[2], 0);
    } else {
      result = OpenGLConfigure(interp, OpenGLPtr, argc-2, argv+2,
			       TK_CONFIG_ARGV_ONLY);
    }
  } 
#ifndef _WIN32
  else if ((c == 'c') && (strncmp(argv[1], "cget", length) == 0)) {
    if (argc == 3) {
      result = Tk_ConfigureValue(interp, OpenGLPtr->tkwin, configSpecs,
				 (char *) OpenGLPtr, argv[2], 0);
    }
    else {
      Tcl_AppendResult(interp, "bad cget command \"", argv[1],
		       "", 
		       (char *) NULL);
      result = TCL_ERROR;
    }
  } 
  else if ((c == 'l') && (strncmp(argv[1], "listvisuals", length) == 0)) 
  {
    result = OpenGLListVisuals(interp, OpenGLPtr);
  } 
  else {
    Tcl_AppendResult(interp, "bad option \"", argv[1],
		     "\":  must be configure, cget, position, or size", 
		     (char *) NULL);
    result = TCL_ERROR;
  }
  
  Tk_Release((ClientData) OpenGLPtr);
#endif
  return result;
}

/*
 *----------------------------------------------------------------------
 *
 * OpenGLConfigure --
 *
 *	This procedure is called to process an argv/argc list in
 *	conjunction with the Tk option database to configure (or
 *	reconfigure) a OpenGL widget.
 *
 * Results:
 *	The return value is a standard Tcl result.  If TCL_ERROR is
 *	returned, then interp->result contains an error message.
 *
 * Side effects:
 *	Configuration information, such as colors, border width,
 *	etc. get set for OpenGLPtr;  old resources get freed,
 *	if there were any.
 *
 *----------------------------------------------------------------------
 */

static int
OpenGLConfigure(interp, OpenGLPtr, argc, argv, flags)
    Tcl_Interp *interp;			/* Used for error reporting. */
    OpenGLClientData *OpenGLPtr;			/* Information about widget. */
    int argc;				/* Number of valid entries in argv. */
    char **argv;			/* Arguments. */
    int flags;				/* Flags to pass to
					 * Tk_ConfigureWidget. */
{
#ifndef _WIN32
  int height, width;

    if (Tk_ConfigureWidget(interp, OpenGLPtr->tkwin, configSpecs,
	    argc, argv, (char *) OpenGLPtr, flags) != TCL_OK) {
	return TCL_ERROR;
    }

    if (sscanf(OpenGLPtr->geometry, "%dx%d", &width, &height) != 2) {
      Tcl_AppendResult(interp, "bad geometry \"", OpenGLPtr->geometry,
		       "\": expected widthxheight", (char *) NULL);
      return TCL_ERROR;
    }

    Tk_GeometryRequest(OpenGLPtr->tkwin, width, height);
    Tk_DefineCursor(OpenGLPtr->tkwin, OpenGLPtr->cursor);
#endif
    return TCL_OK;
}

/*
 *--------------------------------------------------------------
 *
 * OpenGLEventProc --
 *
 *	This procedure is invoked by the Tk dispatcher for various
 *	events on OpenGLs.
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	When the window gets deleted, internal structures get
 *	cleaned up.  When it gets exposed, it is redisplayed.
 *
 *--------------------------------------------------------------
 */

static void
OpenGLEventProc(clientData, eventPtr)
     ClientData clientData;	/* Information about window. */
     XEvent *eventPtr;		/* Information about event. */
{
#ifndef _WIN32
  OpenGLClientData *OpenGLPtr = (OpenGLClientData *) clientData;
  if (eventPtr->type == DestroyNotify) {
    
      glXMakeContextCurrent(OpenGLPtr->display, None, None, 0);
      XSync(OpenGLPtr->display, False);
      //      glXDestroyWindow(OpenGLPtr->display, OpenGLPtr->glx_win);
      glXDestroyContext(OpenGLPtr->display, OpenGLPtr->cx);
      XSync(OpenGLPtr->display, False);

      Tcl_DeleteCommand(OpenGLPtr->interp, Tk_PathName(OpenGLPtr->tkwin));
      OpenGLPtr->tkwin = NULL;
      Tk_EventuallyFree((ClientData) OpenGLPtr, (Tcl_FreeProc*)OpenGLDestroy);
    }
#endif
}


/*
 *----------------------------------------------------------------------
 *
 * OpenGLDestroy --
 *
 *	This procedure is invoked by Tk_EventuallyFree or Tk_Release
 *	to clean up the internal structure of a OpenGL at a safe time
 *	(when no-one is using it anymore).
 *
 * Results:
 *	None.
 *
 * Side effects:
 *	Everything associated with the OpenGL is freed up.
 *
 *----------------------------------------------------------------------
 */

static void
OpenGLDestroy(clientData)
    ClientData clientData;	/* Info about OpenGL widget. */
{
#ifndef _WIN32
    OpenGLClientData *OpenGLPtr = (OpenGLClientData *) clientData;
#ifdef _WIN32
    /* this needs some additional checking first */
    wglDeleteContext(OpenGLPtr->cx);
#endif
    Tk_FreeOptions(configSpecs, (char *) OpenGLPtr, OpenGLPtr->display, 0);
    ckfree((char *) OpenGLPtr);
#endif
}

#ifndef _WIN32
static GLXContext first_context = 0;


GLXContext OpenGLGetContext(interp, name)
    Tcl_Interp* interp;
    char* name;
{
    Tcl_CmdInfo info;
    OpenGLClientData* OpenGLPtr;
#ifdef _WIN32
    PIXELFORMATDESCRIPTOR pfd;
    HDC hDC;
    HWND hWnd;
#endif
    if(!Tcl_GetCommandInfo(interp, name, &info))
      return 0;

    OpenGLPtr=(OpenGLClientData*)info.clientData;

    if (OpenGLPtr->tkwin != 
	Tk_NameToWindow(interp, name, Tk_MainWindow(interp)))
      return 0;
    
    if (OpenGLPtr->display != Tk_Display(OpenGLPtr->tkwin))
      return 0;

    if (OpenGLPtr->display)
      XSync(OpenGLPtr->display, False);

    if (!OpenGLPtr->x11_win) {
      Tk_MakeWindowExist(OpenGLPtr->tkwin);
      XSync(OpenGLPtr->display, False);
      OpenGLPtr->x11_win = Tk_WindowId(OpenGLPtr->tkwin);
    }
    
    if (!OpenGLPtr->x11_win) 
      return 0;

#if 0
    if (!OpenGLPtr->glx_win) {
      printf ("Making context w/ visual id#%d\n", OpenGLPtr->visualid);
      OpenGLPtr->glx_win = 
	glXCreateWindow(OpenGLPtr->display,
			OpenGLPtr->fb_configs[OpenGLPtr->visualid],
			OpenGLPtr->x11_win,
			NULL);
      XSync(OpenGLPtr->display, False);
    }
#endif
    OpenGLPtr->glx_win = OpenGLPtr->x11_win;

    if (!OpenGLPtr->cx) {
#ifdef _WIN32
      hWnd = TkWinGetHWND(Tk_WindowId(tkwin));
      hDC = GetDC(hWnd);
      printf("Trying to SetPixelFormat to %d\n",OpenGLPtr->visualid);
      if (!SetPixelFormat(hDC,OpenGLPtr->visualid,&pfd)) {
	printf("SetPixelFormat() failed; Error code: %d\n",GetLastError());
	return 0;
      }
      OpenGLPtr->cx = wglCreateContext(hDC);
      if (!OpenGLPtr->cx)
	printf("wglCreateContext() returned NULL; Error code: %d\n",
	       GetLastError());
#else

#if 0
      OpenGLPtr->cx = 
	glXCreateNewContext(OpenGLPtr->display,
			    OpenGLPtr->fb_configs[OpenGLPtr->visualid],
			    GLX_RGBA_TYPE,
			    NULL,
			    OpenGLPtr->direct);
#endif
      OpenGLPtr->cx =
	glXCreateContext(OpenGLPtr->display,
			 OpenGLPtr->vi,
			 first_context, OpenGLPtr->direct);
      if (!first_context) first_context = OpenGLPtr->cx;
			 


#endif
    }

    if (!OpenGLPtr->cx) {
      Tcl_AppendResult(interp, "Error making GL context", (char*)NULL);
      return 0;
    }

    if (!OpenGLPtr->glx_win)
      return 0;

    return OpenGLPtr->cx;
}
#endif

void OpenGLSwapBuffers(interp, name)
    Tcl_Interp* interp;
    char* name;
{
#ifndef _WIN32
    Tcl_CmdInfo info;
    OpenGLClientData* OpenGLPtr;
    if(!Tcl_GetCommandInfo(interp, name, &info))
      return;

    OpenGLPtr=(OpenGLClientData*)info.clientData;
    glXSwapBuffers(OpenGLPtr->display, OpenGLPtr->glx_win);
#endif
}


int OpenGLMakeCurrent(interp, name)
    Tcl_Interp* interp;
    char* name;
{
#ifndef _WIN32
    Tcl_CmdInfo info;
    OpenGLClientData* OpenGLPtr;

    if(!Tcl_GetCommandInfo(interp, name, &info))
      return 0;

    OpenGLPtr=(OpenGLClientData*)info.clientData;

    if (!glXMakeContextCurrent(OpenGLPtr->display,
			       OpenGLPtr->glx_win,
			       OpenGLPtr->glx_win,
			       OpenGLPtr->cx)) {
      printf("%s failed make current.\n", Tk_PathName(OpenGLPtr->tkwin));
      return 0;
    }
#endif
    return 1;
}




#define GETCONFIG(attrib, value) \
if(glXGetConfig(OpenGLPtr->display, &vinfo[i], attrib, &value) != 0){\
  Tcl_AppendResult(interp, "Error getting attribute: " #attrib, (char *)NULL); \
  return TCL_ERROR; \
}

#define GETCONFIG_13(attrib, value) \
if(glXGetFBConfigAttrib(OpenGLPtr->display, OpenGLPtr->fb_configs[i], \
                       attrib, &value) != 0) { \
  Tcl_AppendResult(interp, "Error getting attribute: " #attrib,(char *)NULL); \
  return TCL_ERROR; \
}


static int
OpenGLListVisuals(interp, OpenGLPtr)
     Tcl_Interp *interp;
     OpenGLClientData *OpenGLPtr;	
{
#ifndef _WIN32
  int  i;
  int  score=0;
  char buf[200];
  int  id, level, db, stereo, r,g,b,a, depth, stencil, ar, ag, ab, aa;
  //int able;
  char samples_string[20] = "";
#ifdef __sgi
  int  samples_sgis;
#endif
  int nvis;
  XVisualInfo* vinfo=XGetVisualInfo(OpenGLPtr->display, 0, NULL, &nvis);
  if(!vinfo)
  {
    Tcl_AppendResult(interp, "XGetVisualInfo failed", (char *) NULL);
    return TCL_ERROR;
  }


  for(i=0;i<nvis;i++)
  {
    id = vinfo[i].visualid;
    //    GETCONFIG(GLX_FBCONFIG_ID, id);
    GETCONFIG(GLX_LEVEL, level);
    GETCONFIG(GLX_DOUBLEBUFFER, db);
    GETCONFIG(GLX_STEREO, stereo);
    GETCONFIG(GLX_RED_SIZE, r);
    GETCONFIG(GLX_GREEN_SIZE, g);
    GETCONFIG(GLX_BLUE_SIZE, b);
    GETCONFIG(GLX_ALPHA_SIZE, a);
    GETCONFIG(GLX_DEPTH_SIZE, depth);
    GETCONFIG(GLX_STENCIL_SIZE, stencil);
    GETCONFIG(GLX_ACCUM_RED_SIZE, ar);
    GETCONFIG(GLX_ACCUM_GREEN_SIZE, ag);
    GETCONFIG(GLX_ACCUM_BLUE_SIZE, ab);
    GETCONFIG(GLX_ACCUM_ALPHA_SIZE, aa);
    //    GETCONFIG(GLX_RENDER_TYPE, rt);
    // GETCONFIG(GLX_DRAWABLE_TYPE, dt);

    //GETCONFIG(GLX_X_RENDERABLE, able);
    //if (!able) continue;

    score = db?200:0;
    score += stereo?1:0;
    score += r+g+b+a;
    score += depth*5;

#ifdef __sgi
    GETCONFIG(GLX_SAMPLES_SGIS, samples_sgis);
    score += samples_sgis?50:0;
    sprintf(samples_string, "samples=%d, ", samples_sgis);
#endif

    sprintf (buf, "{id=%02x, level=%d, %s%srgba=%d:%d:%d:%d, depth=%d, stencil=%d, accum=%d:%d:%d:%d, %sscore=%d} ",
	     id, level, db?"double, ":"single, ", stereo?"stereo, ":"", 
	     r, g, b, a, depth, stencil, ar, ag, ab, aa,
	     samples_string, score);
    Tcl_AppendResult(interp, buf, (char *)NULL);
  }
#endif
  return TCL_OK;
};

