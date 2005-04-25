#include <tkPort.h>
#include <tkInt.h>
#include <Util/stringUtil.h>
#include <UI/OpenGL.h>
#include <UI/GuiArgs.h>
#include <UI/UserInterface.h>
#include <iostream>
#include <Logging/Log.h>

using namespace SemotusVisum;
/*
 * Information used for argv parsing.
 */

static Tk_ConfigSpec configSpecs[] = {
    {TK_CONFIG_BOOLEAN, "-direct", "direct", "direct",
     "1", Tk_Offset(_OpenGL, direct), 0},
    {TK_CONFIG_INT, "-buffersize", "bufferSize", "BufferSize",
     "8", Tk_Offset(_OpenGL, buffersize), 0},
    {TK_CONFIG_INT, "-level", "level", "level",
     "0", Tk_Offset(_OpenGL, level), 0},
    {TK_CONFIG_BOOLEAN, "-rgba", "rgba", "rgba",
     "1", Tk_Offset(_OpenGL, rgba), 0},
    {TK_CONFIG_BOOLEAN, "-doublebuffer", "doublebuffer", "doublebuffer",
     "0", Tk_Offset(_OpenGL, doublebuffer), 0},
    {TK_CONFIG_BOOLEAN, "-stereo", "glsize", "glsize",
     "0", Tk_Offset(_OpenGL, stereo), 0},
    {TK_CONFIG_INT, "-auxbuffers", "glsize", "glsize",
     "0", Tk_Offset(_OpenGL, auxbuffers), 0},
    {TK_CONFIG_INT, "-redsize", "glsize", "glsize",
     "2", Tk_Offset(_OpenGL, redsize), 0},
    {TK_CONFIG_INT, "-greensize", "glsize", "glsize",
     "2", Tk_Offset(_OpenGL, greensize), 0},
    {TK_CONFIG_INT, "-bluesize", "glsize", "glsize",
     "2", Tk_Offset(_OpenGL, bluesize), 0},
    {TK_CONFIG_INT, "-alphasize", "glsize", "glsize",
     "0", Tk_Offset(_OpenGL, alphasize), 0},
    {TK_CONFIG_INT, "-depthsize", "glsize", "glsize",
     "2", Tk_Offset(_OpenGL, depthsize), 0},
    {TK_CONFIG_INT, "-stencilsize", "glsize", "glsize",
     "0", Tk_Offset(_OpenGL, stencilsize), 0},
    {TK_CONFIG_INT, "-accumredsize", "glsize", "glsize",
     "0", Tk_Offset(_OpenGL, accumredsize), 0},
    {TK_CONFIG_INT, "-accumgreensize", "glsize", "glsize",
     "0", Tk_Offset(_OpenGL, accumgreensize), 0},
    {TK_CONFIG_INT, "-accumbluesize", "glsize", "glsize",
     "0", Tk_Offset(_OpenGL, accumbluesize), 0},
    {TK_CONFIG_INT, "-accumalphasize", "glsize", "glsize",
     "0", Tk_Offset(_OpenGL, accumalphasize), 0},
    {TK_CONFIG_INT, "-visual", "visual", "visual",
     "0", Tk_Offset(_OpenGL, visualid), 0},
    {TK_CONFIG_STRING, "-geometry", "geometry", "Geometry",
     "100x100", Tk_Offset(_OpenGL, geometry), 0},
    {TK_CONFIG_CURSOR, "-cursor", "cursor", "Cursor",
     "left_ptr", Tk_Offset(_OpenGL, cursor), 0},
    {TK_CONFIG_END, (char *) NULL, (char *) NULL, (char *) NULL,
	(char *) NULL, 0, 0}
};

Tk_Window
OpenGL::tkwin = 0;

Window
OpenGL::win = 0;

Display *
OpenGL::dpy = 0;

GLXContext
OpenGL::cx = 0;


string
OpenGL::myname;

vector<XVisualInfo*>
OpenGL::visuals;

int
OpenGL::winx = 640;

int
OpenGL::winy = 512;

GLXFBConfig *
OpenGL::glx_fbconfig = 0;

GLXContext
OpenGL::OpenGLGetContext(Tcl_Interp* interp, char* name ) {

  Log::log( ENTER, "[OpenGL::OpenGLGetContext] entered" );
  
  Tcl_CmdInfo info;
  _OpenGL* OpenGLPtr;
  Tk_Window tkwin;
  Display *dpy=NULL;
  GLXFBConfig *glx_fbconfig = NULL;
  
  if(!Tcl_GetCommandInfo(interp, name, &info))
    return 0;
  OpenGLPtr=(_OpenGL*)info.clientData;
  if(!OpenGLPtr->cx){
    tkwin=Tk_NameToWindow(interp, name, Tk_MainWindow(interp));
    
    if(!dpy)
    {  
      dpy=Tk_Display(tkwin);
    }

    if( dpy == NULL )
    {   
      Log::log( ERROR, "[OpenGL::OpenGLGetContext] display initialization failed" );
      
    }
    
    Log::log( DEBUG, "[OpenGL::OpenGLGetContext] creating OpenGL context" );

    // TEST_CODE

    
    int nelements;
    
    int major, minor;

    if( glXQueryVersion( dpy, &major, &minor ) )
    {
      Log::log( DEBUG, "[OpenGL::OpenGLGetContext] GLX version: " + mkString(major) + "." + mkString(minor) );
    }
    else
    {
      Log::log( ERROR, "[OpenGL::OpenGLGetContext] glxQueryVersion failed -- The client library and the
                         server implementations must have different major version numbers" );	
    }

    glx_fbconfig = glXGetFBConfigs ( dpy, OpenGLPtr->vi->screen, &nelements );

      
    Log::log( DEBUG, "[OpenGL::OpenGLGetContext] screen = " + mkString(OpenGLPtr->vi->screen) + ", nelements = " + mkString(nelements) );

    if( glx_fbconfig == NULL )
    {   
      Log::log( ERROR, "[OpenGL::OpenGLGetContext] glXGetFBConfigs failed" );      
    } 

    OpenGLPtr->cx = glXCreateNewContext(dpy, *glx_fbconfig, GLX_RGBA_TYPE, 0, OpenGLPtr->direct);
    
    // /TEST_CODE

    //OpenGLPtr->cx = glXCreateContext(dpy, OpenGLPtr->vi, 0, OpenGLPtr->direct);
    
    if(!OpenGLPtr->cx){
      Tcl_AppendResult(interp, "Error making GL context", (char*)NULL);
      Log::log( ERROR, "[OpenGL::OpenGLGetContext] error making GL context" );
      return 0;
    }
  }
  
  Log::log( LEAVE, "[OpenGL::OpenGLGetContext] leaving" );
  return OpenGLPtr->cx;
}

/* updated function for glx 1.3 -- broken 
#define GETCONFIG(attrib) \
if(glXGetFBConfigAttrib(dpy, *glx_fbconfig, attrib, &value) != 0){\
  args.error("Error getting attribute: " #attrib); \
  Log::log( ERROR, "[GETCONFIG] error getting attribute" ); \
  return; \
}
*/

#define GETCONFIG(attrib) \
if(glXGetConfig(dpy, &vinfo[i], attrib, &value) != 0){\
  args.error("Error getting attribute: " #attrib); \
  return; \
}

void
OpenGL::listvisuals(GuiArgs& args)
{
  Log::log( ENTER, "[OpenGL::listvisuals] entered" );
  Tk_Window topwin=Tk_NameToWindow(UserInterface::getInterp(),
				   ccast_unsafe(args[2]),
				   Tk_MainWindow(UserInterface::getInterp()));
  if(!topwin)
  {
    cerr << "Unable to locate window!\n";
    return;
  }

  dpy=Tk_Display(topwin);
  int screen=Tk_ScreenNumber(topwin);
  vector<string> visualtags;
  vector<int> scores;
  visuals.clear();
  int nvis;
  XVisualInfo* vinfo=XGetVisualInfo(dpy, 0, NULL, &nvis);

  // check to see if glx_fbconfig is already defined
  // if not, define it and check again
  if( glx_fbconfig == NULL )
  {
    int nelements;
    glx_fbconfig = glXGetFBConfigs ( dpy, 0, &nelements );

    if( glx_fbconfig == NULL )
    { 
      Log::log( ERROR, "[OpenGL::listvisuals] glx_fbconfig is undefined" );
    }
  }
  
  if(!vinfo){
    args.error("XGetVisualInfo failed");
    return;
  }
  int i;
  for(i=0;i<nvis;i++){
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
    string tag(buf);
    GETCONFIG(GLX_DOUBLEBUFFER);
    if(value){
      score+=200;
      tag += "double, ";
    } else {
      tag += "single, ";
    }
    GETCONFIG(GLX_STEREO);
    if(value){
      score+=1;
      tag += "stereo, ";
    }
    tag += "rgba=";
    GETCONFIG(GLX_RED_SIZE);
    tag+=mkString(value)+":";
    score+=value;
    GETCONFIG(GLX_GREEN_SIZE);
    tag+=mkString(value)+":";
    score+=value;
    GETCONFIG(GLX_BLUE_SIZE);
    tag+=mkString(value)+":";
    score+=value;
    GETCONFIG(GLX_ALPHA_SIZE);
    tag+=mkString(value);
    score+=value;
    GETCONFIG(GLX_DEPTH_SIZE);
    tag += ", depth=" + mkString(value);
    score+=value*5;
    GETCONFIG(GLX_STENCIL_SIZE);
    tag += ", stencil="+mkString(value);
    tag += ", accum=";
    GETCONFIG(GLX_ACCUM_RED_SIZE);
    tag += mkString(value) + ":";
    GETCONFIG(GLX_ACCUM_GREEN_SIZE);
    tag += mkString(value) + ":";
    GETCONFIG(GLX_ACCUM_BLUE_SIZE);
    tag += mkString(value) + ":";
    GETCONFIG(GLX_ACCUM_ALPHA_SIZE);
    tag += mkString(value);
#ifdef __sgi
    tag += ", samples=";
    GETCONFIG(GLX_SAMPLES_SGIS);
    if(value)
      score+=50;
#endif
    tag += mkString(value);
    
    tag += ", score=" + mkString(score);
    //cerr << score << ": " << tag << '\n';
    
    visualtags.push_back(tag);
    visuals.push_back(&vinfo[i]);
    scores.push_back(score);
  }
  // Bubble sort, based on score (quality of visual)
#if 0
  for(i=0;(unsigned int)i<scores.size()-1;i++){
    for(unsigned int j=i+1;j<scores.size();j++){
      if(scores[i] < scores[j]){
	// Swap...
	int tmp1=scores[i];
	scores[i]=scores[j];
	scores[j]=tmp1;
	string tmp2=visualtags[i];
	visualtags[i]=visualtags[j];
	visualtags[j]=tmp2;
	XVisualInfo* tmp3=visuals[i];
	visuals[i]=visuals[j];
	visuals[j]=tmp3;
      }
    }
  }
#endif
  args.result(GuiArgs::make_list(visualtags));

  Log::log( LEAVE, "[OpenGL::listvisuals] leaving" );
}

void
OpenGL::clearscreen() {
  //cerr << "In OpenGL::clearscreen, thread id is " << pthread_self() << endl; 
  mkContextCurrent();
  
  glViewport(0,0,winx,winy);
  glClearColor( 0, 0, 0, 1 );
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

}

int
OpenGL::mkContextCurrent() {
  //cerr << "In OpenGL::mkContextCurrent, thread id is " << pthread_self() << endl;    

  Tk_Window new_tkwin =
    Tk_NameToWindow(UserInterface::getInterp(),
		    ccast_unsafe(myname),
		    Tk_MainWindow(UserInterface::getInterp()));
  if(!new_tkwin){
    Log::log( ERROR, "Unable to locate window!" );
    return -1;
  }
  if(tkwin != new_tkwin){

    Log::log( DEBUG, "[OpenGL::mkContextCurrent] assigning new drawable" );
    tkwin=new_tkwin;
    dpy=Tk_Display(tkwin);
    win=Tk_WindowId(tkwin);
    cx=OpenGLGetContext(UserInterface::getInterp(), ccast_unsafe(myname));
    if(!cx){
      Log::log( ERROR, "Unable to create OpenGL Context!" );
      return -1;
    }
  }
  //if ( !glXMakeCurrent(dpy, win, cx) ) {
  if ( !glXMakeContextCurrent(dpy, win, win, cx) ) {
    // this is probably another thread trying to use a context that has 
    // already been made current with another thread -- this doesn't work
    
    Log::log( ERROR, "[OpenGL::mkContextCurrent] Couldn't make context current!" );
    return -1;
  }

  // check to see if context is valid
  Log::log( DEBUG, "[OpenGL::mkContextCurrent] current context is " + mkString(cx) );

  Log::log( DEBUG, "[OpenGL::mkContextCurrent] glXIsDirect = " + mkString(glXIsDirect(dpy, cx)) );
  
  
  glXWaitX();

  return 0;
}

void
OpenGL::setvisual( const string& wname, int which, int width, int height ) {
  tkwin = 0;

  string command = "opengl " + wname + " -visual " +
    mkString((int)visuals[which]->visualid) + " -direct true" +
    " -geometry " + mkString(width) + "x" + mkString(height);
  
  execute( command );

  myname = wname;
}

int
OpenGL::OpenGLCmd(ClientData clientData, Tcl_Interp *interp,
		  int argc, char **argv) {
  Log::log( ENTER, "[OpenGL::OpenGLCmd] entered" );
  XVisualInfo* vi=0;
  Tk_Window mainwin = (Tk_Window) clientData;
  _OpenGL *OpenGLPtr;
  Colormap cmap;
  Tk_Window tkwin;
  int attributes[50];
  int idx=0;
  
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
  
  /*
   * Allocate and initialize the widget record.
   */
  
  OpenGLPtr = (_OpenGL *) ckalloc(sizeof(_OpenGL));
  OpenGLPtr->interp = interp;
  OpenGLPtr->tkwin = tkwin;
  OpenGLPtr->display = Tk_Display(tkwin);
  OpenGLPtr->geometry=0;
  OpenGLPtr->cursor=0;
  
  Tk_CreateEventHandler(OpenGLPtr->tkwin, StructureNotifyMask,
			OpenGL::OpenGLEventProc, (ClientData) OpenGLPtr);
  Tcl_CreateCommand(interp, Tk_PathName(OpenGLPtr->tkwin),
		    OpenGL::OpenGLWidgetCmd,
		    (ClientData) OpenGLPtr, (void (*)(void*)) NULL);
  if (OpenGL::OpenGLConfigure(interp, OpenGLPtr, argc-2,
			      argv+2, 0) != TCL_OK) {
    return TCL_ERROR;
  }
  
  if(OpenGLPtr->visualid){
    int n;
    XVisualInfo tmpl;
    tmpl.visualid=OpenGLPtr->visualid;
    vi=XGetVisualInfo(Tk_Display(tkwin), VisualIDMask, &tmpl, &n);
    if(!vi || n!=1){
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
    
    attributes[idx++]=None;

    //int nelements;
    //GLXFBConfig* glx_fbconfig = glXGetFBConfigs ( Tk_Display(tkwin), Tk_ScreenNumber(tkwin), &nelements );
    
    //vi = glXChooseVisual(Tk_Display(tkwin), Tk_ScreenNumber(tkwin),
    //			 attributes);

    // check to see if glx_fbconfig is already defined
    // if not, define it and check again
    if( glx_fbconfig == NULL )
    {
      int nelements;
      glx_fbconfig = glXGetFBConfigs ( dpy, OpenGLPtr->vi->screen, &nelements );

      if( glx_fbconfig == NULL )
      { 
        Log::log( ERROR, "[OpenGL::listvisuals] glx_fbconfig is undefined" );
      }
    }

      
    vi = glXGetVisualFromFBConfig(Tk_Display(tkwin), *glx_fbconfig);
			 
    if(!vi){
      Tcl_AppendResult(interp, "Error selecting visual", (char*)NULL);
      return TCL_ERROR;
    }
  }
  
  OpenGLPtr->cx=0;
  OpenGLPtr->vi=vi;
  
  cmap = XCreateColormap(Tk_Display(tkwin),
			 RootWindow(Tk_Display(tkwin), vi->screen),
			 vi->visual, AllocNone);
  if( Tk_SetWindowVisual(tkwin, vi->visual, vi->depth, cmap) != 1){
    
    Tcl_AppendResult(interp, "Error setting visual for window", (char*)NULL);
    return TCL_ERROR;
  }
  XSync(Tk_Display(tkwin), False);
  
  interp->result = Tk_PathName(OpenGLPtr->tkwin);

  Log::log( LEAVE, "[OpenGL::OpenGLCmd] leaving" );
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

int
OpenGL::OpenGLWidgetCmd(ClientData clientData, Tcl_Interp *interp,
			int argc, char **argv) {
  
  _OpenGL *OpenGLPtr = (_OpenGL *) clientData;
  int result = TCL_OK;
  int length;
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
  else if ((c == 's') && (strncmp(argv[1], "setvisual", length) == 0)) 
  {
    /* put code for initializing or changing the visual here */
    printf("setvisual called on an opengl tk widget");
  } 
  else {
    Tcl_AppendResult(interp, "bad option \"", argv[1],
		     "\":  must be configure, cget, position, or size", 
		     (char *) NULL);
    result = TCL_ERROR;
  }
  
  Tk_Release((ClientData) OpenGLPtr);
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

int
OpenGL::OpenGLConfigure(Tcl_Interp *interp, _OpenGL *OpenGLPtr, int argc,
			char **argv, int flags) {
  if (Tk_ConfigureWidget(interp, OpenGLPtr->tkwin, configSpecs,
			 argc, argv, (char *) OpenGLPtr, flags) != TCL_OK) {
    return TCL_ERROR;
  }
  
  {
    int height, width;
    if (sscanf(OpenGLPtr->geometry, "%dx%d", &width, &height) != 2) {
      Tcl_AppendResult(interp, "bad geometry \"", OpenGLPtr->geometry,
		       "\": expected widthxheight", (char *) NULL);
      return TCL_ERROR;
    }
    Tk_GeometryRequest(OpenGLPtr->tkwin, width, height);
  }
  
  Tk_DefineCursor( OpenGLPtr->tkwin, OpenGLPtr->cursor );
  
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

void
OpenGL::OpenGLEventProc(ClientData clientData, XEvent *eventPtr) {
  _OpenGL *OpenGLPtr = (_OpenGL *) clientData;
  
  if (eventPtr->type == DestroyNotify) {
    Tcl_DeleteCommand(OpenGLPtr->interp, Tk_PathName(OpenGLPtr->tkwin));
    OpenGLPtr->tkwin = NULL;
    Tk_EventuallyFree((ClientData) OpenGLPtr, (Tcl_FreeProc*)OpenGLDestroy);
  }
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

void
OpenGL::OpenGLDestroy(ClientData clientData) {

  _OpenGL *OpenGLPtr = (_OpenGL *) clientData;
  
  glXDestroyContext(OpenGLPtr->display, OpenGLPtr->cx);
  
  Tk_FreeOptions(configSpecs, (char *) OpenGLPtr, OpenGLPtr->display, 0);
  ckfree((char *) OpenGLPtr);
}

// Revision 1.1  Wed Aug  7 17:19:42 MDT 2002 simpson
// 
//
