#include <UI/UserInterface.h>
#include <UI/SVCallback.h>
#include <UI/OpenGL.h>
#include <UI/uiHelper.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>

#include <tk.h>

namespace SemotusVisum {

Tcl_Interp *
UserInterface::the_interp = NULL;

uiHelper *
UserInterface::the_helper = NULL;

Renderer *
UserInterface::current_renderer = NULL;

Mutex
UserInterface::TclLock("Tcl Lock");


void
UserInterface::initialize( int argc, char ** argv ) {
  cerr << "In UserInterface::initialize, thread id is " << pthread_self() << endl; 
  // a Tcl interpreter is needed for error reporting.
  the_interp = Tcl_CreateInterp();

  // process command line arguments, exit if errors are encountered
  if (Tk_ParseArgv( the_interp, (Tk_Window)NULL, &argc, argv,
		    NULL, 0 ) != TCL_OK ) {
    cerr << the_interp->result << endl;
    exit(1);
  }

  // uiHelper: User interface helper - interfaces with rest of client.
  // initializes drawid, annotationMode, gvID (GROUP_VIEWER), gclID (GET_CLIENT_LIST), 
  // collID (COLLABORATE), chatID, xID (XDISPLAY) via calls to NetDispatchManager
  the_helper = new uiHelper;

  // Waits for Td_AppInit to return, then goes into an event loop until all the windows
  // in the application have been destroyed.  The event loop is built into Tcl,
  // but some customization has been done (i.e. tclUnixNotify).  
  Tk_Main( argc, argv, Tk_AppInit );

  // not sure why this code is after Tk_Main, in Tcl/Tk book is it before
  Tcl_DeleteInterp( the_interp ); // Should never get here...
 
}

/*
void UserInterface::renderer( Renderer * renderer, string type ) {
  if( type == "Image Renderer" ) {
    Log::log( DEBUG, "[UserInterface::renderer] Image rendering was selected" );
    if( current_renderer == NULL ){
      current_renderer = new ImageRenderer;
    }
    
    // this will only work if copy constructors are implemented properly
    *((ImageRenderer *) current_renderer) = *((ImageRenderer *)renderer);
  }
  else if( type == "Geometry Renderer" ) {
    Log::log( DEBUG, "[UserInterface::renderer] Geometry rendering was selected" );
     if( current_renderer == NULL ){
      current_renderer = new GeometryRenderer;
    }

    // this will only work if copy constructors are implemented properly
    *((GeometryRenderer *) current_renderer) = *((GeometryRenderer *)renderer);
  }
  else if( type == "ZTex Renderer" ) {
     Log::log( DEBUG, "[UserInterface::renderer] ZTex rendering was selected" );
  }
  else {
    Log::log( ERROR, "[UserInterface::renderer] no rendering method was selected" );
    
  }
 
}
*/

static void do_lock() {
  UserInterface::lock();
}

static void do_unlock() {
  UserInterface::unlock();
}


// This is a procedure for completing initialization.  It is used by Tk_Main.
int
UserInterface::Tk_AppInit( Tcl_Interp * ) {
  Log::log( ENTER, "[UserInterface::Tk_AppInit] entered, thread id = " + mkString( (int) pthread_self() ) );

  SVCallback *callback = new SVCallback;

  Tcl_SetLock( do_lock, do_unlock );
  if (Tcl_Init(the_interp) == TCL_ERROR )
    return TCL_ERROR;
  if ( Tk_Init(the_interp) == TCL_ERROR )
    return TCL_ERROR;

  printf("Adding SCI extensions to tcl: ");
  fflush(stdout);
  printf("OpenGL widget\n ");
  Tcl_CreateCommand(the_interp,
		    "opengl",
		    &OpenGL::OpenGLCmd,
		    (ClientData) Tk_MainWindow(the_interp),
		    NULL);
  
  // Add callbacks
  add_command( "ui", callback, 0 );
  
  source_once("UI.tcl");

  // makeUI is defined in UI.tcl  
  // It initializes the window by setting up all formatting for how the 
  // window appears as well as linking the features in the window to 
  // the code (functions in the UI files) that executes the features
  // Binds mouse/keyboard/screen events with the appropriate actions
  // (i.e. redraw when anything changes)
  execute("makeUI");

  Log::log( LEAVE, "[UserInterface::Tk_AppInit] leaving" );
  
  return TCL_OK;
}

}
