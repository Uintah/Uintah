#ifndef _opengl_h_
#define _opengl_h_


#include <tcl.h>
#include <tk.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include <string>
#include <vector>

#include <UI/MiscUI.h>
#include <Logging/Log.h>

using namespace std;

namespace SemotusVisum {
/*
 * A data structure of the following type is kept for each OpenGL
 * widget managed by this file:
 */

typedef struct {
    Tk_Window tkwin;		/* Window that embodies the OpenGL.  NULL
				 * means window has been deleted but
				 * widget record hasn't been cleaned up yet. */
    Display *display;		/* X's token for the window's display. */
    Tcl_Interp *interp;		/* Interpreter associated with widget. */
    char* geometry;
    Tk_Cursor cursor;

    /*
     * glXChooseVisual options
     */
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

    GLXContext cx;
    XVisualInfo* vi;

    /*
     * Information used when displaying widget:
     */

} _OpenGL;

class OpenGL {
public:
  static int OpenGLCmd (ClientData clientData, Tcl_Interp *interp,
			int argc, char **argv);

  
  static GLXContext OpenGLGetContext( Tcl_Interp*, char *);
  static void setvisual( const string& wname, int which,
			 int width, int height );
  static void listvisuals(GuiArgs& args);
  static void clearscreen();
  static inline void setsize( const int x, const int y ) {
    winx = x;
    winy = y;
  }
  static int mkContextCurrent();
  static inline void finish() {
    glXSwapBuffers(dpy, win);

    // make sure all previous GL commands are executed
    // glFlush is slightly more efficient, but it may result in some GL commands not being
    // executed before the next command is executed

    //glFlush();
    glFinish();    

    // Make the current thread let go of the context so that it can be used by the next thread
    glXMakeContextCurrent(dpy, None, None, NULL);   
  }
protected:
  
  OpenGL() {}
  ~OpenGL() {}
  
  static int OpenGLConfigure (Tcl_Interp *interp, _OpenGL *OpenGLPtr,
			      int argc, char **argv,
			      int flags);
  static void		OpenGLDestroy (ClientData clientData);
  static void		OpenGLEventProc (ClientData clientData,
					 XEvent *eventPtr);
  static int		OpenGLWidgetCmd(ClientData clientData,
					Tcl_Interp *, int argc, char **argv);
  

  static Tk_Window tkwin;
  static Window win;
  static Display* dpy;
  static GLXContext cx;
  static string myname;
  static vector<XVisualInfo*> visuals;
  static int winx, winy;
  static GLXFBConfig* glx_fbconfig;
};

}
#endif








