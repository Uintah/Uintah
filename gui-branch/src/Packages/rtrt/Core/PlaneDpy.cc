
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/PlaneDpy.h>
#include <Packages/rtrt/Core/MinMax.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <GL/glx.h>
#include <GL/glu.h>
#include <Packages/rtrt/visinfo/visinfo.h>
#include <iostream>
#include <stdlib.h>
#include <values.h>
#include <stdio.h>

using namespace rtrt;
using namespace SCIRun;

namespace rtrt {
  extern Mutex xlock;
} // end namespace rtrt

PlaneDpy::PlaneDpy(const Vector& n, const Point& cen)
    : n(n)
{
    d=n.dot(cen);
    xres=300;
    yres=300;
}

PlaneDpy::~PlaneDpy()
{
}

static void printString(GLuint fontbase, double x, double y,
			char *s, const Color& c)
{
    glColor3f(c.red(), c.green(), c.blue());

    glRasterPos2d(x,y);
    /*glBitmap(0, 0, x, y, 1, 1, 0);*/
    glPushAttrib (GL_LIST_BIT);
    glListBase(fontbase);
    glCallLists((int)strlen(s), GL_UNSIGNED_BYTE, (GLubyte *)s);
    glPopAttrib ();
}

static int calc_width(XFontStruct* font_struct, char* str)
{
    XCharStruct overall;
    int ascent, descent;
    int dir;
    XTextExtents(font_struct, str, (int)strlen(str), &dir, &ascent, &descent, &overall);
    return overall.width;
}

void PlaneDpy::run()
{
    xlock.lock();
    // Open an OpenGL window
    Display* dpy=XOpenDisplay(NULL);
    if(!dpy){
	cerr << "Cannot open display\n";
	Thread::exitAll(1);
    }
    int error, event;
    if ( !glXQueryExtension( dpy, &error, &event) ) {
	cerr << "GL extension NOT available!\n";
	XCloseDisplay(dpy);
	dpy=0;
	Thread::exitAll(1);
    }
    int screen=DefaultScreen(dpy);

    char* criteria="sb, max rgb";
    if(!visPixelFormat(criteria)){
	cerr << "Error setting pixel format for visinfo\n";
	cerr << "Syntax error in criteria: " << criteria << '\n';
	Thread::exitAll(1);
    }
    int nvinfo;
    XVisualInfo* vi=visGetGLXVisualInfo(dpy, screen, &nvinfo);
    if(!vi || nvinfo == 0){
	cerr << "Error matching OpenGL Visual: " << criteria << '\n';
	Thread::exitAll(1);
    }
    Colormap cmap = XCreateColormap(dpy, RootWindow(dpy, screen),
				    vi->visual, AllocNone);
    XSetWindowAttributes atts;
    int flags=CWColormap|CWEventMask|CWBackPixmap|CWBorderPixel;
    atts.background_pixmap = None;
    atts.border_pixmap = None;
    atts.border_pixel = 0;
    atts.colormap=cmap;
    atts.event_mask=StructureNotifyMask|ExposureMask|ButtonPressMask|ButtonReleaseMask|ButtonMotionMask|KeyPressMask;
    Window win=XCreateWindow(dpy, RootWindow(dpy, screen),
			     0, 0, xres, yres, 0, vi->depth,
			     InputOutput, vi->visual, flags, &atts);
    char* p="Plane GUI";
    XTextProperty tp;
    XStringListToTextProperty(&p, 1, &tp);
    XSizeHints sh;
    sh.flags = USSize;
    XSetWMProperties(dpy, win, &tp, &tp, 0, 0, &sh, 0, 0);

    XMapWindow(dpy, win);

    GLXContext cx=glXCreateContext(dpy, vi, NULL, True);
    if(!glXMakeCurrent(dpy, win, cx)){
	cerr << "glXMakeCurrent failed!\n";
    }
    glShadeModel(GL_FLAT);
    for(;;){
	XEvent e;
	XNextEvent(dpy, &e);
	if(e.type == MapNotify)
	    break;
    }
    XFontStruct* fontInfo = XLoadQueryFont(dpy, 
        "-adobe-helvetica-bold-r-normal--17-120-100-100-p-88-iso8859-1");
    if (fontInfo == NULL) {
        cerr << "no font found\n";
	Thread::exitAll(1);
    }
    Font id = fontInfo->fid;
    unsigned int first = fontInfo->min_char_or_byte2;
    unsigned int last = fontInfo->max_char_or_byte2;
    GLuint fontbase = glGenLists((GLuint) last+1);
    if (fontbase == 0) {
        printf ("out of display lists\n");
        exit (0);
    }
    glXUseXFont(id, first, last-first+1, fontbase+first);
    int textheight=fontInfo->descent+fontInfo->ascent;
    
    bool redraw=true;
    int starty=0;
    xlock.unlock();
    for(;;){
	if(redraw){
	    glViewport(0, 0, xres, yres);
	    glClearColor(0, 0, 0, 1);
	    glClear(GL_COLOR_BUFFER_BIT);
	    glMatrixMode(GL_PROJECTION);
	    glLoadIdentity();
	    gluOrtho2D(-1, 1, -1, 1);
	    glMatrixMode(GL_MODELVIEW);
	    glLoadIdentity();
	    glColor3f(0,0,1);
	    glBegin(GL_LINES);
	    glVertex2f(0, -1);
	    glVertex2f(0, 1);
	    glEnd();
	    for(int i=0;i<4;i++){
		int s=i*yres/4;
		int e=(i+1)*yres/4;
		glViewport(0, s, xres, e-s);
		double th=double(textheight+1)/(e-s);
		double v;
		double wid=2;
		char* name;
		switch(i){
		case 3: v=n.x(); name="X"; break;
		case 2: v=n.y(); name="Y"; break;
		case 1: v=n.z(); name="Z"; break;
		case 0: v=d; name="D"; wid=20; break;
		}
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		if(i==0)
		    gluOrtho2D(-10, 10, 0, 1);
		else
		    gluOrtho2D(-1, 1, 0, 1);
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		glColor3f(1,1,1);
		glBegin(GL_LINES);
		glVertex2f(v, th);
		glVertex2f(v, 1);
		glEnd();
		char buf[100];
		sprintf(buf, "%s: %g", name, v);
		int w=calc_width(fontInfo, buf);
		printString(fontbase, v-w/wid/yres, 1./yres, buf, Color(1,1,1));
	    }
	    redraw=false;
	}
	XEvent e;
	XNextEvent(dpy, &e);	
	switch(e.type){
	case Expose:
	    redraw=true;
	    break;
	case ConfigureNotify:
	    yres=e.xconfigure.height;
	    if(e.xconfigure.width != xres){
		xres=e.xconfigure.width;
		redraw=true;
	    }
	    break;
	case ButtonPress:
	case ButtonRelease:
	    switch(e.xbutton.button){
	    case Button1:
		starty=e.xbutton.y;
		move(e.xbutton.x, e.xbutton.y);
		redraw=true;
		break;
	    case Button2:
		break;
	    case Button3:
		break;
	    }
	    break;
	case MotionNotify:
	    switch(e.xmotion.state&(Button1Mask|Button2Mask|Button3Mask)){
	    case Button1Mask:
		move(e.xbutton.x, starty);
		redraw=true;
		break;
	    case Button2Mask:
		break;
	    case Button3Mask:
		break;
	    }
	    break;
	default:
	    cerr << "Unknown event, type=" << e.type << '\n';
	}
    }
}

void PlaneDpy::move(int x, int y)
{
    float xn=float(x)/xres;
    float yn=float(y)/yres;
    if(yn>.75){
	d=xn*20-10;
    } else if(yn>.5){
	n.z(xn*2-1);
    } else if(yn>.25){
	n.y(xn*2-1);
    } else {
	// X...
	n.x(xn*2-1);
    }
}
