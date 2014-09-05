
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/VolumeDpy.h>
#include <Packages/rtrt/Core/VolumeBase.h>
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
#include <unistd.h>

using namespace rtrt;
using SCIRun::Mutex;
using SCIRun::Thread;

namespace rtrt {
  extern Mutex xlock;
} // end namespace rtrt

VolumeDpy::VolumeDpy(float isoval)
    : isoval(isoval)
{
    xres=400;
    yres=100;
    hist=0;
}

VolumeDpy::~VolumeDpy()
{
}

void VolumeDpy::attach(VolumeBase* vol)
{
    vols.add(vol);
}

void VolumeDpy::run()
{
  sleep(3);
    // Compute the global minmax
    if(vols.size()==0)
	exit(0);
    datamax=-MAXFLOAT;
    datamin=MAXFLOAT;
    for(int i=0;i<vols.size();i++){
	float min, max;
	vols[i]->get_minmax(min, max);
	datamin=Min(min, datamin);
	datamax=Max(max, datamax);
    }
    if(isoval == -123456){
	isoval=(datamin+datamax)*0.5;
    }
    new_isoval=isoval;
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
    char* p="Volume histogram";
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
    

    bool need_hist=true;
    bool redraw=true;
    bool redraw_isoval=false;
    xlock.unlock();
    for(;;){
	if(need_hist){
	    need_hist=false;
	    compute_hist(fontbase);
	    redraw=true;
	}
	if(redraw || redraw_isoval){
	    if(redraw)
		redraw_isoval=false;
	    draw_hist(fontbase, fontInfo, redraw_isoval);
	    redraw=false;
	    redraw_isoval=false;
	}
	XEvent e;
	XNextEvent(dpy, &e);	
	switch(e.type){
	case Expose:
	    // Ignore expose events, since we will be refreshing
	    // constantly anyway
	    redraw=true;
	    break;
	case ConfigureNotify:
	    yres=e.xconfigure.height;
	    if(e.xconfigure.width != xres){
		xres=e.xconfigure.width;
		need_hist=true;
	    } else {
		redraw=true;
	    }
	    break;
	case ButtonPress:
	case ButtonRelease:
	    switch(e.xbutton.button){
	    case Button1:
		move(e.xbutton.x, e.xbutton.y);
		redraw_isoval=true;
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
		move(e.xbutton.x, e.xbutton.y);
		redraw_isoval=true;
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

void VolumeDpy::compute_hist(unsigned int fid)
{
    glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
    glViewport(0, 0, xres, yres);
    glClearColor(0, 0, .2, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, 1, 0, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
	    
    printString(fid, .1, .5, "Recomputing histogram...\n", Color(1,1,1));
    glFlush();
    if(hist){
	delete[] hist;
    }
    int nhist=xres;
    hist=new int[nhist];
    for(int i=0;i<nhist;i++){
	hist[i]=0;
    }
    int* tmphist=new int[nhist];

    for(int i=0;i<vols.size();i++){
	for(int j=0;j<nhist;j++){
	    tmphist[j]=0;
	}
	vols[i]->compute_hist(nhist, tmphist, datamin, datamax);
	for(int j=0;j<nhist;j++){
	    hist[j]+=tmphist[j];
	}
    }
    delete[] tmphist;
    int* hp=hist;
    int max=0;
    for(int i=0;i<nhist;i++){
	if(*hp>max)
	    max=*hp;
	hp++;
    }
    histmax=max;
    cerr << "Done building histogram\n";
}
    
static int calc_width(XFontStruct* font_struct, char* str)
{
    XCharStruct overall;
    int ascent, descent;
    int dir;
    XTextExtents(font_struct, str, (int)strlen(str), &dir, &ascent, &descent, &overall);
    return overall.width;
}

void VolumeDpy::draw_hist(unsigned int fid, XFontStruct* font_struct,
			  bool redraw_isoval)
{
    int descent=font_struct->descent;
    int textheight=font_struct->descent+font_struct->ascent;
    if(!redraw_isoval){
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glViewport(0, 0, xres, yres);
	glClearColor(0, 0, 0, 1);
	glClear(GL_COLOR_BUFFER_BIT);
	int nhist=xres;
	int s=2;
	int e=yres-2;
	int h=e-s;
	glViewport(0, s, xres, e-s);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, xres, -float(textheight)*histmax/(h-textheight), histmax);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glColor3f(0,0,1);
	glBegin(GL_LINES);
	for(int i=0;i<nhist;i++){
	    glVertex2i(i, 0);
	    glVertex2i(i, hist[i]);
	}
	glEnd();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, xres, 0, h);
	char buf[100];
	sprintf(buf, "%g", datamin);
	printString(fid, 2, descent+1, buf, Color(0,1,1));
	sprintf(buf, "%g", datamax);
	int w=calc_width(font_struct, buf);
	printString(fid, xres-2-w, descent+1, buf, Color(0,1,1));
    }

    glColorMask(GL_TRUE, GL_TRUE, GL_FALSE, GL_FALSE);
    int s=2;
    int e=yres-2;
    int h=e-s;
    glViewport(0, s, xres, e-s);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(datamin, datamax, 0, h);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glColor3f(0,0,0);
    glRectf(datamin, 0, datamax, h);
    glColorMask(GL_TRUE, GL_FALSE, GL_FALSE, GL_FALSE);
    glColor3f(.8,0,0);
    glBegin(GL_LINES);
    glVertex2f(new_isoval, 0);
    glVertex2f(new_isoval, h);
    glEnd();

    char buf[100];
    sprintf(buf, "%g", new_isoval);
	
    int w=calc_width(font_struct, buf);
    float wid=(datamax-datamin)*w/xres;
    float x=new_isoval-wid/2.;
    float left=datamin+(datamax-datamin)*2/xres;
    if(x<left)
	x=left;
    printString(fid, x, descent+1, buf, Color(1,0,0));

    glFinish();
    int errcode;
    while((errcode=glGetError()) != GL_NO_ERROR){
	cerr << "We got an error from GL: " << (char*)gluErrorString(errcode) << endl;
    }
}

void VolumeDpy::move(int x, int)
{
    float xn=float(x)/xres;
    float val=datamin+xn*(datamax-datamin);
    if(val<datamin)
	val=datamin;
    if(val>datamax)
	val=datamax;
    new_isoval=val;
}

void VolumeDpy::animate(bool& changed)
{
    if(isoval != new_isoval){
	isoval=new_isoval;
	changed=true;
    }
}
