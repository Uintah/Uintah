
#include <Packages/rtrt/Core/HVolumeBrickColor.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Runnable.h>
#include <Core/Math/MiscMath.h>
#include <Core/Math/MinMax.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <GL/glx.h>
#include <GL/glu.h>
#include <Packages/rtrt/visinfo/visinfo.h>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#define XK_LATIN1
#include <X11/keysymdef.h>
#include "FontString.h"

using namespace rtrt;
using namespace std;
using namespace SCIRun;

namespace rtrt {
  extern Mutex io_lock_;
  extern Mutex xlock;

  class HVolumeBrickColorDpy : public Runnable {
    HVolumeBrickColor* vol;
    void drawit(GLuint fid, XFontStruct* font_struct);
    void move(int x, int y, int ox);
    double min, max;
    int xres, yres;
    void putit(XFontStruct* font_struct, GLuint fid, double y,
	       double val, const Color& color);
    int x,y;
  public:
    HVolumeBrickColorDpy(HVolumeBrickColor* vol);
    virtual ~HVolumeBrickColorDpy();
    virtual void run();
  };
} // end namespace rtrt

HVolumeBrickColor::HVolumeBrickColor(char* filebase, int np,
				     double Ka, double Kd, double Ks,
				     double specpow, double refl,
				     double dt)
    : filebase(filebase), Ka(Ka/255.), Kd(Kd/255.), Ks(Ks/255.),
      specpow(specpow), refl(refl), dt(dt), work(0)
{
    nn=false;
    grid=false;
    char buf[200];
    sprintf(buf, "%s.hdr", filebase);
    ifstream in(buf);
    if(!in){
	cerr << "Error opening header: " << buf << '\n';
	exit(1);
    }
    in >> nx >> ny >> nz;
    double x,y,z;
    in >> x >> y >> z;
    min=Point(x,y,z);
    in >> x >> y >> z;
    Point max(x,y,z);
    if(!in){
	cerr << "Error reading header: " << buf << '\n';
	exit(1);
    }
    datadiag=max-min;
    sdiag=datadiag/Vector(nx-1,ny-1,nz-1);

#define L1 4
#define L2 5
    unsigned long totalx=(nx+L2*L1-1)/(L2*L1);
    unsigned long totaly=(ny+L2*L1-1)/(L2*L1);
    unsigned long totalz=(nz+L2*L1-1)/(L2*L1);

    xidx=new unsigned long[nx];

    for(unsigned long x=0;x<nx;x++){
	unsigned long m1x=x%L1;
	unsigned long xx=x/L1;
	unsigned long m2x=xx%L2;
	unsigned long m3x=xx/L2;
	xidx[x]=m3x*totaly*totalz*L2*L2*L2*L1*L1*L1+m2x*L2*L2*L1*L1*L1+m1x*L1*L1;
	xidx[x]*=3;
    }
    yidx=new unsigned long[ny];
    for(unsigned long y=0;y<ny;y++){
	unsigned long m1y=y%L1;
	unsigned long yy=y/L1;
	unsigned long m2y=yy%L2;
	unsigned long m3y=yy/L2;
	yidx[y]=m3y*totalz*L2*L2*L2*L1*L1*L1+m2y*L2*L1*L1*L1+m1y*L1;
	yidx[y]*=3;
    }
    zidx=new unsigned long[nz];
    for(unsigned long z=0;z<nz;z++){
	unsigned long m1z=z%L1;
	unsigned long zz=z/L1;
	unsigned long m2z=zz%L2;
	unsigned long m3z=zz/L2;
	zidx[z]=m3z*L2*L2*L2*L1*L1*L1+m2z*L1*L1*L1+m1z;
	zidx[z]*=3;
    }

    unsigned long totalsize=totalx*totaly*totalz*L2*L2*L2*L1*L1*L1*3;
    cerr << "totalsize=" << totalsize << '\n';
    blockdata=new unsigned char[totalsize];
    if(!blockdata){
	cerr << "Error allocating data array\n";
	exit(1);
    }
    sprintf(buf, "%s.brick", filebase);
    //ifstream bin(buf);
    int bin_fd  = open (buf, O_RDONLY);
    if(bin_fd == -1){
      //ifstream din(filebase);
      int din_fd = open (filebase, O_RDONLY);
	if(din_fd == -1){
	    cerr << "Error opening data file: " << filebase << '\n';
	    exit(1);
	}
	indata=new unsigned char[(unsigned long)nx*ny*nz*3];
	if(!indata){
	    cerr << "Error allocating data array\n";
	    exit(1);
	}
	double start=SCIRun::Time::currentSeconds();
	cerr << "Reading " << filebase << "...";
	cerr.flush();
	//read(din.rdbuf()->fd(), indata, sizeof(unsigned char)*nx*ny*nz*3);
	read(din_fd, indata, sizeof(unsigned char)*nx*ny*nz*3);
	double dt=SCIRun::Time::currentSeconds()-start;
	cerr << "done in " << dt << " seconds (" << nx*ny*nz*3/dt/1024/1024 << " MB/sec)\n";
	int s = close(din_fd);
	if(s == -1) {
	    cerr << "Error reading data file: " << filebase << '\n';
	    exit(1);
	}
	cerr << "Done reading data\n";

	int bnp=np>2?2:np;
	cerr << "Bricking data with " << bnp << " processors\n";
	// <<<<< bigler >>>>>
	//work=WorkQueue("Bricking", nx, bnp, false, 5);
	work.refill(nx, bnp, 5);
	Parallel<HVolumeBrickColor> phelper(this, &HVolumeBrickColor::brickit);
	Thread::parallel(phelper, bnp, true);

	//ofstream bout(buf);
	int bout_fd = open (buf, O_WRONLY | O_CREAT | O_TRUNC,
		    S_IRUSR|S_IWUSR|S_IRGRP|S_IWGRP|S_IROTH|S_IWOTH);
	if (bout_fd == -1) {
	  cerr << "Error in opening " << buf << " for writing.\n";
	  exit(1);
	}
	cerr << "Writing " << buf << "...";
	start=SCIRun::Time::currentSeconds();
	//write(bout.rdbuf()->fd(),blockdata, sizeof(unsigned char)*totalsize);
	write(bout_fd, blockdata, sizeof(unsigned char)*totalsize);
	dt=SCIRun::Time::currentSeconds()-start;
	cerr << "done in " << dt << " seconds (" << (double)totalsize/dt/1024/1024 << " MB/sec)\n";
	delete[] indata;
    } else {
	cerr << "Reading " << buf << "(" << totalsize << " bytes)...";
	cerr.flush();
	double start=SCIRun::Time::currentSeconds();
	//read(bin.rdbuf()->fd(), blockdata, sizeof(unsigned char)*totalsize);
	read(bin_fd, blockdata, sizeof(unsigned char)*totalsize);
	double dt=SCIRun::Time::currentSeconds()-start;
	cerr << "done in " << dt << " seconds (" << (double)totalsize/dt/1024/1024 << " MB/sec)\n";
	int s = close(bin_fd);
	if(s == -1) {
	    cerr << "Error reading data file: " << filebase << '\n';
	    exit(1);
	}
    } 		   
    HVolumeBrickColorDpy* dpy=new HVolumeBrickColorDpy(this);
    new Thread(dpy, "HVolumeBrickColor display thread\n");
}

void HVolumeBrickColor::shade(Color& result, const Ray& ray,
			      const HitInfo& hit, int depth, 
			      double atten, const Color& accumcolor,
			      Context* cx)
{
    Point p(ray.origin()+ray.direction()*(hit.min_t+dt));
    if(nn){
	Vector sp((p-min)/datadiag*Vector(nx-1,ny-1,nz-1));
	int x=(int)(sp.x()+.5);
	int y=(int)(sp.y()+.5);
	int z=(int)(sp.z()+.5);
	if(x<0 || x>=nx || y<0 || y>=ny || z<0 || z>=nz){
	    result=Color(1,0,0);
	} else {
	    unsigned long idx=xidx[x]+yidx[y]+zidx[z];
	    Color c(blockdata[idx], blockdata[idx+1], blockdata[idx+2]);
	    if(grid && ((int)p.x()%10==0 || (int)p.y()%10==0 || (int)p.z()%10==0))
		c=Color(1,0,1);
	    Color ambient(c*Ka);
	    Color diffuse(c*Kd);
	    Color specular(c*Ks);
	    phongshade(result, diffuse, specular, specpow, refl,
		       ray, hit, depth,  atten,
		       accumcolor, cx);
	}
    } else {
	Vector sp((p-min)/datadiag*Vector(nx-1,ny-1,nz-1));
	int x=(int)sp.x();
	int y=(int)sp.y();
	int z=(int)sp.z();
	if(x<0 || x>=nx-1 || y<0 || y>=ny-1 || z<0 || z>=nz-1){
	    result=Color(1,0,0);
	} else {
	    float fx=sp.x()-x;
	    float fy=sp.y()-y;
	    float fz=sp.z()-z;
	    float fx1=1-fx;
	    float fy1=1-fy;
	    float fz1=1-fz;
	    unsigned long idx000=xidx[x]+yidx[y]+zidx[z];
	    Color c000(blockdata[idx000], blockdata[idx000+1], blockdata[idx000+2]);
	    unsigned long idx001=xidx[x]+yidx[y]+zidx[z+1];
	    Color c001(blockdata[idx001], blockdata[idx001+1], blockdata[idx001+2]);
	    unsigned long idx010=xidx[x]+yidx[y+1]+zidx[z];
	    Color c010(blockdata[idx010], blockdata[idx010+1], blockdata[idx010+2]);
	    unsigned long idx011=xidx[x]+yidx[y+1]+zidx[z+1];
	    Color c011(blockdata[idx011], blockdata[idx011+1], blockdata[idx011+2]);
	    unsigned long idx100=xidx[x+1]+yidx[y]+zidx[z];
	    Color c100(blockdata[idx100], blockdata[idx100+1], blockdata[idx100+2]);
	    unsigned long idx101=xidx[x+1]+yidx[y]+zidx[z+1];
	    Color c101(blockdata[idx101], blockdata[idx101+1], blockdata[idx101+2]);
	    unsigned long idx110=xidx[x+1]+yidx[y+1]+zidx[z];
	    Color c110(blockdata[idx110], blockdata[idx110+1], blockdata[idx110+2]);
	    unsigned long idx111=xidx[x+1]+yidx[y+1]+zidx[z+1];
	    Color c111(blockdata[idx111], blockdata[idx111+1], blockdata[idx111+2]);
	    Color c00(c000*fz1+c001*fz);
	    Color c01(c010*fz1+c011*fz);
	    Color c10(c100*fz1+c101*fz);
	    Color c11(c110*fz1+c111*fz);

	    Color c0(c00*fy1+c01*fy);
	    Color c1(c10*fy1+c11*fy);

	    Color c(c0*fx1+c1*fx);
	    if(grid && ((int)p.x()%10==0 || (int)p.y()%10==0 || (int)p.z()%10==0))
		c=Color(1,0,1);
	    Color ambient(c*Ka);
	    Color diffuse(c*Kd);
	    Color specular(c*Ks);
	    phongshade(result, diffuse, specular, specpow, refl,
		       ray, hit, depth, atten,
		       accumcolor, cx);
	}
    }
}

HVolumeBrickColor::~HVolumeBrickColor()
{
    if(blockdata)
	delete[] blockdata;
}

void HVolumeBrickColor::brickit(int proc)
{
    unsigned long nynz=ny*nz;
    int sx, ex;
    while(work.nextAssignment(sx, ex)){
	for(unsigned long x=sx;x<ex;x++){
	    io_lock_.lock();
	    cerr << "processor " << proc << ": " << x << " of " << nx-1 << "\n";
	    io_lock_.unlock();
	    for(int y=0;y<ny;y++){
		unsigned long idx=x*nynz*3+y*nz*3;
		for(int z=0;z<nz;z++){
		    unsigned char r=indata[idx];
		    unsigned char g=indata[idx+1];
		    unsigned char b=indata[idx+2];
		    unsigned long idx0=xidx[x]+yidx[y]+zidx[z];
		    blockdata[idx0]=r;
		    blockdata[idx0+1]=g;
		    blockdata[idx0+2]=b;
		    
		    idx+=3;
		}
	    }
	}
    }
}

HVolumeBrickColorDpy::HVolumeBrickColorDpy(HVolumeBrickColor* vol)
    : vol(vol)
{
    Point dmax(vol->min+vol->datadiag);
    min=Min(Min(vol->min.x(), vol->min.y(), vol->min.z()),
	    Min(dmax.x(), dmax.y(), dmax.z()));
    max=Max(Max(vol->min.x(), vol->min.y(), vol->min.z()),
	    Max(dmax.x(), dmax.y(), dmax.z()));
    double range=max-min;
    min-=range*0.2;
    max+=range*0.2;
    xres=500;
    yres=100;
}

HVolumeBrickColorDpy::~HVolumeBrickColorDpy()
{
}

void HVolumeBrickColorDpy::run()
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
    char* p="HVolumeBrickColor aligner";
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
    XFontStruct* fontInfo = XLoadQueryFont(dpy, __FONTSTRING__);

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
    

    bool redraw=true;
    xlock.unlock();
    for(;;){
	if(redraw){
	    drawit(fontbase, fontInfo);
	    redraw=false;
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
		redraw=true;
	    } else {
		redraw=true;
	    }
	    break;
	case KeyPress:
	    switch(XKeycodeToKeysym(dpy, e.xkey.keycode, 0)){
	    case XK_g:
		vol->grid=!vol->grid;
		break;
	    case XK_n:
		vol->nn=!vol->nn;
		break;
	    }
	    break;
	case ButtonPress:
	case ButtonRelease:
	    switch(e.xbutton.button){
	    case Button1:
		move(e.xbutton.x, e.xbutton.y, e.xbutton.x);
		y=e.xbutton.y;
		x=e.xbutton.x;
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
		move(e.xbutton.x, y, x);
		x=e.xbutton.x;
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

void HVolumeBrickColorDpy::putit(XFontStruct* font_struct, GLuint fid,
				 double y, double val, const Color& color)
{
    char buf[100];
    sprintf(buf, "%g", val);
    int w=calc_width(font_struct, buf);
    double x=(val-min)/(max-min)*xres-w/2.;
    printString(fid, x, y, buf, color);
}

void HVolumeBrickColorDpy::drawit(GLuint fid, XFontStruct* font_struct)
{
    glViewport(0, 0, xres, yres);
    glClearColor(0, 0, 0, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(min, max, 0, 6);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glColor3f(0,0,1);
    Point dmax(vol->min+vol->datadiag);
    glBegin(GL_LINES);
    glVertex2f(vol->min.x(), 1);
    glVertex2f(vol->min.x(), 2);
    glVertex2f(dmax.x(), 1);
    glVertex2f(dmax.x(), 2);

    glVertex2f(vol->min.y(), 3);
    glVertex2f(vol->min.y(), 4);
    glVertex2f(dmax.y(), 3);
    glVertex2f(dmax.y(), 4);

    glVertex2f(vol->min.z(), 5);
    glVertex2f(vol->min.z(), 6);
    glVertex2f(dmax.z(), 5);
    glVertex2f(dmax.z(), 6);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, xres, 0, 6);

    putit(font_struct, fid, 0.1, vol->min.x(), Color(.5,0,0));
    putit(font_struct, fid, 0.1, dmax.x(), Color(1,0,0));
    putit(font_struct, fid, 2.1, vol->min.y(), Color(.5,0,0));
    putit(font_struct, fid, 2.1, dmax.y(), Color(1,0,0));
    putit(font_struct, fid, 4.1, vol->min.z(), Color(.5,0,0));
    putit(font_struct, fid, 4.1, dmax.z(), Color(1,0,0));

    glFinish();
    int errcode;
    while((errcode=glGetError()) != GL_NO_ERROR){
	cerr << "We got an error from GL: " << (char*)gluErrorString(errcode) << endl;
    }
}

void HVolumeBrickColorDpy::move(int x, int y, int ox)
{
    float xn=float(x)/xres;
    float val=min+xn*(max-min);
    Point dmax(vol->min+vol->datadiag);
    if(y<yres/3){
	// Z
	double deltamin=Abs(val-vol->min.z())/Abs(max-min);
	double deltamax=Abs(val-dmax.z())/Abs(max-min);
	if(deltamin < .1 && deltamin<deltamax){
	    vol->min=Point(vol->min.x(), vol->min.y(), val);
	} else if(deltamax < .1){
	    dmax=Point(dmax.x(), dmax.y(), val);
	} else {
	    float xn=float(x-ox)/xres;
	    float dval=xn*(max-min);
	    vol->min=Point(vol->min.x(), vol->min.y(), vol->min.z()+dval);
	    dmax=Point(dmax.x(), dmax.y(), dmax.z()+dval);
	}
    } else if(y<(2*yres)/3){
	// Y
	double deltamin=Abs(val-vol->min.y())/Abs(max-min);
	double deltamax=Abs(val-dmax.y())/Abs(max-min);
	if(deltamin < .1 && deltamin<deltamax){
	    vol->min=Point(vol->min.x(), val, vol->min.z());
	} else if(deltamax < .1){
	    dmax=Point(dmax.x(), val, dmax.z());
	} else {
	    float xn=float(x-ox)/xres;
	    float dval=xn*(max-min);
	    vol->min=Point(vol->min.x(), vol->min.y()+dval, vol->min.z());
	    dmax=Point(dmax.x(), dmax.y()+dval, dmax.z());
	}
    } else {
	// X
	double deltamin=Abs(val-vol->min.x())/Abs(max-min);
	double deltamax=Abs(val-dmax.x())/Abs(max-min);
	if(deltamin< .1 && deltamin<deltamax){
	    vol->min=Point(val, vol->min.y(), vol->min.z());
	} else if(deltamax < .1){
	    dmax=Point(val, dmax.y(), dmax.z());
	} else {
	    float xn=float(x-ox)/xres;
	    float dval=xn*(max-min);
	    vol->min=Point(vol->min.x()+dval, vol->min.y(), vol->min.z());
	    dmax=Point(dmax.x()+dval, dmax.y(), dmax.z());
	}
    }
    vol->datadiag=dmax-vol->min;
}

