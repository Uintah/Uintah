
/*
 *  OpenGL.cc: Render geometry using opengl
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   December 1994
 *
 *  Copyright (C) 1994 SCI Group
 */


#include <Modules/Salmon/Renderer.h>
#include <Modules/Salmon/Roe.h>
#include <Modules/Salmon/Ball.h>
#include <Modules/Salmon/Salmon.h>
#include <Modules/Salmon/SalmonGeom.h>
#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Classlib/Timer.h>
#include <Geom/Geom.h>
#include <Geom/GeomOpenGL.h>
#include <Geom/RenderMode.h>
#include <Geom/Light.h>
#include <Malloc/Allocator.h>
#include <Math/Trig.h>
#include <TCL/TCLTask.h>
#include <tcl/tcl/tcl.h>
#include <tcl/tk/tk.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include <strstream.h>
#include <fstream.h>

#ifdef __sgi
#include <X11/extensions/SGIStereo.h>
#endif

const int STRINGSIZE=200;
class OpenGLHelper;

class OpenGL : public Renderer {
    Tk_Window tkwin;
    Window win;
    Display* dpy;
    GLXContext cx;
    char* strbuf;
    int maxlights;
    DrawInfoOpenGL* drawinfo;
    WallClockTimer fpstimer;
    double current_time;

    int old_stereo;

    void redraw_obj(Salmon* salmon, Roe* roe, GeomObj* obj);
    void pick_draw_obj(Salmon* salmon, Roe* roe, GeomObj* obj);
    OpenGLHelper* helper;
    clString my_openglname;
public:
    OpenGL();
    virtual ~OpenGL();
    virtual clString create_window(Roe* roe,
				   const clString& name,
				   const clString& width,
				   const clString& height);
    virtual void redraw(Salmon*, Roe*, double tbeg, double tend,
			int ntimesteps, double frametime);
    virtual void get_pick(Salmon*, Roe*, int, int, GeomObj*&, GeomPick*&);
    virtual void hide();
    virtual void dump_image(const clString&);
    virtual void put_scanline(int y, int width, Color* scanline, int repeat=1);

    clString myname;
    void redraw_loop();
    Semaphore send_sema;
    Semaphore recv_sema;

    Salmon* salmon;
    Roe* roe;
    double tbeg;
    double tend;
    int nframes;
    double framerate;
    void redraw_frame();
    // these functions were added to clean things up a bit...

protected:
    
    void initState(void);

};

static OpenGL* current_drawer=0;
static const int pick_buffer_size = 512;
static const double pick_window = 5.0;

extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);
extern Tcl_Interp* the_interp;

static Renderer* make_OpenGL()
{
    return scinew OpenGL;
}

static int query_OpenGL()
{
    TCLTask::lock();
    int have_opengl=glXQueryExtension(Tk_Display(Tk_MainWindow(the_interp)),
				      NULL, NULL);
    TCLTask::unlock();
    return have_opengl;
}

RegisterRenderer OpenGL_renderer("OpenGL", &query_OpenGL, &make_OpenGL);

OpenGL::OpenGL()
: tkwin(0), send_sema(0), recv_sema(0)
{
    strbuf=scinew char[STRINGSIZE];
    drawinfo=scinew DrawInfoOpenGL;
    fpstimer.start();
}

OpenGL::~OpenGL()
{
    fpstimer.stop();
    delete[] strbuf;
}

clString OpenGL::create_window(Roe*,
			       const clString& name,
			       const clString& width,
			       const clString& height)
{
    myname=name;
    width.get_int(xres);
    height.get_int(yres);
    static int direct=1;
    int d=direct;
    direct=0;
    return "opengl "+name+" -geometry "+width+"x"+height+" -doublebuffer true -direct "+(d?"true":"false")+" -rgba true -redsize 2 -greensize 2 -bluesize 2 -depthsize 2";
}

void OpenGL::initState(void)
{
    
}

class OpenGLHelper : public Task {
    OpenGL* opengl;
public:
    OpenGLHelper(char* name, OpenGL* opengl);
    virtual ~OpenGLHelper();
    virtual int body(int);
};

OpenGLHelper::OpenGLHelper(char*, OpenGL* opengl)
: Task(name, 1, DEFAULT_PRIORITY), opengl(opengl)
{
}

OpenGLHelper::~OpenGLHelper()
{
}

int OpenGLHelper::body(int)
{
    opengl->redraw_loop();
    return 0;
}

void OpenGL::redraw(Salmon* s, Roe* r, double _tbeg, double _tend,
		    int _nframes, double _framerate)
{
    // This is the first redraw - if there is not an OpenGL thread,
    // start one...
    if(!helper){
	my_openglname=clString("OpenGL: ")+myname;
	helper=new OpenGLHelper(my_openglname(), this);
	helper->activate(0);
    }
    salmon=s;
    roe=r;
    tbeg=_tbeg;
    tend=_tend;
    nframes=_nframes;
    framerate=_framerate;
    send_sema.up(); // Tell it to redraw...
    recv_sema.down(); // Wait until it is done...
}


void OpenGL::redraw_loop()
{
    // Get window information
    void* spec;
    if(salmon->lookup_specific("opengl_context", spec)){
	cx=(GLXContext)spec;
	TCLTask::lock(); // Unlock after MakeCurrent
    } else {
	TCLTask::lock();
	tkwin=Tk_NameToWindow(the_interp, myname(), Tk_MainWindow(the_interp));
	if(!tkwin){
	    cerr << "Unable to locate window!\n";
	    TCLTask::unlock();
	    return;
	}
	dpy=Tk_Display(tkwin);
	win=Tk_WindowId(tkwin);
	cx=OpenGLGetContext(the_interp, myname());
	if(!cx){
	    cerr << "Unable to create OpenGL Context!\n";
	    TCLTask::unlock();
	    return;
	}
    }
    fprintf(stderr, "dpy=%p, win=%p, cx=%p\n", dpy, win, cx);
    glXMakeCurrent(dpy, win, cx);
    glXWaitX();
    current_drawer=this;
    GLint data[1];
    glGetIntegerv(GL_MAX_LIGHTS, data);
    maxlights=data[0];
    TCLTask::unlock();

    // Tell the Roe that we are started...
    TimeThrottle throttle;
    throttle.start();
    double newtime=0;
    while(1){
	if(roe->inertia_mode){
	    cerr << "Inertia mode...";
	    cerr << "framerate=" << framerate << endl;
	    double current_time=throttle.time();
	    if(framerate==0)
		framerate=30;
	    double frametime=1./framerate;
	    double delta=current_time-newtime;
	    cerr << "delta=" << delta << endl;
	    if(delta > 1.5*frametime){
		cerr << "REALLY backing off..." << endl;
		framerate=1./delta;
		frametime=delta;
		newtime=current_time;
		cerr << "now=" << framerate << endl;
	    } if(delta > .85*frametime){
		cerr << "Backing off framerate.." << endl;
		framerate*=.9;
		cerr << "now=" << framerate << endl;
		frametime=1./framerate;
		newtime=current_time;
	    } else if(delta < .5*frametime){
		cerr << "Advancing off framrate..." << endl;
		framerate*=1.1;
		if(framerate>30)
		    framerate=30;
		cerr << "now=" << framerate << endl;
		frametime=1./framerate;
		newtime=current_time;
	    }
	    newtime+=frametime;
	    throttle.wait_for_time(newtime);
	    while(send_sema.try_down()) { /* Nothing */}
	    View view(roe->view.get());
	    view.eyep(view.eyep()+(view.eyep()-view.lookat())*0.01);
	    roe->view.set(view);
	} else {
	    send_sema.down(); // Wait to be woken up...
	    newtime=throttle.time();
	}
	redraw_frame();
	recv_sema.up();
    }
}

void OpenGL::redraw_frame()
{
    // Start polygon counter...
    WallClockTimer timer;
    timer.clear();
    timer.start();


    initState();

    // Get the window size
    xres=Tk_Width(tkwin);
    yres=Tk_Height(tkwin);

    // Make ourselves current
    if(current_drawer != this){
	current_drawer=this;
	TCLTask::lock();
	glXMakeCurrent(dpy, win, cx);
	TCLTask::unlock();
    }

    TCLTask::lock();

    // Clear the screen...
    glViewport(0, 0, xres, yres);
    Color bg(roe->bgcolor.get());
    glClearColor(bg.r(), bg.g(), bg.b(), 1);

    // Setup the view...
    View view(roe->view.get());
    double aspect=double(xres)/double(yres);
    double fovy=RtoD(2*Atan(aspect*Tan(DtoR(view.fov()/2.))));

    drawinfo->reset();

    // Get a lock on the geometry database...
    salmon->geomlock.read_lock();

    // Compute znear and zfar...
    double znear;
    double zfar;
    if(compute_depth(roe, view, znear, zfar)){


	// Set up graphics state
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_NORMALIZE);
	glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	
	clString globals("global");
	roe->setState(drawinfo,globals);

	int errcode;
	while((errcode=glGetError()) != GL_NO_ERROR){
	    cerr << "We got an error from GL: " << (char*)gluErrorString(errcode) << endl;
	}

	// Do the redraw loop for each time value
	double dt=(tend-tbeg)/nframes;
	double frametime=framerate==0?0:1./framerate;
	TimeThrottle throttle;
	throttle.start();
#ifdef __sgi
	int do_stereo=roe->do_stereo.get();
	if(do_stereo && !old_stereo){
	    int first_event, first_error;
	    if(!XSGIStereoQueryExtension(dpy, &first_event, &first_error)){
		do_stereo=0;
		cerr << "Stereo not supported!\n";
	    }
	    glXWaitX();
	    old_stereo=do_stereo;
	    int height=492; // Magic numbers from the man pages
	    int offset=532;
	    int mode=STEREO_TOP;
	    XClearWindow(dpy, win);
#if 0
	    if(!XSGISetStereoMode(dpy, win, height, offset, mode)){
		cerr << "Cannot set stereo mode!\n";
		do_stereo=0;
	    }
#endif
	    //	    system("/usr/gfx/setmon STR_TOP");
	    XSync(dpy, 0);
	    glXWaitX();
	}
	if(old_stereo && !do_stereo){
//	    system("/usr/gfx/setmon 72HZ");
#if 0
	    if(!XSGISetStereoMode(dpy, win, 0, 0, STEREO_OFF)){
		cerr << "Cannot set stereo mode!\n";
		do_stereo=0;
	    }
#endif
	    old_stereo=do_stereo;
	}
	Vector eyesep(0,0,0);
	if(do_stereo){
	    aspect/=2;	
	    double eye_sep_dist=0.025/2;
	    Vector u, v;
	    view.get_viewplane(aspect, 1.0, u, v);
	    u.normalize();
	    double zmid=(znear+zfar)/2.;
	    eyesep=u*eye_sep_dist*zmid;
	}
#endif /* __sgi */
	for(int t=0;t<nframes;t++){
#ifdef __sgi
	    if(do_stereo){
		XSGISetStereoBuffer(dpy, win, STEREO_BUFFER_LEFT);
		//		XClearWindow(dpy, win);
		XSync(dpy, 0);
		glXWaitX();
	    }
#endif
	    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	    double modeltime=t*dt+tbeg;
	    roe->set_current_time(modeltime);

	    // Setup view...
	    glViewport(0, 0, xres, yres);
	    glMatrixMode(GL_PROJECTION);
	    glLoadIdentity();
	    gluPerspective(fovy, aspect, znear, zfar);
	    
	    glMatrixMode(GL_MODELVIEW);
	    glLoadIdentity();
	    Point eyep(view.eyep());
	    Point lookat(view.lookat());
#ifdef __sgi
	    if(do_stereo){
		eyep-=eyesep;
		lookat-=eyesep;
	    }
#endif
	    Vector up(view.up());
	    gluLookAt(eyep.x(), eyep.y(), eyep.z(),
		      lookat.x(), lookat.y(), lookat.z(),
		      up.x(), up.y(), up.z());

	    // Set up Lighting
	    glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	    Lighting& l=salmon->lighting;
	    int idx=0;
	    int i;
	    for(i=0;i<l.lights.size();i++){
		Light* light=l.lights[i];
		light->opengl_setup(view, drawinfo, idx);
	    }
	    for(i=0;i<idx && i<maxlights;i++)
		glEnable(GL_LIGHT0+i);
	    for(;i<maxlights;i++)
		glDisable(GL_LIGHT0+i);

	    // now set up the fog stuff

	    glFogi(GL_FOG_MODE,GL_LINEAR);
	    glFogf(GL_FOG_START,float(znear));
	    glFogf(GL_FOG_END,float(zfar));
	    // now make the Roe setup its clipping planes...
	    roe->setClip(drawinfo);
	    
	    // Draw it all...
	    current_time=modeltime;
#ifdef REAL_STEREO
	    if(do_stereo){
		glDrawBuffer(GL_BACK_LEFT);
	    } else {
		glDrawBuffer(GL_BACK);
	    }
#endif
	    roe->do_for_visible(this, (RoeVisPMF)&OpenGL::redraw_obj);
#ifdef __sgi
	    if(do_stereo){
		glXWaitGL();
		//		XClearWindow(dpy, win);
		XSGISetStereoBuffer(dpy, win, STEREO_BUFFER_RIGHT);
		glXWaitX();
		//		glDrawBuffer(GL_BACK_RIGHT);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Setup view...
		glViewport(0, 0, xres, yres);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(fovy, aspect, znear, zfar);
		
		glMatrixMode(GL_MODELVIEW);
		glLoadIdentity();
		Point eyep(view.eyep());
		Point lookat(view.lookat());
		Vector up(view.up());
		eyep+=eyesep;
		lookat+=eyesep;
		gluLookAt(eyep.x(), eyep.y(), eyep.z(),
			  lookat.x(), lookat.y(), lookat.z(),
			  up.x(), up.y(), up.z());

		roe->do_for_visible(this, (RoeVisPMF)&OpenGL::redraw_obj);
	    }
#endif

	    // Wait for the right time before swapping buffers
	    //TCLTask::unlock();
	    double realtime=t*frametime;
	    //throttle.wait_for_time(realtime);
	    //TCLTask::lock();
	    TCL::execute("update idletasks");

	    // Show the pretty picture
	    glXSwapBuffers(dpy, win);
	    //	    glXWaitGL();
	}
	throttle.stop();
	double fps=nframes/throttle.time();
	int fps_whole=(int)fps;
	int fps_hund=(int)((fps-fps_whole)*100);
	ostrstream str(strbuf, STRINGSIZE);
	str << roe->id << " setFrameRate " << fps_whole << "." << fps_hund << '\0';
	TCL::execute(str.str());
	roe->set_current_time(tend);
    } else {
	// Just show the cleared screen
	roe->set_current_time(tend);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glXSwapBuffers(dpy, win);
	//	glXWaitGL();
    }
    salmon->geomlock.read_unlock();

    // Look for errors
    int errcode;
    while((errcode=glGetError()) != GL_NO_ERROR){
	cerr << "We got an error from GL: " << (char*)gluErrorString(errcode) << endl;
    }

    // Report statistics
    timer.stop();
    fpstimer.stop();
    double fps=nframes/fpstimer.time();
    fps+=0.05; // Round to nearest tenth
    int fps_whole=(int)fps;
    int fps_tenths=(int)((fps-fps_whole)*10);
    fpstimer.clear();
    fpstimer.start(); // Start it running for next time
    ostrstream str(strbuf, STRINGSIZE);
    str << roe->id << " updatePerf \"";
    str << drawinfo->polycount << " polygons in " << timer.time()
	<< " seconds\" \"" << drawinfo->polycount/timer.time()
	    << " polygons/second\"" << " \"" << fps_whole << "."
		<< fps_tenths << " frames/sec\""	<< '\0';
    TCL::execute(str.str());
    TCLTask::unlock();
}

void OpenGL::hide()
{
    tkwin=0;
    if(current_drawer==this)
	current_drawer=0;
}

void OpenGL::get_pick(Salmon*, Roe* roe, int x, int y,
		      GeomObj*& pick_obj, GeomPick*& pick_pick)
{
    pick_obj=0;
    pick_pick=0;
    // Make ourselves current
    if(current_drawer != this){
	current_drawer=this;
	TCLTask::lock();
	glXMakeCurrent(dpy, win, cx);
	TCLTask::unlock();
    }
    // Setup the view...
    View view(roe->view.get());
    double aspect=double(xres)/double(yres);
    double fovy=RtoD(2*Atan(aspect*Tan(DtoR(view.fov()/2.))));

    //    drawinfo->reset();

    // Compute znear and zfar...
    double znear;
    double zfar;
    if(compute_depth(roe, view, znear, zfar)){
	// Setup picking...
	TCLTask::lock();
	int errcode;
	while((errcode=glGetError()) != GL_NO_ERROR){
	    cerr << "We got an error from GL: " << (char*)gluErrorString(errcode) << endl;
	}
	GLuint pick_buffer[pick_buffer_size];
	glSelectBuffer(pick_buffer_size, pick_buffer);
	glRenderMode(GL_SELECT);
	glInitNames();
#if (_MIPS_SZPTR == 64)
	glPushName(0);
	glPushName(0);
#else
	glPushName(0);
#endif

	while((errcode=glGetError()) != GL_NO_ERROR){
	    cerr << "We got an error from GL: " << (char*)gluErrorString(errcode) << endl;
	}
	glViewport(0, 0, xres, yres);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	GLint viewport[4];
	glGetIntegerv(GL_VIEWPORT, viewport);
	gluPickMatrix(x, viewport[3]-y, pick_window, pick_window, viewport);
	gluPerspective(fovy, aspect, znear, zfar);
	while((errcode=glGetError()) != GL_NO_ERROR){
	    cerr << "We got an error from GL: " << (char*)gluErrorString(errcode) << endl;
	}
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Point eyep(view.eyep());
	Point lookat(view.lookat());
	Vector up(view.up());
	while((errcode=glGetError()) != GL_NO_ERROR){
	    cerr << "We got an error from GL: " << (char*)gluErrorString(errcode) << endl;
	}
	gluLookAt(eyep.x(), eyep.y(), eyep.z(),
		  lookat.x(), lookat.y(), lookat.z(),
		  up.x(), up.y(), up.z());

	drawinfo->lighting=0;
	drawinfo->set_drawtype(DrawInfoOpenGL::Flat);
	drawinfo->pickmode=1;

	// Draw it all...
	while((errcode=glGetError()) != GL_NO_ERROR){
	    cerr << "We got an error from GL: " << (char*)gluErrorString(errcode) << endl;
	}
	roe->do_for_visible(this, (RoeVisPMF)&OpenGL::pick_draw_obj);
#if (_MIPS_SZPTR == 64)
	glPopName();
	glPopName();
#else
	glPopName();
#endif

	glFlush();
	int hits=glRenderMode(GL_RENDER);
	while((errcode=glGetError()) != GL_NO_ERROR){
	    cerr << "We got an error from GL: " << (char*)gluErrorString(errcode) << endl;
	}
	TCLTask::unlock();
	GLuint min_z;
#if (_MIPS_SZPTR == 64)
	unsigned long hit_obj=0;
	unsigned long hit_pick=0;
#else
	GLuint hit_obj=0;
	GLuint hit_pick=0;
#endif
	cerr << "hits=" << hits << endl;
	if(hits >= 1){
	    int idx=0;
	    min_z=0;
	    for (int h=0; h<hits; h++) {
		int nnames=pick_buffer[idx++];
		GLuint z=pick_buffer[idx++];
		if (nnames > 1 && (h==0 || z < min_z)) {
		    min_z=z;
		    idx++; // Skip Max Z
#if (_MIPS_SZPTR == 64)
		    unsigned int ho1=pick_buffer[idx++];
		    unsigned int ho2=pick_buffer[idx++];
		    hit_obj=((long)ho1<<32)|ho2;
		    idx+=nnames-4; // Skip to the last one...
		    unsigned int hp1=pick_buffer[idx++];
		    unsigned int hp2=pick_buffer[idx++];
		    hit_pick=((long)hp1<<32)|hp2;
#else
		    hit_obj=pick_buffer[idx++];
		    idx+=nnames-2; // Skip to the last one...
		    hit_pick=pick_buffer[idx++];
#endif
		} else {
		    idx+=nnames+1;
		}
	    }
	}
	pick_obj=(GeomObj*)hit_obj;
	pick_pick=(GeomPick*)hit_pick;
	cerr << "pick_pick=" << pick_pick << endl;
    }
}

void OpenGL::dump_image(const clString& name) {
    ofstream dumpfile(name());
    GLint vp[4];
    glGetIntegerv(GL_VIEWPORT,vp);
    int n=3*vp[2]*vp[3];
    unsigned char* pxl=scinew unsigned char[n];
    glReadBuffer(GL_FRONT);
    glReadPixels(0,0,vp[2],vp[3],GL_RGB,GL_UNSIGNED_BYTE,pxl);
    dumpfile.write(pxl,n);
    delete[] pxl;
}

void OpenGL::put_scanline(int y, int width, Color* scanline, int repeat)
{
    float* pixels=scinew float[width*3];
    float* p=pixels;
    int i;
    for(i=0;i<width;i++){
	*p++=scanline[i].r();
	*p++=scanline[i].g();
	*p++=scanline[i].b();
    }
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glTranslated(-1, -1, 0);
    glScaled(2./xres, 2./yres, 1.0);
    glDepthFunc(GL_ALWAYS);
    glDrawBuffer(GL_FRONT);
    for(i=0;i<repeat;i++){
	glRasterPos2i(0, y+i);
	glDrawPixels(width, 1, GL_RGB, GL_FLOAT, pixels);
    }
    glDepthFunc(GL_LEQUAL);
    glDrawBuffer(GL_BACK);
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    delete[] pixels;
}

void OpenGL::pick_draw_obj(Salmon* salmon, Roe*, GeomObj* obj)
{
#if (_MIPS_SZPTR == 64)
    unsigned long o=(unsigned long)obj;
    unsigned int o1=(o>>32)&0xffffffff;
    unsigned int o2=o&0xffffffff;
    glPopName();
    glPopName();
    glPushName(o1);
    glPushName(o2);
#else
    glLoadName((GLuint)obj);
#endif
    obj->draw(drawinfo, salmon->default_matl.get_rep(), current_time);
}

void OpenGL::redraw_obj(Salmon* salmon, Roe* roe, GeomObj* obj)
{
    drawinfo->roe = roe;
    obj->draw(drawinfo, salmon->default_matl.get_rep(), current_time);
}

void Roe::setState(DrawInfoOpenGL* drawinfo,clString tclID)
{
    clString val;
    clString type(tclID+"-"+"type");
    clString lighting(tclID+"-"+"light");
    clString fog(tclID+"-"+"fog");
    clString debug(tclID+"-"+"debug");

    clString use_clip(tclID+"-"+"clip");

    if (!get_tcl_stringvar(id,type,val)) {
	cerr << "Error illegal name!\n";
	return;
    }
    else {
	if(val == "Wire"){
	    drawinfo->set_drawtype(DrawInfoOpenGL::WireFrame);
	    drawinfo->lighting=0;
	} else if(val == "Flat"){
	    drawinfo->set_drawtype(DrawInfoOpenGL::Flat);
	    drawinfo->lighting=0;
	} else if(val == "Gouraud"){
	    drawinfo->set_drawtype(DrawInfoOpenGL::Gouraud);
	    drawinfo->lighting=1;
	}
	else if (val == "Default") {
	    drawinfo->currently_lit=drawinfo->lighting;
	    drawinfo->init_lighting(drawinfo->lighting);
	    return; // if they are using the default, con't change
	} else {
	    cerr << "Unknown shading(" << val << "), defaulting to phong" << endl;
	    drawinfo->set_drawtype(DrawInfoOpenGL::Gouraud);
	    drawinfo->lighting=1;
	}

	// now see if they want a bounding box...

	if (get_tcl_stringvar(id,debug,val)) {
	    if (val == "0")
		drawinfo->debug = 0;
	    else
		drawinfo->debug = 1;
	}	
	else {
	    cerr << "Error, no debug level set!\n";
	    drawinfo->debug = 0;
	}

	if (get_tcl_stringvar(id,use_clip,val)) {
	    if (val == "0")
		drawinfo->check_clip = 0;
	    else
		drawinfo->check_clip = 1;
	}	
	else {
	    cerr << "Error, no clipping info\n";
	    drawinfo->check_clip = 0;
	}

	drawinfo->init_clip(); // set clipping 

	if (!get_tcl_stringvar(id,lighting,val))
	    cerr << "Error, no lighting!\n";
	else {
	    if (val == "0"){
		drawinfo->lighting=0;
	    }
	    else if (val == "1") {
		drawinfo->lighting=1;
	    }
	    else {
		cerr << "Unknown lighting setting(" << val << "\n";
	    }

	    if (get_tcl_stringvar(id,fog,val)) {
		if (val=="0"){
		    drawinfo->fog=0;
		}
		else {
		    drawinfo->fog=1;
		}
	    }
	    else {
		cerr << "Fog not defined properly!\n";
		drawinfo->fog=0;
	    }

	    //		drawinfo->pickmode=0;            
	}
    }
    drawinfo->pickmode=0;
    drawinfo->currently_lit=drawinfo->lighting;
    drawinfo->init_lighting(drawinfo->lighting);
	
    
}

void Roe::setDI(DrawInfoOpenGL* drawinfo,clString name)
{
    ObjTag* vis;

    if (visible.lookup(name,vis)){
	setState(drawinfo,to_string(vis->tagid));
    }
}

// set the bits for the clipping planes that are on...

void Roe::setClip(DrawInfoOpenGL* drawinfo)
{
    clString val;
    int i;

    drawinfo->clip_planes = 0; // set them all of for default
    clString num_clip("clip-num");

    if (get_tcl_stringvar(id,"clip-visible",val) && 
	get_tcl_intvar(id,num_clip,i)) {

	int cur_flag = CLIP_P5;
	if ( (i>0 && i<7) ) {
	    while(i--) {
		
		clString vis("clip-visible-"+to_string(i+1));


		if (get_tcl_stringvar(id,vis,val)) {
		    if (val == "1") {
			double plane[4];
			clString nx("clip-normal-x-"+to_string(i+1));
			clString ny("clip-normal-y-"+to_string(i+1));
			clString nz("clip-normal-z-"+to_string(i+1));
			clString nd("clip-normal-d-"+to_string(i+1));
			
			int rval=0;

			rval = get_tcl_doublevar(id,nx,plane[0]);
			rval = get_tcl_doublevar(id,ny,plane[1]);
			rval = get_tcl_doublevar(id,nz,plane[2]);
			rval = get_tcl_doublevar(id,nd,plane[3]);
			
			double mag = plane[0]*plane[0] +
			    plane[1]*plane[1] +
				plane[2]*plane[2];
			plane[0] /= mag;
			plane[1] /= mag;
			plane[2] /= mag;
			plane[3] = -plane[3]; // so moves in planes direction...
			glClipPlane(GL_CLIP_PLANE0+i,plane);

			if (drawinfo->check_clip)
			    glEnable(GL_CLIP_PLANE0+i);
			else
			    glDisable(GL_CLIP_PLANE0+i);
			    
			drawinfo->clip_planes |= cur_flag;

			if (!rval ) {
			    cerr << "Error, variable is hosed!\n";
			}
		    }
		    else {
			glDisable(GL_CLIP_PLANE0+i);
		    }

		}
		cur_flag >>= 1; // shift the bit we are looking at...
	    }
	}
    }
}


void GeomSalmonItem::draw(DrawInfoOpenGL* di, Material *m, double time)
{
    // here we need to query the roe with our name and give it our
    // di so it can change things if they need to be...
    di->roe->setDI(di,name);

    // lets get the childs bounding box, and draw it...

    BBox bb;
    //    child->reset_bbox();
    child->get_bounds(bb);
    if(!bb.valid())
	return;

    // might as well try and draw the arcball also...

    Point min,max;

    min = bb.min();
    max = bb.max();
    if (!di->debug)
	child->draw(di,m,time);

    if (di->debug) {

	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	glEnable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glDepthMask(GL_FALSE);

	glColor4f(1.0,0.0,1.0,0.2);

	glDisable(GL_LIGHTING);	

	glBegin(GL_QUADS);
	//front
	glVertex3d(max.x(),min.y(),max.z());
	//	glColor4f(0.0,1.0,0.0,0.8);
	glVertex3d(max.x(),max.y(),max.z());
	glColor4f(0.0,1.0,0.0,0.2);
	glVertex3d(min.x(),max.y(),max.z());
	glVertex3d(min.x(),min.y(),max.z());
	//back
	glVertex3d(max.x(),max.y(),min.z());
	glVertex3d(max.x(),min.y(),min.z());
	//	glColor4f(1.0,0.0,0.0,0.8);
	glVertex3d(min.x(),min.y(),min.z());
	glColor4f(0.0,1.0,0.0,0.2);
	glVertex3d(min.x(),max.y(),min.z());

	glColor4f(1.0,0.0,0.0,0.2);

	//left
	glVertex3d(min.x(),min.y(),max.z());
	glVertex3d(min.x(),max.y(),max.z());
	glVertex3d(min.x(),max.y(),min.z());
	//	glColor4f(1.0,0.0,0.0,0.8);
	glVertex3d(min.x(),min.y(),min.z());
	glColor4f(1.0,0.0,0.0,0.2);

	//right
	glVertex3d(max.x(),min.y(),min.z());
	glVertex3d(max.x(),max.y(),min.z());
	//	glColor4f(0.0,1.0,0.0,0.8);
	glVertex3d(max.x(),max.y(),max.z());
	glColor4f(1.0,0.0,0.0,0.2);
	glVertex3d(max.x(),min.y(),max.z());


	glColor4f(0.0,0.0,1.0,0.2);

	//top
	glVertex3d(min.x(),max.y(),max.z());
	//	glColor4f(0.0,1.0,0.0,0.8);
	glVertex3d(max.x(),max.y(),max.z());
	glColor4f(0.0,0.0,1.0,0.2);
	glVertex3d(max.x(),max.y(),min.z());
	glVertex3d(min.x(),max.y(),min.z());
	//bottom
	//	glColor4f(1.0,0.0,0.0,0.8);
	glVertex3d(min.x(),min.y(),min.z());
	glColor4f(0.0,0.0,1.0,0.2);
	glVertex3d(max.x(),min.y(),min.z());
	glVertex3d(max.x(),min.y(),max.z());
	glVertex3d(min.x(),min.y(),max.z());

	glEnd();
	glDisable(GL_BLEND);
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glDisable(GL_CULL_FACE);
    }
}

