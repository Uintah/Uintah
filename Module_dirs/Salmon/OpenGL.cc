
#include <Modules/Salmon/Renderer.h>
#include <Modules/Salmon/Roe.h>
#include <Modules/Salmon/Salmon.h>
#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Classlib/Timer.h>
#include <Geom/Geom.h>
#include <Geom/GeomOpenGL.h>
#include <Math/Trig.h>
#include <TCL/TCLTask.h>
#include <tcl/tcl7.3/tcl.h>
#include <tcl/tk3.6/tk.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include <values.h>

class OpenGL : public Renderer {
    Tk_Window tkwin;
    Window win;
    Display* dpy;
    GLXContext cx;
    clString myname;
    double aspect;
public:
    OpenGL();
    virtual ~OpenGL();
    virtual clString create_window(const clString& name,
				   const clString& width,
				   const clString& height);
    virtual void redraw(Salmon*, Roe*);
    virtual void hide();
};

static OpenGL* current_drawer=0;
extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);
extern Tcl_Interp* the_interp;

static Renderer* make_OpenGL()
{
    return new OpenGL;
}

RegisterRenderer OpenGL_renderer("OpenGL", &make_OpenGL);

OpenGL::OpenGL()
: tkwin(0)
{
}

OpenGL::~OpenGL()
{
}

clString OpenGL::create_window(const clString& name,
			       const clString& width,
			       const clString& height)
{
    myname=name;
    double w,h;
    width.get_double(w);
    height.get_double(h);
    aspect=(double)w/(double)h;
    return "opengl "+name+" -geometry "+width+"x"+height+" -doublebuffer true -direct true -rgba true -redsize 2 -greensize 2 -bluesize 2 -depthsize 2";
}

void OpenGL::redraw(Salmon* salmon, Roe* roe)
{
    // Start polygon counter...
    WallClockTimer timer;
    timer.clear();
    timer.start();

    // Get window information
    if(!tkwin){
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
	TCLTask::unlock();
    }

    // Make ourselves current
    if(current_drawer != this){
	current_drawer=this;
	TCLTask::lock();
	glXMakeCurrent(dpy, win, cx);
	TCLTask::unlock();
    }

    // Setup lighting
    NOT_FINISHED("OpenGL::setup_lighting");
    GLfloat light_position0[] = { 500,500,-100,1};
    glLightfv(GL_LIGHT0, GL_POSITION, light_position0);
    GLfloat light_position1[] = { -50,-100,100,1};
    glLightfv(GL_LIGHT1, GL_POSITION, light_position1);

    // Clear the screen...
    glClearColor(0,0,0,1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Setup the view...
    glLoadIdentity();
    double fovy=RtoD(2*Atan(aspect*Tan(DtoR(roe->view.fov/2.))));

    DrawInfoOpenGL drawinfo;
    drawinfo.polycount=0;

    // Compute znear and zfar...
    double znear=MAXDOUBLE;
    double zfar=-MAXDOUBLE;
    BBox bb;
    roe->get_bounds(bb);
    if(bb.valid()) {
	// We have something to draw...
	Point min(bb.min());
	Point max(bb.max());
	Point eyep(roe->view.eyep);
	Vector dir(roe->view.lookat-eyep);
	dir.normalize();
	double d=-(eyep.x()*dir.x()+eyep.y()*dir.y()+eyep.z()*dir.z());
	for(int ix=0;ix<1;ix++){
	    for(int iy=0;iy<1;iy++){
		for(int iz=0;iz<1;iz++){
		    Point p(ix?max.x():min.x(),
			    iy?max.y():min.y(),
			    iz?max.z():min.z());
		    double dist=p.x()*dir.x()+p.y()*dir.y()+p.z()*dir.z()+d;
		    znear=Min(znear, dist);
		    zfar=Max(zfar, dist);
		}
	    }
	}
	if(znear <= 0){
	    if(zfar <= 0){
		// Everything is behind us - it doesn't matter what we do
		znear=1.0;
		zfar=2.0;
	    } else {
		znear=zfar*.001;
	    }
	}
	gluPerspective(fovy, aspect, znear, zfar);
	Point lookat(roe->view.lookat);
	Vector up(roe->view.up);
	gluLookAt(eyep.x(), eyep.y(), eyep.z(),
		  lookat.x(), lookat.y(), lookat.z(),
		  up.x(), up.y(), up.z());


	// Draw it all...
	drawinfo.push_matl(salmon->default_matl.get_rep());
	HashTableIter<int,HashTable<int, GeomObj*>*> iter(&salmon->portHash);
	for (iter.first(); iter.ok(); ++iter) {
	    HashTable<int, GeomObj*>* serHash=iter.get_data();
	    HashTableIter<int, GeomObj*> serIter(serHash);
	    for (serIter.first(); serIter.ok(); ++serIter) {
		GeomObj *geom=serIter.get_data();

		// Look up this object by name and see if it is supposed to be
		// displayed...
		geom->draw(&drawinfo);
	    }
	}
	drawinfo.pop_matl();
    }

    // Show the pretty picture
    TCLTask::lock();
    glXSwapBuffers(dpy, win);
    TCLTask::unlock();

    // Report statistics
    timer.stop();
    clString perf1(to_string(drawinfo.polycount)+" polygons in "
		  +to_string(timer.time())+" seconds");
    clString perf2(to_string(double(drawinfo.polycount)/timer.time())
		   +" polygons/second");
    static clString q("\"");
    static clString s(" ");
    static clString c("updatePerf ");
    TCL::execute(c+roe->id+s+q+perf1+q+s+q+perf2+q);
}

void OpenGL::hide()
{
    tkwin=0;
    if(current_drawer==this)
	current_drawer=0;
}
