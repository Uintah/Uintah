
#include <Modules/Salmon/Renderer.h>
#include <Modules/Salmon/Roe.h>
#include <Modules/Salmon/Salmon.h>
#include <Classlib/HashTable.h>
#include <Classlib/NotFinished.h>
#include <Classlib/Timer.h>
#include <Geom/Geom.h>
#include <Geom/GeomOpenGL.h>
#include <Geom/Light.h>
#include <Math/Trig.h>
#include <TCL/TCLTask.h>
#include <tcl/tcl7.3/tcl.h>
#include <tcl/tk3.6/tk.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>

class OpenGL : public Renderer {
    Tk_Window tkwin;
    Window win;
    Display* dpy;
    GLXContext cx;
    clString myname;
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
    width.get_int(xres);
    height.get_int(yres);
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

    TCLTask::lock();

    // Clear the screen...
    glClearColor(0,0,0,1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Setup the view...
    View view(roe->view.get());
    double aspect=double(xres)/double(yres);
    double fovy=RtoD(2*Atan(aspect*Tan(DtoR(view.fov/2.))));

    DrawInfoOpenGL drawinfo;
    drawinfo.polycount=0;

    // Compute znear and zfar...
    double znear;
    double zfar;
    if(compute_depth(roe, view, znear, zfar)){
	glViewport(0, 0, xres, yres);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fovy, aspect, znear, zfar);
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	Point eyep(view.eyep);
	Point lookat(view.lookat);
	Vector up(view.up);
	gluLookAt(eyep.x(), eyep.y(), eyep.z(),
		  lookat.x(), lookat.y(), lookat.z(),
		  up.x(), up.y(), up.z());

	// Set up Lighting
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
	Lighting& l=salmon->lighting;
	int idx=0;
	for(int i=0;i<l.lights.size();i++){
	    Light* light=l.lights[i];
	    light->opengl_setup(view, &drawinfo, idx);
	}
	for(i=0;i<idx && i<GL_MAX_LIGHTS;i++)
	    glEnable(GL_LIGHT0+i);
	for(;i<GL_MAX_LIGHTS;i++)
	    glDisable(GL_LIGHT0+i);

	// Set up graphics state
	glDepthFunc(GL_LEQUAL);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_NORMALIZE);

	clString shading(roe->shading.get());
	if(shading == "wire"){
	    drawinfo.drawtype=DrawInfoOpenGL::WireFrame;
	    drawinfo.lighting=0;
	} else if(shading == "flat"){
	    drawinfo.drawtype=DrawInfoOpenGL::Flat;
	    drawinfo.lighting=0;
	} else if(shading == "gouraud"){
	    drawinfo.drawtype=DrawInfoOpenGL::Gouraud;
	    drawinfo.lighting=1;
	} else if(shading == "phong"){
	    drawinfo.drawtype=DrawInfoOpenGL::Phong;
	    drawinfo.lighting=1;
	} else {
	    cerr << "Unknown shading(" << shading << "), defaulting to gouraud" << endl;
	    drawinfo.drawtype=DrawInfoOpenGL::Gouraud;
	    drawinfo.lighting=1;
	}
	drawinfo.currently_lit=0;
	if(drawinfo.lighting)
	    glEnable(GL_LIGHTING);
	else
	    glDisable(GL_LIGHTING);

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
