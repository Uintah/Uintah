/*
 *  OpenGL_Ex.cc:  Pops up an OpenGL window for SCIRun use
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#include <Classlib/Array2.h>
#include <Classlib/NotFinished.h>
#include <Classlib/String.h>
#include <Dataflow/Module.h>
#include <Datatypes/ScalarFieldRGchar.h>
#include <Datatypes/ScalarFieldRGBase.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Datatypes/ScalarFieldPort.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include <Geom/Color.h>
#include <Geom/GeomOpenGL.h>
#include <Malloc/Allocator.h>
#include <Math/MinMax.h>
#include <Math/MiscMath.h>
#include <TCL/TCLTask.h>
#include <TCL/TCLvar.h>
#include <TCL/TCL.h>
#include <tcl/tcl/tcl.h>
#include <tcl/tk/tk.h>
#include <iostream.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

extern Tcl_Interp* the_interp;
extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);

class OpenGL_Ex : public Module {
    ScalarFieldIPort *iport;
    ScalarFieldOPort *oport;
    ScalarFieldHandle last_sfIH;	// last input fld
    ScalarFieldRG* last_sfrg;		// just a convenience

    double x_pixel_size;
    double y_pixel_size;
    double z_pixel_size;
    int x_win_min;
    int x_win_max;
    int y_win_min;
    int y_win_max;
    int z_win_min;
    int z_win_max;

    int cleared;
    Array2<char> image;

    clString myid;
    Tk_Window tkwin;
    Window win;
    Display* dpy;
    GLXContext cx;
    int tcl_execute;
public:
    OpenGL_Ex(const clString& id);
    OpenGL_Ex(const OpenGL_Ex&, int deep);
    virtual ~OpenGL_Ex();
    virtual Module* clone(int deep);
    virtual void execute();
    void set_str_vars();
    void tcl_command( TCLArgs&, void * );
    void redraw_all();
    int makeCurrent();
};

extern "C" {
Module* make_OpenGL_Ex(const clString& id)
{
    return scinew OpenGL_Ex(id);
}
}

OpenGL_Ex::OpenGL_Ex(const clString& id)
: Module("OpenGL_Ex", id, Source), tcl_execute(0)
{
    // Create the input port
    myid=id;
    iport = scinew ScalarFieldIPort(this, "SFRG", ScalarFieldIPort::Atomic);
    add_iport(iport);
}

OpenGL_Ex::OpenGL_Ex(const OpenGL_Ex& copy, int deep)
: Module(copy, deep), tcl_execute(0)
{
    NOT_FINISHED("OpenGL_Ex::OpenGL_Ex");
}

OpenGL_Ex::~OpenGL_Ex()
{
}

Module* OpenGL_Ex::clone(int deep)
{
    return scinew OpenGL_Ex(*this, deep);
}

void OpenGL_Ex::execute()
{
    ScalarFieldHandle sfIH;
    iport->get(sfIH);
    if (!sfIH.get_rep()) return;
    if (!tcl_execute && (sfIH.get_rep() == last_sfIH.get_rep())) return;
    ScalarFieldRG *sfrg;
    if ((sfrg=sfIH->getRG()) == 0) return;
    if (sfIH.get_rep() != last_sfIH.get_rep()) {	// new field came in
	int nx, ny, nz;		// just so my fingers don't get tired ;)
	nx = sfrg->nx;
	ny = sfrg->ny;
	nz = sfrg->nz;
	image.newsize(nx,ny);
	image.initialize(0);

	int mid_z = nz/2;
	double max, min;
	max=min=sfrg->grid(0,0,mid_z);

	int i,j;
	for (i=0; i<nx; i++) {
	    for (j=0; j<ny; j++) {
		if(max>sfrg->grid(i,j,mid_z)) max=sfrg->grid(i,j,mid_z);
		if(min<sfrg->grid(i,j,mid_z)) min=sfrg->grid(i,j,mid_z);
	    }
	}

	double scale=255/(max-min);
	for (i=0; i<nx; i++) {
	    for (j=0; j<ny; j++) {
		image(i,j)=(char)((sfrg->grid(i,j,mid_z)-min)*scale);
	    }
	}

	last_sfIH=sfIH;
	last_sfrg=sfrg;
	Point pmin;
	Point pmax;
	last_sfIH->get_bounds(pmin, pmax);
	Vector dims(pmax-pmin);
	double win_scale = 512.0/Max(dims.x(), dims.y(), dims.z());
	x_pixel_size=dims.x()/nx*win_scale;
	y_pixel_size=dims.y()/ny*win_scale;
	z_pixel_size=dims.z()/nz*win_scale;
	x_win_min=256-(x_pixel_size*nx/2.);
	y_win_min=256-(y_pixel_size*ny/2.);
	z_win_min=256-(z_pixel_size*nz/2.);
	x_win_max=256+(x_pixel_size*nx/2.);
	y_win_max=256+(y_pixel_size*ny/2.);
	z_win_max=256+(z_pixel_size*nz/2.);
	redraw_all();
    }
    tcl_execute=0;
}

void OpenGL_Ex::redraw_all() {
    
    cleared=1;
    cerr << "In redraw_all()\n";
    if (!makeCurrent()) return;
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    // if the handle was empty, just flush the buffer (to clear the window)
    //    and return.
    if (!last_sfIH.get_rep()) {
	glFlush();
        glXMakeCurrent(dpy, None, NULL);
	TCLTask::unlock();
	return;
    }

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glPixelZoom(x_pixel_size, y_pixel_size);
    glRasterPos2i(310+x_win_min, 565-y_win_min);
    char *pixels=&(image(0,0));
    glDrawPixels(last_sfrg->nx, last_sfrg->ny, GL_LUMINANCE, GL_BYTE, pixels);

    int errcode;
    while((errcode=glGetError()) != GL_NO_ERROR){
	cerr << "plot_matrices got an error from GL: " << (char*)gluErrorString(errcode) << endl;
    }
    glXMakeCurrent(dpy, None, NULL);
    TCLTask::unlock();
}	    

void OpenGL_Ex::tcl_command(TCLArgs& args, void* userdata) {
    if (args[1] == "redraw_all") {
	reset_vars();
	redraw_all();
    } else {
        Module::tcl_command(args, userdata);
    }
}

int OpenGL_Ex::makeCurrent() {
    TCLTask::lock();
    clString myname(clString(".ui")+id+".gl.gl");
    tkwin=Tk_NameToWindow(the_interp, myname(), Tk_MainWindow(the_interp));
    if(!tkwin){
	cerr << "Unable to locate window!\n";
	TCLTask::unlock();
	return 0;
    }
    dpy=Tk_Display(tkwin);
    win=Tk_WindowId(tkwin);
    cx=OpenGLGetContext(the_interp, myname());
    if(!cx){
	cerr << "Unable to create OpenGL Context!\n";
	TCLTask::unlock();
	return 0;
    }
    if (!glXMakeCurrent(dpy, win, cx))
	    cerr << "*glXMakeCurrent failed.\n";

    // Clear the screen...
    glViewport(0, 0, 599, 599);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 599, 599, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    return 1;
}
