//static char *id="@(#) $Id$";

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

#include <SCICore/Containers/Array2.h>
#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Datatypes/ScalarFieldRGchar.h>
#include <SCICore/Datatypes/ScalarFieldRGBase.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Geom/GeomOpenGL.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Math/MiscMath.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/TclInterface/TCL.h>
#include <tcl.h>
#include <tk.h>
#include <iostream>
using std::cerr;
using std::endl;
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

extern Tcl_Interp* the_interp;
extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Containers;
using namespace SCICore::Math;

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
    virtual ~OpenGL_Ex();
    virtual void execute();
    void set_str_vars();
    void tcl_command( TCLArgs&, void * );
    void redraw_all();
    int makeCurrent();
};

Module* make_OpenGL_Ex(const clString& id) {
  return new OpenGL_Ex(id);
}

OpenGL_Ex::OpenGL_Ex(const clString& id)
: Module("OpenGL_Ex", id, Source), tcl_execute(0)
{
    // Create the input port
    myid=id;
    iport = scinew ScalarFieldIPort(this, "SFRG", ScalarFieldIPort::Atomic);
    add_iport(iport);
}

OpenGL_Ex::~OpenGL_Ex()
{
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
    tkwin=Tk_NameToWindow(the_interp, const_cast<char *>(myname()), Tk_MainWindow(the_interp));
    if(!tkwin){
	cerr << "Unable to locate window!\n";
	TCLTask::unlock();
	return 0;
    }
    dpy=Tk_Display(tkwin);
    win=Tk_WindowId(tkwin);
    cx=OpenGLGetContext(the_interp, const_cast<char *>(myname()));
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

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.6  1999/10/07 02:06:48  sparker
// use standard iostreams and complex type
//
// Revision 1.5  1999/08/25 03:47:48  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:45  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:19:42  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:29  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:43  mcq
// Initial commit
//
// Revision 1.3  1999/07/07 21:10:22  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/04/28 20:51:12  dav
// deleted some files that are dependent on DaveW files
//
// Revision 1.1.1.1  1999/04/24 23:12:33  dav
// Import sources
//
//
