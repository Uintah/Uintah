//static char *id="@(#) $Id$";

/*
 *  ExtractSubmatrix.cc:  Visual matrix editor
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   November 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Util/NotFinished.h>
#include <Containers/String.h>
#include <Dataflow/Module.h>
#include <CoreDatatypes/DenseMatrix.h>
#include <CoreDatatypes/Matrix.h>
#include <CommonDatatypes/MatrixPort.h>
#include <CoreDatatypes/SymSparseRowMatrix.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include <Geom/Color.h>
#include <Geom/GeomOpenGL.h>
#include <Malloc/Allocator.h>
#include <TclInterface/TCLTask.h>
#include <TclInterface/TCLvar.h>
#include <TclInterface/TCL.h>
#include <tcl.h>
#include <tk.h>
#include <iostream.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

// Everything will be done in C++ -- the tcl UI will just be used for display
// (i.e. all the flags and the OpenGL drawing will be done from here).
// tcl will just handle the toggles for which matrices are being graphed

extern Tcl_Interp* the_interp;
extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);

namespace PSECommon {
namespace Modules {

using namespace PSECommon::Dataflow;
using namespace PSECommon::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

class ExtractSubmatrix : public Module {
    MatrixIPort *iport;
    MatrixOPort *oport;
    MatrixHandle mH;
    MatrixHandle mHO;
    TCLint minRow;
    TCLint minCol;
    TCLint maxRow;
    TCLint maxCol;
    TCLint ntrows;
    TCLint ntcols;
    TCLint always;

    int sendFlag;
    int lastMinR;
    int lastMinC;
    int lastMaxR;
    int lastMaxC;

    int drawRect;
    int eraseRect;

    clString myid;
    Tk_Window tkwin;
    Window win;
    Display* dpy;
    GLXContext cx;
public:
    ExtractSubmatrix(const clString& id);
    ExtractSubmatrix(const ExtractSubmatrix&, int deep);
    virtual ~ExtractSubmatrix();
    virtual Module* clone(int deep);
    virtual void execute();
    void tcl_command( TCLArgs&, void * );
    void plot_matrices();
    void plot_rect();
    int makeCurrent(int &, int &, int &, int &, int &);
    SymSparseRowMatrix* buildSymSparseRow(SymSparseRowMatrix *m);
    DenseMatrix* buildDense(DenseMatrix* m);
};

static ExtractSubmatrix* current_drawer=0;

Module* make_ExtractSubmatrix(const clString& id) {
  return new ExtractSubmatrix(id);
}

ExtractSubmatrix::ExtractSubmatrix(const clString& id)
: Module("ExtractSubmatrix", id, Source), drawRect(0), eraseRect(0), tkwin(0), 
  minRow("minRow", id, this), minCol("minCol", id, this), 
  maxRow("maxRow", id, this), maxCol("maxCol", id, this), mHO(0),
  ntrows("ntrows", id, this), ntcols("ntcols", id, this), sendFlag(0),
  always("always", id, this)
{
    // Create the input port
    myid=id;
    iport = scinew MatrixIPort(this, "Matrix", MatrixIPort::Atomic);
    add_iport(iport);
    oport = scinew MatrixOPort(this, "Matrix", MatrixIPort::Atomic);
    add_oport(oport);
}

ExtractSubmatrix::ExtractSubmatrix(const ExtractSubmatrix& copy, int deep)
: Module(copy, deep), drawRect(0), eraseRect(0),
  minRow("minRow", id, this), minCol("minCol", id, this), 
  maxRow("maxRow", id, this), maxCol("maxCol", id, this), mHO(0),
  ntrows("ntrows", id, this), ntcols("ntcols", id, this), sendFlag(0),
  always("always", id, this)
{
    NOT_FINISHED("ExtractSubmatrix::ExtractSubmatrix");
}

ExtractSubmatrix::~ExtractSubmatrix()
{
}

Module* ExtractSubmatrix::clone(int deep)
{
    return scinew ExtractSubmatrix(*this, deep);
}

void ExtractSubmatrix::execute()
{
    iport->get(mH);
    if (!mH.get_rep()) return;
    if (mH->nrows() != ntrows.get() || mH->ncols() != ntcols.get()) {
	if (mH->nrows() != ntrows.get()) {
	    ntrows.set(mH->nrows());
	}
	if (mH->ncols() != ntcols.get()) {
	    ntcols.set(mH->ncols());
	}
	return;	// we know we're gonna get a needexecute call from tcl...
    }
    plot_matrices();
    drawRect=1;
    plot_rect();
    if (sendFlag || always.get()) {
	if (mH->getDense()) {
	    mHO=buildDense(mH->getDense());
	} else {
	    mHO=buildSymSparseRow(mH->getSymSparseRow());
	}
	sendFlag=0;
    }
    oport->send(mHO);
}

SymSparseRowMatrix* ExtractSubmatrix::buildSymSparseRow(SymSparseRowMatrix *m){
    int maxR=maxRow.get()-1;
    int maxC=maxCol.get()-1;
    int minR=minRow.get()-1;
    int minC=minCol.get()-1;
    Array1<double> val;
    Array1<int> ridx;
    Array1<int> cidx;
    Array1<int> in_rows;
    in_rows.add(0);
    int r,c;
    for (r=minR; r<=maxR; r++) {
	Array1<int> idx;
	Array1<double> v;
	m->getRowNonzeros(r, idx, v);
	int added=0;
	for (c=0; c<idx.size() && idx[c]<=maxC; c++) {
	    if (idx[c]>=minC) {
		added++;
		val.add(v[c]);
		ridx.add(r-minR);
		cidx.add(idx[c]-minC);
	    }
	}
	in_rows.add(added+in_rows[in_rows.size()-1]);
    }
    SymSparseRowMatrix* ssrm = 
	scinew SymSparseRowMatrix(maxR-minR+1, maxC-minC+1, in_rows, cidx);
    for (int z=0; z<val.size(); z++)
	ssrm->put(ridx[z], cidx[z], val[z]);
    return ssrm;
}

DenseMatrix* ExtractSubmatrix::buildDense(DenseMatrix* m) {
    int maxR=maxRow.get()-1;
    int maxC=maxCol.get()-1;
    int minR=minRow.get()-1;
    int minC=minCol.get()-1;
    DenseMatrix* dm = scinew DenseMatrix(maxR-minR+1, maxC-minC+1);
    for (int r=0; r<=maxR-minR; r++) {
	for (int c=0; c<=maxC-minC; c++)
	    dm->put(r, c, m->get(r+minR, c+minC));
    }
    return dm;
}

void ExtractSubmatrix::plot_rect() {

cerr << "In plot_rect()\n";
    int xres, yres, nrows, ncols, haveMat;
    if (!makeCurrent(xres, yres, nrows, ncols, haveMat)) return;

    // if the handle was empty, just flush the buffer (to clear the window)
    //    and return.
    if (!haveMat) {
	glFlush();
        glXMakeCurrent(dpy, None, NULL);
	TCLTask::unlock();
	return;
    }

    // plot the rect
    glLogicOp(GL_XOR);
    glEnable(GL_LOGIC_OP);
    glColor3d(1,1,1);

    if (eraseRect) { // if it was a move, "erase" our last rectangle
	glBegin(GL_LINE_STRIP);
	glVertex2i(lastMinC-1, lastMinR-1);
	glVertex2i(lastMinC-1, lastMaxR);
	glVertex2i(lastMaxC, lastMaxR);
	glVertex2i(lastMaxC, lastMinR-1);
	glVertex2i(lastMinC-1, lastMinR-1);
	glEnd();
	eraseRect=0;
    }	

    // now, draw the current rectangle
    if (drawRect) {
	glBegin(GL_LINE_STRIP);
	glVertex2i(minCol.get()-1, minRow.get()-1);
	glVertex2i(minCol.get()-1, maxRow.get());
	glVertex2i(maxCol.get(), maxRow.get());
	glVertex2i(maxCol.get(), minRow.get()-1);
	glVertex2i(minCol.get()-1, minRow.get()-1);
	glEnd();
	drawRect=0;
    }

    // and save the location that we just drew at
    lastMinC=minCol.get();
    lastMaxC=maxCol.get();
    lastMinR=minRow.get();
    lastMaxR=maxRow.get();
    glDisable(GL_LOGIC_OP);
    glFlush();
    int errcode;
    while((errcode=glGetError()) != GL_NO_ERROR){
	cerr << "plot_rect got an error from GL: " << (char*)gluErrorString(errcode) << endl;
    }
    glXMakeCurrent(dpy, None, NULL);
    TCLTask::unlock();
}

void ExtractSubmatrix::plot_matrices() {

cerr << "In plot_matrices()\n";
    int xres, yres, nrows, ncols, haveMat;
    if (!makeCurrent(xres, yres, nrows, ncols, haveMat)) return;
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    // if the handle was empty, just flush the buffer (to clear the window)
    //    and return.
    if (!haveMat) {
	glFlush();
        glXMakeCurrent(dpy, None, NULL);
	TCLTask::unlock();
	return;
    }

    // plot each matrix's points
    int ps=(xres/nrows)+1;
    glPointSize(ps);
    glBegin(GL_POINTS);
    glColor3d(1, 0, 0);	// red
    Array1<int> idx;
    Array1<double> val;
    for (int r=0; r<nrows; r++) {
	mH->getRowNonzeros(r, idx, val);
	if (ps != 1) {
	    for (int c=0; c<idx.size(); c++) {
		glVertex2f(idx[c]+.5, r+.5);
	    }
	} else {
	    for (int c=0; c<idx.size(); c++) {
		glVertex2i(idx[c], r);
	    }
	}
    }
    glEnd();
    glFlush();
    int errcode;
    while((errcode=glGetError()) != GL_NO_ERROR){
	cerr << "plot_matrices got an error from GL: " << (char*)gluErrorString(errcode) << endl;
    }
    glXMakeCurrent(dpy, None, NULL);
    TCLTask::unlock();
}

void ExtractSubmatrix::tcl_command(TCLArgs& args, void* userdata) {
    if (args[1] == "redrawRect") {
	reset_vars();	
	if (always.get()) {
	    want_to_execute();
	} else {
	    eraseRect=1;
	    drawRect=1;
	    plot_rect();
	}
    } else if (args[1] == "redrawMatrices") {
	reset_vars();
	plot_matrices();
	drawRect=1;
	plot_rect();
    } else if (args[1] == "send") {
	sendFlag=1;
	want_to_execute();
    } else {
        Module::tcl_command(args, userdata);
    }
}

int ExtractSubmatrix::makeCurrent(int &xres, int &yres, int &nrows,
				  int &ncols, int &haveMat) {
    haveMat=1;
    if (!mH.get_rep()) haveMat=0;

    if (haveMat) {
	nrows=mH->nrows();
	ncols=mH->ncols();
    } else {
	nrows=0;
	ncols=0;
    }
    TCLTask::lock();
//    if(!tkwin){
	clString myname(clString(".ui")+id+".submatrix.gl");
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
//    }
    
    // Get the window size
    xres=Tk_Width(tkwin);
    yres=Tk_Height(tkwin);

//    cerr << "*xres="<<xres<<"  yres="<<yres<<"\n";
    // Make ourselves current
//    if(current_drawer != this){
        current_drawer=this;
    if (!glXMakeCurrent(dpy, win, cx))
//	    cerr << "*glXMakeCurrent succeeded.\n";
//	else
	    cerr << "*glXMakeCurrent failed.\n";
//	cerr << "***cx=" << cx << endl;
//    }
    

    // Clear the screen...
    glViewport(0, 0, xres-1, yres-1);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, nrows, ncols, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    return 1;
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:57:45  mcq
// Initial commit
//
// Revision 1.3  1999/07/07 21:10:23  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/04/27 22:57:50  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//
