//static char *id="@(#) $Id$";

/*
 *  VisualizeMatrix.cc:  Visual matrix editor
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   November 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <SCICore/Util/NotFinished.h>
#include <SCICore/Containers/String.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/CoreDatatypes/DenseMatrix.h>
#include <SCICore/CoreDatatypes/Matrix.h>
#include <PSECore/CommonDatatypes/MatrixPort.h>
#include <SCICore/CoreDatatypes/SymSparseRowMatrix.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Geom/GeomOpenGL.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/TclInterface/TCL.h>
#include <tcl.h>
#include <tk.h>
#include <iostream.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#define SNOOP_SIZE 40

// Everything will be done in C++ -- the tcl UI will just be used for display
// (i.e. all the flags and the OpenGL drawing will be done from here).
// tcl will just handle the toggles for which matrices are being graphed

extern Tcl_Interp* the_interp;
extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using SCICore::Containers::to_string;

class VisualizeMatrix : public Module {
    TCLint numMatrices;  // number of matrix iports
    Array1<TCLstring *> nrow;
    Array1<TCLstring *> ncol;
    Array1<TCLstring *> type;
    Array1<TCLstring *> density;
    Array1<TCLstring *> condition;
    Array1<TCLstring *> symm;
    Array1<TCLstring *> posdef;
    Array1<TCLint *> scale;
    Array1<TCLint *> shown;
    Array1<Color> ctable;
    Array1<MatrixIPort *> iport;
    Array1<MatrixHandle> matHandle;
    
    TCLint snoopX;
    TCLint snoopY;
    TCLint snoopOn;
    TCLstring snoopRender;

    int snoopWidth;
    GLXContext lastContext;
    double elementsPerPixel;
    double lastSnX1;
    double lastSnX2;
    double lastSnY1;
    double lastSnY2;

//    int tcl_requested;
//    int have_ever_executed;
    int tcl_requested;
    int maxRows;
    int maxCols;
    
    int drawRect;
    int eraseRect;

    GLuint base;

    clString myid;
    Tk_Window tkwin;
    Window win;
    Display* dpy;
    GLXContext cx;
public:
    VisualizeMatrix(const clString& id);
    VisualizeMatrix(const VisualizeMatrix&, int deep);
    virtual ~VisualizeMatrix();
    virtual Module* clone(int deep);
    virtual void connection(Module::ConnectionMode, int, int);
    virtual void execute();
    void tcl_command( TCLArgs&, void * );
    void plot_matrices();
    void plot_snoop();
    void plot_snoop_rect();
    void print_it();
    void makeRasterFont(void);
    void printString(char *s);
};

Module* make_VisualizeMatrix(const clString& id) {
  return new VisualizeMatrix(id);
}

VisualizeMatrix::VisualizeMatrix(const clString& id)
: Module("VisualizeMatrix", id, Source), numMatrices("numMatrices", id, this),
  tcl_requested(0), drawRect(0), eraseRect(0), tkwin(0), 
  snoopX("snoopX", id, this), snoopY("snoopY", id, this), 
  snoopOn("snoopOn", id, this), lastContext(0),
  snoopRender("snoopRender", id, this)
{
    // Create the input port
    myid=id;
//    numMatrices.set(0);
    MatrixIPort *mi = scinew MatrixIPort(this, "Matrix", MatrixIPort::Atomic);
    iport.add(mi);
    add_iport(mi);
    ctable.add(Color(1, 0, 0));
    ctable.add(Color(0, 1, 0));
    ctable.add(Color(0, 0, 1));
    ctable.add(Color(1, 1, 0));
    ctable.add(Color(1, 0, 1));
    ctable.add(Color(0, 1, 1));
    ctable.add(Color(.7, .3, 0));
}

VisualizeMatrix::VisualizeMatrix(const VisualizeMatrix& copy, int deep)
: Module(copy, deep), numMatrices("numMatrices", id, this),
  tcl_requested(0), drawRect(0), eraseRect(0), tkwin(0),
  snoopX("snoopX", id, this), snoopY("snoopY", id, this), 
  snoopOn("snoopOn", id, this), lastContext(0),
  snoopRender("snoopRender", id, this)
{
    NOT_FINISHED("VisualizeMatrix::VisualizeMatrix");
}

VisualizeMatrix::~VisualizeMatrix()
{
}

Module* VisualizeMatrix::clone(int deep)
{
    return scinew VisualizeMatrix(*this, deep);
}

void VisualizeMatrix::connection(ConnectionMode mode, int which_port,
				 int output)
{
    if (output) {
	error("No output ports... how was an output connection made??");
	return;
    }
//    if (have_ever_executed) {
//        NOT_FINISHED("Can't add new ports after having executed... sorry!\n");
//        return;
//    }
    if (mode==Disconnected) {
        numMatrices.set(numMatrices.get()-1);
        remove_iport(which_port);
        delete iport[which_port];
        iport.remove(which_port);
    } else {
        numMatrices.set(numMatrices.get()+1);
	MatrixIPort *mi = scinew MatrixIPort(this, "Matrix", 
					     MatrixIPort::Atomic);
	iport.add(mi);
	add_iport(mi);
    }
    want_to_execute();
}
        
void VisualizeMatrix::execute()
{
    int i;
    int last_matsize=matHandle.size();

    for (i=0; i<matHandle.size(); i++) {
	matHandle[i]=0;
    }
    matHandle.resize(numMatrices.get());
    for (i=0; i<numMatrices.get(); i++) {
	iport[i]->get(matHandle[i]);
    }

    // make sure we have the right number of variables allocated
    if (numMatrices.get() > last_matsize) {
	int j;
	for (j=last_matsize; j<numMatrices.get(); j++) {
	    clString dummy;
	    dummy = "nrow"+to_string(j);
	    nrow.add(scinew TCLstring(dummy, myid, this));
	    dummy =  "ncol"+to_string(j);
	    ncol.add(scinew TCLstring(dummy, myid, this));
	    dummy = "density"+to_string(j);
	    density.add(scinew TCLstring(dummy, myid, this));
	    dummy = "scale"+to_string(j);
	    scale.add(scinew TCLint(dummy, myid, this));
	    scale[j]->set(1);
	    dummy = "shown"+to_string(j);
	    shown.add(scinew TCLint(dummy, myid, this));
	    shown[j]->set(1);
	    dummy = "symm"+to_string(j);
	    symm.add(scinew TCLstring(dummy, myid, this));
	    dummy = "posdef"+to_string(j);
	    posdef.add(scinew TCLstring(dummy, myid, this));
	    dummy = "condition"+to_string(j);
	    condition.add(scinew TCLstring(dummy, myid, this));
	    dummy = "type"+to_string(j);
	    type.add(scinew TCLstring(dummy, myid, this));
	}		     
    }
    if (numMatrices.get() < last_matsize) {
	int j;
	for (j=last_matsize; j>numMatrices.get(); j--) {
	    free (nrow[j]);
	    free (ncol[j]);
	    free (density[j]);
	    free (scale[j]);
	    free (shown[j]);
	    free (symm[j]);
	    free (posdef[j]);
	    free (condition[j]);
	    free (type[j]);
	}
	nrow.resize(j);
	ncol.resize(j);
	density.resize(j);
	scale.resize(j);
	shown.resize(j);
	symm.resize(j);
	posdef.resize(j);
	condition.resize(j);
	type.resize(j);
    }

    // now go through and assign all those variables to have the right values
    for (i=0; i<numMatrices.get(); i++) {
	int Nrow, Ncol;
	double Sparseness, Condition;
	clString Type="Dense";
	clString Symm="Symmetric";
	clString Posdef="Positive Definite";
	Nrow=Ncol=Sparseness=Condition=0;
	if (matHandle[i].get_rep()) {
	    Matrix *m = matHandle[i].get_rep();
	    Nrow=m->nrows();
	    Ncol=m->ncols();
//	    if (!m->is_symmetric) Symm="Not Symmetric";
//	    if (!m->is_posdef) Posdef="Not Positive Definite";
//	    Condition=m->condition();
	    if (m->getDense()) {
		Type="Dense";
		Sparseness=1.;
		Condition=0;	// gotta get a fast way to compute this!
	    } else {
		Type="Sparse";	
		Sparseness=1./(m->getSymSparseRow())->density();
		// fast way...
		if (Symm=="Symmetric" && Posdef=="Positive Definite") {	
		    double minD, maxD;
		    minD=maxD=m->get(0,0);
		    for (int r=0; r<m->nrows(); r++) {
			double x=m->get(r,r);
			if (x>maxD) maxD=x;
			if (x<minD) minD=x;
		    }
		    Condition=maxD/minD;
		} else {
		    Condition=0;	// need a fast way here too!
		}
	    }
	}
	nrow[i]->set(to_string(Nrow));
	ncol[i]->set(to_string(Ncol));
	symm[i]->set(Symm);
	posdef[i]->set(Posdef);
	density[i]->set(to_string(Sparseness));
	condition[i]->set(to_string(Condition));
	type[i]->set(Type);
    }
    plot_matrices();
    plot_snoop();	
    if (snoopOn.get()) {
	drawRect=1;
	plot_snoop_rect();
    }
}

void VisualizeMatrix::makeRasterFont(void)
{
    if (cx && (cx != lastContext)) {
	lastContext=cx;
    } else {
	return;
    }
//cerr << "Generating new raster font list!\n";
    XFontStruct *fontInfo;
    Font id;
    unsigned int first, last;
    Display *xdisplay;

    xdisplay = dpy;
    fontInfo = XLoadQueryFont(xdisplay, 
        "-*-helvetica-medium-r-normal--12-*-*-*-p-67-iso8859-1");
    if (fontInfo == NULL) {
        printf ("no font found\n");
        exit (0);
    }

    id = fontInfo->fid;
    first = fontInfo->min_char_or_byte2;
    last = fontInfo->max_char_or_byte2;

    base = glGenLists((GLuint) last+1);
    if (base == 0) {
        printf ("out of display lists\n");
        exit (0);
    }
    glXUseXFont(id, first, last-first+1, base+first);
/*    *height = fontInfo->ascent + fontInfo->descent;
    *width = fontInfo->max_bounds.width;  */
}

void VisualizeMatrix::printString(char *s)
{
    glPushAttrib (GL_LIST_BIT);
    glListBase(base);
    glCallLists(strlen(s), GL_UNSIGNED_BYTE, (GLubyte *)s);
    glPopAttrib ();
}

void VisualizeMatrix::print_it() {
    cerr << "C REPORT: \n";
    cerr << "NumMatrices: "<<numMatrices.get()<<"\n";
    for (int i=0; i<numMatrices.get(); i++) {
	cerr << "Matrix "<<i<<"\n";
	cerr << "  nrow: "<<nrow[i]->get()<<"\n";
	cerr << "  ncol: "<<ncol[i]->get()<<"\n";
	cerr << "  type: "<<type[i]->get()<<"\n";
	cerr << "  density: "<<density[i]->get()<<"\n";
	cerr << "  condition: "<<condition[i]->get()<<"\n";
	cerr << "  symm: "<<symm[i]->get()<<"\n";
	cerr << "  posdef: "<<posdef[i]->get()<<"\n";
	cerr << "  scale: "<<scale[i]->get()<<"\n";
	cerr << "  shown: "<<shown[i]->get()<<"\n";
    }    
}

void VisualizeMatrix::tcl_command(TCLArgs& args, void* userdata) {
//    cerr << "snoopOn.get="<<snoopOn.get()<<"\n";
    if (args[1] == "redrawSnoop") {
	reset_vars();
	if (snoopOn.get()) {
	    drawRect=1;
	    eraseRect=1;
	    plot_snoop_rect();
	}
	plot_snoop();
    } else if (args[1] == "snoop_dying") {
	lastContext=0;
    } else if (args[1] == "redrawMatrices") {
	reset_vars();
	plot_matrices();
	if (snoopOn.get()) {
	    drawRect=1;
	    plot_snoop_rect();
	}
    } else if (args[1] == "redrawSnoopRect") {
	reset_vars();
	if (snoopOn.get()) {
	    drawRect=1;
	    plot_snoop_rect();
	} else {
	    eraseRect=1;
	    plot_snoop_rect();
	}
    } else {
        Module::tcl_command(args, userdata);
    }
}

void VisualizeMatrix::plot_snoop() {
//    cerr << "In plot_snoop\n";
    TCLTask::lock();
//    if(!tkwin){
	clString myname(clString(".ui")+id+".plot.snoop.gl");
	tkwin=Tk_NameToWindow(the_interp, 
			      const_cast<char *>(myname()),
			      Tk_MainWindow(the_interp));
	if(!tkwin){
	    cerr << "Unable to locate window!\n";
	    TCLTask::unlock();
	    return;
	}
	dpy=Tk_Display(tkwin);
	win=Tk_WindowId(tkwin);
	cx=OpenGLGetContext(the_interp, const_cast<char *>(myname()));
	if(!cx){
	    cerr << "Unable to create OpenGL Context!\n";
	    TCLTask::unlock();
	    return;
	}
//    }
    
    // Get the window size
    int xres=Tk_Width(tkwin);
    int yres=Tk_Height(tkwin);
    
    
//    cerr << "*xres="<<xres<<"  yres="<<yres<<"\n";
    // Make ourselves current
    if (!glXMakeCurrent(dpy, win, cx))
//	    cerr << "*glXMakeCurrent succeeded.\n";
//	else
	    cerr << "*glXMakeCurrent failed.\n";
//	cerr << "***cx=" << cx << endl;
//    }
    

    // Clear the screen...
    glViewport(0, 0, xres, yres);
    glClearColor(1, 1, 1, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    // plot each matrix's points
    int i;
    maxCols=0;
    maxRows=0;
    for (i=0; i<numMatrices.get(); i++) {
	if (shown[i]->get()) {	// find max and min of only the shown matrices
	    if (!matHandle[i].get_rep()) continue;
	    if (matHandle[i]->nrows()>maxRows) maxRows=matHandle[i]->nrows();
	    if (matHandle[i]->ncols()>maxCols) maxCols=matHandle[i]->ncols();
	}	
    }
//cerr << "plot_snoop maxRows="<<maxRows<<"\n";
    if (!maxRows || !maxCols) {
        glXMakeCurrent(dpy, None, NULL);
	TCLTask::unlock();
	return;
    }

    if (maxRows<SNOOP_SIZE)
	snoopWidth=maxRows/2;
    else
	snoopWidth=SNOOP_SIZE/2;
    int newSnX1=snoopX.get()*elementsPerPixel - snoopWidth;
    int newSnY1=snoopY.get()*elementsPerPixel - snoopWidth;
    int newSnX2=snoopX.get()*elementsPerPixel + snoopWidth;
    int newSnY2=snoopY.get()*elementsPerPixel + snoopWidth;
    if (newSnX1<0) {
	newSnX2-=newSnX1;	// it's negative, so subtracting it is addative
	newSnX1=0;
    }
    if (newSnY1<0) {
	newSnY2-=newSnY1;	// it's negative, so subtracting it is addative
	newSnY1=0;
    }
    if (newSnX2>maxRows) {
	newSnX1-=(newSnX2-maxRows-1);
	newSnX2=maxRows-1;
    }
    if (newSnY2>maxRows) {
	newSnY1-=(newSnY2-maxRows-1);
	newSnY2=maxRows-1;
    }
//    cerr << "Snoop rect dims: ("<<newSnX1<<", "<<newSnY1<<") (";
//    cerr << newSnX2 << ", "<<newSnY2<<")\n";
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(newSnX1, newSnX2, newSnY2, newSnY1, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glPointSize(4);
//    glBegin(GL_POINTS);
    makeRasterFont();

    clString rndMode(snoopRender.get());
    if (rndMode == "scaled_dots" || rndMode == "uniform_dots") {
	glBegin(GL_POINTS);
    }
    for (i=0; i<numMatrices.get(); i++) {
	if (i>=7) break;
	Matrix *m=matHandle[i].get_rep();
	if (!m) continue;
	if (!(shown[i]->get())) continue;
	if (rndMode == "uniform_dots" || rndMode == "uniform_text") {
	    glColor3d(ctable[i].r(), ctable[i].g(), ctable[i].b());
	}
	double maxVal, minVal;
	if (rndMode == "scaled_dots" || rndMode == "scaled_text") {
	    maxVal = m->maxValue();
	    minVal = m->minValue();
	}
//	cerr << "maxVal="<<maxVal<<"   minVal="<<minVal<<"\n";
	Array1<int> idx;
	Array1<double> val;
	double ysc, xsc;
	if (scale[i]->get()) {
	    ysc=maxRows*1./m->nrows();
	    xsc=maxCols*1./m->ncols();		
	} else {
	    ysc=xsc=1;
	}
//	if (matHandle[i]->nrows()>maxRows) maxRows=matHandle[i]->nrows();
	for (int r=newSnY1/ysc; r<m->nrows() && r<newSnY2/ysc; r++) {
	    m->getRowNonzeros(r, idx, val);
	    for (int c=0; c<idx.size(); c++) {
		if (idx[c]>=newSnX1/xsc && idx[c]<=newSnX2/xsc) {
		    if (rndMode == "scaled_dots" || rndMode == "scaled_text") {
			double lum=(val[c]-minVal)/(maxVal-minVal);
			glColor3d(ctable[i].r()*lum,
				  ctable[i].g()*lum,
				  ctable[i].b()*lum);
		    }	
		    if (rndMode == "scaled_text" || rndMode == "uniform_text"){
			glRasterPos2i(idx[c]*xsc, r*ysc);
			char s[20];
			sprintf(s, "%.5g", val[c]);
			printString(s);
		    }
		    if (rndMode == "uniform_dots" || rndMode == "scaled_dots"){
			glVertex2i(idx[c]*xsc, r*ysc);
		    }
		}	
	    }	
	}
    }
    if (rndMode == "scaled_dots" || rndMode == "uniform_dots") {
	glEnd();	// GL_POINTS
    }
    glFlush();
    glPointSize(1);
    int errcode;
    while((errcode=glGetError()) != GL_NO_ERROR){
	cerr << "Snoop got an error from GL: " << (char*)gluErrorString(errcode) << endl;
    }
    glXMakeCurrent(dpy, None, NULL);
    TCLTask::unlock();
}

void VisualizeMatrix::plot_snoop_rect() {
//    cerr << "In plot_snoop_rect\n";
    TCLTask::lock();
//    if(!tkwin){
	clString myname(clString(".ui")+id+".plot.gl");
	tkwin=Tk_NameToWindow(the_interp,
			      const_cast<char *>(myname()),
			      Tk_MainWindow(the_interp));
	if(!tkwin){
	    cerr << "Unable to locate window!\n";
	    TCLTask::unlock();
	    return;
	}
	dpy=Tk_Display(tkwin);
	win=Tk_WindowId(tkwin);
	cx=OpenGLGetContext(the_interp, const_cast<char *>(myname()));
	if(!cx){
	    cerr << "Unable to create OpenGL Context!\n";
	    TCLTask::unlock();
	    return;
	}
//    }
    
    // Get the window size
    int xres=Tk_Width(tkwin);
    int yres=Tk_Height(tkwin);

//    cerr << "*xres="<<xres<<"  yres="<<yres<<"\n";
    // Make ourselves current
    if (!glXMakeCurrent(dpy, win, cx))
//	    cerr << "*glXMakeCurrent succeeded.\n";
//	else
	    cerr << "*glXMakeCurrent failed.\n";
//	cerr << "***cx=" << cx << endl;
//    }
    

//cerr << "plot_snoop_rect maxRows="<<maxRows<<"\n";
    if (!maxRows || !maxCols) {
	TCLTask::unlock();
        glXMakeCurrent(dpy, None, NULL);
	return;
    }

    glViewport(0, 0, xres, yres);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, maxRows, maxCols, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glLogicOp(GL_XOR);
    glEnable(GL_LOGIC_OP);
    glColor3d(1,1,1);

    if (maxRows<SNOOP_SIZE)
	snoopWidth=maxRows/2;
    else
	snoopWidth=SNOOP_SIZE/2;
    int newSnX1=snoopX.get()*elementsPerPixel - snoopWidth;
    int newSnY1=snoopY.get()*elementsPerPixel - snoopWidth;
    int newSnX2=snoopX.get()*elementsPerPixel + snoopWidth;
    int newSnY2=snoopY.get()*elementsPerPixel + snoopWidth;
    if (newSnX1<0) {
	newSnX2-=newSnX1;	// it's negative, so subtracting it is addative
	newSnX1=0;
    }
    if (newSnY1<0) {
	newSnY2-=newSnY1;	// it's negative, so subtracting it is addative
	newSnY1=0;
    }
    if (newSnX2>maxRows) {
	newSnX1-=(newSnX2-maxRows-1);
	newSnX2=maxRows-1;
    }
    if (newSnY2>maxRows) {
	newSnY1-=(newSnY2-maxRows-1);
	newSnY2=maxRows-1;
    }
    if (eraseRect) { // if it was a move, "erase" our last rectangle
	glBegin(GL_LINE_STRIP);
	glVertex2i(lastSnX1, lastSnY1);
	glVertex2i(lastSnX1, lastSnY2);
	glVertex2i(lastSnX2, lastSnY2);
	glVertex2i(lastSnX2, lastSnY1);
	glVertex2i(lastSnX1, lastSnY1);
	glEnd();
//	cerr << "Erased snoop rect at: ("<<lastSnX1<<", "<<lastSnY1<<") (";
//	cerr << lastSnX2 << ", "<<lastSnY2<<")\n";
	eraseRect=0;
    }	

    // now, draw the current rectangle
    if (drawRect) {
	glBegin(GL_LINE_STRIP);
	glVertex2i(newSnX1, newSnY1);
	glVertex2i(newSnX1, newSnY2);
	glVertex2i(newSnX2, newSnY2);
	glVertex2i(newSnX2, newSnY1);
	glVertex2i(newSnX1, newSnY1);
	glEnd();
//	cerr << "Drew snoop rect at: ("<<newSnX1<<", "<<newSnY1<<") (";
//	cerr << newSnX2 << ", "<<newSnY2<<")\n";
	drawRect=0;
    }

    // and save the location that we just drew at
    lastSnX1=newSnX1;
    lastSnX2=newSnX2;
    lastSnY1=newSnY1;
    lastSnY2=newSnY2;
    glDisable(GL_LOGIC_OP);
    glFlush();
    int errcode;
    while((errcode=glGetError()) != GL_NO_ERROR){
	cerr << "Rect got an error from GL: " << (char*)gluErrorString(errcode) << endl;
    }
    glXMakeCurrent(dpy, None, NULL);
    TCLTask::unlock();
}

void VisualizeMatrix::plot_matrices() {
    TCLTask::lock();
//    if(!tkwin){
	clString myname(clString(".ui")+id+".plot.gl");
	tkwin=Tk_NameToWindow(the_interp,
			      const_cast<char *>(myname()),
			      Tk_MainWindow(the_interp));
	if(!tkwin){
	    cerr << "Unable to locate window!\n";
	    TCLTask::unlock();
	    return;
	}
	dpy=Tk_Display(tkwin);
	win=Tk_WindowId(tkwin);
	cx=OpenGLGetContext(the_interp, const_cast<char *>(myname()));
	if(!cx){
	    cerr << "Unable to create OpenGL Context!\n";
	    TCLTask::unlock();
	    return;
	}
//    }
    
    // Get the window size
    int xres=Tk_Width(tkwin);
    int yres=Tk_Height(tkwin);

//    cerr << "*xres="<<xres<<"  yres="<<yres<<"\n";
    // Make ourselves current
    if (!glXMakeCurrent(dpy, win, cx))
//	    cerr << "*glXMakeCurrent succeeded.\n";
//	else
	    cerr << "*glXMakeCurrent failed.\n";
//	cerr << "***cx=" << cx << endl;
//    }
    

    // Clear the screen...
    glViewport(0, 0, xres, yres);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    // plot each matrix's points
    int i;
    int maxCols=0;
    int maxRows=0;
    for (i=0; i<numMatrices.get(); i++) {
	if (shown[i]->get()) {	// find max and min of only the shown matrices
	    if (!matHandle[i].get_rep()) continue;
	    if (matHandle[i]->nrows()>maxRows) maxRows=matHandle[i]->nrows();
	    if (matHandle[i]->ncols()>maxCols) maxCols=matHandle[i]->ncols();
	}	
    }

    if (!maxRows || !maxCols) {
        glXMakeCurrent(dpy, None, NULL);
	TCLTask::unlock();
	return;
    }

    elementsPerPixel=maxRows*1./xres;
//    cerr << "rect elementsPerPixel="<<elementsPerPixel<<"\n";

    glViewport(0, 0, xres, yres);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, maxRows, maxCols, 0, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glBegin(GL_POINTS);
    for (i=0; i<numMatrices.get(); i++) {
	if (i>=7) break;
	Matrix *m=matHandle[i].get_rep();
	if (!m) continue;
	if (!(shown[i]->get())) continue;
	glColor3d(ctable[i].r(), ctable[i].g(), ctable[i].b());
	Array1<int> idx;
	Array1<double> val;
	if (!scale[i]->get()) {
	    for (int r=0; r<m->nrows(); r++) {
		m->getRowNonzeros(r, idx, val);
		for (int c=0; c<idx.size(); c++) {
		    glVertex2i(idx[c], r);
		}
//		for (int c=0; c<m->ncols(); c++) {
//		    if (m->get(r,c) != 0) {	// it's a non-zero, plot it
//			glVertex2i(r, c);
//		    }
//		}
	    }
	} else {
	    if (matHandle[i]->nrows()>maxRows) maxRows=matHandle[i]->nrows();
	    double ysc=maxRows*1./m->nrows();
	    double xsc=maxCols*1./m->ncols();		
	    for (int r=0; r<m->nrows(); r++) {
		m->getRowNonzeros(r, idx, val);
		for (int c=0; c<idx.size(); c++) {
		    glVertex2i(idx[c]*xsc, r*ysc);
		}
//		for (int c=0; c<m->ncols(); c++) {
//		    if (m->get(r,c) != 0) {	// it's a non-zero, plot it
//			glVertex2i(r*xsc, c*ysc);
//		    }
//		}
	    }	    
	}
    }
    glEnd();
    for (i=0; i<numMatrices.get(); i++) {
	if (i>=7) break;
	Matrix *m=matHandle[i].get_rep();
	if (!m) continue;
	if (!(shown[i]->get())) continue;
	if (scale[i]->get()) continue;
	glColor3d(ctable[i].r(), ctable[i].g(), ctable[i].b());
	glBegin(GL_LINE_STRIP);
	glVertex2i(0, m->nrows());
	glVertex2i(m->ncols(), m->nrows());
	glVertex2i(m->ncols(), 0);
	glEnd();
    }
    glFlush();
    int errcode;
    while((errcode=glGetError()) != GL_NO_ERROR){
	cerr << "Plot got an error from GL: " << (char*)gluErrorString(errcode) << endl;
    }
    glXMakeCurrent(dpy, None, NULL);
    TCLTask::unlock();
}

#if 0
void myinit (void) 
{
    makeRasterFont ();
    glShadeModel (GL_FLAT);    
}

void display(void)
{
    GLfloat white[3] = { 1.0, 1.0, 1.0 };
    int i, j;
    char teststring[33];

    glClear(GL_COLOR_BUFFER_BIT);
    glColor3fv(white);
    for (i = 32; i < 127; i += 32) {
        glRasterPos2i(20, 200 - 18*(GLint) i/32);
        for (j = 0; j < 32; j++)
            teststring[j] = (char) (i+j);
        teststring[32] = 0;
        printString(teststring);
    }
    glRasterPos2i(20, 100);
    printString("The quick brown fox jumps");
    glRasterPos2i(20, 82);
    printString("over a lazy dog.");
    glFlush ();
}

void myReshape(int w, int h)
{
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho (0.0, (GLfloat) w, 0.0, (GLfloat) h, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

/*  Main Loop
 *  Open window with initial window size, title bar, 
 *  RGBA display mode, and handle input events.
 */
#endif

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.2  1999/08/17 06:37:32  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:57:46  mcq
// Initial commit
//
// Revision 1.3  1999/07/07 21:10:24  dav
// added beginnings of support for g++ compilation
//
// Revision 1.2  1999/04/27 22:57:50  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:31  dav
// Import sources
//
//
