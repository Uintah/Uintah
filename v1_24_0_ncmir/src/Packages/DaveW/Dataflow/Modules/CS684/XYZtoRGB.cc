/*
 *  XYZtoRGB.cc:  Convert and XYZ image to an RGB image
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Packages/DaveW/Core/Datatypes/CS684/ImageR.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array2.h>
#include <Core/Containers/String.h>
#include <Dataflow/Ports/VoidStarPort.h>
#include <Core/Datatypes/VoidStar.h>
#define Colormap XColormap
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include <Core/Geom/GeomOpenGL.h>
#include <tcl.h>
#include <tk.h>
#undef Colormap
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/Color.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Persistent/Pstreams.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/TCL.h>

#include <iostream>
using std::cerr;
using std::endl;
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#undef MAX

extern Tcl_Interp* the_interp;
extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);

namespace DaveW {
using namespace DaveW;
using namespace SCIRun;

class XYZtoRGB : public Module {
    VoidStarIPort *iXYZ;
    int xyzGen;

    VoidStarHandle xyzHandle;
    ImageXYZ* xyz;
    Array1<unsigned char> image;

    Tk_Window tkwin;
    Window win;
    Display* dpy;
    GLXContext cx;

    GuiDouble WX, WY, RX, RY, GX, GY, BX, BY, MAX;
    int tcl_exec;
    int init;
    int NX;
    int NY;
public:
    void parallel_raytrace(int proc);

    XYZtoRGB(const clString& id);
    virtual ~XYZtoRGB();
    virtual void execute();
    void tcl_command( TCLArgs&, void * );
    void makeRGB();
    void redraw();
    int makeCurrent();
};

static XYZtoRGB* current_drawer=0;

extern "C" Module* make_XYZtoRGB(const clString& id)
{
    return scinew XYZtoRGB(id);
}

static clString module_name("XYZtoRGB");

XYZtoRGB::XYZtoRGB(const clString& id)
: Module("XYZtoRGB", id, Source), 
  WX("wx", id, this), WY("wy", id, this), 
  RX("rx", id, this), RY("ry", id, this), 
  GX("gx", id, this), GY("gy", id, this), 
  BX("bx", id, this), BY("by", id, this), 
  MAX("max", id, this)
{
    // Create the input port
    iXYZ = scinew VoidStarIPort(this, "ImageXYZ", VoidStarIPort::Atomic);
    add_iport(iXYZ);
    NX=NY=128;
    image.resize(128*128*3);
    image.initialize(20);
    init=1;
}

XYZtoRGB::~XYZtoRGB()
{
}

void XYZtoRGB::execute()
{
    VoidStarHandle xyzHandle;
    iXYZ->get(xyzHandle);
    if (!xyzHandle.get_rep()) return;
    if (!(xyz = dynamic_cast<ImageXYZ *>(xyzHandle.get_rep()))) return;

    NY=xyz->xyz.dim1();
    NX=xyz->xyz.dim2()/3;

    // New scene -- have to build widgets
    if (xyz->generation != xyzGen) {
	image.resize(NY*NX*3);
	tcl_exec=1;
    }

    // Only ~really~ raytrace if tcl said to
    if (tcl_exec) {
	reset_vars();
	makeRGB();
	redraw();
    }
}

void mat3x3invert(double c[3][3], double inv[3][3]) {
    double denom=c[0][0]*c[1][1]*c[2][2] - 
	c[0][0]*c[1][2]*c[2][1]-
	    c[1][0]*c[0][1]*c[2][2]+
		c[1][0]*c[0][2]*c[2][1]+
		    c[2][0]*c[0][1]*c[1][2]-
			c[2][0]*c[0][2]*c[1][1];
    if (Abs(denom)<0.00001) {
	cerr << "Denom too small!\n";
    }
    double denomInv=1./denom;
    inv[0][0]=(c[1][1]*c[2][2]-c[1][2]*c[2][1])/denomInv;
    inv[0][1]=(c[0][1]*c[2][2]-c[0][2]*c[2][1])/denomInv;
    inv[0][2]=(c[0][1]*c[1][2]-c[0][2]*c[1][1])/denomInv;
    inv[1][0]=(c[1][2]*c[2][0]-c[1][0]*c[2][2])/denomInv;
    inv[1][1]=(c[0][0]*c[2][2]-c[0][2]*c[2][0])/denomInv;
    inv[1][2]=(c[0][2]*c[1][0]-c[0][0]*c[1][2])/denomInv;
    inv[2][0]=(c[1][0]*c[2][1]-c[1][1]*c[2][0])/denomInv;
    inv[2][1]=(c[0][1]*c[2][0]-c[0][0]*c[2][1])/denomInv;
    inv[2][2]=(c[0][0]*c[1][1]-c[0][1]*c[1][0])/denomInv;
}
	    
void XYZtoRGB::makeRGB() {
    double toRGB[3][3];
    double wY;
    wY=1./MAX.get();

    double toXYZ[3][3];
    double Cr, Cg, Cb;
    double wx, wy, rx, ry, gx, gy, bx, by;
    double YoveryD;

    wy=WY.get(); wx=WX.get(); rx=RX.get(); ry=RY.get();
    gx=GX.get(); gy=GY.get(); bx=BX.get(); by=BY.get();

    double D=(rx * (gy - by) + gx * (by - ry) + bx * (ry - gy));
//    YoveryD=(wY/wy)/D;
    YoveryD=(1./wy)/D;
    Cr=YoveryD  * (wx * (gy - by) - wy * (gx - bx) +
		     gx * by - bx * gy);
    Cg=YoveryD  * (wx * (by - ry) - wy * (bx - rx) +
		     bx * ry - rx * by);
    Cb=YoveryD  * (wx * (ry - gy) - wy * (rx - gx) +
		     rx * gy - gx * ry);
    toXYZ[0][0] = rx * Cr;   toXYZ[0][1] = gx * Cg;   toXYZ[0][2] = bx * Cb; 
    toXYZ[1][0] = ry * Cr;   toXYZ[1][1] = gy * Cg;   toXYZ[1][2] = by * Cb; 
    toXYZ[2][0] = (1.0 - (rx + ry)) * Cr;
    toXYZ[2][1] = (1.0 - (gx + gy)) * Cg;
    toXYZ[2][2] = (1.0 - (bx + by)) * Cb; 

#if 0
    toXYZ[0][0] = rx;
    toXYZ[0][1] = gx;
    toXYZ[0][2] = bx;
    toXYZ[1][0] = ry;
    toXYZ[1][1] = gy;
    toXYZ[1][2] = by;
    toXYZ[2][0] = 1-(rx+ry);
    toXYZ[2][1] = 1-(gx+gy);
    toXYZ[2][2] = 1-(bx+by);
#endif

    mat3x3invert(toXYZ, toRGB);

    cerr << "XYZ->RGB matrix: \n   ";
    for (int ii=0; ii<3; ii++)
	for (int jj=0; jj<3; jj++)
	    cerr << toRGB[ii][jj] <<"  ";
	cerr << "\n   ";
    cerr << "\n";

    int count=0;
    for (int i=0; i<NY; i++) {
	for (int j=0; j<NX*3; j+=3, count+=3) {
	    for (int k=0; k<3; k++) {
		double v=toRGB[k][0]*xyz->xyz(i,j)+
		    toRGB[k][1]*xyz->xyz(i,j+1)+
		    toRGB[k][2]*xyz->xyz(i,j+2);
		image[count+k]=(int) Max(0.0, Min(255.0, v*255*wY));
	    }
	}
    }
}

void XYZtoRGB::redraw() {
    makeCurrent();
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glDrawPixels(NX, NY, GL_RGB, GL_UNSIGNED_BYTE, 
		 &(image[0]));
    glXSwapBuffers(dpy,win);
    int errcode;
    while((errcode=glGetError()) != GL_NO_ERROR){
	cerr << "XYZtoRGB got an error from GL: " << (char*)gluErrorString(errcode) << endl;
    }
    glXMakeCurrent(dpy, None, NULL);
    TCLTask::unlock();
}	    

void XYZtoRGB::tcl_command(TCLArgs& args, void* userdata) {
    if (args[1] == "redraw") {
	reset_vars();
	redraw();
    } else if (args[1] == "tcl_exec") {
	tcl_exec=1;
	want_to_execute();
    } else if (args[1] == "save") {
	FILE *f=fopen("/tmp/rgbImage.raw", "wb");
	if(!f) {
	    cerr << "Error opening /tmp/rgbImage.raw\n";
	    return;
	}
	for (int i=0; i<NX*NY*3; i+=3) {
	    fprintf(f,"%c%c%c", image[i], image[i+1], image[i+2]);
	}
	fclose(f);
	char scall[200];
	sprintf(scall, "rawtorle -w %d -h %d -n 3 /tmp/rgbImage.raw -o image.rle\n", NX, NY);
	cerr << "Call system: "<<scall;
//	system(scall);
    } else {
        Module::tcl_command(args, userdata);
    }
}

int XYZtoRGB::makeCurrent() {
    TCLTask::lock();
    clString myname(clString(".ui")+id+".gl.gl");
    char *myn=strdup(myname());
    tkwin=Tk_NameToWindow(the_interp, myn, Tk_MainWindow(the_interp));
    if(!tkwin){
	cerr << "Unable to locate window!\n";
	glXMakeCurrent(dpy, None, NULL);
	TCLTask::unlock();
	return 0;
    }
    dpy=Tk_Display(tkwin);
    win=Tk_WindowId(tkwin);
    cx=OpenGLGetContext(the_interp, myn);
    if(!cx){
	cerr << "Unable to create OpenGL Context!\n";
	glXMakeCurrent(dpy, None, NULL);
	TCLTask::unlock();
	return 0;
    }
    current_drawer=this;
    if (!glXMakeCurrent(dpy, win, cx))
	    cerr << "*glXMakeCurrent failed.\n";

    // Clear the screen...
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-.5, 511.5, -.5, 511.5, -1, 1);
    glViewport(0, 0, 512, 512);
    glClearColor(.2,.2,.2,0);
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glRasterPos2d(256-NX/2, 256-NY/2);
    return 1;
} // End namespace DaveW
}

// $Log
