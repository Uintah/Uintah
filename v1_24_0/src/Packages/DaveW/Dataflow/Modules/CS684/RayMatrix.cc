
/*
 *  RayMatrix.cc:  Render and edit params of a Ray Matrix image
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Core/Containers/Array1.h>
#include <Core/Containers/Array2.h>
#include <Core/Containers/String.h>
#include <Dataflow/Ports/GeometryPort.h>
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
#include <Packages/DaveW/Core/Datatypes/CS684/ImageR.h>
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

extern Tcl_Interp* the_interp;
extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);

namespace DaveW {
using namespace DaveW;
using namespace SCIRun;

class RayMatrix : public Module {
    VoidStarIPort *iRM;
    VoidStarOPort *oXYZ;
    int rmGen;

    GuiString tclFName;
    GuiString tclSpectrum;
    GuiString tclLType;
    GuiDouble tclDiff;
    GuiDouble tclSpec;
    GuiDouble scale;
    GuiInt tclMin, tclMax, tclNum;

    VoidStarHandle rmHandle;
    VoidStarHandle xyzHandle;
    ImageRM* rm;
    ImageXYZ* xyz;
    Array1<unsigned char> image;

    Tk_Window tkwin;
    Window win;
    Display* dpy;
    GLXContext cx;

    clString tclCommand;
    Array1<clString> tclArgs;

    int init;
    int NX;
    int NY;

    int min, max, num;
    double spacing;
public:
    RayMatrix(const clString& id);
    virtual ~RayMatrix();
    virtual void execute();
    void tcl_command( TCLArgs&, void * );
    void makeRGB();
    void redraw();
    int makeCurrent();
};

static RayMatrix* current_drawer=0;

extern "C" Module* make_RayMatrix(const clString& id)
{
    return scinew RayMatrix(id);
}

static clString module_name("RayMatrix");

RayMatrix::RayMatrix(const clString& id)
: Module("RayMatrix", id, Source), tclFName("tclFname", id, this),
  tclSpectrum("tclSpectrum", id, this), tclLType("tclLType", id, this),
  tclDiff("tclDiff", id, this), tclSpec("tclSpec", id, this),
  tclMin("tclMin", id, this), tclMax("tclMax", id, this),
  tclNum("tclNum", id, this), xyz(0), scale("scale", id, this)
{
    // Create the input port
    iRM = scinew VoidStarIPort(this, "ImageRM", VoidStarIPort::Atomic);
    add_iport(iRM);
    oXYZ = scinew VoidStarOPort(this, "ImageXYZ", VoidStarIPort::Atomic);
    add_oport(oXYZ);
    NX=NY=128;
    image.resize(128*128*3);
    image.initialize(20);
    init=1;
    tclCommand="none";
    tclArgs.resize(3);
    xyzHandle=0;
}

RayMatrix::~RayMatrix()
{
}

void RayMatrix::execute()
{
    VoidStarHandle rmHandle;
    iRM->get(rmHandle);
    if (!rmHandle.get_rep()) return;
    if (!(rm = dynamic_cast<ImageRM *>(rmHandle.get_rep()))) return;

    NY=rm->pix.dim1();
    NX=rm->pix.dim2();

    // New scene -- have to build widgets
    if (rm->generation != rmGen) {
	image.resize(NY*NX*3);
	tclCommand="new";
	tclMin.set(rm->min);
	tclMax.set(rm->max);
	tclNum.set(rm->num);
	if (tclSpectrum.get() == clString("")) {
	    tclSpectrum.set(rm->lightName[0]);
	    tclLType.set("light");
	}
	reset_vars();
	rmGen=rm->generation;
    }

    if (tclCommand == "print") {
	if (!rm) {
	    cerr << "Can't print matrices -- no raymatrix yet...\n";
	    return;
	}
	cerr << "Here's RM...\n";
	cerr << "-- ks -- : ";
	int ii;
	for (ii=0; ii<rm->ks.size(); ii++) cerr <<rm->ks[ii]<<" ";
	cerr << "\n";
	cerr << "-- kd --: ";
	for (ii=0; ii<rm->kd.size(); ii++) cerr <<rm->kd[ii]<<" ";
	cerr << "\n";
	if (!rm->LS) {
	    cerr << "no LS\n";
	} else {
	    cerr << "-- LS --\n";
	    rm->LS->print();
	}
	cerr << "-- MS --\n";
	if (!rm->MS) {
	    cerr << "no MS\n";
	} else {
	    rm->MS->print();
	}
	cerr << "-- KAS --\n";
	if (!rm->KAS) {
	    cerr << "no KAS\n";
	} else {
	    rm->KAS->print();
	}
	cerr << "RM is "<<rm->pix.dim1()<<"x"<<rm->pix.dim2()<<"\n";
	for (ii=0; ii<rm->pix.dim1(); ii++) {
	    for (int jj=0; jj<rm->pix.dim2(); jj++) {
		cerr << "\n\nPIXEL ("<<ii<<","<<jj<<")...\n";
		cerr << "  E: ";
		int ne=rm->pix(ii,jj).E.size();
		int kk;
		for (kk=0; kk<ne; kk++) cerr << rm->pix(ii,jj).E[kk] <<" ";
		cerr << "\n";
//		cerr << "  S0: ";
//		int ns=rm->pix(ii,jj).S0.size();
//		for (kk=0; kk<ns; kk++) cerr << rm->pix(ii,jj).S0[kk] <<" ";
//		cerr << "\n";
		cerr << "  R:  ";
	        rm->pix(ii,jj).R.print();
		int ns=rm->pix(ii,jj).S.size();
		cerr << " "<<ns<<" S matrices...\n";
		for (kk=0; kk<ns; kk++) {
		    cerr << "  S"<<kk<<":  ";
		    rm->pix(ii,jj).S[kk].print();
		}
		int nd=rm->pix(ii,jj).D.size();
		cerr << " "<<nd<<" 'D' matrices...\n";
		for (kk=0; kk<nd; kk++) {
		    cerr << "  'D"<<kk<<"':  ";
		    rm->pix(ii,jj).D[kk].print();
		}
	    }
	}
	cerr << "\n\n";
	return;
    }
    // Only ~really~ raytrace if tcl said to
    if (tclCommand != "none") {

	cerr << "Got command: "<<tclCommand<<"\n";
	cerr << "Min="<<tclMin.get()<<"\n";
	cerr << "Max="<<tclMax.get()<<"\n";
	cerr << "Num="<<tclNum.get()<<"\n";
	cerr << "Spec="<<tclSpec.get()<<"\n";
	cerr << "FName="<<tclFName.get()<<"\n";
	cerr << "tclSpectrum="<<tclSpectrum.get()<<"\n";
	cerr << "tclLType="<<tclLType.get()<<"\n";

	min=tclMin.get(); 
	max=tclMax.get(); 
	num=tclNum.get();

	spacing=(min-max)/(num-1.);
	
	// these are the cases where we have to recompute all our spectra
	if (tclCommand == "ex" || tclCommand == "changeparam" || 
	    tclCommand == "new") {
	    cerr << "Recomputing all spectra...\n";
	    rm->bldSpecMatrices();
	}


// NOTE -- SHOULD PROBABLY BE DOING THIS FOR MATERIALS, NOT JUST SPECTA!
// but for now, we'll just throw away the ->reflectivity for the material
// and use the same ks, kd we were using for the material in this slot
// before...


	// this is the case where we have to recompute one spectrum
	if (tclCommand == "changefile") {
	    cerr << "Reading spectrum from file...\n";
	    Spectrum s;
	    Piostream* stream=auto_istream(tclFName.get());
	    if (!stream) {
		cerr << "ERROR -- no such file: "<<tclFName.get()<<"\n";
		return;
	    }
	    Pio(*stream, s);
	    if (tclLType.get() == "ambient") {
		rm->ka=s;
		rm->bldSpecAmbientMatrix();
	    } else if (tclLType.get() == "light") {
		int i;
		for (i=0; i<rm->lightName.size(); i++) 
		    if (rm->lightName[i] == tclSpectrum.get()) break;
		if (i==rm->lightName.size()) {
		    cerr << "Spectrum: "<<tclSpectrum.get()<<" not among lights.\n";
		    return;
		}
		rm->lightSpec[i]=s;
		rm->bldSpecLightMatrix(i);
	    } else if (tclLType.get() == "material") {
		int i;
		for (i=0; i<rm->matlName.size(); i++) 
		    if (rm->matlName[i] == tclSpectrum.get()) break;
		if (i==rm->matlName.size()) {
		    cerr << "Spectrum: "<<tclSpectrum.get()<<" not among matls.\n";
		    return;
		}
		rm->matlSpec[i]=s;
		rm->bldSpecMaterialMatrix(i);
	    } else {
		cerr << "ERROR -- unknown tclLType: "<<tclLType.get()<<"\n";
		return;
	    }
	}

	// these are the cases where we have to build R
	if (tclCommand == "ex" || tclCommand == "changediff" ||
	    tclCommand == "changespec" || tclCommand == "new") {
	    cerr << "Building R...\n";
	    int i;
	    int changed=1;
	    if (tclCommand == "changediff" || tclCommand == "changespec") {
		for (i=0; i<rm->matlName.size(); i++)
		    if (rm->matlName[i] == "tclSpectrum") break;
		if (i==rm->matlName.size()) {
		    cerr << "Spectrum: "<<tclSpectrum.get()<<" not among matls.\n";
		    return;
		}
		if (tclCommand == "changediff") {
		    if (rm->kd[i] == tclDiff.get()) changed=0;
		    else {
			rm->kd[i]=tclDiff.get();
			rm->ks[i]=(1-tclDiff.get());
		    }
		} else {
		    if (rm->ks[i] == tclSpec.get()) changed=0;
		    else {
			rm->ks[i]=tclSpec.get();
			rm->kd[i]=(1-tclSpec.get());
		    }
		}
	    }
	    rm->bldPixelR();

	    TextPiostream stream("/tmp/tmp.rm", Piostream::Write);
	    VoidStarHandle rmh=rm;
	    Pio(stream, rmh);
	}

	// for ALL cases we we have to build XYZ and RGB

	rm->bldPixelSpectrum();
	rm->bldPixelXYZandRGB();
	xyz=new ImageXYZ;
	xyz->xyz.resize(NY,NX*3);
	for (int yy=0; yy<NY; yy++) {
	    for (int xx=0; xx<NX; xx++) {
		xyz->xyz(yy,xx*3)=rm->pix(yy,xx).xyz.x();
		xyz->xyz(yy,xx*3+1)=rm->pix(yy,xx).xyz.y();
		xyz->xyz(yy,xx*3+2)=rm->pix(yy,xx).xyz.z();
	    }
	}
	xyzHandle=xyz;
	tclCommand = "none";
	makeRGB();
	redraw();
    }
    oXYZ->send(xyzHandle);
}

// from rgb of each pixel, put them in image to be rendered
void RayMatrix::makeRGB() {
    int count=0;
    for (int i=0; i<NY; i++) {
	for (int j=0; j<NX*3; j+=3, count+=3) {
	    image[count]=rm->pix(i,j/3).c.r();
	    image[count+1]=rm->pix(i,j/3).c.g();
	    image[count+2]=rm->pix(i,j/3).c.b();
	}
    }
}

void RayMatrix::redraw() {
    makeCurrent();
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glDrawPixels(NX, NY, GL_RGB, GL_UNSIGNED_BYTE, 
		 &(image[0]));
    glXSwapBuffers(dpy,win);
    int errcode;
    while((errcode=glGetError()) != GL_NO_ERROR){
	cerr << "RayMatrix got an error from GL: " << (char*)gluErrorString(errcode) << endl;
    }
    glXMakeCurrent(dpy, None, NULL);
    TCLTask::unlock();
}	    

void RayMatrix::tcl_command(TCLArgs& args, void* userdata) {
    if (args[1] == "redraw") {
	reset_vars();
	redraw();
    } else if (args[1] == "ex") {
	tclCommand="ex";
	want_to_execute();
    } else if (args[1] == "print") {
	tclCommand="print";
	want_to_execute();
    } else if (args[1] == "changesparam") {
	tclCommand="changesparam";
	want_to_execute();
    } else if (args[1] == "changediff") {
	tclCommand="changediff";
	want_to_execute();
    } else if (args[1] == "changespec") {
	tclCommand="changediff";
	want_to_execute();
    } else if (args[1] == "changefile") {
	tclCommand="changefile";
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

int RayMatrix::makeCurrent() {
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
    double sc=scale.get();
    glRasterPos2d(256-NX*sc/2, 256-NY*sc/2);
    glPixelZoom(sc, sc);
    return 1;
} // End namespace DaveW
}

// $Log
