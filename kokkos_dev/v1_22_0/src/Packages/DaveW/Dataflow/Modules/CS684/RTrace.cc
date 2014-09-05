
/*
 *  RTrace.cc:  Take in a scene (through a VoidStarPort), and output the scene
 *		   Geometry and a rendering window
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
#include <Packages/DaveW/Core/Datatypes/CS684/DRaytracer.h>
#include <Packages/DaveW/Core/Datatypes/CS684/RTPrims.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/TCL.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/Timer.h>
#include <Dataflow/Widgets/ViewWidget.h>

#include <iostream>
using std::cerr;
using std::endl;
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

extern int global_numbounces;

extern Tcl_Interp* the_interp;
extern "C" GLXContext OpenGLGetContext(Tcl_Interp*, char*);

namespace DaveW {
using namespace DaveW;
using namespace SCIRun;

class RTrace : public Module {
    VoidStarIPort *iRT;
    VoidStarOPort *oRT;
    VoidStarOPort *oXYZ;
    VoidStarOPort *oRM;
    GeometryOPort *ogeom;
    DRaytracer *rt;
    int rtGen;

    VoidStarHandle xyzHandle;
    VoidStarHandle rmHandle;
    Array1<double> image;
    Array1<unsigned char> rawImage;
    Array1<unsigned char> clampedImage;
    Array2<double> xyz;
    unsigned char *pixels;
    int camera_id;
    ViewWidget* camera_widget;

    Tk_Window tkwin;
    Window win;
    Display* dpy;
    GLXContext cx;

    CrowdMonitor widget_lock;

    GuiInt numProc;
    GuiInt maxProc;
    GuiInt nx;
    GuiInt ny;
    GuiInt ns;
    GuiDouble app;
    GuiDouble specMin, specMax;
    GuiDouble scale;
    GuiInt specNum;
    GuiInt abrt;
    GuiDouble bGlMin;
    GuiDouble bGlMax;
    GuiDouble bmin;
    GuiDouble bmax;
    GuiInt tweak;

    int widgetMoved;
    int tcl_exec;
    int init;
    int NX;
    int NY;

    int np;
public:
    void parallel_raytrace(int proc);

    RTrace(const clString& id);
    virtual ~RTrace();
    virtual void execute();
    void tcl_command( TCLArgs&, void * );
    void histogram();
    void bound();
    void redraw();
    int makeCurrent();
    void buildCamera();
    void updateCamera();
    void getCamera();
    virtual void widget_moved2(int last, void *userdata);    
};

static RTrace* current_drawer=0;

extern "C" Module* make_RTrace(const clString& id)
{
    return scinew RTrace(id);
}

static clString module_name("RTrace");

RTrace::RTrace(const clString& id)
: Module("RTrace", id, Source), widget_lock("RTrace widget lock"),
  nx("nx", id, this), ny("ny", id, this),
  ns("ns", id, this), abrt("abrt", id, this), app("app", id, this), 
  camera_id(0), tcl_exec(0), widgetMoved(0),
  specMin("specMin", id, this), specMax("specMax", id, this),
  specNum("specNum", id, this), tweak("tweak", id, this),
  bGlMin("bGlMin", id, this), bGlMax("bGlMax", id, this),
  bmin("bmin", id, this), bmax("bmax", id, this),
  numProc("numProc", id, this), maxProc("maxProc", id, this),
  scale("scale", id, this)
{
    // Create the input port
    iRT = scinew VoidStarIPort(this, "DRaytracer", VoidStarIPort::Atomic);
    add_iport(iRT);
    // Create the output port
    oRT = scinew VoidStarOPort(this, "DRaytracer", VoidStarIPort::Atomic);
    add_oport(oRT);
    ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
    oXYZ = scinew VoidStarOPort(this, "ImageXYZ", VoidStarIPort::Atomic);
    add_oport(oXYZ);
    oRM = scinew VoidStarOPort(this, "RayMatrix", VoidStarIPort::Atomic);
    add_oport(oRM);
    init=0;
}

RTrace::~RTrace()
{
}

void RTrace::updateCamera() {
    rt->camera.view=camera_widget->GetView();
    rt->camera.init=0;
    cerr << "Camera: eye="<<rt->camera.view.eyep()<<"  lookat="<<rt->camera.view.lookat()<<"  fov="<<rt->camera.view.fov()*(180./3.14159)<<"\n";    
}

void RTrace::buildCamera() {
    if (camera_id) ogeom->delObj(camera_id);
    camera_widget->SetView(rt->camera.view);
    GeomObj* go=camera_widget->GetWidget();
    camera_id=ogeom->addObj(go, "Camera", &widget_lock);
    cerr << "Camera: eye="<<rt->camera.view.eyep()<<"  lookat="<<rt->camera.view.lookat()<<"  fov="<<rt->camera.view.fov()*(180./3.14159)<<"\n";
}

void RTrace::execute()
{
    if (!init) {
	np=Thread::numProcessors();
	cerr << "Found "<<np<<" processors...\n";
	maxProc.set(np);
	reset_vars();
	camera_widget=scinew ViewWidget(this, &widget_lock, 0.1);
	char *ud=new char[20];
	sprintf(ud, "w0");
	camera_widget->userdata=(void *) ud;
	NX=nx.get();
	NY=ny.get();
	xyz.resize(NY,NX*3);
	xyz.initialize(0);
	image.resize(NX*NY*3);
	rawImage.resize(NX*NY*3);
	clampedImage.resize(NX*NY*3);
	for (int i=0; i<image.size(); i++) 
	    image[i]=clampedImage[i]=rawImage[i]=0;
	pixels=&(rawImage[0]);
	rtGen=0;
	init=1;
    }	
    View v=camera_widget->GetView();

    VoidStarHandle RTHandle;
    iRT->get(RTHandle);
    if (!RTHandle.get_rep()) return;
    if (!(rt = dynamic_cast<DRaytracer *>(RTHandle.get_rep()))) return;

    // New scene -- have to build widgets
    if (rt->generation != rtGen) {
	cerr << "making camera...\n";
	buildCamera();
	NX=rt->nx; nx.set(NX);
	NY=rt->ny; ny.set(NY);
	app.set(rt->camera.apperture);
	int nsInt=(int) Sqrt(rt->ns);
	ns.set(nsInt);
	rt->ns=nsInt*nsInt;
	specNum.set(rt->specNum);
	specMax.set(rt->specMax);
	specMin.set(rt->specMin);
    } else {
	rt->nx=NX=nx.get();
	rt->ny=NY=ny.get();
	rt->ns=ns.get()*ns.get();
	rt->camera.apperture=app.get();
	rt->specNum=specNum.get();
	rt->specMin=specMin.get();
	rt->specMax=specMax.get();
    }

    camera_widget->SetAspectRatio(ny.get()*1./nx.get());

    if (widgetMoved) {
	updateCamera();
    }

    // Only ~really~ raytrace if tcl said to
    if (tcl_exec) {
	xyz.resize(NY,NX*3);
	image.resize(NX*NY*3);
	rawImage.resize(NX*NY*3);
	clampedImage.resize(NX*NY*3);
	xyz.initialize(0);
	for(int ii=0; ii<NX*NY*3; ii++) 
	    image[ii]=clampedImage[ii]=rawImage[ii]=0;
	pixels=&(rawImage[0]);
	rt->preRayTrace();
	
	np=numProc.get();
	cerr << "Using "<<np<<" processors...\n";
	Thread::parallel(Parallel<RTrace>(this, &RTrace::parallel_raytrace),
			 np, true);
	pixels=&(rawImage[0]);
	if (abrt.get())	abrt.set(0);
//	for (int i=0; i<NX*NY*3; i++) clampedImage[i]=image[i]=rawImage[i]=0;
	reset_vars();
	histogram();
	bound();
//	cerr << "Total number of rays = "<<global_numbounces<<"   Avg bounces = "<<global_numbounces*1./(rt->nx*rt->ny*rt->ns)<<"\n";
    }

    // reset all flags
    rtGen=rt->generation; widgetMoved=0; tcl_exec=0;
    RTHandle=rt;
    oRT->send(RTHandle);    
    xyzHandle=scinew ImageXYZ(xyz);
    oXYZ->send(xyzHandle);
    rmHandle=rt->irm;
    oRM->send(rmHandle);
    return;
}

void RTrace::parallel_raytrace(int proc) {
    CPUTimer myThreadTime;
    myThreadTime.start();
    int sy=proc*NY/np;
    int ey=(proc+1)*NY/np;
    int stepSize=1;
    if (ey-sy > 10) stepSize=(ey-sy)/10;
    int lastY;
    for (int y=sy; y<ey; y+=stepSize) {	    
	lastY=y+stepSize;
	if (lastY>ey) lastY=ey;
	reset_vars();
	if (abrt.get()) {
	    break;
	}
	if (proc == (np/2+1)) update_progress(y-sy,ey-sy);
	rt->rayTrace(0,NX,y,lastY,&(image[0]),&(rawImage[0]), &(xyz(0,0)));
	redraw();
    }
    myThreadTime.stop();
    timer.add(myThreadTime.time());
}

void RTrace::widget_moved2(int last, void *)
{
    if (last && !abort_flag) {
	abort_flag=1;
	widgetMoved=1;
	want_to_execute();
    }
}

#if 0
void RTrace::bound() {
    if (!init) {
	NX=NY=128;
	xyz.resize(NY,NX*3);
	xyz.initialize(0);
	image.resize(NX*NY*3);
	rawImage.resize(NX*NY*3);
	clampedImage.resize(NX*NY*3);
	for (int i=0; i<image.size(); i++) 
	    image[i]=clampedImage[i]=rawImage[i]=0;
	pixels=&(rawImage[0]);
    }

    clString m=bmethod.get();
    int i;
    double rmax, gmax, bmax, rmin, gmin, bmin;
    rmax=gmax=bmax=0;
    rmin=gmin=bmin=1000000;
    for (i=0; i<NX*NY*3; i+=3) {
	if (image[i] > rmax) rmax=image[i];
	if (image[i+1] > gmax) gmax=image[i+1];
	if (image[i+2] > bmax) bmax=image[i+2];
	if (image[i] < rmin) rmin=image[i];
	if (image[i+1] < gmin) gmin=image[i+1];
	if (image[i+2] < bmin) bmin=image[i+2];
    }
    double max=Max(rmax, gmax, bmax);
    double min=Min(rmin, gmin, bmin);
    cerr << "rmax="<<rmax<<" rmin="<<rmin<<" gmax="<<gmax<<" gmin="<<gmin<<" bmax="<<bmax<<" bmin="<<bmin<<" min="<<min<<" max="<<max<<"\n";
    double dr=rmax-rmin;
    double dg=gmax-gmin;
    double db=bmax-bmin;
    double dd=max-min;
    double idr, idg, idb, idd;
    idr=idg=idb=idd=0;
    if (dr > .001) idr=255/dr;
    if (dg > .001) idg=255/dg;
    if (db > .001) idb=255/db;
    if (dd > .001) idd=255/dd;

    if (m=="indepRescale") {
	for (i=0; i<NX*NY*3; i+=3) {
	    clampedImage[i]=(image[i]-rmin)*idr;
	    clampedImage[i+1]=(image[i+1]-gmin)*idg;
	    clampedImage[i+2]=(image[i+2]-bmin)*idb;
	}
    } else if (m=="indepClamp") {
	for (i=0; i<NX*NY*3; i+=3) {
	    if (image[i]<0) clampedImage[i]=0;
	    else clampedImage[i]=(image[i]-rmin)*idr;
	    if (image[i+1]<0) clampedImage[i+1]=0;
	    else clampedImage[i+1]=(image[i+1]-gmin)*idg;
	    if (image[i+2]<0) clampedImage[i+2]=0;
	    else clampedImage[i+2]=(image[i+2]-bmin)*idb;
	}
    } else if (m=="compRescale") {
	for (i=0; i<NX*NY*3; i+=3) {
	    clampedImage[i]=(image[i]-min)*idd;
	    clampedImage[i+1]=(image[i+1]-min)*idd;
	    clampedImage[i+2]=(image[i+2]-min)*idd;
	}
    } else if (m=="compClamp") {
	for (i=0; i<NX*NY*3; i+=3) {
	    if (image[i]<0) clampedImage[i]=0;
	    else clampedImage[i]=(image[i]-min)*idd;
	    if (image[i+1]<0) clampedImage[i+1]=0;
	    else clampedImage[i+1]=(image[i+1]-min)*idd;
	    if (image[i+2]<0) clampedImage[i+2]=0;
	    else clampedImage[i+2]=(image[i+2]-min)*idd;
	}
    } else {
	cerr << "Unkown method: "<<m<<"\n";
	return;
    }
    pixels=&(clampedImage[0]);
    redraw();
}
#endif

void RTrace::histogram() {
    if (!init) {
	NX=NY=128;
	xyz.resize(NY,NX*3);
	xyz.initialize(0);
	image.resize(NX*NY*3);
	rawImage.resize(NX*NY*3);
	clampedImage.resize(NX*NY*3);
	for (int i=0; i<image.size(); i++) 
	    image[i]=clampedImage[i]=rawImage[i]=0;
	pixels=&(rawImage[0]);
    }
    double min, max;
    min=max=image[0];
    for (int i=0; i<NX*NY*3; i++)
	if (image[i] > max) max=image[i];
	else if (image[i] < min) min=image[i];

//    bGlMin.set(min);
//    bGlMax.set(max);
//    tweak.set(!tweak.get());
    reset_vars();
    cerr << "min="<<min<<" max="<<max<<"\n";
}

void RTrace::bound() {
    double selMin=bmin.get();
    double selMax=bmax.get();
    cerr << "selMin="<<selMin<<"  selMax="<<selMax<<"\n";
    double dd=selMax-selMin;
    double idd=0;
    if (dd > .001) idd=255/dd;
    for (int i=0; i<NX*NY*3; i++) {
	if (image[i]<selMin) clampedImage[i]=0;
	else if (image[i]>selMax) clampedImage[i]=255;
	else clampedImage[i]=(image[i]-selMin)*idd;
    }
    pixels=&(clampedImage[0]);
    redraw();
}

void RTrace::redraw() {
    if (!init) {
	NX=NY=128;
	image.resize(NX*NY*3);
	rawImage.resize(NX*NY*3);
	clampedImage.resize(NX*NY*3);
	for (int i=0; i<image.size(); i++) 
	    image[i]=clampedImage[i]=rawImage[i]=0;
	pixels=&(rawImage[0]);
    }
    makeCurrent();
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glDrawPixels(NX, NY, GL_RGB, GL_UNSIGNED_BYTE, pixels);
    glXSwapBuffers(dpy,win);
    int errcode;
    while((errcode=glGetError()) != GL_NO_ERROR){
	cerr << "RTrace got an error from GL: " << (char*)gluErrorString(errcode) << endl;
    }
    glXMakeCurrent(dpy, None, NULL);
    TCLTask::unlock();
}	    

void RTrace::tcl_command(TCLArgs& args, void* userdata) {
    if (args[1] == "redraw") {
	reset_vars();
	redraw();
    } else if (args[1] == "bound") {
	reset_vars();
	bound();
    } else if (args[1] == "sendXYZ") {
	want_to_execute();
    } else if (args[1] == "tcl_exec") {
	tcl_exec=1;
	want_to_execute();
    } else if (args[1] == "save") {
	FILE *f=fopen("/tmp/image.raw", "wb");
	if(!f) {
	    cerr << "Error opening /tmp/image.raw\n";
	    return;
	}
	for (int i=0; i<NX*NY*3; i+=3) {
	    fprintf(f,"%c%c%c", pixels[i], pixels[i+1], pixels[i+2]);
	}
	fclose(f);
	char scall[200];
	sprintf(scall, "rawtorle -w %d -h %d -n 3 /tmp/image.raw -o image.rle\n", NX, NY);
	cerr << "Call system: "<<scall;
//	system(scall);
    } else if (args[1] == "widgetMoved") {
	reset_vars();
        widgetMoved=1;
	want_to_execute();
    } else {
        Module::tcl_command(args, userdata);
    }
}

int RTrace::makeCurrent() {
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
//    glTranslated(256-NX/2, 256-NY/2, 0);
    glLoadIdentity();
    double sc=scale.get();
    glRasterPos2d(256-NX*sc/2, 256-NY*sc/2);

    glPixelZoom(sc, sc);

//    glMatrixMode(GL_PROJECTION);
//    int minx=(512-NX)/2;
//    int miny=(512-NY)/2;
//    glViewport(minx, miny, NX, NY);


//    glMatrixMode(GL_MODELVIEW);
//    glLoadIdentity();
    return 1;
} // End namespace DaveW
}

// $Log
