/*
 *  RTRTViewer.cc:  Real Time Ray Tracer rendering engine
 *
 *  Rendering engine of the real time ray tracer.  This module takes a scene  
 *  file from the input port and renders it.
 *
 *
 *  Written by:
 *   James Bigler
 *   Department of Computer Science
 *   University of Utah
 *   May 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>

#include <Core/Thread/Thread.h>
#include <Core/Thread/Parallel.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/BV2.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Dpy.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/Image.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/rtrt.h>
#include <Packages/rtrt/Dataflow/Ports/ScenePort.h>
#include <GL/glut.h>
#include <dlfcn.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>

#include <Packages/rtrt/Core/Gui.h>

namespace rtrt {

#define RTRT_THREAD_GROUP 128
  
using namespace SCIRun;
using namespace std;

class RTRTViewer : public Module {
public:
  RTRTViewer(GuiContext *ctx);
  virtual ~RTRTViewer();
  virtual void execute();
  void tcl_command(GuiArgs& args, void* userdata);
  bool startSoundThread;
  bool show_gui;
private:
  Scene *current_scene, *next_scene;
  RTRT *rtrt_engine;
  bool first_time;
  
  GuiInt nworkers;
  GuiInt xres_gui, yres_gui; 	// default to 400
  // 0 for frames, 1 for frameless
  GuiInt render_mode; 		// default to 0
  // 0 for none, 1 for BV1, 2 for BV2, 3 for grid
  GuiInt scene_opt_type;	// default to 1
  GuiInt gridcellsize_gui;	// default to 4

  void start_rtrt();
  void stop_rtrt();

  SceneIPort *in_scene_port;
  void GlutHelper(int, int, int);
};

static string widget_name("RTRTViewer Widget");
int mainWindowId;


DECLARE_MAKER(RTRTViewer)

RTRTViewer::RTRTViewer(GuiContext* ctx)
: Module("RTRTViewer", ctx, Filter, "Render", "rtrt"),
  current_scene(0), next_scene(0), rtrt_engine(0),
  nworkers(ctx->subVar("nworkers")),
  xres_gui(ctx->subVar("xres_gui")),
  yres_gui(ctx->subVar("yres_gui")),
  render_mode(ctx->subVar("render_mode")),
  scene_opt_type(ctx->subVar("scene_opt_type")),
  gridcellsize_gui(ctx->subVar("gridcellsize_gui")),
  startSoundThread(false),show_gui(true),first_time(true)
{
  //  inColorMap = scinew ColorMapIPort( this, "ColorMap",
  //				     ColorMapIPort::Atomic);
  //  add_iport( inColorMap);
}

RTRTViewer::~RTRTViewer()
{
}

void RTRTViewer::execute()
{
  reset_vars();
  int xres=xres_gui.get();
  int yres=yres_gui.get();

  // get the scene
  in_scene_port = (SceneIPort*) get_iport("Scene");
  SceneContainerHandle handle;
  if(!in_scene_port->get(handle)){
    std::cerr<<"Didn't get a handle\n";
    return;
  } else {
    std::cerr<<"Got handle!\n";
  }
  next_scene = handle.get_rep()->get_scene();
  if (next_scene == 0) {
    std::cerr<<"Didn't get a non null scene pointer, bailing\n";
    return;
  }
}

void RTRTViewer::start_rtrt() {
  cout << "Starting the rtrt\n";
  reset_vars();

  if (current_scene != 0) {
    // then we already have a scene running
    std::cerr<<"Already running a scene. Stop it then start again.\n";
    return;
  } else {
    current_scene = next_scene;
  }
  if (!current_scene) {
    std::cerr<<"No scene to render.\n";
    return;
  }

  rtrt_engine = new RTRT();

  int xres=xres_gui.get();
  int yres=yres_gui.get();
  int use_bv=scene_opt_type.get();
  int gridcellsize=gridcellsize_gui.get();
  int c0 = 0;
  int c1 = 0;
  int ncounters=0;
  bool bench=false;
  ShadowType shadow_mode = No_Shadows;
  bool no_aa=false;
  double bvscale=.3;
  char* criteria1="db, stereo, max rgb, max accumrgb";
  char* criteria2="db, max rgb, max accumrgb";
  double light_radius=-1;
  
  bool do_frameless;
  if (render_mode.get() == 0)
    do_frameless = false;
  else
    do_frameless = true;
  bool display_frames=true;
  
  Camera usercamera(Point(1,0,0), Point(0,0,0), Vector(0,0,1), 60);
  bool use_usercamera = false;
  
  // extract the parameters from the tcl code
  rtrt_engine->nworkers = nworkers.get();
  rtrt_engine->np = rtrt_engine->nworkers;

  cout << "xres = "<<xres<<", yres = "<<yres<<endl;
  // set the scenes rtrt_engine pointer
  current_scene->set_rtrt_engine(rtrt_engine);
  
  if(shadow_mode > Uncached_Shadows ){
    cerr << "Unknown shadow mode: " << shadow_mode << '\n';
    exit(1);
  }
  current_scene->no_aa = no_aa;
  
  if(use_bv){
    if(current_scene->nprims() > 1){
      if(use_bv==1){
	current_scene->set_object(new BV1(current_scene->get_object()));
      } else if(use_bv==2){
	current_scene->set_object(new BV2(current_scene->get_object()));
      } else if(use_bv==3){
	current_scene->set_object(new Grid(current_scene->get_object(), gridcellsize));
      } else {
	cerr << "WARNING: Unknown bv method\n";
      }
    }
  }
  
  if(!current_scene->get_image(0)){
    Image* image0=new Image(xres, yres, false);
    current_scene->set_image(0, image0);
    Image* image1=new Image(xres, yres, false);
    current_scene->set_image(1, image1);
  } else {
    current_scene->get_image(0)->resize_image(xres,yres);
    current_scene->get_image(1)->resize_image(xres,yres);
  }
  if(use_usercamera){
    usercamera.setup();
    *current_scene->get_camera(0)=usercamera;
  }
  current_scene->copy_camera(0);
  
  if(light_radius != -1){
    for(int i=0;i<current_scene->nlights();i++)
      current_scene->light(i)->radius=light_radius;
  }
  
  cerr << "Preprocessing\n";
  int pp_size=0;
  int scratchsize=0;
  current_scene->preprocess(bvscale, pp_size, scratchsize);
  cerr << "Done\n";
  
  // initialize jitter masks 
  
  for(int ii=0;ii<1000;ii++) {
    rtrt_engine->Gjitter_vals[ii] = drand48()*2 - 1.0;
    while (fabs(rtrt_engine->Gjitter_vals[ii]) >= 0.85) // sort of possion...
      rtrt_engine->Gjitter_vals[ii] = drand48()*2 - 1.0;
    
    rtrt_engine->Gjitter_vals[ii] *= 0.25;
    
    rtrt_engine->Gjitter_valsb[ii] = drand48() - 1.0;
    while (fabs(rtrt_engine->Gjitter_valsb[ii]) >= 0.85)// sort of possion...
      rtrt_engine->Gjitter_valsb[ii] = drand48() - 1.0;
    
    rtrt_engine->Gjitter_valsb[ii] *= 0.25;
  }
  
  // Start up display thread...
  Dpy* dpy=new Dpy(current_scene, criteria1, criteria2, rtrt_engine->nworkers,
		   bench, ncounters, c0, c1, 1.0, 1.0, display_frames,
		   pp_size, scratchsize, false, do_frameless==true, false);

  Gui * gui = new Gui();
  Gui::setActiveGui( gui );
  gui->setDpy( dpy );

  /* <<<< bigler >>>> */
  Thread* t = new Thread(dpy, "Display thread");
  t->detach();
  
  // Start up worker threads...
  for(int i=0;i<rtrt_engine->nworkers;i++){
    char buf[100];
    sprintf(buf, "worker %d", i);

    /* <<<< bigler >>>> */
    Thread* t= new Thread(new Worker(dpy, current_scene, i,
				     pp_size, scratchsize,
				     ncounters, c0, c1),
			  buf);
    t->detach();
  }

  if (first_time){
    cout << "about to start GlutHelper()\n";
//      Thread::parallel(Parallel<RTRTViewer>(this,&RTRTViewer::GlutHelper),
//  		     1,false);
    Thread::parallel(this, &RTRTViewer::GlutHelper, 1, false, xres, yres);
    cout << "thread started\n";
    first_time = false;
  }
} // end start_rtrt()

void RTRTViewer::GlutHelper(int, int xres, int yres) {

  // Initialize GLUT and GLUI stuff.
  printf("start glut inits\n");
  glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE);
  glutInitWindowPosition( 0, 0 );
  mainWindowId = glutCreateWindow("RTRT");
  glutReshapeWindow (xres,yres);

  glutKeyboardFunc( Gui::handleKeyPressCB );
  glutSpecialFunc( Gui::handleSpecialKeyCB );
  glutMouseFunc( Gui::handleMouseCB );
  glutMotionFunc( Gui::handleMouseMotionCB );
  glutSpaceballMotionFunc( Gui::handleSpaceballMotionCB );
  glutSpaceballRotateFunc( Gui::handleSpaceballRotateCB );
  glutSpaceballButtonFunc( Gui::handleSpaceballButtonCB );

  glutReshapeFunc( Gui::handleWindowResizeCB );
  glutDisplayFunc( Gui::redrawBackgroundCB );

  // Must do this after glut is initialized.
  Gui::createMenus( mainWindowId, startSoundThread, show_gui);

  cout << "running glutMainLoop()\n";
  glutMainLoop();
}

void RTRTViewer::stop_rtrt() {
  cout << "Stoping the rtrt\n";
  if (rtrt_engine) {
    rtrt_engine->exit_clean(1);
    delete(rtrt_engine);
  }
  current_scene = 0;
}

// This is called when the tcl code explicity calls a function besides
// needexecute.
void RTRTViewer::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2) {
    args.error("Streamline needs a minor command");
    return;
  }
  if(args[1] == "start_rtrt") {
    start_rtrt();
  } else if(args[1] == "stop_rtrt") {
    stop_rtrt();
  } else {
    Module::tcl_command(args, userdata);
  }
}


} // End namespace rtrt

