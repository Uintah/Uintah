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
#include <dlfcn.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>

// for scene 1
#include <Packages/rtrt/Core/BouncingSphere.h>
#include <Packages/rtrt/Core/DielectricMaterial.h>
#include <Packages/rtrt/Core/Checker.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/Phong.h>
#include <Packages/rtrt/Core/Rect.h>


namespace rtrt {

#define RTRT_THREAD_GROUP 128
  
using namespace SCIRun;
using namespace std;

class RTRTViewer : public Module {
public:
  RTRTViewer(const string& id);
  virtual ~RTRTViewer();
  virtual void execute();
  void tcl_command(TCLArgs& args, void* userdata);

private:
  Scene *current_scene, *next_scene;
  
  GuiInt nworkers;

  void start_rtrt();
  void stop_rtrt();

  Scene* make_scene_1();
  SceneIPort *in_scene_port;
};

static string widget_name("RTRTViewer Widget");
 
extern "C" Module* make_RTRTViewer(const string& id) {
  return scinew RTRTViewer(id);
}

RTRTViewer::RTRTViewer(const string& id)
: Module("RTRTViewer", id, Filter, "Render", "rtrt"),
  current_scene(0), next_scene(0),
  nworkers("nworkers",id,this)
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

  // get the scene
  in_scene_port = (SceneIPort*) get_iport("Scene");
  SceneHandle handle;
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
    //    scene = make_scene_1();
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

  RTRT* rtrt_engine = new RTRT();
  int displayproc=0;

  int xres=128;//360;
  int yres=128;//360;
  int use_bv=1;
  int gridcellsize=4;
  int c0 = 0;
  int c1 = 0;
  int ncounters=0;
  bool bench=false;
  int shadow_mode=-1;
  bool no_aa=false;
  double bvscale=.3;
  char* criteria1="db, stereo, max rgb, max accumrgb";
  char* criteria2="db, max rgb, max accumrgb";
  double light_radius=-1;
  
  bool do_frameless=false;
  bool logframes=false;
  bool display_frames=true;
  
  Camera usercamera(Point(1,0,0), Point(0,0,0), Vector(0,0,1), 60);
  bool use_usercamera = false;
  
  // extract the parameters from the tcl code
  rtrt_engine->nworkers = nworkers.get();
  rtrt_engine->np = rtrt_engine->nworkers;
#if 0
  if(strcmp(argv[i], "-nobv")==0){
    use_bv=0;
  } else if(strcmp(argv[i], "-bv")==0){
    i++;
    use_bv=atoi(argv[i]);
  } else if(strcmp(argv[i], "-gridcellsize")==0){
    i++;
    gridcellsize=atoi(argv[i]);
  } else if(strcmp(argv[i], "-scene")==0){
    i++;
    scenename=argv[i];
    scene_argc=argc-i;
    scene_argv=argv+i;
    break;
  } else if(strcmp(argv[i], "-perfex")==0){
    i++;
    ncounters=sscanf(argv[i], "%d,%d", &c0, &c1);
    if(ncounters==1){
      if(c0 >= 32 || c0<0){
	cerr << "Illegal counter number: " << c0 << '\n';
	exit(1);
      }
      cerr << "enabling 1 counter: " << c0 << '\n';
    } else if(ncounters==2){
      if(c0 >= 32 || c0<0){
	cerr << "Illegal counter number: " << c0 << '\n';
	exit(1);
      }
      if(c1 >= 32 || c1<0){
	cerr << "Illegal counter number: " << c0 << '\n';
	exit(1);
      }
      if((c0 <16 && c1<16) || (c0>=16 && c1>=16)){
	cerr << "Illegal counter combination: " << c0 << " and " << c1 << '\n';
	exit(1);
      }
      cerr << "enabling 2 counters: " << c0 << " and " << c1 << '\n';
    } else {
      cerr << "Error parsing counter numbers: " << argv[i] << '\n';
      exit(1);
    }
  } else if(strcmp(argv[i], "-visual")==0){
    i++;
    criteria1=argv[i];
  } else if(strcmp(argv[i], "-bench")==0){
    bench=true;
  } else if(strcmp(argv[i], "-no_shadows")==0){
    shadow_mode=0;
  } else if(strcmp(argv[i], "-no_aa")==0){
    no_aa=true;
  } else if(strcmp(argv[i], "-bvscale")==0){
    i++;
    bvscale=atof(argv[i]);
  } else if(strcmp(argv[i], "-light")==0){
    i++;
    light_radius=atof(argv[i]);
  } else if(strcmp(argv[i], "-res")==0){
    i++;
    if(sscanf(argv[i], "%dx%d", &xres, &yres) != 2){
      cerr << "Error parsing resolution: " << argv[i] << '\n';
      exit(1);
    }
  } else if (strcmp(argv[i],"-nchnk")==0){
    i++;
    rtrt_engine->NUMCHUNKS = 1<<atoi(argv[i]);
    cerr << rtrt_engine->NUMCHUNKS << endl;
  } else if (strcmp(argv[i],"-clstr")==0){
    i++;
    rtrt_engine->clusterSize = atoi(argv[i]);
  } else if (strcmp(argv[i],"-noshuffle")==0){
    rtrt_engine->shuffleClusters=0;
  } else if (strcmp(argv[i],"-udp")==0){
    i++;
    rtrt_engine->updatePercent = atof(argv[i]);
  } else if (strcmp(argv[i],"-displayless")==0) {
    display_frames = false;
  } else if (strcmp(argv[i],"-frameless")==0) {
    do_frameless=true;
    i++;
    if (argv[i][0] == '-') {
      i--; // just use default mode...
    } else {
      if (0==strcmp(argv[i],"Hilbert")) {
	rtrt_engine->framelessMode = 0; // 0 is hilbert, outch
      } else if (0 ==strcmp(argv[i],"Scan")) {
	rtrt_engine->framelessMode = 1; // 1 is pixel interleaving
      } else {
	cerr << "Woah - bad frameless argument, Hilbert or Scan\n";
      }
    }
  } else if(strcmp(argv[i],"-jitter")==0) {
    rtrt_engine->do_jitter=1;
  } else if(strcmp(argv[i], "-logframes") == 0){
    logframes=true;
  } else if(strcmp(argv[i], "-eye") == 0){
    Point p;
    i++;
    p.x(atof(argv[i]));
    i++;
    p.y(atof(argv[i]));
    i++;
    p.z(atof(argv[i]));
    usercamera.set_eye(p);
    use_usercamera=true;
  } else if(strcmp(argv[i], "-lookat") == 0){
    Point p;
    i++;
    p.x(atof(argv[i]));
    i++;
    p.y(atof(argv[i]));
    i++;
    p.z(atof(argv[i]));
    usercamera.set_lookat(p);
    use_usercamera=true;
  } else if(strcmp(argv[i], "-up") == 0){
    Vector v;
    i++;
    v.x(atof(argv[i]));
    i++;
    v.y(atof(argv[i]));
    i++;
    v.z(atof(argv[i]));
    usercamera.set_up(v);
    use_usercamera=true;
  } else if(strcmp(argv[i], "-fov") == 0){
    i++;
    double fov=atof(argv[i]);
    usercamera.set_fov(fov);
    use_usercamera=true;
  } else {
    cerr << "Unknown option: " << argv[i] << '\n';
    usage(argv[0]);
    exit(1);
  }
#endif
  
  // set the scenes rtrt_engine pointer
  current_scene->set_rtrt_engine(rtrt_engine);
  
  if(shadow_mode != -1)
    current_scene->shadow_mode = shadow_mode;
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
  
  current_scene->logframes=logframes;
  
  // Start up display thread...
  Dpy* dpy=new Dpy(current_scene, criteria1, criteria2, rtrt_engine->nworkers,
		   bench, ncounters, c0, c1, 1.0, 1.0, display_frames,
		   do_frameless==true);
  /* <<<< bigler >>>> */
  Thread* t = new Thread(dpy, "Display thread");
  
  // Start up worker threads...
  for(int i=0;i<rtrt_engine->nworkers;i++){
    char buf[100];
    sprintf(buf, "worker %d", i);
#if 0
    /* Thread* t=*/new Thread(new Worker(dpy, current_scene, i,
					 pp_size, scratchsize,
					 ncounters, c0, c1),
			      buf);
    //t->migrate(i);
#endif
    /* <<<< bigler >>>> */
    Thread* t= new Thread(new Worker(dpy, current_scene, i,
				     pp_size, scratchsize,
				     ncounters, c0, c1),
			  buf);
  }
}

void RTRTViewer::stop_rtrt() {
  cout << "Stoping the rtrt\n";
  cout << "But not really....\n";
}

// This is called when the tcl code explicity calls a function besides
// needexecute.
void RTRTViewer::tcl_command(TCLArgs& args, void* userdata)
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

Scene* RTRTViewer::make_scene_1() {
    Camera cam(Point(4,4,1.7), Point(0,0,0),
		 Vector(0,0,1), 60.0);

    double bgscale=0.5;
    double ambient_scale=.5;

    Color bgcolor(bgscale*108/255., bgscale*166/255., bgscale*205/255.);

    Material* matl0=new Phong(Color(0,0,0), Color(.2,.2,.2), Color(.3,.3,.3), 100, .5);
    Material* matl00=new Phong(Color(0,0,0), Color(.2,.2,.2), Color(.3,.3,.3), 10, 0);
    Material* matl1=new Checker(new Phong(Color(.05,.05,.05), Color(.2,.2,.5), Color(.1,.1,.1), 0, .1),
				new Phong(Color(.05,.0,0), Color(.2,.2,.2), Color(.1,.1,.1), 0, .1),
				Vector(1,1.1,0), Vector(-1.1,1,0));
    Object* obj1=new Rect(matl1, Point(0,0,0), Vector(20,0,0), Vector(0,20,0));
    
    Group* group=new Group();
    group->add(obj1);
    group->add(new BouncingSphere(matl00, Point(0,0,1.5), .5, Vector(0,0,1.2)));
    group->add(new BouncingSphere(matl0, Point(0,0,2.5), .5, Vector(0,0,1.4)));
    group->add(new BouncingSphere(matl00, Point(0,0,3.5), .5, Vector(0,0,1.6)));
    group->add(new BouncingSphere(matl0, Point(0,0,.5), .5, Vector(0,0,1)));

    Plane groundplane ( Point(0, 0, 0), Vector(1, 0, 0) );
    Scene* scene=new Scene(group, cam,
			   bgcolor, Color(0,0,0), bgcolor, groundplane,
			   ambient_scale);
    scene->shadow_mode=1;
    return scene;
}

} // End namespace rtrt

