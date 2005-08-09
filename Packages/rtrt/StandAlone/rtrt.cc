
/*
 * Real time ray-tracer
 */

#include <Core/Thread/Thread.h>
#include <Core/Thread/ThreadGroup.h>
#include <Core/Persistent/Pstreams.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/BV1.h>
#include <Packages/rtrt/Core/BV2.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/Dpy.h>
#include <Packages/rtrt/Core/DpyBase.h>
#include <Packages/rtrt/Core/Grid.h>
#include <Packages/rtrt/Core/Grid2.h>
#include <Packages/rtrt/Core/Group.h>
#include <Packages/rtrt/Core/HierarchicalGrid.h>
#include <Packages/rtrt/Core/Image.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/PPMImage.h>
#include <Packages/rtrt/Core/Trigger.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/rtrt.h>
#include <Packages/rtrt/Core/DpyGui.h>

#include <sys/stat.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>

using namespace rtrt;
using namespace std;

using SCIRun::Thread;
using SCIRun::ThreadGroup;

bool use_pm = true;
extern bool pin;
#ifdef __sgi
#include <sys/types.h>
#include <sys/pmo.h>
#include <sys/attributes.h>
#include <sys/conf.h>
#include <sys/hwgraph.h>
#include <sys/stat.h>
#include <invent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void mld_alloc(long size, int nmld, 
		      pmo_handle_t*& mlds, pmo_handle_t& mldset)
{
  mlds = new pmo_handle_t[nmld];

  for(int i=0; i<nmld; i++) {
    mlds[i] = mld_create( 0, size );
    if ((long)mlds[i] < 0) {
      perror("mld_create()");
      cerr << "Try using -nomempolicy\n";
      exit(1);
    }
  }
  mldset = mldset_create( mlds, nmld );
  if ((long) mldset < 0) {
    perror("mldset_create");
    cerr << "Try using -nomempolicy\n";
    exit(1);
  }

  if ( mldset_place( mldset, TOPOLOGY_FREE, 0, 0, RQMODE_ADVISORY ) < 0) {
    perror("mldset_place");
    fprintf( stderr, "set: %p nmld: %d ( ", (void *)mldset, nmld );
    for(int i=0; i<nmld; i++)
      fprintf( stderr, "%d ", mlds[i] );
    fprintf( stderr, ")\n" );
    cerr << "Try using -nomempolicy\n";
    exit(1);
  }
}
#endif

int       mainWindowId = -1;

static void usage(char* progname)
{
  cerr << "usage: " << progname << " [options] -scene name [scene options]\n";
  cerr << "Options:\n";
  cerr << " -serialize       - Write the generated scene to disk using" << endl
       << "                    object serialization (Pio)" << endl;
  cerr << " -np n            - Set the number of worker processes to use.\n";
  cerr << "                    The actual number of processors used will be\n";
  cerr << "                    one more than n.\n";
  cerr << " -nobv            - use no bounding volume acceleration hierarchy\n";
  cerr << " -bv n            - Use a specific bounding volume hierarchy\n";
  cerr << "     n=0: None - same as -nobv\n";
  cerr << "     n=1: Use bounding volume tree, implementation #1. (the\n";
  cerr << "          default)\n";
  cerr << "     n=2: Use bounding volume tree, implementation #2.\n";
  cerr << "          (doesn't work yet)\n";
  cerr << "     n=3: Use Grid traversal\n";
  cerr << "     n=4: Use Hierarchical Grid traversal\n";
  cerr << " -gridcellsize n  - Set the number of gridcells for -bv 3\n";
  cerr << " -hgridcellsize n n n - Set the number of gridcells at level 1, 2 and 3 -bv 4\n";
  cerr << " -minobjs n n - Set the number of objects in each grid cell for -bv 4\n";
  cerr << " -perfex n1 n2    - Collect performance counters n1 and n2 for\n";
  cerr << "                   each frame.\n";
  cerr << " -visual criteria - Uses criteria for selecting the OpenGL\n";
  cerr << "                    visual for the image display.  See\n";
  cerr << "                    visinfo/findvis.1 for more info.\n";
  cerr << " -bench           - Disable frame display, and just render the\n";
  cerr << "                    image 100 times for timing purposes.\n";
  cerr << " -no_shadows      - Turn off shadows\n";
  cerr << " -shadows mode    - Select mode for shadows\n";
  cerr << "                     o 0 - No Shadows\n";
  cerr << "                     o 1 - Single Soft Shadows\n";
  cerr << "                     o 2 - Hard Shadows\n";
  cerr << "                     o 3 - Glass Shadows\n";
  cerr << "                     o 4 - Soft Shadows\n";
  cerr << "                     o 5 - Uncached Shadows\n";
  cerr << " -no_aa           - Turn off accumulation buffer anti-aliasing\n";
  cerr << " -bvscale         - Controls bounding volume scale factor for\n";
  cerr << "                    the soft shadow method.\n";
  cerr << " -light           - Specifies the radius of the light source for\n";
  cerr << "                    soft shadows.\n";
  cerr << " -res             - Sets the initial resolution of the image, in\n";
  cerr << "                    the form mxn (i.e. 100x100)\n";
  cerr << " -frameless       - Hilbert or Scan, hilbert has to be pwr2xpwr2\n";
  cerr << " -nchnk           - 1<<val is the number of chunks \n";
  cerr << "                    (for hilbert log2(xres*yres) for 1 pixel bins\n";
  cerr << " -clstr           - cluster size - what a pixel is for frameless\n";
  cerr << " -noshuffle       - don't randomize chunks - much uglier\n";
  cerr << " -udp             - update rate - how often to synchronuze cameras\n";
  cerr << "                    as a fraction of pixels per/proc\n";
  cerr << " -sound           - start sound thread\n";
  cerr << " -fullscreen      - run in full screen mode\n";
  cerr << " -jitter          - jittered masks - fixed table for now\n";
  cerr << " -stereo          - display in stereo mode\n";
  cerr << " -worker_gltest   - calls run_gl_test from worker threads\n";
  cerr << " -display_gltest  - calls run_gl_test from display thread\n";
  cerr << " -displayless     - do not display and write a frame to displayless.ppm\n";
  cerr << " -rserver         - Send the display to a remote display\n";
  cerr << " -bgcolor [value] - Override the scene's background with this one\n";
  cerr << "                    value can be:\n";
  cerr << "                     o white\n";
  cerr << "                     o black\n";
  cerr << "                     o [r] [g] [b] - floats from [0..1]\n";
  cerr << " -nomempolicy     - Do system default for memory distribution.\n";
  cerr << " -pin             - Assign each worker to a specific processor.\n"
       << "                    Use only if you have the entire machine to yourself.\n";
  cerr << " -sils            - Draw the silhouettes.\n";
  cerr << " -silvalue [float] - Value to use for silhouette drawing.\n";
  cerr << " -ambientlevel    - the level of ambient light.\n";
  
  exit(1);
}

#if 0
namespace rtrt {
  int rtrt_nworkers()
  {
    return nworkers;
  }
} // end namespace rtrt
#endif

// ick, global variables for frameless rendering...

//extern int NUMCHUNKS;
//extern double updatePercent;
//extern int clusterSize;
//extern int shuffleClusters;
//extern int np;
//extern int framelessMode; // mode for frameless rendering...

//extern double Gjitter_vals[1000];
//extern double Gjitter_valsb[1000];

//extern int do_jitter;

#if HAVE_IEEEFP_H
#include <ieeefp.h>
#endif

bool fullscreen = false;

int
main(int argc, char* argv[])
{
#if HAVE_IEEEFP_H
  //    fpsetmask(FP_X_OFL|FP_X_DZ|FP_X_INV);
#endif

  RTRT* rtrt_engine = new RTRT();

  int xres=360;
  int yres=360;
  int use_bv=1;
  char* scenename=0;

  int gridcellsize = -1;
  int gridcellsizeL2=4;
  int gridcellsizeL3=4;
  int minObjs1 = 20;
  int minObjs2 = 20;

  int c0;
  int c1;
  int ncounters=0;
  bool bench=false;
  ShadowType shadow_mode = No_Shadows;
  bool override_scene_shadow_mode = false;
  bool no_aa=false;
  double bvscale=.3;
  char* criteria1="db, stereo, max rgb, max accumrgb";
  char* criteria2="db, max rgb, max accumrgb";
  double light_radius=-1;
  
  bool do_frameless=false;
  bool display_frames=true;
  bool serialize_scene = false;
  bool startSoundThread = false;

  bool show_gui = false;
  bool rserver=false;

  Color bgcolor;
  bool override_scene_bgcolor = false;

  int maxdepth = 8;
  bool override_maxdepth = false;

  float ambient_level = -1;

  bool do_sils = false;
  float sil_value = -1;

  Camera usercamera(Point(1,0,0), Point(0,0,0), Vector(0,0,1), 60);
  bool use_usercamera = false;

  bool stereo = false;
  
  int scene_argc=-1;
  char** scene_argv=0;
  for(int i=1;i<argc;i++){
    if(strcmp(argv[i], "-np")==0){
      i++;
      rtrt_engine->nworkers=atoi(argv[i]);
    } else if(strcmp(argv[i], "-nomempolicy") == 0){
      use_pm=false;
    } else if(strcmp(argv[i], "-rserver") == 0){
      rserver=true;
    } else if(strcmp(argv[i], "-pin") == 0){
      pin=true;
    } else if(strcmp(argv[i], "-nobv")==0){
      use_bv=0;
    } else if(strcmp(argv[i], "-bv")==0){
      i++;
      use_bv=atoi(argv[i]);
    } else if(strcmp(argv[i], "-gridcellsize")==0){
      i++;
      gridcellsize=atoi(argv[i]);
    } else if(strcmp(argv[i], "-hgridcellsize")==0){
      i++;
      gridcellsize=atoi(argv[i++]);
      gridcellsizeL2=atoi(argv[i++]);
      gridcellsizeL3=atoi(argv[i]);
    } else if(strcmp(argv[i], "-minobjs")==0){
      i++;
      minObjs1=atoi(argv[i++]);
      minObjs2=atoi(argv[i]);
    } else if(strcmp(argv[i], "-scene")==0){
      i++;
      scenename=argv[i];
      scene_argc=argc-i;
      scene_argv=argv+i;
      break;
    } else if(strcmp(argv[i], "-serialize")==0){
      serialize_scene = true;
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
      shadow_mode = No_Shadows;
      override_scene_shadow_mode = true;
    } else if(strcmp(argv[i], "-shadows")==0){
      i++;
      shadow_mode = (ShadowType)atoi(argv[i]);
      override_scene_shadow_mode = true;
    } else if(strcmp(argv[i], "-bgcolor")==0){
      i++;
      if(strcmp(argv[i], "white")==0){
	bgcolor = Color(1,1,1);
      } else if(strcmp(argv[i], "black")==0){
	bgcolor = Color(0,0,0);
      } else {
	// read the three colors
	bgcolor = Color(atof(argv[i]), atof(argv[++i]), atof(argv[++i]));
      }
      override_scene_bgcolor = true;
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
    } else if (strcmp(argv[i],"-raydepth")==0 ||
	       strcmp(argv[i],"-maxdepth")==0) {
      i++;
      maxdepth = atoi(argv[i]);
      override_maxdepth = true;
    } else if (strcmp(argv[i],"-ambientlevel")==0) {
      ambient_level = atof(argv[++i]);
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
    } else if(strcmp(argv[i],"-fullscreen")==0) {
      fullscreen = true;
    } else if(strcmp(argv[i],"-jitter")==0) {
      rtrt_engine->do_jitter=1;
    } else if(strcmp(argv[i],"-sound")==0) {
      startSoundThread = true;
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
    } else if (strcmp(argv[i], "-worker_gltest") == 0) {
      rtrt_engine->worker_run_gl_test = true;
    } else if (strcmp(argv[i], "-display_gltest") == 0) {
      rtrt_engine->display_run_gl_test = true;
    } else if (strcmp(argv[i], "-show_gui") == 0 || strcmp(argv[i], "-showgui") == 0 || strcmp(argv[i], "-glut") == 0) {
      show_gui = true;
    } else if (strcmp(argv[i], "-stereo") == 0) {
      stereo = true;
    } else if (strcmp(argv[i], "-sils") == 0) {
      do_sils = true;
    } else if (strcmp(argv[i], "-silvalue") == 0) {
      sil_value = atof(argv[++i]);
    } else {
      cerr << "Unknown option: " << argv[i] << '\n';
      usage(argv[0]);
      exit(1);
    }
  }
  
  if(!scenename){
    cerr << "Must specify a scene with '-scene'\n\n";
    usage(argv[0]);
    exit(1);
  }

#ifdef __sgi
  if(use_pm){
    long mempernode = 300*1024*1024;
    // How can we tell if it should be 2 or 4 processors per node?
    int numnodes = static_cast<int>(Thread::numProcessors()/4);
    pmo_handle_t *mlds=0;
    pmo_handle_t mldset=0;
    mld_alloc(mempernode, numnodes, mlds, mldset);
    policy_set_t ps;
    pm_filldefault(&ps);
    ps.placement_policy_name = "PlacementRoundRobin";
    ps.placement_policy_args = (void*)mldset;
    pmo_handle_t policy = pm_create(&ps);
    if(policy == -1){
      perror("pm_create");
      exit(1);
    }
    pmo_handle_t old1=pm_setdefault(policy, MEM_DATA);
    cerr << "old1=" << old1 << '\n';
    pmo_handle_t old2=pm_setdefault(policy, MEM_TEXT);
    cerr << "old2=" << old2 << '\n';
    // STACK is left as first touch...
  }
#endif
  char scenefile[MAXPATHLEN];
  char pioscenefile[MAXPATHLEN];
  SceneHandle sh;

  // test for pio'd version
  sprintf(pioscenefile, "./%s.scn", scenename);
  Scene* scene = 0;
  struct stat buf;
  if (stat(pioscenefile, &buf) != -1) {
    cerr << "pio read: " << pioscenefile << endl;
    sh = scene;
    SCIRun::Piostream *str;
    str = new SCIRun::FastPiostream (pioscenefile, SCIRun::Piostream::Read);
    //str = new SCIRun::TextPiostream (pioscenefile, SCIRun::Piostream::Read);
    SCIRun::Pio(*str, sh);
    scene = sh.get_rep();
  } else {
    // First try .mo
    sprintf(scenefile, "./%s.mo", scenename);
    void* handle=dlopen(scenefile, RTLD_NOW);
    if(!handle){
      // Try exactly whatever they passed in
      handle=dlopen(scenename, RTLD_NOW);
      if(!handle){
        cerr << "Error opening scene: " << scenename << '\n';
        cerr << dlerror() << '\n';
        exit(1);
      }
    }
    void* scene_fn=dlsym(handle, "make_scene");
    if(!scene_fn){
      cerr << "Scene file found, but make_scene() function not found\n";
      exit(1);
    }
    Scene* (*make_scene)(int,char**,int) = (Scene*(*)(int,char**,int))scene_fn;
    scene=(*make_scene)(scene_argc, scene_argv, rtrt_engine->nworkers);
  }
  if(!scene){
    cerr << "Scene creation failed!\n";
    exit(1);
  }
  
  if (serialize_scene) {
    // Create a stream to save to.
    SCIRun::Piostream *str;
    char scnfile[MAXPATHLEN];
    
    // test for pio'd version
    sprintf(scnfile, "./%s.scn", scenename);
    str = new SCIRun::FastPiostream (scnfile,SCIRun::Piostream::Write);
    //str = new SCIRun::TextPiostream (scnfile,SCIRun::Piostream::Write);

    // Write it out.
    sh = scene;
    SCIRun::Pio(*str, sh);
    delete str;
    cerr << "Saved scene to " << scenename << ".scn" << endl;
    exit(0);
  }

  // set the scenes rtrt_engine pointer
  scene->set_rtrt_engine(rtrt_engine);
  
  if(shadow_mode > Uncached_Shadows ){
    cerr << "Unknown shadow mode: " << shadow_mode << '\n';
    exit(1);
  }

  if (override_scene_shadow_mode)
    scene->shadow_mode = shadow_mode;

  if (override_scene_bgcolor)
    scene->set_bgcolor(bgcolor);

  if (ambient_level >= 0)
    scene->setAmbientLevel(ambient_level);
  
  scene->no_aa=no_aa;

  // Turn on silhouettes
  if (do_sils) {
    scene->display_sils = 1;
    scene->store_depth = 1;
  }

  // Change the sil_value from the default one
  if (sil_value >= 0)
    scene->max_depth = sil_value;
  

  if (override_maxdepth || scene->maxdepth == -1)
    // Override what the scene specifies,
    // or set it if the scene has no default (i.e. maxdepth == -1)
    scene->maxdepth = maxdepth;

  if(use_bv){
    if(scene->nprims() > 1){
      cerr << "*********************************************************\n";
      cerr << " WARNING WARNING WARNING WARNING WARNING WARNING WARNING\n";
      cerr << "             if you have multiple timesteps you\n";
      cerr << "             should use \"-bv 0 or -nobv\"!!!\n";
      cerr << "*********************************************************\n";
      if(use_bv==1){
	scene->set_object(new BV1(scene->get_object()));
      } else if(use_bv==2){
	scene->set_object(new BV2(scene->get_object()));
      } else if(use_bv==3){
	  if (gridcellsize == -1) {
	      rtrt::Array1<Object*> prims;	
	      scene->get_object()->collect_prims(prims);
	      gridcellsize = (int)ceil(pow(prims.size(),1./3.));
	      cerr << "PS " << prims.size() << " GSS " << gridcellsize << endl;
	  }
	  scene->set_object(new Grid(scene->get_object(), gridcellsize));
      } else if(use_bv==4){
	  scene->set_object(new HierarchicalGrid( scene->get_object(), 
						  gridcellsize, 
						  gridcellsizeL2,
						  gridcellsizeL3,
						  minObjs1, minObjs2, 
						  rtrt_engine->nworkers ));
      } else if(use_bv==5){
	Object *obj = scene->get_object();
	Group *g = dynamic_cast<Group *>(obj);
	if (g)
	  for (int i=0; i<g->objs.size(); i++) {
	    Group *gg = new Group;
	    gg->add(new BV1(g->objs[i]));
	    g->objs[i] = gg;
	  }
	else
	  scene->set_object(new BV1(obj));
      } else if(use_bv==6){
                  scene->set_object(new Grid2( scene->get_object(), 
                                               gridcellsize));
      } else {
	cerr << "WARNING: Unknown bv method\n";
      }
    }
  } else {
    scene->set_object( scene->get_object() );
  }
  
  if(!scene->get_image(0)){
    Image* image0=new Image(xres, yres, false);
    scene->set_image(0, image0);
    Image* image1=new Image(xres, yres, false);
    scene->set_image(1, image1);
  }
  if(use_usercamera){
    usercamera.setup();
    *scene->get_camera(0)=usercamera;
  }
  scene->copy_camera(0);
  
  if(light_radius != -1){
    for(int i=0;i<scene->nlights();i++)
      scene->light(i)->radius=light_radius;
  }
  
  cerr << "Preprocessing\n";
  int pp_size=0;
  int scratchsize=0;
  scene->preprocess(bvscale, pp_size, scratchsize);
  cerr << "Done\n";

  // initialize jitter masks 
  
  for(int ii=0;ii<1000;ii++) {
    rtrt_engine->Gjitter_vals[ii] = drand48()*2 - 1.0;
    while (fabs(rtrt_engine->Gjitter_vals[ii]) >= 0.85) // sort of possion...
      rtrt_engine->Gjitter_vals[ii] = drand48()*2 - 1.0;
    
    rtrt_engine->Gjitter_vals[ii] *= 0.25;
    
    rtrt_engine->Gjitter_valsb[ii] = drand48()*2 - 1.0;
    while (fabs(rtrt_engine->Gjitter_valsb[ii]) >= 0.85)// sort of possion...
      rtrt_engine->Gjitter_valsb[ii] = drand48()*2 - 1.0;
    
    rtrt_engine->Gjitter_valsb[ii] *= 0.25;
  }
  
  double aspectRatio = (double)yres/(double)xres;

  // 0.5625 is the 9 to 5 (ish?) aspect ratio.
  scene->get_camera(0)->setWindowAspectRatio( aspectRatio );
  scene->get_camera(1)->setWindowAspectRatio( aspectRatio );

  // Start up display thread...
  Dpy* dpy=new Dpy(scene, rtrt_engine, criteria1, criteria2,
                   bench, ncounters, c0, c1, 1.0, 1.0, display_frames,
		   pp_size, scratchsize, fullscreen, do_frameless==true,
                   rserver, stereo );

  ThreadGroup *rtrt_engine_tg = new ThreadGroup("rtrt engine group");
  
  //////////////////////////////////////////////////////////////////
  // Setup the DpyGui thread

  DpyGui* dpygui = new DpyGui();
  dpygui->setDpy(dpy);
  dpygui->setRTRTEngine(rtrt_engine);
  dpygui->set_resolution(xres, yres);
  
  //////////////////////////////////////////////////////////////////
  // This is the glut glui stuff

  if (show_gui)
    dpygui->startDefaultGui();

  /*  bigler */
  new Thread(dpygui, "DpyGui", rtrt_engine_tg);
  new Thread(dpy, "Render Display", rtrt_engine_tg);

  // Start up worker threads...
  for(int i=0;i<rtrt_engine->nworkers;i++){
    char buf[100];
    sprintf(buf, "worker %d", i);
    Worker * worker = new Worker(dpy, scene, i,
				 pp_size, scratchsize,
				 ncounters, c0, c1);
    new Thread( worker, buf, rtrt_engine_tg);
  } // end for (create workers)

  // Add the other displays to our GUI for cleanup.
  scene->attach_displays_to_engine(dpygui);
  
  // If we return now, we won't be able to access stdin, or CNTR-C and
  // stuff.
  rtrt_engine_tg->join();
  
  return 0;
}

