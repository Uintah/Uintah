
/*
 * Real time ray-tracer
 */

#include <Core/Thread/Thread.h>
#include <Core/Thread/ThreadGroup.h>
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
#include <dlfcn.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/param.h>

using namespace rtrt;
using namespace std;
//using std::endl;

//static int nworkers=1;

using SCIRun::Thread;
using SCIRun::ThreadGroup;

static void usage(char* progname)
{
    cerr << "usage: " << progname << " [options] -scene name [scene options]\n";
    cerr << "Options:\n";
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
    cerr << " -gridcellsize n  - Set the number of gridcells for -bv 3\n";
    cerr << " -perfex n1 n2    - Collect performance counters n1 and n2 for\n";
    cerr << "                   each frame.\n";
    cerr << " -visual criteria - Uses criteria for selecting the OpenGL\n";
    cerr << "                    visual for the image display.  See\n";
    cerr << "                    visinfo/findvis.1 for more info.\n";
    cerr << " -bench           - Disable frame display, and just render the\n";
    cerr << "                    image 100 times for timing purposes.\n";
    cerr << " -no_shadows      - Turn off shadows\n";
    cerr << " -shadows mode    - Select mode for shadows\n";
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
    cerr << " -jitter          - jittered masks - fixed table for now\n";
    cerr << " -worker_gltest   - calls run_gl_test from worker threads\n";
    cerr << " -display_gltest  - calls run_gl_test from display thread\n";
    cerr << " -displayless     - do not display and write a frame to displayless\n";

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

int main(int argc, char* argv[])
{
  RTRT* rtrt_engine = new RTRT();

  int xres=360;
  int yres=360;
  int use_bv=1;
  char* scenename=0;
  int gridcellsize=4;
  int c0;
  int c1;
  int ncounters=0;
  bool bench=false;
  ShadowType shadow_mode = No_Shadows;
  bool no_aa=false;
  double bvscale=.3;
  char* criteria1="db, stereo, max rgb, max accumrgb";
  char* criteria2="db, max rgb, max accumrgb";
  double light_radius=-1;
  
  bool do_frameless=false;
  bool display_frames=true;
  
  Camera usercamera(Point(1,0,0), Point(0,0,0), Vector(0,0,1), 60);
  bool use_usercamera;
  
  int scene_argc=-1;
  char** scene_argv=0;

  int num_rtrt = 1;
  for(int i=1;i<argc;i++){
    if(strcmp(argv[i], "-np")==0){
      i++;
      rtrt_engine->nworkers=atoi(argv[i]);
      rtrt_engine->np = rtrt_engine->nworkers;
    } else if(strcmp(argv[i], "-nr")==0){
      i++;
      num_rtrt=atoi(argv[i]);
    } else if(strcmp(argv[i], "-nobv")==0){
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
      shadow_mode = No_Shadows;
    } else if(strcmp(argv[i], "-shadows")==0){
      i++;
      shadow_mode = (ShadowType)atoi(argv[i]);
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
    } else {
      cerr << "Unknown option: " << argv[i] << '\n';
      usage(argv[0]);
      exit(1);
    }
  }
  
  if(!scenename){
    cerr << "Must specify a scene with -scene\n";
    exit(1);
  }
  char scenefile[MAXPATHLEN];
  sprintf(scenefile, "./%s.mo", scenename);
  void* handle=dlopen(scenefile, RTLD_NOW);
  if(!handle){
    cerr << "Error opening scene: " << scenename << '\n';
    cerr << dlerror() << '\n';
    exit(1);
  }
  void* scene_fn=dlsym(handle, "make_scene");
  if(!scene_fn){
    cerr << "Scene file found, but make_scene() function not found\n";
    exit(1);
  }
  Scene* (*make_scene)(int,char**,int) = (Scene*(*)(int,char**,int))scene_fn;
  for (int nr = 0; nr < num_rtrt; nr++) {
    Scene* scene=(*make_scene)(scene_argc, scene_argv, rtrt_engine->nworkers);
    if(!scene){
      cerr << "Scene creation failed!\n";
      exit(1);
    }
  
    // set the scenes rtrt_engine pointer
    scene->set_rtrt_engine(rtrt_engine);
  
    if(shadow_mode > Uncached_Shadows ){
      cerr << "Unknown shadow mode: " << shadow_mode << '\n';
      exit(1);
    }

    scene->no_aa=no_aa;
  
    if(use_bv){
      if(scene->nprims() > 1){
	if(use_bv==1){
	  scene->set_object(new BV1(scene->get_object()));
	} else if(use_bv==2){
	  scene->set_object(new BV2(scene->get_object()));
	} else if(use_bv==3){
	  scene->set_object(new Grid(scene->get_object(), gridcellsize));
	} else {
	  cerr << "WARNING: Unknown bv method\n";
	}
      }
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
    
      rtrt_engine->Gjitter_valsb[ii] = drand48() - 1.0;
      while (fabs(rtrt_engine->Gjitter_valsb[ii]) >= 0.85)// sort of possion...
	rtrt_engine->Gjitter_valsb[ii] = drand48() - 1.0;
    
      rtrt_engine->Gjitter_valsb[ii] *= 0.25;
    }
  
    //  ThreadGroup *group = new ThreadGroup("rtrt group");

    // Start up display thread...
    Dpy* dpy=new Dpy(scene, criteria1, criteria2, rtrt_engine->nworkers, bench,
		     ncounters, c0, c1, 1.0, 1.0, display_frames,
		     pp_size, scratchsize, false, do_frameless==true, false);
    /* <<<< bigler >>>> */
    char buf1[100];
    sprintf(buf1, "Display thread %d", nr);
    (new Thread(dpy, buf1))->detach();
    //new Thread(dpy, "Display thread", group);
    
    // Start up worker threads...
    for(int i=0;i<rtrt_engine->nworkers;i++){
      char buf[100];
      sprintf(buf, "worker %d of rtrt(%d)", i, nr);
#if 1
      (new Thread(new Worker(dpy, scene, i,
			     pp_size, scratchsize,
			     ncounters, c0, c1), buf))->detach();
      //			     ncounters, c0, c1), buf, group);
#endif
    }
  }

  //  group->join();
  //  cout << "Threads exited" << endl;
  
}

