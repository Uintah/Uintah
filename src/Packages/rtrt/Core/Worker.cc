
#include <Packages/rtrt/Core/Worker.h>

#include <Packages/rtrt/Core/CutGroup.h>
#include <Packages/rtrt/Core/Dpy.h>
#include <Packages/rtrt/Core/Image.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Mutex.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/CycleMaterial.h>
#include <Packages/rtrt/Core/InvisibleMaterial.h>
#include <Packages/rtrt/Core/PhongMaterial.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/PerProcessorContext.h>
#include <Packages/rtrt/Core/MusilRNG.h>
#include <Packages/rtrt/Core/Context.h>
#include <iostream>
#ifdef __sgi
#include <sys/sysmp.h>
#endif
#include <unistd.h>
#if 0
#include <SpeedShop/api.h>
#endif

using namespace rtrt;
using namespace SCIRun;
using namespace std;

extern void run_gl_test();

extern Mutex io_lock_;

#define CSCALE 5.e3
/*#define CMAP(t) Color(t*CSCALE,0,1-(t)*CSCALE)*/
#define CMAP(t) Color(t*CSCALE,t*CSCALE,t*CSCALE)

namespace rtrt {
  extern Mutex io_lock_;
  
} // end namespace rtrt

extern void hilbert_i2c( int n, int m, long int r, int a[]);

Worker::Worker(Dpy* dpy, Scene* scene, int num, int pp_size, int scratchsize,
	       int ncounters, int c0, int c1)
  : dpy(dpy), num(num), scene(scene), ncounters(ncounters), c0(c0), c1(c1),
    stop_(false), useAddSubBarrier_(false)
{
  if(dpy){
    dpy->register_worker(num, this);
    dpy->get_barriers( barrier, addSubThreads_ );
  }
  stats[0]=new Stats(1000);
  stats[1]=new Stats(1000);
  ppc=new PerProcessorContext(pp_size, scratchsize);
}

Worker::~Worker()
{
  delete ppc;
}

Stats* Worker::get_stats(int idx)
{
  return stats[idx];
}

// this stuff is just in global variables for now...

//int NUMCHUNKS=1<<16;
//double updatePercent=0.5;
//int clusterSize=1;
//int shuffleClusters=1;

//int np=4;

extern float Galpha;

//int framelessMode = 1; // default is the other mode...

//int do_jitter=0;

//double Gjitter_vals[1000],Gjitter_valsb[1000];

bool pin = false;

void Worker::run()
{
  if(pin)
    Thread::self()->migrate((num+2)%Thread::numProcessors());
#if 0
  io_lock_.lock();
  cerr << "worker pid " << getpid() << '\n';
  io_lock_.unlock();
#endif
  if (scene->get_rtrt_engine()->worker_run_gl_test)
    run_gl_test();
  if(ncounters)
    counters=new Counters(ncounters, c0, c1);
  else
    counters=0;
  int np = scene->get_rtrt_engine()->np;
  
#if 1
  rendering_scene=0;
  int showing_scene=1;
  
  // make them from -1 to 1
  
  // jittered masks for this stuff...
  double jitter_vals[1000];
  double jitter_valsb[1000];
  
  // make them from -1 to 1
  
  for(int ii=0;ii<1000;ii++) {
    jitter_vals[ii] = scene->get_rtrt_engine()->Gjitter_vals[ii];
    jitter_valsb[ii] = scene->get_rtrt_engine()->Gjitter_valsb[ii];
  }
  
  if (!dpy->doing_frameless()) {
    
    for(;;){
      if( useAddSubBarrier_ ) {
	//cout << "stopping for threads, will wait for: " 
	//   << oldNumWorkers_+1 << " threads\n";

	useAddSubBarrier_ = false;
	addSubThreads_->wait( oldNumWorkers_+1 );
	// stop if you have been told to stop.  

	//cout << "stop is " << stop_ << "\n";

	if( stop_ ) {// I don't think this one ever is called
	  //cout << "Thread: " << num << " stopping\n";
	  my_thread_->exit();
	  cout << "Worker.cc: should never get here!\n";
	  return;
	}
      }

      stats[showing_scene]->add(SCIRun::Time::currentSeconds(), Color(0,1,0));

      // Sync.
      //cout << "b" << num << ": " << dpy->get_num_procs()+1 << "\n";
      barrier->wait(dpy->get_num_procs()+1);

      // exit if you are supposed to
      if (scene->get_rtrt_engine()->stop_execution()) {
	return;
      }	

      counters->end_frame();

      Stats* st=stats[rendering_scene];
      st->reset();

      // Sync.
      //cout << "B" << num << "\n";
      barrier->wait(dpy->get_num_procs()+1);

      int hotSpotMode = scene->getHotSpotsMode();

#if 0
      //////////////////////
      // For debugging... //
      int shadow_mode_at_beginning = scene->shadow_mode;
      Camera cam_at_beg = *(scene->get_camera(rendering_scene));
      //////////////////////
#endif

      st->add(SCIRun::Time::currentSeconds(), Color(1,0,0));
      Image* image=scene->get_image(rendering_scene);
      Camera* camera=scene->get_camera(rendering_scene);
      int    xres=image->get_xres();
      int    halfXres = xres / 2;
      int    yres=image->get_yres();
      bool   stereo=image->get_stereo();
      double ixres=1./xres;
      double iyres=1./yres;
      double xoffset=scene->xoffset;
      double yoffset=scene->yoffset;
      int    stile, etile;
      int    n = 0;
      WorkQueue& work=scene->work;
      // <<<< bigler >>>>
      //st->add(Thread::currentSeconds(), Color(0,0,0));
      st->add(SCIRun::Time::currentSeconds(), Color(0,0,0));
      int xtilesize=scene->xtilesize;
      int ytilesize=scene->ytilesize;
      int nx=(xres+xtilesize-1)/xtilesize;
      Context cx(this, scene, st);
      while(work.nextAssignment(stile, etile)){

	Ray   ray;
	Color result;

	for(int tile=stile;tile<etile;tile++){
	  int ytile=tile/nx;
	  int xtile=tile%nx;
	  int sx=xtile*xtilesize;
	  int ex=(xtile+1)*xtilesize;
	  int sy=ytile*ytilesize;
	  int ey=(ytile+1)*ytilesize;
	  if(ey>yres)
	    ey=yres;
	  if(ex>xres)
	    ex=xres;
	  st->npixels+=(ex-sx)*(ey-sy);
	  if(stereo){
	    static bool warned = false;
	    if( !warned ) {
	      warned = true;
	      cout << "WARNING: Stereo not implemented now!\n";
	    }
	  } else {
	    if (scene->get_rtrt_engine()->do_jitter) {
	      Color sum;
	      Color resulta,resultb; // 4 samples
	      double stime = 0;
	      if( hotSpotMode )
		stime = SCIRun::Time::currentSeconds();
	      for(int y=sy;y<ey;y++){
		for(int x=sx;x<ex;x++){
		  // do central ray plus 3 jittered samples...
		  camera->makeRay(ray, x+xoffset -0.25 + jitter_vals[x], 
				  y+yoffset -0.25 + jitter_valsb[y], ixres, iyres);
		  traceRay(sum, ray, 0, 1.0, Color(0,0,0), &cx);
		  
		  camera->makeRay(ray, x+xoffset +0.25 + jitter_vals[x], 
				  y+yoffset -0.25 + jitter_valsb[y], ixres, iyres);
		  traceRay(result, ray, 0, 1.0, Color(0,0,0), &cx);
		  
		  camera->makeRay(ray, x+xoffset +0.25 + jitter_vals[x], 
				  y+yoffset +0.25 + jitter_valsb[y], ixres, iyres);
		  traceRay(resulta, ray, 0, 1.0, Color(0,0,0), &cx);
		  
		  camera->makeRay(ray, x+xoffset -0.25 + jitter_vals[x], 
				  y+yoffset +0.25 + jitter_valsb[y], ixres, iyres);
		  traceRay(resultb, ray, 0, 1.0, Color(0,0,0), &cx);
		  
		  if( (hotSpotMode == 1) ||
		      (hotSpotMode == 2 && (x < halfXres) ) ){
		    double etime=SCIRun::Time::currentSeconds();
		    double t=etime-stime;	
		    stime=etime;
		    (*image)(x,y).set(CMAP(t));
		  } else {
		    sum = (sum+result+resulta+resultb)*0.25;
		    (*image)(x,y).set(sum);
		  }
		}
	      }
	      // end if( do_jitter )
	    } else {
	      double stime;
	      bool   transMode = scene->doTransmissionMode();
	      if( hotSpotMode ) {
		stime = SCIRun::Time::currentSeconds();
		if( hotSpotMode == 1){
		  // Hot Spot Mode: 1
		  for(int y=sy;y<ey;y++){
		    for(int x=sx;x<ex;x++){
		      camera->makeRay(ray, x+xoffset, y+yoffset, ixres, iyres);
		      traceRay(result, ray, 0, 1.0, Color(0,0,0), &cx);
		      double etime=SCIRun::Time::currentSeconds();
		      double t=etime-stime;	
		      stime=etime;
		      (*image)(x,y).set(CMAP(t));
		    }
		  }
		} else {
		  // Hot Spot Mode: 2
		  int ex1, sx2;

		  if( sx > halfXres ){
		    ex1 = 0;
		    sx2 = sx;
		  } else {
		    ex1 = Min( ex, halfXres );
		    sx2 = halfXres;
		  }

		  for(int y=sy;y<ey;y++){
		    // Draw scanline up to middle of screen.
		    for(int x=sx;x<ex1;x++){
		      camera->makeRay(ray, x+xoffset, y+yoffset, ixres, iyres);
		      traceRay(result, ray, 0, 1.0, Color(0,0,0), &cx);
		      double etime=SCIRun::Time::currentSeconds();
		      double t=etime-stime;	
		      stime=etime;
		      (*image)(x,y).set(CMAP(t));
		    }
		    // Draw scanline past middle of screen.
		    for(int x=sx2;x<ex;x++){
		      if( transMode && (y % 2 == 0) ){
			(*image)(x,y).set(Color(0,0,0));
			continue;
		      }
		      camera->makeRay(ray, x+xoffset, y+yoffset, ixres, iyres);
		      traceRay(result, ray, 0, 1.0, Color(0,0,0), &cx);
		      (*image)(x,y).set(result);
		    }
		  }
		} // end else hot spot mode 2.
	      } else {
		for(int y=sy;y<ey;y++){
		  for(int x=sx;x<ex;x++){
		    if( transMode && (y % 2 == 0) ){
		      (*image)(x,y).set(Color(0,0,0));
		      continue;
		    }
		    camera->makeRay(ray, x+xoffset, y+yoffset, ixres, iyres);
		    traceRay(result, ray, 0, 1.0, Color(0,0,0), &cx);
		    (*image)(x,y).set(result);
		  }
		}
	      }
	    }
	  }
	}
	st->add(SCIRun::Time::currentSeconds(), (n%2)?Color(0,0,0):Color(1,1,1));
	n++;
      }

#if 0
      if( cam_at_beg != *(scene->get_camera(rendering_scene)) )
	{
	  cout << "ERROR camera changed in the middle of the rendering!\n";
	}
      if( shadow_mode_at_beginning != scene->shadow_mode )
	{
	  cout << "ERROR shadow mode changed!\n";
	}
#endif

      rendering_scene=1-rendering_scene;
      showing_scene=1-showing_scene;
    }

  } else { // FRAMELES RENDERING...
    // Read comment in WorkerFrameless.cc
#   include<Packages/rtrt/Core/WorkerFrameless.cc>
  }
#endif
} // end run()

void
Worker::traceRay(Color& result, Ray& ray, int depth,
		 double atten, const Color& accumcolor,
		 Context* cx)
{
  HitInfo hit;
  Object* obj=scene->get_object();
  cx->stats->ds[depth].nrays++;
  obj->intersect(ray, hit, &cx->stats->ds[depth], ppc);
  if(hit.was_hit){
    cx->ppc = ppc;
    hit.hit_obj->get_matl()->shade(result, ray, hit, depth,
				   atten, accumcolor, cx);
  } else {
    cx->stats->ds[depth].nbg++;
    scene->get_bgcolor( ray.direction(), result );
  }
}

void Worker::traceRay(Color& result, Ray& ray, int depth,
		      double atten, const Color& accumcolor,
		      Context* cx, double &dist)
{
  HitInfo hit;
  Object* obj=scene->get_object();
  cx->stats->ds[depth].nrays++;
  obj->intersect(ray, hit, &cx->stats->ds[depth], ppc);
  if(hit.was_hit){
    cx->ppc = ppc;
    hit.hit_obj->get_matl()->shade(result, ray, hit, depth,
				   atten, accumcolor, cx);
    dist=hit.min_t;
  } else {
    cx->stats->ds[depth].nbg++;
    scene->get_bgcolor( ray.direction(), result );
    dist = MAXDOUBLE;
  }
}

void Worker::traceRay(Color& result, Ray& ray, int depth,
		      double atten, const Color& accumcolor,
		      Context* cx, Object* obj)
{
  HitInfo hit;
  cx->stats->ds[depth].nrays++;
  obj->intersect(ray, hit, &cx->stats->ds[depth], ppc);
  if(hit.was_hit){
    hit.hit_obj->get_matl()->shade(result, ray, hit, depth,
				   atten, accumcolor, cx);
  } else {
    cx->stats->ds[depth].nbg++;
    scene->get_bgcolor( ray.direction(), result );
  }
}

void Worker::traceRay(Color& result, Ray& ray, int depth,
		      double atten, const Color& accumcolor,
		      Context* cx, Object* obj, double &dist)
{
  HitInfo hit;
  cx->stats->ds[depth].nrays++;
  obj->intersect(ray, hit, &cx->stats->ds[depth], ppc);
  if(hit.was_hit){
    hit.hit_obj->get_matl()->shade(result, ray, hit, depth,
				   atten, accumcolor, cx);
    dist = hit.min_t;
  } else {
    cx->stats->ds[depth].nbg++;
    scene->get_bgcolor( ray.direction(), result );
    dist = MAXDOUBLE;
  }
}

void Worker::traceRay(Color& result, Ray& ray,
		      Point& hitpos, Object*& hitobj)
{
  HitInfo hit;
  Context cx(this, scene, stats[0]);
  scene->get_object()->intersect(ray, hit, &cx.stats->ds[0], ppc);
  if(hit.was_hit){
    hit.hit_obj->get_matl()->shade(result, ray, hit, 0,
				   0.0, Color(0,0,0), &cx);
    hitpos=ray.origin()+ray.direction()*hit.min_t;
    hitobj=hit.hit_obj;
  } else {
    hitobj=0;
    scene->get_bgcolor( ray.direction(), result );
  }
}

void
Worker::syncForNumThreadChange( int oldNumWorkers, bool stop /* = false */ )
{
  //cout << "W" << num << " sync" << oldNumWorkers << "," << stop << "\n";
  oldNumWorkers_ = oldNumWorkers;
  useAddSubBarrier_ = true;
  stop_ = stop;
}
