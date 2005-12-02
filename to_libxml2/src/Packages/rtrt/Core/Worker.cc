
#include <Packages/rtrt/Core/Worker.h>

#include <Packages/rtrt/Core/CutGroup.h>
#include <Packages/rtrt/Core/Dpy.h>
#include <Packages/rtrt/Core/DpyPrivate.h>
#include <Packages/rtrt/Core/Image.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/Ray.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/HitInfo.h>
#include <Packages/rtrt/Core/Object.h>
#include <Packages/rtrt/Core/Material.h>
#include <Packages/rtrt/Core/CycleMaterial.h>
#include <Packages/rtrt/Core/InvisibleMaterial.h>
#include <Packages/rtrt/Core/PhongMaterial.h>
#include <Packages/rtrt/Core/Light.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/PerProcessorContext.h>
#include <Packages/rtrt/Core/MusilRNG.h>
#include <Packages/rtrt/Core/Context.h>
#include <Packages/rtrt/Core/rtrt.h>
#include <Packages/rtrt/Core/Names.h>

#include <Core/Thread/Barrier.h>
#include <Core/Thread/Mutex.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Time.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

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
    stop_(false), useAddSubBarrier_(false), rendering_scene(0)
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

extern float Galpha;

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
  
  if (!dpy->doing_frameless()) {

    int showing_scene = 1 - rendering_scene;
    
    // jittered masks for this stuff...
    double jitter_vals[1000];
    double jitter_valsb[1000];
  
    // make them from -1 to 1
  
    for(int ii=0;ii<1000;ii++) {
      jitter_vals[ii] = scene->get_rtrt_engine()->Gjitter_vals[ii];
      jitter_valsb[ii] = scene->get_rtrt_engine()->Gjitter_valsb[ii];
    }
  
    for(;;) {
      if( useAddSubBarrier_ ) {
	//cout << "stopping for threads, will wait for: " 
	//   << oldNumWorkers_+1 << " threads\n";

	useAddSubBarrier_ = false;
	addSubThreads_->wait( oldNumWorkers_+1 );

	// stop if you have been told to stop.  
	if( stop_ ) {
          //	  cerr << "Thread: " << num << " stopping\n";
	  return;
	}
      }

      stats[showing_scene]->add(SCIRun::Time::currentSeconds(), Color(0,1,0));

      // Sync.
      //cout << "b" << num << ": " << dpy->get_num_procs()+1 << "\n";
      barrier->wait(dpy->get_num_procs()+1);

      counters->end_frame();

      Stats* st=stats[rendering_scene];
      st->reset();

      // Sync.
      //cout << "B" << num << "\n";
      barrier->wait(dpy->get_num_procs()+1);

      int hotSpotsMode = dpy->rtrt_engine->hotSpotsMode;
      bool do_jitter = dpy->rtrt_engine->do_jitter;

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
      // This is used for the jittered code
      ASSERT(1000 > (xtilesize * ytilesize * 4));
      int nx=(xres+xtilesize-1)/xtilesize;
      Context cx(scene, st, ppc, rendering_scene, num);

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
	    ///////////////////////////////////////////////////////
	    // Stereo
	    ///////////////////////////////////////////////////////
	    double stime = 0;
	    if( hotSpotsMode )
	      stime = SCIRun::Time::currentSeconds();

	    if (!do_jitter) {
              Color Rcolor;
              Ray rayR;
	      //////////////////////////
	      // Non jitter
	      for(int y=sy;y<ey;y++){
		for(int x=sx;x<ex;x++){
		  camera->makeRayLR(ray, rayR, x+xoffset, y+yoffset,
                                    ixres, iyres);
		  traceRay(result, ray, 0, 1.0, Color(0,0,0), &cx);
		  traceRay(Rcolor, rayR, 0, 1.0, Color(0,0,0), &cx);
		  if( !hotSpotsMode ) {
		    (*image)(x,y).set(result);
		    (*image)(x,y+yres).set(Rcolor);
		  } else {
		    double etime=SCIRun::Time::currentSeconds();
		    double dt=(etime-stime)*0.5;
		    stime=etime;
		    (*image)(x,y).set(CMAP(dt));
		    (*image)(x,y+yres).set(CMAP(dt));
		  }
		}
	      }
	    } else {
	      ///////////////////////////
	      // Jitter
              Color Lcolor, Rcolor;
              Color Lcolor_sum, Rcolor_sum;
              Ray rayR;
              
              // We use the pixel coordinate to index into the
              // jittered samples.  We only have 1000 of these
              // jittered samples, so we need to make sure that we
              // loop back the indicies when x gets too large.
              //
              // The previous implementation reused the same x
              // jittered samples for every row (they still had
              // different y offets, though).  I'm going to use a
              // contiguous chunk of indicies for the whole tile, now.
              // This code will get you into trouble when the number
              // of pixels in a given tile gets above 1000.
              int xj_index = sx % (1000-(xtilesize*ytilesize*4));
              int yj_index = sy % (1000-(ytilesize*ytilesize*4));
	      for(int y=sy;y<ey;y++){
		for(int x=sx;x<ex;x++){
                  camera->makeRayLR(ray, rayR,
                                    x+xoffset - 0.25 + jitter_vals[xj_index],
                                    y+yoffset - 0.25 + jitter_valsb[yj_index],
                                    ixres, iyres);
                  traceRay(Lcolor_sum, ray, 0, 1.0, Color(0,0,0), &cx);
                  traceRay(Rcolor_sum, rayR, 0, 1.0, Color(0,0,0), &cx);
                  // Increment for the next set of pixels
                  xj_index++; yj_index++;
                  
                  camera->makeRayLR(ray, rayR,
                                    x+xoffset + 0.25 + jitter_vals[xj_index], 
                                    y+yoffset - 0.25 + jitter_valsb[yj_index],
                                    ixres, iyres);
                  traceRay(Lcolor, ray, 0, 1.0, Color(0,0,0), &cx);
                  traceRay(Rcolor, rayR, 0, 1.0, Color(0,0,0), &cx);
                  Lcolor_sum += Lcolor;
                  Rcolor_sum += Rcolor;
                  xj_index++; yj_index++;
                  
                  camera->makeRayLR(ray, rayR,
                                    x+xoffset + 0.25 + jitter_vals[xj_index],
                                    y+yoffset + 0.25 + jitter_valsb[yj_index],
                                    ixres, iyres);
                  traceRay(Lcolor, ray, 0, 1.0, Color(0,0,0), &cx);
                  traceRay(Rcolor, rayR, 0, 1.0, Color(0,0,0), &cx);
                  Lcolor_sum += Lcolor;
                  Rcolor_sum += Rcolor;
                  xj_index++; yj_index++;
                  
                  camera->makeRayLR(ray, rayR,
                                    x+xoffset - 0.25 + jitter_vals[xj_index],
                                    y+yoffset + 0.25 + jitter_valsb[yj_index],
                                    ixres, iyres);
                  traceRay(Lcolor, ray, 0, 1.0, Color(0,0,0), &cx);
                  traceRay(Rcolor, rayR, 0, 1.0, Color(0,0,0), &cx);
                  Lcolor_sum += Lcolor;
                  Rcolor_sum += Rcolor;
                  xj_index++; yj_index++;

		  if( !hotSpotsMode ) {
		    (*image)(x,y).set(Lcolor_sum*0.25f);
		    (*image)(x,y+yres).set(Rcolor_sum*0.25f);
		  } else {
		    double etime=SCIRun::Time::currentSeconds();
		    double dt=(etime-stime)*0.5;
		    stime=etime;
		    (*image)(x,y).set(CMAP(dt));
		    (*image)(x,y+yres).set(CMAP(dt));
		  }
		} // x loop
	      } // y loop
	    } // jitter section
	  } else {
	    ///////////////////////////////////////////////////////
	    // Mono
	    ///////////////////////////////////////////////////////
	    if (do_jitter) {
	      Color result2, result3, result4; // 4 samples
	      double stime = 0;
	      if( hotSpotsMode )
		stime = SCIRun::Time::currentSeconds();
              // See comments above for these two variables
              int xj_index = sx % (1000-(xtilesize*ytilesize*4));
              int yj_index = sy % (1000-(ytilesize*ytilesize*4));
	      for(int y=sy;y<ey;y++){
		for(int x=sx;x<ex;x++){
		  // do central ray plus 3 jittered samples...
                  camera->makeRay(ray,
                                  x+xoffset - 0.25 + jitter_vals[xj_index],
                                  y+yoffset - 0.25 + jitter_valsb[yj_index],
                                  ixres, iyres);
                  traceRay(result, ray, 0, 1.0, Color(0,0,0), &cx);
                  // Increment for the next set of pixels
                  xj_index++; yj_index++;
                  
                  camera->makeRay(ray,
                                  x+xoffset + 0.25 + jitter_vals[xj_index], 
                                  y+yoffset - 0.25 + jitter_valsb[yj_index],
                                  ixres, iyres);
                  traceRay(result2, ray, 0, 1.0, Color(0,0,0), &cx);
                  xj_index++; yj_index++;
                  
                  camera->makeRay(ray,
                                  x+xoffset + 0.25 + jitter_vals[xj_index],
                                  y+yoffset + 0.25 + jitter_valsb[yj_index],
                                  ixres, iyres);
                  traceRay(result3, ray, 0, 1.0, Color(0,0,0), &cx);
                  xj_index++; yj_index++;
                  
                  camera->makeRay(ray,
                                  x+xoffset - 0.25 + jitter_vals[xj_index],
                                  y+yoffset + 0.25 + jitter_valsb[yj_index],
                                  ixres, iyres);
                  traceRay(result4, ray, 0, 1.0, Color(0,0,0), &cx);
                  xj_index++; yj_index++;
		  
		  if( (hotSpotsMode == RTRT::HotSpotsOn) ||
		      (hotSpotsMode == RTRT::HotSpotsHalfScreen &&
                       (x < halfXres) ) ) {
		    double etime=SCIRun::Time::currentSeconds();
		    double t=etime-stime;	
		    stime=etime;
		    (*image)(x,y).set(CMAP(t));
		  } else {
		    result = (result+result2+result3+result4)*0.25f;
		    (*image)(x,y).set(result);
		  }
		}
	      }
	      // end if( do_jitter )
	    } else {
	      double stime;
	      bool   transMode = dpy->rtrt_engine->frameMode == RTRT::OddRows;
	      if( hotSpotsMode ) {
		stime = SCIRun::Time::currentSeconds();
		if( hotSpotsMode == RTRT::HotSpotsOn){
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
	      } else if (scene->store_depth) {
                // This stores off the depth stuff
                double dist;
		for(int y=sy;y<ey;y++){
		  for(int x=sx;x<ex;x++){
		    camera->makeRay(ray, x+xoffset, y+yoffset, ixres, iyres);
		    traceRay(result, ray, 0, 1.0, Color(0,0,0), &cx, dist);
		    (*image)(x,y).set(result);
                    image->set_depth(x,y,dist);
		  }
		}
                
              } else {
                /////////////////////////////////////////////////
                // Here's the most default case.  Do simple single sample
                /////////////////////////////////////////////////
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

      rendering_scene=1-rendering_scene;
      showing_scene=1-showing_scene;
    }

  } else { // FRAMELES RENDERING...
    renderFrameless();
    // Force the return if we exit from this function
    return;
  }
} // end run()

#ifdef DEBUG
void
Worker::traceRay(Color& result, Ray& ray, int depth,
		 double atten, const Color& accumcolor,
		 Context* cx)
{
  HitInfo hit;
  Object* obj=cx->scene->get_object();
  cx->stats->ds[depth].nrays++;

  if (cx->ppc->debug()) cerr << "Worker::traceRay::starting intersection with ray: dir("<<ray.direction()<<"), origin("<<ray.origin()<<")\n";

  obj->intersect(ray, hit, &cx->stats->ds[depth], cx->ppc);

  if(hit.was_hit){
    if (cx->ppc->debug()) cerr << "Worker::traceRay::object ("<<Names::getName(hit.hit_obj)<<") was hit "<<hit.min_t<<" away.\n";

    hit.hit_obj->get_matl()->shade(result, ray, hit, depth,
				   atten, accumcolor, cx);
  } else {
    if (cx->ppc->debug()) cerr << "Worker::traceRay:: no object hit\n";
    cx->stats->ds[depth].nbg++;
    cx->scene->get_bgcolor( ray.direction(), result );
  }
  if (cx->ppc->debug()) cerr << "Done Shading. result = ("<<result<<")\n";
}

#else

void
Worker::traceRay(Color& result, Ray& ray, int depth,
		 double atten, const Color& accumcolor,
		 Context* cx)
{
  HitInfo hit;
  Object* obj=cx->scene->get_object();
  cx->stats->ds[depth].nrays++;

  obj->intersect(ray, hit, &cx->stats->ds[depth], cx->ppc);

  if(hit.was_hit){
    hit.hit_obj->get_matl()->shade(result, ray, hit, depth,
				   atten, accumcolor, cx);
  } else {
    cx->stats->ds[depth].nbg++;
    cx->scene->get_bgcolor( ray.direction(), result );
  }
}
#endif // ifdef DEBUG

void Worker::traceRay(Color& result, Ray& ray, int depth,
		      double atten, const Color& accumcolor,
		      Context* cx, double &dist)
{
  HitInfo hit;
  Object* obj = cx->scene->get_object();
  cx->stats->ds[depth].nrays++;
  obj->intersect(ray, hit, &cx->stats->ds[depth], cx->ppc);
  if(hit.was_hit){
    hit.hit_obj->get_matl()->shade(result, ray, hit, depth,
				   atten, accumcolor, cx);
    dist=hit.min_t;
  } else {
    cx->stats->ds[depth].nbg++;
    cx->scene->get_bgcolor( ray.direction(), result );
    dist = MAXDOUBLE;
  }
}

void Worker::traceRay(Color& result, Ray& ray, int depth,
		      double atten, const Color& accumcolor,
		      Context* cx, Object* obj)
{
  HitInfo hit;
  cx->stats->ds[depth].nrays++;
  obj->intersect(ray, hit, &cx->stats->ds[depth], cx->ppc);
  if(hit.was_hit){
    hit.hit_obj->get_matl()->shade(result, ray, hit, depth,
				   atten, accumcolor, cx);
  } else {
    cx->stats->ds[depth].nbg++;
    cx->scene->get_bgcolor( ray.direction(), result );
  }
}

void Worker::traceRay(Color& result, Ray& ray, int depth,
		      double atten, const Color& accumcolor,
		      Context* cx, Object* obj, double &dist)
{
  HitInfo hit;
  cx->stats->ds[depth].nrays++;
  obj->intersect(ray, hit, &cx->stats->ds[depth], cx->ppc);
  if(hit.was_hit){
    hit.hit_obj->get_matl()->shade(result, ray, hit, depth,
				   atten, accumcolor, cx);
    dist = hit.min_t;
  } else {
    cx->stats->ds[depth].nbg++;
    cx->scene->get_bgcolor( ray.direction(), result );
    dist = MAXDOUBLE;
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

// Having this code in the file made things confusing.
#include <Packages/rtrt/Core/WorkerFrameless.cc>
