
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

void Worker::run()
{
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
#if 0
  int np=Thread::numProcessors();
  int p=(50+num)%np;
  io_lock_.lock();
  cerr << "Mustrun: " << p << "(pid=" << getpid() << ")\n";
  io_lock_.unlock();
  if(sysmp(MP_MUSTRUN, p) == -1){
    perror("sysmp - MP_MUSTRUN");
  }
#endif
  
#if 1
  int rendering_scene=0;
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
      int xres=image->get_xres();
      int yres=image->get_yres();
      bool stereo=image->get_stereo();
      double ixres=1./xres;
      double iyres=1./yres;
      double xoffset=scene->xoffset;
      double yoffset=scene->yoffset;
      int stile, etile;
      int n=0;
      WorkQueue& work=scene->work;
      // <<<< bigler >>>>
      //st->add(Thread::currentSeconds(), Color(0,0,0));
      st->add(SCIRun::Time::currentSeconds(), Color(0,0,0));
      int xtilesize=scene->xtilesize;
      int ytilesize=scene->ytilesize;
      int nx=(xres+xtilesize-1)/xtilesize;
      Context cx(this, scene, st);
      while(work.nextAssignment(stile, etile)){
	Ray ray;
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
	    double stime;
	    if(scene->doHotSpots())
	      // <<<< bigler >>>>
	      //stime=Thread::currentSeconds();
	      stime=SCIRun::Time::currentSeconds();
	    int ny_total = image->get_yres();
	    for(int y=sy;y<ey;y++){
	      for(int x=sx;x<ex;x++){
		camera->makeRayL(ray, x+xoffset, y+yoffset, ixres, iyres);
		traceRay(result, ray, 0, 1.0, Color(0,0,0), &cx);
		if(scene->doHotSpots()){
		  // <<<< bigler >>>>
		  //double etime=Thread::currentSeconds();
		  double etime=SCIRun::Time::currentSeconds();
		  double t=etime-stime;	
		  stime=etime;
		  (*image)(x,y).set(CMAP(t));
		} else {
		  (*image)(x,y).set(result);
		}
	      }
	      for(int x=sx;x<ex;x++){
		camera->makeRayR(ray, x+xoffset, y+yoffset, ixres, iyres);
		traceRay(result, ray, 0, 1.0, Color(0,0,0), &cx);
		if(scene->doHotSpots()){
		  //double etime=Thread::currentSeconds();
		  double etime=SCIRun::Time::currentSeconds();
		  double t=etime-stime;	
		  stime=etime;
		  (*image)(x,y+ny_total).set(CMAP(t));
		} else {
		  (*image)(x,y+ny_total).set(result);
		}
	      }
	    }
	  } else {
	    if (scene->get_rtrt_engine()->do_jitter) {
	      Color sum;
	      Color resulta,resultb; // 4 samples
	      double stime;
	      if(scene->doHotSpots())
		//stime=Thread::currentSeconds();
		stime=SCIRun::Time::currentSeconds();
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
		  
		  sum = (sum+result+resulta+resultb)*0.25;
		  if(scene->doHotSpots()){
		    //double etime=Thread::currentSeconds();
		    double etime=SCIRun::Time::currentSeconds();
		    double t=etime-stime;	
		    stime=etime;
		    (*image)(x,y).set(CMAP(t));
		  } else {
		    (*image)(x,y).set(sum);
		  }
		}
	      }
	    } else {
	      double stime;
	      if(scene->doHotSpots()){
		stime=SCIRun::Time::currentSeconds();
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
		bool transMode = scene->doTransmissionMode();
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
	//st->add(Thread::currentSeconds(), (n%2)?Color(0,0,0):Color(1,1,1));
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

  } else { // we're doing frameles rendering...
    int clusterSize = scene->get_rtrt_engine()->clusterSize;
    showing_scene=0; // you just need 1 for this...
    
    Camera rcamera;
    Camera *camera = &rcamera; // keep a copy...
    
    // figure out the pixels that each proc has to
    // draw

    Array1<int> xpos;
    Array1<int> ypos;
      
    Image* image=scene->get_image(rendering_scene);
    int xres=image->get_xres();
    int yres=image->get_yres();

    int nwork=0;

    if (scene->get_rtrt_engine()->framelessMode == 0) { // do hilbert curve stuff...
	
      int uxres = xres/clusterSize;
      int uyres = yres/clusterSize;
	
      // don't just use 1 interval, have the procs
      // interleave themselves
	
      int chunksize = uxres*uyres/scene->get_rtrt_engine()->NUMCHUNKS;
      int chunki = scene->get_rtrt_engine()->NUMCHUNKS/np;  // num chunk intervals
      nwork = chunksize*chunki*clusterSize*clusterSize;
      int slopstart=0;
      int totslop=0;
	
      // permute the chunk order...
	
      Array1<int> chunkorder(chunki);
      Array1<int> chunksizes(chunki,chunksize);
	
      if (chunki*np != scene->get_rtrt_engine()->NUMCHUNKS) {
	slopstart = chunksize*chunki*np;
	totslop = scene->get_rtrt_engine()->NUMCHUNKS*chunksize - np*chunki*chunksize;
	if (!num) {
	  io_lock_.lock();
	  cerr<< "Slop: " << slopstart << " " << totslop << endl;
	  io_lock_.unlock();
	}
	  
	// just do pixel interleaving of this stuff...
      }
	
      for(int ii=0;ii<chunki;ii++) {
	chunkorder[ii] = ii; // first i chunks
      }
#if 1
	
      if (scene->get_rtrt_engine()->shuffleClusters) {
	  
	io_lock_.lock();
	for(int jj=0;jj<chunkorder.size();jj++) {
	  int swappos = drand48()*chunkorder.size();
	  if (swappos == chunki) {
	    cerr << "Bad Swap!\n";
	  }
	  int tmp = chunkorder[swappos];
	  chunkorder[swappos] = chunkorder[jj];
	  chunkorder[jj] = tmp;
	    
	  // also swap the sizes...
	  tmp = chunksizes[swappos];
	  chunksizes[swappos] = chunksizes[jj];
	  chunksizes[jj] = tmp;
	}
	io_lock_.unlock();
      }
#endif
	
      if (!num) {
	cerr << uxres << " " << uyres << endl;
	cerr << clusterSize << " " << chunksize << " " << nwork<< endl;
      }
      
      xpos.resize(nwork);
      ypos.resize(nwork);
	
      int coords[2];
	
      int bits=1;
      int sz=1<<bits;
	
      while(sz != uxres) {
	bits++;
	sz = sz<<1;
      }
	
      // force the tables to be built for hilbert...
	
      io_lock_.lock();
      hilbert_i2c(2,bits,0,coords);
      io_lock_.unlock();
#if 0	
      Pixel procColor;
	
      procColor.r = (1.0/np)*num*255.0;
      procColor.g = (1.0/np)*(np-num)*255.0;
      procColor.b = (num&1)?0:255;
#endif
      int index=0;
      int basei = num*chunksize; // start at proc, bump up
	
      // the chunks should really be permuted...
	
      //io_lock_.lock();
      for(int chnk=0;chnk<chunki;chnk++) {
	  
	basei = num*chunksize + chunkorder[chnk]*chunksize*np;
	  
	for(int i=0;i<chunksize;i++) {
	  hilbert_i2c(2,bits,basei+i,coords);
	    
	  for(int yii=0;yii<clusterSize;yii++) {
	    for (int xii=0;xii<clusterSize;xii++) {
	      xpos[index] = coords[0]*clusterSize + xii;
	      ypos[index] = coords[1]*clusterSize + yii;
	      index++;
	    }
	  }
	}
      }
      //io_lock_.unlock();
#if 1
      if (totslop) {
	int pos = slopstart + num;
	  
	if (!num) {
	  cerr << "Doing Slop: " << pos << endl;
	}
	  
	while(pos < uxres*uyres) {
	    
	  hilbert_i2c(2,bits,pos,coords);
	    
	  for(int yii=0;yii<clusterSize;yii++) {
	    for (int xii=0;xii<clusterSize;xii++) {
	      xpos.add(coords[0]*clusterSize + xii);
	      ypos.add(coords[1]*clusterSize + yii);
	    }	
	  }	
	  pos += np;
	}
	  
      }
#endif
    } else { // just use scan line stuff...

      // divide the screen into clustersize
      // pieces (along x), then interleave
      // the procs on these intervals

	// you have slop if these things don't divide
	// evenly, take care of that later...
	
      int numintervals = xres*yres/clusterSize;
      int iperproc = numintervals/np;

	// assume no slop for now...

      Array1<int> myintervals(iperproc);
      Array1<int> mysizes(iperproc);

      for(int i=0;i<iperproc;i++) {
	myintervals[i] = (i*np+ (num+i)%np)*clusterSize; 
	mysizes[i] = clusterSize;
      }

      // need to add clusters for the slop...

      if (scene->get_rtrt_engine()->shuffleClusters) {
	io_lock_.lock(); // drand48 isn't thread safe...
	for(int i=0;i<myintervals.size();i++) {
	  int swappos = drand48()*myintervals.size();
	  if (swappos != i) {
	    int tmp = myintervals[i];
	    myintervals[i] = myintervals[swappos];
	    myintervals[swappos] = tmp;

	    tmp = mysizes[i];
	    mysizes[i] = mysizes[swappos];
	    mysizes[swappos] = tmp;
	  }
	}
	io_lock_.unlock();
      }

      xpos.resize(iperproc*clusterSize);
      ypos.resize(iperproc*clusterSize);

      int pi=0;

      io_lock_.lock();
      for(int chunk = 0;chunk<myintervals.size();chunk++) {
	int yp = myintervals[chunk]/xres;
	int xp = myintervals[chunk]-yp*xres;

	// ok, do a "swap" after you compute these...

#if 0
	int pibase=pi;
#endif

	for(int pos=0;pos<mysizes[chunk];pos++) {
	  if (xp + pos >= xres) {
	    xpos[pi] = (xp+pos)%xres;
	    ypos[pi] = yp+1;
	  } else {
	    xpos[pi] = xp+pos;
	    ypos[pi] = yp;
	  }
	  pi++;
	}

	for(int pos=0;pos<mysizes[chunk];pos++) {
	  int swappos = drand48()*mysizes[chunk];
	  if (swappos != pos) {
	    int tmp = xpos[pos];
	    xpos[pos] = xpos[swappos];
	    xpos[swappos] = tmp;

	    tmp = ypos[pos];
	    ypos[pos] = ypos[swappos];
	    ypos[swappos] = tmp;
	  }
	}

      }
      io_lock_.unlock();
      nwork = pi;
    }
	
    int camerareload = nwork*scene->get_rtrt_engine()->updatePercent;
    int cameracounter = camerareload - drand48()*camerareload; // random
      
    float alpha=1.0; // change it after 1st iteration...
      
    scene->get_rtrt_engine()->cameralock.lock();
    //Camera* rcamera=scene->get_camera(rendering_scene);
    rcamera = *scene->get_camera(rendering_scene); // copy
    int synch_frameless = dpy->synching_frameless();
    scene->get_rtrt_engine()->cameralock.unlock();
      
    int iteration=0;
      
    Array1< Color > lastC;
    Array1< Color > lastCs; // sum...
    Array1< Color > lastCa;
    Array1< Color > lastCb;
    Array1< Color > lastCc;
    Array1< Color > lastCd;

      // lets precompute jittered masks for each pixel as well...

      // lets just make some fixed number of masks...

    double jOff[110][4]; // 10 by 10 at the most

    double jPosX[100];     // X posistion
    double jPosY[100];     // Y posistion
      
    Array1< int > jitterMask; // jitter pattern used by this one...
      
    lastC.resize(xpos.size());
    lastCa.resize(xpos.size());
    lastCb.resize(xpos.size());
    lastCc.resize(xpos.size());
    lastCd.resize(xpos.size());
    lastCs.resize(xpos.size());
    jitterMask.resize(xpos.size());

      // ok, now build the masks and offsets...
      // assume 4 samples for now...
      
    int sampX=2,sampY=2;

    jPosX[0] = -0.5/sampX;
    jPosX[1] = 0.5/sampX;
    jPosX[2] = 0.5/sampX;
    jPosX[3] = -0.5/sampX;

    jPosY[0] = -0.5/sampY;
    jPosY[1] = -0.5/sampY;
    jPosY[2] = 0.5/sampY;
    jPosY[3] = 0.5/sampY;

    io_lock_.lock(); // compute jittered offsets now...

    for(int ii=0;ii<110;ii++) {
      double val;
      while ((val=drand48()) > 0.9) ; // get a number
      jOff[ii][0] = 0.5*(val-0.5)/sampX;
      while ((val=drand48()) > 0.9) ; // get a number
      jOff[ii][1] =  0.5*(val-0.5)/sampX;
      while ((val=drand48()) > 0.9) ; // get a number
      jOff[ii][2] =  0.5*(val-0.5)/sampX;
      while ((val=drand48()) > 0.9) ; // get a number
      jOff[ii][3] =  0.5*(val-0.5)/sampX;
    }

    for(int ii=0;ii<jitterMask.size();ii++)
      jitterMask[ii] = drand48()*100;

    io_lock_.unlock();

#if 1     
    Array1<Color>* clrs[4] = {&lastCa,&lastCb,&lastCc,&lastCd};
#endif
    int wcI=0;
    //int doAverage=0;

    for(;;){
      // exit if you are supposed to
      if (scene->get_rtrt_engine()->stop_execution())
	Thread::exit();
      
      //stats[showing_scene]->add(Thread::currentSeconds(), Color(0,1,0));
      iteration++;
      Stats* st=stats[rendering_scene];
      //st->reset();
      //barrier->wait(dpy->get_num_procs()+1);
      //st->add(Thread::currentSeconds(), Color(1,0,0));
	
      double ixres=1./xres;
      double iyres=1./yres;
      double xoffset=scene->xoffset;
      double yoffset=scene->yoffset;
#if 0
      int stile, etile;
      int n=0;
#endif
      //st->add(Thread::currentSeconds(), Color(0,0,0));
#if 0
      int xtilesize=scene->xtilesize;
      int ytilesize=scene->ytilesize;
      int nx=(xres+xtilesize-1)/xtilesize;
#endif

      Context cx(this, scene, st);

      int synch_change=0;

      if (synch_frameless) {
	barrier->wait(dpy->get_num_procs()+1);
      }

      if (!scene->get_rtrt_engine()->do_jitter) {
	  
	  double stime;
	  if(scene->doHotSpots())
	    //stime=Thread::currentSeconds();
	      stime=SCIRun::Time::currentSeconds();
	for(int ci=0;ci<xpos.size();ci++) { 
	  Ray ray;
	  Color result;
	    
	  if (!synch_frameless && --cameracounter <= 0) {
	    cameracounter = camerareload;
	    scene->get_rtrt_engine()->cameralock.lock();
	    //Camera* rcamera=scene->get_camera(rendering_scene);
	    rcamera = *scene->get_camera(rendering_scene); // copy
	    alpha = Galpha; // set this for blending...
	    //synch_frameless = dpy->synching_frameless();
	    if (synch_frameless != dpy->synching_frameless()) {
	      synch_change=1;
	      synch_frameless = 1-synch_frameless;
	      cerr << synch_frameless << " Synch Change!\n";
	    }
	    scene->get_rtrt_engine()->cameralock.unlock();
	  }
	    
	  int x = xpos[ci];
	  int y = ypos[ci];
	  if (!synch_change) {
	    camera->makeRay(ray, x+xoffset, y+yoffset, ixres, iyres);
	    traceRay(result, ray, 0, 1.0, Color(0,0,0), &cx);
	      
#if 1
	      
	    Color wc;
	    wc = lastC[ci]*(1-alpha) + result*alpha;
	    lastC[ci] = wc;
	    
	    if(scene->doHotSpots()){
	      //double etime=Thread::currentSeconds();
		double etime=SCIRun::Time::currentSeconds();
		double t=etime-stime;	
		stime=etime;
		image->set(x,y, CMAP(t));
	    } else {
	      image->set(x,y,wc);
	    }
#else
	    Color wc = lastC[ci] + (*clrs[wcI&1])[ci]*float(-0.5);
	    wc = wc + result*0.5;
	    lastC[ci] = wc;
	    (*clrs[wcI&1])[ci] = result;
	    if(scene->doHotSpots()){
	      //double etime=Thread::currentSeconds();
		double etime=SCIRun::Time::currentSeconds();
		double t=etime-stime;	
		stime=etime;
		image->set(x,y, CMAP(t));
	    } else {
	      image->set(x,y, wc);
	    }
#endif
#if 0
	  } else {
	    (*image)(x,y) = procColor;
#endif
	  }
	    
	}
	wcI++;

	if (synch_frameless) { // grab a camera...
#if 1
	  if (synch_change) {
	    io_lock_.lock();
	    cerr << num << " Doing double block!\n";
	    io_lock_.unlock();
	    barrier->wait(dpy->get_num_procs()+1);
	    io_lock_.lock();
	    cerr << num << " out of double block!\n";
	    io_lock_.unlock();
	  } // just do 2 of them here...
#endif
	  synch_change=0;

	  barrier->wait(dpy->get_num_procs()+1);
	  
	  scene->get_rtrt_engine()->cameralock.lock();
	  rcamera = *scene->get_camera(rendering_scene); // copy
	  alpha = Galpha; // set this for blending...
	  synch_frameless = dpy->synching_frameless();
	  if (synch_frameless != dpy->synching_frameless()) {
	    synch_change=1;
	    synch_frameless = 1-synch_frameless;
	  }
	  scene->get_rtrt_engine()->cameralock.unlock();
	  
	}
      } else { // doing jittering...

#if 0	  
	if (wcI>3) {
	  doAverage = 1;
	}
#endif

	for(int ji=0;ji<2;ji++) {
	  double xia=jPosX[ji]+xoffset;
	  double yia=jPosY[ji]+yoffset;

	  double xib=jPosX[ji+2]+xoffset;
	  double yib=jPosY[ji+2]+yoffset;
	  int oji=(ji+1)&1;
	  double stime;
	  if(scene->doHotSpots())
	    //stime=Thread::currentSeconds();
	      stime=SCIRun::Time::currentSeconds();
	  for(int ci=0;ci<xpos.size();ci++) { 
	    Ray ray;
	    Color result;
	    Color resultb;
	      
	    if (!synch_frameless && --cameracounter <= 0) {
	      cameracounter = camerareload;
	      scene->get_rtrt_engine()->cameralock.lock();
	      rcamera = *scene->get_camera(rendering_scene); // copy
	      alpha = Galpha; // set this for blending...
	      if (synch_frameless != dpy->synching_frameless()) {
		synch_change=1;
		synch_frameless = 1-synch_frameless;
		cerr << synch_frameless << " Synch Change!\n";
	      }
	      scene->get_rtrt_engine()->cameralock.unlock();
	    }
	      
	    int x = xpos[ci];
	    int y = ypos[ci];
	    int mi=jitterMask[ci];
	    if (!synch_change) {
	      camera->makeRay(ray, x+xia+jOff[mi][ji], y+yia+jOff[mi][ji], ixres, iyres);
	      traceRay(result, ray, 0, 1.0, Color(0,0,0), &cx);

	      camera->makeRay(ray, x+xib+jOff[mi][ji+2], y+yib+jOff[mi][ji+2], ixres, iyres);
	      traceRay(resultb, ray, 0, 1.0, Color(0,0,0), &cx);
#if 0
	      
	      Color wc;
	      wc = lastC[ci]*(1-alpha) + result*alpha;
	      lastC[ci] = wc;
	    
	      if(scene->doHotSpots()){
		//double etime=Thread::currentSeconds();
		  double etime=SCIRun::Time::currentSeconds();
		  double t=etime-stime;	
		  stime=etime;
		  set->set(x,y, CMAP(t));
	      } else {
		image->set(x,y, wc);
	      }
#else
	      //do this, then blend into the frame buffer
	      // first compute average

#if 1
	      // do the full average all of the time...
	      Color sc = (*clrs[oji])[ci]*0.5 + (result + resultb)*0.25;
	      //lastCs[ci] = sc;
	      (*clrs[ji])[ci] = (result + resultb)*0.5;
	      if(scene->doHotSpots()){
		//double etime=Thread::currentSeconds();
		  double etime=SCIRun::Time::currentSeconds();
		  double t=etime-stime;	
		  stime=etime;
		  image->set(x,y, CMAP(t));
	      } else {
		image->set(x,y, sc);
	      }
#else
	      if (doAverage) {
		
		Color sc = lastCs[ci] + (*clrs[wcI&3])[ci]*float(-0.25); // subtract off
		sc = sc + result*0.25;
		lastCs[ci] = sc;
		Color wc = lastC[ci]*(1-alpha) + sc*alpha;
		lastC[ci] = wc;
		(*clrs[wcI&3])[ci] = result;
		if(scene->doHotSpots()){
		  //double etime=Thread::currentSeconds();
		    double etime=SCIRun::Time::currentSeconds();
		    double t=etime-stime;	
		    stime=etime;
		    image->set(x,y, CMAP(t));
		} else {
		  image->set(x,y, wc);
		}
	      } else {
		Color sc = lastCs[ci] + (*clrs[wcI&3])[ci]*float(-0.25); // subtract off
		sc = sc + result*0.25;
		if (!wcI)
		  lastCs[ci] = result*0.25;
		else
		  lastCs[ci] += result*0.25;
		  
		lastC[ci] = result;
		(*clrs[wcI&3])[ci] = result;
		if(scene->doHotSpots()){
		  //double etime=Thread::currentSeconds();
		    double etime=SCIRun::Time::currentSeconds();
		    double t=etime-stime;	
		    stime=etime;
		    image->set(x,y, CMAP(t));
		} else {
		  image->set(x,y, result);
		}
	      }
#endif
#endif
	      }
	  }

	  wcI++;

	  if (synch_frameless) { // grab a camera...
	    if (synch_change) {
	      io_lock_.lock();
	      cerr << num << " Doing double block!\n";
	      io_lock_.unlock();
	      barrier->wait(dpy->get_num_procs()+1);
	      io_lock_.lock();
	      cerr << num << " out of double block!\n";
	      io_lock_.unlock();
	    } // just do 2 of them here...

	    synch_change=0;

	    barrier->wait(dpy->get_num_procs()+1);
	      
	    scene->get_rtrt_engine()->cameralock.lock();
	    rcamera = *scene->get_camera(rendering_scene); // copy
	    alpha = Galpha; // set this for blending...
	    synch_frameless = dpy->synching_frameless();
	    if (synch_frameless != dpy->synching_frameless()) {
	      synch_change=1;
	      synch_frameless = 1-synch_frameless;
	    }
	    scene->get_rtrt_engine()->cameralock.unlock();
	  }
	}
      }
    }
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
