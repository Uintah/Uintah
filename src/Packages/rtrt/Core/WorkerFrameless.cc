//
// Usually I don't like to have a code fragment in a separate file, 
// but in this case, I find it too confusing to search through the
// Worker.cc file to find things.  I can never tell if I am in the
// frameless side, or the framed side.  So I am putting the frameless
// loop here.
//


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
	  int swappos = (int)(drand48()*chunkorder.size());
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
	  int swappos = (int)(drand48()*myintervals.size());
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
	  int swappos = (int)(drand48()*mysizes[chunk]);
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
	
    int camerareload = (int)(nwork*scene->get_rtrt_engine()->updatePercent);
    int cameracounter = (int)(camerareload - drand48()*camerareload); // random
      
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
      jitterMask[ii] = (int)(drand48()*100.0);

    io_lock_.unlock();

    Array1<Color>* clrs[4] = {&lastCa,&lastCb,&lastCc,&lastCd};

    int wcI=0;

    for(;;){  
      // exit if you are supposed to
      if (scene->get_rtrt_engine()->stop_execution()) { Thread::exit(); }
      
      iteration++;
      Stats* st=stats[rendering_scene];
	
      double ixres=1./xres;
      double iyres=1./yres;
      double xoffset=scene->xoffset;
      double yoffset=scene->yoffset;

      Context cx(this, scene, st);

      int synch_change=0;

      if (synch_frameless) {
	barrier->wait(dpy->get_num_procs()+1);
      }

      int hotSpotMode = scene->getHotSpotsMode();

      if (!scene->get_rtrt_engine()->do_jitter) {
	
	double stime = 0;
	if(hotSpotMode)
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
	    
	    if( (hotSpotMode == 1) ||
		(hotSpotMode == 2 && (x < xres/2) ) ){
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
	    if( (hotSpotMode == 1) ||
		(hotSpotMode == 2 && (x < xres/2) ) ){
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
	  double stime = 0;
	  if( hotSpotMode )
	    stime = SCIRun::Time::currentSeconds();
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
	    
	      if( (hotSpotMode == 1) ||
		  (hotSpotMode == 2 && (x < xres/2) ) ){
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
	      if( (hotSpotMode == 1) ||
		  (hotSpotMode == 2 && (x < xres/2) ) ){
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
		if( (hotSpotMode == 1) ||
		    (hotSpotMode == 2 && (x < xres/2) ) ){
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

		if( (hotSpotMode == 1) ||
		    (hotSpotMode == 2 && (x < xres/2) ) ){
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
