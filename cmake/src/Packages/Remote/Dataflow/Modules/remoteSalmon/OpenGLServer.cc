
/*
 *  OpenGLServer.cc: add a network daemon to the opengl renderer
 *
 *  Written by:
 *   David Hart
 *   Department of Computer Science
 *   University of Utah
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Packages/Remote/Dataflow/Modules/remoteSalmon/OpenGLServer.h>

namespace Remote {
using namespace SCIRun;
using Remote::Tools::DtoR;
using Remote::Tools::RtoD;

extern Tcl_Interp* the_interp;
Array1<float> *triangles;

//----------------------------------------------------------------------
static Renderer* make_OpenGLServer()
{
    return scinew OpenGLServer;
}

//----------------------------------------------------------------------
RegisterRenderer OpenGLServer_renderer("OpenGLServer",
  &query_OpenGL, &make_OpenGLServer);

//----------------------------------------------------------------------
OpenGLServer::OpenGLServer():
  gl_in_mb("OpenGL incoming state mailbox", 5),
  gl_out_mb("OpenGL incoming state mailbox", 5)
{
  socketserver = new socketServer(this);
  socketserverthread = new Thread(socketserver, "socketServer");
}

//----------------------------------------------------------------------
void OpenGLServer::collect_triangles(Salmon* /*salmon*/,
  Roe* /*roe*/, GeomObj* obj)
{
  //obj->io(*dumpfile);
  //Array1<float> v;
  obj->get_triangles(*triangles);
  cerr << "found " << (*triangles).size()/3 << "  triangles" << endl;
  //obj->draw(drawinfo, salmon->default_matl.get_rep(), current_time);
}

//----------------------------------------------------------------------
void
OpenGLServer::real_getData(int datamask, FutureValue<GeometryData*>* result)
{
  cerr << "*************** REAL_GETDATA" << endl;
  GeometryData* res = new GeometryData;
  if(datamask&GEOM_VIEW){
    res->view=new View(lastview);
    res->xres=xres;
    res->yres=yres;
    res->znear=znear;
    res->zfar=zfar;
  }
  if(datamask&(GEOM_COLORBUFFER|GEOM_DEPTHBUFFER)){
    TCLTask::lock();
  }
  if(datamask&GEOM_COLORBUFFER){
    ColorImage* img = res->colorbuffer = new ColorImage(xres, yres);
    float* data=new float[xres*yres*3];
    cerr << "xres=" << xres << ", yres=" << yres << endl;
    WallClockTimer timer;
    timer.start();
    glReadPixels(0, 0, xres, yres, GL_RGB, GL_FLOAT, data);
    timer.stop();
    cerr << "done in " << timer.time() << " seconds\n";
    float* p=data;
    for(int y=0;y<yres;y++){
      for(int x=0;x<xres;x++){
	img->put_pixel(x, y, Color(p[0], p[1], p[2]));
	p+=3;
      }
    }
    delete[] data;
  }
  if(datamask&GEOM_DEPTHBUFFER){
    //DepthImage* img=res->depthbuffer=new DepthImage(xres, yres);
    unsigned int* data = new unsigned int[xres*yres];
    cerr << "reading depth...\n";
    WallClockTimer timer;
    timer.start();
    glReadPixels(0, 0, xres, yres,
      GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, data);
    timer.stop();
    cerr << "done in " << timer.time() << " seconds\n";
    /*
    unsigned int* p=data;
    for(int y=0;y<yres;y++){
      for(int x=0;x<xres;x++){
	img->put_pixel(x, y, (*p++)*(1./4294967295.));
      }
    }
    */
    //delete[] data;

				// this is a very dangerous hack
    res->depthbuffer = (DepthImage*)data;
  }
  if(datamask&(GEOM_COLORBUFFER|GEOM_DEPTHBUFFER)){
    GLenum errcode;
    while((errcode=glGetError()) != GL_NO_ERROR){
      cerr << "We got an error from GL: "
	   << (char*)gluErrorString(errcode) << endl;
    }
    TCLTask::unlock();
  }
  if ( datamask & GEOM_TRIANGLES ) {
    triangles = new Array1<float>;
    //dumpfile = new TextPiostream("dumpfile", Piostream::Write);
    roe->do_for_visible(this, (RoeVisPMF)&OpenGLServer::collect_triangles);
    //delete dumpfile;
    res->depthbuffer = (DepthImage*)triangles;
  }

  result->send(res);
}

//----------------------------------------------------------------------
void
OpenGLServer::redraw_loop() {

  int r;
				// Tell the Roe that we are started...
  TimeThrottle throttle;
  throttle.start();
  double newtime = 0;
    
  while(1) {
    int nreply=0;
    if (roe->inertia_mode) {
      double current_time = throttle.time();
      if (framerate == 0)
	framerate = 30;
      double frametime = 1./framerate;
      double delta = current_time-newtime;
      if (delta > 1.5*frametime) {
	framerate = 1./delta;
	frametime = delta;
	newtime = current_time;
      }
      else if (delta > .85*frametime) {
	framerate *= .9;
	frametime = 1./framerate;
	newtime=current_time;
      }
      else if(delta < .5*frametime){
	framerate*=1.1;
	if (framerate > 30)
	  framerate = 30;
	frametime=1./framerate;
	newtime=current_time;
      }
      newtime += frametime;
      throttle.wait_for_time(newtime);

      while (send_mb.tryReceive(r)) {
	if (r == DO_PICK) {
	  real_get_pick(salmon, roe, send_pick_x,
	    send_pick_y, ret_pick_obj, ret_pick_pick,
	    ret_pick_index);
	  recv_mb.send(PICK_DONE);
	} else if (r == DO_GETDATA) {
	  GetReq req(get_mb.receive());
	  real_getData(req.datamask, req.result);
	} else if (r == DO_GETGLSTATE) {
	  int which = send_mb.receive();
	  getGLState(which);
	} else {
				// Gobble them up...
	  nreply++;
	}
      }

				// you want to just rotate around the
				// current rotation axis - the current
				// quaternion is roe->ball->qNow the
				// first 3 components of this

      roe->ball->SetAngle(newtime*roe->angular_v);

      View tmpview(roe->rot_view);
	    
      Transform tmp_trans;
      HMatrix mNow;
      roe->ball->Value(mNow);
      tmp_trans.set(&mNow[0][0]);
	    
      Transform prv = roe->prev_trans;
      prv.post_trans(tmp_trans);
	    
      HMatrix vmat;
      prv.get(&vmat[0][0]);
	    
      Point y_a(vmat[0][1],vmat[1][1],vmat[2][1]);
      Point z_a(vmat[0][2],vmat[1][2],vmat[2][2]);
	    
      tmpview.up(y_a.vector());
      tmpview.eyep((z_a*(roe->eye_dist)) + tmpview.lookat().vector());
	    
      roe->view.set(tmpview);	    
    } else {
      
      for (;;) {
	int r=send_mb.receive();
	if (r == DO_PICK) {
	  real_get_pick(salmon, roe, send_pick_x, send_pick_y,
	    ret_pick_obj, ret_pick_pick, ret_pick_index);
	  recv_mb.send(PICK_DONE);
	} else if (r== DO_GETDATA) {
	  GetReq req(get_mb.receive());
	  real_getData(req.datamask, req.result);
	} else if (r == DO_GETGLSTATE) {
	  int which = send_mb.receive();
	  getGLState(which);
	} else {
	  nreply++;
	  break;
	}
      }
      
      newtime=throttle.time();
      throttle.stop();
      throttle.clear();
      throttle.start();
    }
    redraw_frame();
    for(int i=0;i<nreply;i++)
      recv_mb.send(REDRAW_DONE);
  }
}

//----------------------------------------------------------------------
void OpenGLServer::getGLState(int which) {
  TCLTask::lock();
  
  cerr << "OpenGL::getGLState called" << endl;

  glXMakeCurrent(dpy, win, cx);
  //glXWaitX();
  
  View view(roe->view.get());

  double aspect = double(xres)/double(yres);
  double fovy = RtoD(2*Atan(aspect*Tan(DtoR(view.fov()/2.))));

  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  gluPerspective(fovy, aspect, znear, zfar);
  glMatrixMode(GL_MODELVIEW);
  glLoadIdentity();

  gluLookAt(view.eyep().x(), view.eyep().y(), view.eyep().z(),
    view.lookat().x(), view.lookat().y(), view.lookat().z(),
    view.up().x(), view.up().y(), view.up().z());
  
  switch(which) {
    
  case GL_VIEWPORT: {
    int* viewport = (int*)gl_in_mb.receive();
    glGetIntegerv(GL_VIEWPORT, viewport);
    gl_out_mb.send((void*)viewport);
    cerr << endl;
  }
  break;
  
  case GL_MODELVIEW_MATRIX: {
    double* modelmat = (double*)gl_in_mb.receive();
    glGetDoublev(GL_MODELVIEW_MATRIX, modelmat);
    gl_out_mb.send((void*)modelmat);
    cerr << endl;
  }
  break;
  
  case GL_PROJECTION_MATRIX: {
    double* projmat = (double*)gl_in_mb.receive();
    glGetDoublev(GL_PROJECTION_MATRIX, projmat);
    gl_out_mb.send((void*)projmat);
    cerr << endl;
  }
  break;
  
  default: {
    cerr << "OpenGL::getGLState: dont' know what you're talkin' 'bout"
	 << endl;
  }
  break;
  }
  
  GLenum errcode;
  while((errcode=glGetError()) != GL_NO_ERROR){
    cerr << "We got an error from GL: "
	 << (char*)gluErrorString(errcode) << endl;
  }
  
  TCLTask::unlock();
}
} // End namespace Remote

