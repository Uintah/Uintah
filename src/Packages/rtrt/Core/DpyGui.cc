

#include <Packages/rtrt/Core/DpyGui.h>
#include <Packages/rtrt/Core/Dpy.h>
#include <Packages/rtrt/Core/DpyPrivate.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/ExternalUIInterface.h>
#include <Packages/rtrt/Core/rtrt.h>

#include <Core/Thread/Thread.h>

#include <sgi_stl_warnings_off.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <sgi_stl_warnings_on.h>

using namespace rtrt;
using namespace SCIRun;
using namespace std;

#ifdef HAVE_EXC
#include <libexc.h>
#elif defined(__GNUC__) && defined(__linux)
#include <execinfo.h>
#endif

namespace rtrt {
  void print_stack() {
    static const int MAXSTACK = 100;
#ifdef HAVE_EXC
    // Use -lexc to print out a stack trace
    static const int MAXNAMELEN = 1000;
    __uint64_t addrs[MAXSTACK];
    char* cnames_str = new char[MAXSTACK*MAXNAMELEN];
    char* names[MAXSTACK];
    for(int i=0;i<MAXSTACK;i++)
      names[i]=cnames_str+i*MAXNAMELEN;
    int nframes = trace_back_stack(0, addrs, names, MAXSTACK, MAXNAMELEN);
    if(nframes == 0){
      fprintf(stderr, "Backtrace not available!\n");
    } else {
      fprintf(stderr, "Backtrace:\n");
      for(int i=0;i<nframes;i++)
        fprintf(stderr, "0x%p: %s\n", (void*)addrs[i], names[i]);
    }
#elif defined(__GNUC__) && defined(__linux)
    static void *addresses[MAXSTACK];
    int n = backtrace( addresses, MAXSTACK );
    if (n == 0){
      fprintf(stderr, "Backtrace not available!\n");
    } else {
      fprintf(stderr, "Backtrace:\n");
      char **names = backtrace_symbols( addresses, n );
      for ( int i = 0; i < n; i++ )
        {
          fprintf (stderr, "%s\n", names[i]);
        } 
      free(names);
    } 
#endif
  }
}

DpyGui::DpyGui():
  DpyBase("RTRT DpyGui"),
  rtrt_dpy(0),
  rtrt_engine(0),
  ui_mutex("DpyGui::ui_mutex")
{
  cleaned = true;

  resize_xres = xres;
  resize_yres = yres;
}

DpyGui::~DpyGui() {
}

void DpyGui::addExternalUIInterface(ExternalUIInterface* ui_interface) {
  ui_mutex.lock();
  // See if the pointer is already in there
  if (find(ext_uis.begin(), ext_uis.end(), ui_interface) == ext_uis.end()) {
    ext_uis.push_back(ui_interface);
  }
  ui_mutex.unlock();
}

void DpyGui::removeExternalUIInterface(ExternalUIInterface* ui_interface) {
  ui_mutex.lock();
  ext_uis.erase(remove(ext_uis.begin(), ext_uis.end(), ui_interface),
                ext_uis.end());
  ui_mutex.unlock();
}

void DpyGui::set_resolution(const int width, const int height) {
  // Be sure not to pick wacked out values.
  if (width > 0 && height > 0) {
    resize_xres = width;
    resize_yres = height;

    if (!opened) {
      xres = width;
      yres = height;
    }
  }
}

void DpyGui::stopUIs() {
  ui_mutex.lock();
  //  cerr << "DpyGui::stopUIs::calling stop on all guis\n";
  for(size_t i = 0; i < ext_uis.size(); i++) {
    if (ext_uis[i])
      ext_uis[i]->stop();
  }
  //  cerr << "DpyGui::stopUIs::finished\n";
  ui_mutex.unlock();
}

void DpyGui::run() {
  //  cerr << "DpyGui::run(): start\n";
  open_events_display();
  //  cerr << "DpyGui::run(): after open_events_display\n";
  
  init();
  //  cerr << "DpyGui::run(): after init\n";
  
  for(;;) {
    if (should_close()) {
      cleaned = false;
      cleanup();
      //      cerr << "DpyGui::run(): about to return\n";
      //      for(;;) {}
      return;
    }

    // Check to see if we need to resize the window.  We need to give
    // preference to external resize events over what Dpy thinks it
    // should be.
    if (resize_xres != xres || resize_yres != yres) {
      cerr << "resize ("<<resize_xres<<", "<<resize_yres<<") res ("<<xres<<", "<<yres<<")\n";
      resize(resize_xres, resize_yres);
    } else if (rtrt_dpy->priv->xres != xres || rtrt_dpy->priv->yres != yres) {
      //      resize(rtrt_dpy->priv->xres, rtrt_dpy->priv->yres);
    }

    // Do some events
    wait_and_handle_events();
  }
}

void DpyGui::init() {
  // This lets Dpy know that it can create its window, because you
  // have to wait for the parent (this here).
  rtrt_dpy->release(win);
  //  cerr << "DpyGui::init::parentSema up\n";
}
  
void DpyGui::cleanup() {
  if (cleaned) return;
  else cleaned = true;

  //  cerr << "DpyGui::cleanup called\n";
  
  // Close the GG thread
  stopUIs();

  // Close the children
  rtrt_dpy->stop();

  // Wait for the child to stop rendering
  rtrt_dpy->wait_on_close();
  //  cerr << "rtrt_dpy->wait_on_close() finished\n";
  
  // Can't delete it for now, because it will cause a recursive lock
  // when doing Thread::exitAll().

  //  delete(rtrt_dpy);

  close_display();
  //  cerr << "DpyGui::cleanup::close_display finished\n";
}

bool DpyGui::should_close() {
  return on_death_row ||
    (rtrt_engine && rtrt_engine->exit_engine);
}

void DpyGui::resize(const int width, const int height) {
  xres = resize_xres = width;
  yres = resize_yres = height;
  if (width != xres || height != yres) {
    // Check to see if the XServer already had the right geometry before resizing
    XWindowAttributes win_attr;
    XGetWindowAttributes(dpy, win, &win_attr);
    if (win_attr.width != xres || win_attr.height != yres)
      XResizeWindow(dpy, win, xres, yres);
  }

  if (rtrt_dpy->display_frames) {
    //    cerr << "Setting Dpy's resolution to ("<<xres<<", "<<yres<<")\n";
    // Need to make sure the Dpy class has the same resolution we do.
    rtrt_dpy->priv->xres = xres;
    rtrt_dpy->priv->yres = yres;
    // Update the camera's aspect ratio
    rtrt_dpy->guiCam_->setWindowAspectRatio((double)yres/xres);
    //    print_stack();
  }
}

#if 1

void DpyGui::key_pressed(unsigned long key) {
  switch (key) {
  case XK_g:
    stopUIs();
    break;
  case XK_r:
    redraw = true;
    break;
  case XK_q:
    cleaned = false;
    rtrt_engine->stop_engine();
    break;
  case XK_Escape:
    Thread::exitAll(0);
    break;
  }
}

#else

void DpyGui::key_pressed( unsigned long key ) {

  switch( key ){

  // KEYPAD KEYS USED FOR MOVEMENT

  case '+':
    if (shift_pressed) {
      // increase planet orbit speed
      if (ORBIT_SPEED<.02) ORBIT_SPEED=1;
      else ORBIT_SPEED*=1.9;
      cerr << "orbit speed: " << ORBIT_SPEED << endl;
    } else if (activeGui->ctrlDown_) {
      // increase planet rotate speed
      ROTATE_SPEED*=1.1;
    } else {
      // SPEED up or slow down
      activeGui->stealth_->accelerate();
    }
    break;
  case '-':
    if (activeGui->shiftDown_) {
      // decrease planet orbit speed
      if (ORBIT_SPEED<.1) ORBIT_SPEED=0;
      else ORBIT_SPEED*=.6;
      cerr << "orbit speed: " << ORBIT_SPEED << endl;
    } else if (activeGui->ctrlDown_) {
      // decrease planet rotate speed
      ROTATE_SPEED*=.6;
    } else {
      activeGui->stealth_->decelerate();
    }
    break;
  // PITCH up and down
  case '8':
    cout << "pitchdown\n";
    activeGui->stealth_->pitchDown();
    break;
  case '2':
    cout << "pitchup\n";
    activeGui->stealth_->pitchUp();
    break;
  // SLIDE left and right
  case '9':
    activeGui->stealth_->slideRight();
    break;
  case '7':
    activeGui->stealth_->slideLeft();
    break;
  // TURN left and right
  case '4':
    activeGui->stealth_->turnLeft();
    break;
  case '5':    // STOP rotations (pitch/turn)
    activeGui->stealth_->stopPitchAndRotate();
    break;
  case '6':
    activeGui->stealth_->turnRight();
    break;
  // SLOW down and STOP
  case '.':
    activeGui->stealth_->slowDown();
    break;
  case ' ':
  case '0':
    activeGui->stealth_->stopAllMovement();
    break;
  // ACCELERATE UPWARDS or DOWNWARDS
  case '*': 
    activeGui->stealth_->goUp();   // Accelerate UP
    break;
  case '/': 
    activeGui->stealth_->goDown(); // Accelerate DOWN
    break;
  case 'q':
    activeGui->quit();
    break;
  case 'G': // Toggle Display of Gui
    activeGui->toggleGui();
    break;
  case 'g':
    cout << "Toggling Gravity.  If you want to increase framerate, use 'F'\n";
    activeGui->stealth_->toggleGravity();
    break;
  case 't':
    if( activeGui->dpy_->scene->hotSpotMode_ )
      activeGui->dpy_->scene->hotSpotMode_ = 0;
    else
      activeGui->dpy_->scene->hotSpotMode_ = 1;
    break;
  case 'T':
    if( activeGui->dpy_->scene->hotSpotMode_ )
      activeGui->dpy_->scene->hotSpotMode_ = 0;
    else
      activeGui->dpy_->scene->hotSpotMode_ = 2;
    break;
  case 'Q':
    activeGui->beQuiet_ = !activeGui->beQuiet_;
    break;
  case 's':
    activeGui->cycleShadowMode();
    break;
  case 'h':
    activeGui->cycleAmbientMode();
    break;
  case 'z':
    handleMenuCB( TOGGLE_RIGHT_BUTTON_MENU );
    break;
  case 'v':
    {
      if(activeGui->priv->followPath) { activeGui->priv->followPath = false; }
      activeGui->stealth_->stopAllMovement();

      // Animate lookat point to center of BBox...
      Object* obj= activeGui->dpy_->scene->get_object();
      BBox bbox;
      obj->compute_bounds(bbox, 0);
      if(bbox.valid()){
	activeGui->camera_->set_lookat(bbox.center());
        
	// Move forward/backwards until entire view is in scene...
	// change this a little, make it so that the FOV must
	// be 60 deg...
	// 60 degrees sucks - try 40...
        // Let user specify using gui.

	const double FOVtry = activeGui->fovValue_;

	Vector diag(bbox.diagonal());
	double w=diag.length();
	Vector lookdir(activeGui->camera_->get_lookat() -
		       activeGui->camera_->get_eye()); 
	lookdir.normalize();
	const double scale = 1.0/(2*tan(DtoR(FOVtry/2.0)));
	double length = w*scale;
	activeGui->camera_->set_fov(FOVtry);
	activeGui->camera_->set_eye( activeGui->camera_->get_lookat() -
				    lookdir*length );
	activeGui->camera_->setup();
	activeGui->fovSpinner_->set_float_val( FOVtry );

	Point origin;
	Vector lookdir2;
	Vector up;
	Vector side;
	double fov;
	activeGui->camera_->getParams(origin, lookdir2, up, side, fov);
	lookdir2.normalize();
	up.normalize();
	side.normalize();
	// Move the lights that are fixed to the eye
	for(int i = 0; i < activeGui->dpy_->scene->nlights(); i++) {
	  Light *light = activeGui->dpy_->scene->light(i);
	  if (light->fixed_to_eye) {
	    //	    light->updatePosition(light->get_pos() + dir*scl);
	    light->updatePosition(origin, 
				  Vector(side*light->eye_offset_basis.x()+
					 up*light->eye_offset_basis.y()+
					 lookdir2*light->eye_offset_basis.z()),
				  lookdir2);
	  }
	}
      }
    }
    break;
  case 13: // Enter
    if (activeGui->shiftDown_) {
      // toggle holo room on/off
      activeGui->dpy_->holoToggle_ = !activeGui->dpy_->holoToggle_;
      cout << "holo room is now " << activeGui->dpy_->holoToggle_ << endl;
    } else {
      activeGui->camera_->flatten(); // Right yourself (0 pitch, 0 roll)
    }
    break;
  case 'x':
    traverseRouteCB(-1);
    break;
  case 'a':
    activeGui->priv->animate =! activeGui->priv->animate;
    cout << "animate is now " << activeGui->priv->animate << "\n";
    break;
  case 'c':
    activeGui->camera_->print();
    break;
  case 'n':
    activeGui->camera_->scale_eyesep(0.9);
    cerr << "camera->eyesep="<<activeGui->camera_->get_eyesep()<<"\n";
    break;
  case 'm':
    activeGui->camera_->scale_eyesep(1.1);
    cerr << "camera->eyesep="<<activeGui->camera_->get_eyesep()<<"\n";
    break;
  case 'o':
    printf("Number materials: %d\n",activeGui->dpy_->scene->nmaterials());
    for (int m=0; m<activeGui->dpy_->scene->nmaterials(); m++) {
      CycleMaterial * cm =
	dynamic_cast<CycleMaterial*>(activeGui->dpy_->scene->get_material(m));
      if (cm) { cm->next(); printf("Got a cycle material!\n");}
    }
    break;

  case 'J': // toggle on/off "Jitter On Stop" moded...
    toggleAutoJitterCB( -1 );
    break;
  case 'j': // toggle on/off continuous jittered sampling...
    toggleJitterCB( -1 );
    break;

  case 'e':
    activeGui->dpy_->nstreams++;
    break;
  case 'E':
    if(activeGui->dpy_->nstreams > 1)
      activeGui->dpy_->nstreams--;
    break;

  case 'r':
    activeGui->displayRStats_ = !activeGui->displayRStats_;
    break;
  case 'p':
    activeGui->displayPStats_ = !activeGui->displayPStats_;
    break;

  case 'f':
#if 0
    if( activeGui->dpy_->fullScreenMode_ )
      activeGui->dpy_->toggleRenderWindowSize_ = true;
    else
      cout << "Can't toggle to full res on non-full screen mode.\n";
#else
    activeGui->priv->show_frame_rate = !activeGui->priv->show_frame_rate;
#endif
    break;

  case 27: // Escape key... need to find a symbolic name for this...
    activeGui->quit();
    break;
  case 'S':
    activeGui->priv->stereo=!(activeGui->priv->stereo);
    break;
#if 0
    // below is for blending "pixels" in
    // frameless rendering...

  case 'y': // sychronizing mode for frameless...
    synch_frameless = !synch_frameless;  //1-synch_frameless;
    //doing_frameless = 1-doing_frameless; // just toggle...
    cerr << synch_frameless << " Synch?\n";
    break;
#endif
  case 'W':
    cerr << "Saving raw image file\n";
    dpy->scene->get_image(dpy->priv->showing_scene)->save("images/image.raw");
    break;
  case 'w':
    cerr << "Saving ppm image file\n";
    dpy->priv->dumpFrame = 1;
    break;
  case 'M':
    switch (activeGui->dpy_->priv->dumpFrame)
      {
      case 0:
      case 1: // Start
        cerr << "Saving every frame to ppm image\n";
        dpy->priv->dumpFrame = -1;
        break;
      case -1: // Stop
      case -2:
      case -3:
        cerr << "Stopping saving every frame to ppm image\n";
        dpy->priv->dumpFrame = -3;
        break;
      }
    break;
  case 'd':
    {
      bool dd = activeGui->scene()->display_depth;
      bool ds = activeGui->scene()->display_sils;
      activeGui->scene()->display_depth = !dd && !ds;
      activeGui->scene()->display_sils = dd && !ds;
      activeGui->scene()->store_depth = !ds;
    }
    break;
  case 'D':
    {
      bool ds = activeGui->scene()->display_sils;
      activeGui->scene()->display_depth = false;
      activeGui->scene()->display_sils = !ds;
      activeGui->scene()->store_depth = !ds;
    }
    break;
  default:
    printf("unknown regular key %d\n", key);
    break;
  }
} // end key_pressed()


#endif

void DpyGui::button_pressed(MouseButton button,
                            const int mouse_x, const int mouse_y) {

  switch(button) {
  case MouseButton2:
    {
      double xpos = 2.0*mouse_x/xres - 1.0;
      double ypos = 1.0 - 2.0*mouse_y/yres;
      rotate_from = projectToSphere(xpos, ypos);
    }
    break;
  } // end switch
}

void DpyGui::button_released(MouseButton /*button*/,
                             const int /*x*/, const int /*y*/) {
}


void DpyGui::button_motion(MouseButton button,
                           const int mouse_x, const int mouse_y) {
  switch(button) {
  case MouseButton2:
    {
      double xpos = 2.0*mouse_x/xres - 1.0;
      double ypos = 1.0 - 2.0*mouse_y/yres;
      Vector to(projectToSphere(xpos, ypos));
      //cerr << "Transforming from "<<rotate_from<<" to "<<to<<"\n";
      Transform trans;
      trans.load_identity();
      if (trans.rotate(to, rotate_from)) {
        rotate_from = to;
        
        // Perform the transform
        rtrt_dpy->guiCam_->transform(trans, Camera::LookAt);
      }
    }
    break;
  } // end switch
}

Vector DpyGui::projectToSphere(double x, double y, double radius) const
{
  x /= radius;
  y /= radius;
  double rad2 = x*x+y*y;
  if(rad2 > 1){
    double rad = sqrt(rad2);
    x /= rad;
    y /= rad;
    return Vector(x,y,0);
  } else {
    double z = sqrt(1-rad2);
    return Vector(x,y,z);
  }
}

