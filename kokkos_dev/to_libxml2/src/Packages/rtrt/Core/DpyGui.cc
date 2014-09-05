

#include <Packages/rtrt/Core/DpyGui.h>
#include <Packages/rtrt/Core/Dpy.h>
#include <Packages/rtrt/Core/DpyPrivate.h>
#include <Packages/rtrt/Core/Camera.h>
#include <Packages/rtrt/Core/ExternalUIInterface.h>
#include <Packages/rtrt/Core/rtrt.h>
#include <Packages/rtrt/Core/Scene.h>
#include <Packages/rtrt/Core/BBox.h>
#include <Packages/rtrt/Core/Image.h>
#include <Packages/rtrt/Core/Gui.h>
#include <Packages/rtrt/Core/Context.h>
#include <Packages/rtrt/Core/PerProcessorContext.h>
#include <Packages/rtrt/Core/Stats.h>
#include <Packages/rtrt/Core/Worker.h>
#include <Packages/rtrt/Core/Stealth.h>

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

  scene = 0;
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
  cerr << "DpyGui::stopUIs::calling stop on all guis\n";
  for(size_t i = 0; i < ext_uis.size(); i++) {
    if (ext_uis[i])
      ext_uis[i]->stop();
  }
  cerr << "DpyGui::stopUIs::finished\n";
  ui_mutex.unlock();
}

void DpyGui::startDefaultGui() {
  // I'm not decided on if I should stop all the guis and then start
  // up the default one.  I think, I'll leave this upto the default
  // gui to decide this.

  if (GGT::getActiveGGT())
    // A window already exists
    return;
  
  GGT* ggt = new GGT();

  ggt->setDpy( rtrt_dpy );
  ggt->setDpyGui( this );
  
  (new Thread(ggt, "Glut Glui Thread"))->detach();
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

void DpyGui::key_pressed(unsigned long key) {
  switch (key) {
  case XK_a:
    rtrt_dpy->priv->animate =! rtrt_dpy->priv->animate;
    cout << "animate is now " << rtrt_dpy->priv->animate << "\n";
    break;
  case XK_c:
    if (shift_pressed) {
      cout << "\nEnter new camera parameters > ";
      // Get a line from the command line
      char buf[200];
      if (fgets(buf, 200, stdin))
        // Send it to camera to parse
        if (!rtrt_dpy->guiCam_->read(buf))
          cerr << "Parameters didn't affect any change.  Check syntax.\n";
      fflush(stdin);
    } else {
      rtrt_dpy->guiCam_->print();
    }
    break;
  case XK_d:
    if (shift_pressed) {
      bool ds = rtrt_dpy->scene->display_sils;
      rtrt_dpy->scene->display_depth = false;
      rtrt_dpy->scene->display_sils = !ds;
      rtrt_dpy->scene->store_depth = !ds;
    } else {
      bool dd = rtrt_dpy->scene->display_depth;
      bool ds = rtrt_dpy->scene->display_sils;
      rtrt_dpy->scene->display_depth = !dd && !ds;
      rtrt_dpy->scene->display_sils = dd && !ds;
      rtrt_dpy->scene->store_depth = !ds;
    }
    if (rtrt_dpy->scene->display_depth)
      cerr << "Displaying depth\n";
    else if (rtrt_dpy->scene->display_sils)
      cerr << "Displaying sils\n";
    else
      cerr << "Displaying normally\n";
    break;
  case XK_f:
    if (shift_pressed)
      rtrt_dpy->synch_frameless = 1 - rtrt_dpy->synch_frameless;
    break;
  case XK_j:
    rtrt_engine->do_jitter = !rtrt_engine->do_jitter;
    break;
  case XK_g:
    if (shift_pressed)
      startDefaultGui();
    break;
  case XK_m:
    switch (rtrt_dpy->priv->dumpFrame) {
    case 0:
    case 1: // Start
      cerr << "Saving every frame to ppm image\n";
      rtrt_dpy->priv->dumpFrame = -1;
      break;
    case -1: // Stop
    case -2:
    case -3:
      cerr << "Stopping movie\n";
      rtrt_dpy->priv->dumpFrame = -3;
      break;
    }
    break;
  case XK_p:
    if (shift_pressed)
      // Decrease number of threads
      rtrt_dpy->change_nworkers(-1);
    else
      rtrt_dpy->change_nworkers(1);
    break;
  case XK_q:
    cleaned = false;
    rtrt_engine->stop_engine();
    break;
  case XK_r:
    redraw = true;
    break;
  case XK_s:
    if (shift_pressed) {
      rtrt_dpy->shadowMode_ =
        ShadowBase::decrement_shadow_type(rtrt_dpy->shadowMode_);
    } else {
      rtrt_dpy->shadowMode_ =
        ShadowBase::increment_shadow_type(rtrt_dpy->shadowMode_);
    }
    cout << "Shadow mode now "
         << ShadowBase::shadowTypeNames[rtrt_dpy->shadowMode_]<<"\n";
    break;
  case XK_t:
    if( rtrt_engine->hotSpotsMode != RTRT::HotSpotsOff)
      rtrt_engine->hotSpotsMode = RTRT::HotSpotsOff;
    else if (shift_pressed)
      rtrt_engine->hotSpotsMode = RTRT::HotSpotsHalfScreen;
    else
      rtrt_engine->hotSpotsMode = RTRT::HotSpotsOn;
    break;
  case XK_v:
    // Autoview
    {
      if(rtrt_dpy->priv->followPath) { rtrt_dpy->priv->followPath = false; }
      rtrt_dpy->stealth_->stopAllMovement();

      // Animate lookat point to center of BBox...
      Object* obj= rtrt_dpy->scene->get_object();
      BBox bbox;
      obj->compute_bounds(bbox, 0);
      if(bbox.valid()) {
        rtrt_dpy->guiCam_->autoview(bbox);
      }
    }
    break;
  case XK_w:
    if (shift_pressed) {
      cerr << "Saving raw image file\n";
      rtrt_dpy->scene->get_image(rtrt_dpy->priv->showing_scene)->
        save("images/image.raw");
    } else {
      cerr << "Saving ppm image file\n";
      rtrt_dpy->priv->dumpFrame = 1;
    }
    break;

  //////////////////////////////////////
  // Stealth stuff
  case XK_KP_Add: // Keypad +
    rtrt_dpy->stealth_->accelerate();
    break;
  case XK_KP_Subtract: // Keypad -
    rtrt_dpy->stealth_->decelerate();
    break;
  case XK_KP_End: // Keypad 1
    break;
  case XK_KP_Down:  // Keypad 2
    rtrt_dpy->stealth_->pitchUp();
    break;
  case XK_KP_Page_Down: // Keypad 3
    break;
  case XK_KP_Left: // Keypad 4
    rtrt_dpy->stealth_->turnLeft();
    break;
  case XK_KP_Begin: // Keypad 5
    rtrt_dpy->stealth_->stopPitchAndRotate();
    break;
  case XK_KP_Right: // Keypad 6
    rtrt_dpy->stealth_->turnRight();
    break;
  case XK_KP_Home: // Keypad 7
    rtrt_dpy->stealth_->slideLeft();
    break;
  case XK_KP_Up: // Keypad 8
    rtrt_dpy->stealth_->pitchDown();
    break;
  case XK_KP_Page_Up: // Keypad 9
    rtrt_dpy->stealth_->slideRight();
    break;
  case XK_KP_Insert: // Keypad 0
    rtrt_dpy->stealth_->stopAllMovement();
    break;
  case XK_KP_Delete: // Keypad .
    rtrt_dpy->stealth_->slowDown();
    break;
  case XK_KP_Multiply: // Keypad *
    rtrt_dpy->stealth_->goUp();
    break;
  case XK_KP_Divide: // Keypad /
    rtrt_dpy->stealth_->goDown();
    break;
  case XK_KP_Enter: // Keypad Enter
    break;
  case XK_Num_Lock:
    break;

  ////////////////////////////////////
  case XK_Escape:
    Thread::exitAll(0);
    break;
  default:
    cerr << "DpyGui::key_pressed: Unknown key: "<<key<<" (";
    cerr << hex << key << dec << ")\n";
  }
}

void DpyGui::button_pressed(MouseButton button,
                            const int mouse_x, const int mouse_y) {

  switch(button) {
  case MouseButton1:
    {
      if( control_pressed ) {
        // Turn on debug stuff
        Camera *C = rtrt_dpy->scene->get_camera( 0 );
        Ray ray;
        C->makeRay( ray, mouse_x, rtrt_dpy->priv->yres-mouse_y, 
                    1.0/rtrt_dpy->priv->xres,
                    1.0/rtrt_dpy->priv->yres );
        // Turn on debugging
        // DpyGui should really have its own.
        rtrt_dpy->ppc->debugOn();
        Color result;
        Stats stats(1000);
        Context cx(rtrt_dpy->scene, &stats, rtrt_dpy->ppc,
                   1-rtrt_dpy->priv->showing_scene, -1);
        Worker::traceRay(result, ray, 0, 1.0, Color(0,0,0), &cx);
        rtrt_dpy->ppc->debugOff();
        return;
      }
    }
  case MouseButton3:
    {
      last_x = mouse_x;
      last_y = mouse_y;
    }
    break;
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
  case MouseButton1: // Translate
    {
      double xmotion =  double(last_x-mouse_x)/xres;
      double ymotion = -double(last_y-mouse_y)/yres;
      // This could be more clever to translate pixel changed into
      // real world changes.
      double translate_speed = 1;
      Vector translation(xmotion*translate_speed, ymotion*translate_speed, 0);
      
      // Perform the transform
      rtrt_dpy->guiCam_->translate(translation);

      last_x = mouse_x;
      last_y = mouse_y;
    }
    break;
  case MouseButton2: // Rotate
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
        if (shift_pressed)
          // Around the eye point
          rtrt_dpy->guiCam_->transform(trans, Camera::Eye);
        else
          rtrt_dpy->guiCam_->transform(trans, Camera::LookAt);
      }
    }
    break;
  case MouseButton3: 
    if (shift_pressed) {
      ////////////////////////////////////////
      // Dolly
      
      double xmotion = -double(last_x-mouse_x)/xres;
      double ymotion = -double(last_y-mouse_y)/yres;
      double scale;
      // This could be come a prameter later
      double dolly_speed = 5;
      
      if (Abs(xmotion)>Abs(ymotion))
        scale=xmotion;
      else
        scale=ymotion;
      scale *= dolly_speed;
      
      rtrt_dpy->guiCam_->dolly(scale);
    } else {
      ////////////////////////////////////////
      // Zoom

      double xmotion= double(last_x - mouse_x)/xres;
      double ymotion= double(last_y - mouse_y)/yres;
      double scale;
      // This could be come a prameter later
      double fov_speed = 10;
      if (Abs(xmotion) > Abs(ymotion))
        scale = xmotion;
      else
        scale = ymotion;
      scale *= fov_speed;
      
      if (scale < 0)
        scale = 1/(1-scale);
      else
        scale += 1;
      
      rtrt_dpy->guiCam_->scaleFOV(scale);
    }
    last_x = mouse_x;
    last_y = mouse_y;
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

