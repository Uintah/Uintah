

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
  case XK_t:
    if( rtrt_engine->hotSpotsMode != RTRT::HotSpotsOff)
      rtrt_engine->hotSpotsMode = RTRT::HotSpotsOff;
    else if (shift_pressed)
      rtrt_engine->hotSpotsMode = RTRT::HotSpotsHalfScreen;
    else
      rtrt_engine->hotSpotsMode = RTRT::HotSpotsOn;
    break;
  case XK_Escape:
    Thread::exitAll(0);
    break;
  }
}

void DpyGui::button_pressed(MouseButton button,
                            const int mouse_x, const int mouse_y) {

  switch(button) {
  case MouseButton1:
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
        rtrt_dpy->guiCam_->transform(trans, Camera::LookAt);
      }
    }
    break;
  case MouseButton3: 
    if (shift_pressed) {
      ////////////////////////////////////////
      // Dolly
      
      double xmotion = -double(last_x-mouse_x)/xres;
      double ymotion = double(last_y-mouse_y)/yres;
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

