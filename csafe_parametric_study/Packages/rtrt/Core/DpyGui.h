
#ifndef RTRT_DPYGUI_H
#define RTRT_DPYGUI_H

#include <Packages/rtrt/Core/DpyBase.h>
#include <Packages/rtrt/Core/MouseCallBack.h>
#include <Core/Thread/Mutex.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace rtrt {

  class Dpy;
  class ExternalUIInterface;
  class RTRT;
  
class DpyGui: public DpyBase {
public:
  DpyGui();
  virtual ~DpyGui();

  void setDpy(Dpy* newdpy) {
    rtrt_dpy = newdpy;
  }

  void setRTRTEngine(RTRT* new_rtrt_engine) {
    rtrt_engine = new_rtrt_engine;
  }

  void addExternalUIInterface(ExternalUIInterface* ui_interface);
  void removeExternalUIInterface(ExternalUIInterface* ui_interface);

  // This function is designed to be called externally.  It is in
  // charge of resizing everthing.
  virtual void set_resolution(const int width, const int height);
  
  // Start up the default gui thread.
  void startDefaultGui();
  
protected:
  Dpy* rtrt_dpy;
  RTRT* rtrt_engine;
  std::vector<ExternalUIInterface*> ext_uis;
  SCIRun::Mutex ui_mutex;

  // Tells all the UIs to stop
  void stopUIs();

  // Inherited from DpyBase
  virtual void run();
  virtual void init();
  virtual void cleanup();
  virtual bool should_close();

  virtual void key_pressed(unsigned long key);

  virtual void button_pressed(MouseButton button, const int x, const int y);
  virtual void button_released(MouseButton button, const int x, const int y);
  virtual void button_motion(MouseButton button, const int x, const int y);

  virtual void resize(const int width, const int height);

  // These are used for the mouse callbacks
  int last_x, last_y;
  // This keeps track of the last rotation for the camera
  Vector rotate_from;
  // This helps with the arc ball mouse rotations
  Vector projectToSphere(double x, double y, double radius = 0.8) const;
  
  // These variables are for external resizing
  int resize_xres, resize_yres;

};

  
} // end namespace rtrt

#endif
