
#ifndef VOLUMEDPY_H
#define VOLUMEDPY_H 1

#include <X11/Xlib.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/DpyBase.h>

namespace rtrt {

using SCIRun::Runnable;

class VolumeBase;

class VolumeDpy : public DpyBase {
 protected:
  Array1<VolumeBase*> vols;
  int* hist;
  int histmax;
  float datamin, datamax;
  
  void compute_hist();
  virtual void draw_hist(bool redraw_hist);

  virtual void init();
  
  // event variable and handlers
  bool need_hist; // default: true
  bool redraw_isoval; // default: false
  virtual void display();
  virtual void resize(const int width, const int height);
  virtual void button_released(MouseButton button, const int x, const int y);
  virtual void button_motion(MouseButton button, const int x, const int y);

  void move_isoval(const int x);
  float new_isoval;
public:
  float isoval;
  VolumeDpy(float isoval=-123456);
  void attach(VolumeBase*);
  virtual ~VolumeDpy();
  virtual void animate(bool& changed);

  // This will set new_isoval to new_val if it lies within the range
  // [datamin, datamax].
  void change_isoval(float new_val);
  // This changes what datamin and datamax are
  void set_minmax(float min, float max);
};

} // end namespace rtrt


#endif
