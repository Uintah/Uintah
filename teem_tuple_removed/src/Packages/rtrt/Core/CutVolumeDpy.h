
#ifndef CUTVOLUMEDPY_H
#define CUTVOLUMEDPY_H 1

#include <X11/Xlib.h>
#include <Packages/rtrt/Core/VolumeDpy.h>
#include <Packages/rtrt/Core/ColorMap.h>

/*
This is a new VolumeDpy used for CutPlanes.
The histogram is colored according to a ColorMap.
The Dpy can modify and save the associated ColorMap.
*/

namespace rtrt {

using SCIRun::Runnable;

class CutVolumeDpy : public VolumeDpy {
 protected:
  virtual void draw_hist(bool redraw_hist);

  virtual void button_released(MouseButton button, const int x, const int y);
  virtual void button_motion(MouseButton button, const int x, const int y);
  virtual void key_pressed(unsigned long key);

  void insertcolor(int x);
  void selectcolor(int x);
  void movecolor(int x);
  int selcolor;
 public:
  CutVolumeDpy(float isoval=-123456, ColorMap *cmap=0);

  ColorMap *cmap;
};

} // end namespace rtrt


#endif



