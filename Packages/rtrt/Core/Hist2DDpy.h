
#ifndef RTRT_Hist2DDPY_H
#define RTRT_Hist2DDPY_H 1

#include <Core/Thread/Runnable.h>
#include <X11/Xlib.h>
#include <Packages/rtrt/Core/Array1.h>

namespace rtrt {

class VolumeVGBase;
using SCIRun::Runnable;
  
struct ImplicitLine {
  float a;
  float b;
  float c;
  bool operator != (const ImplicitLine& x) const;
  float operator()(float x, float y) const;
};

class Hist2DDpy : public Runnable {
  Array1<VolumeVGBase*> vols;
  int** hist;
  int histmax;
  int xres, yres;
  float vdatamin, vdatamax, gdatamin, gdatamax;
  void compute_hist(unsigned int fid);
  void draw_hist(unsigned int fid, XFontStruct* font_struct,
		 bool redraw_hist);
  enum BTN {
    Press, Release, Motion
  };
  void move(int x, int y, BTN what);
  ImplicitLine new_isoline;
  ImplicitLine clipline;
  ImplicitLine new_clipline;
  bool clip;
  bool new_clip;
  
  bool have_line;
  float px0, py0, px1, py1;
  void set_p();
  void set_line();
  int whichp;
  bool have_p;
  
  void set_clip();
public:
  ImplicitLine isoline;
  Hist2DDpy();
  Hist2DDpy(float a, float b, float c);
  void attach(VolumeVGBase*);
  virtual ~Hist2DDpy();
  virtual void run();
  void animate(bool& changed);
};

} // end namespace rtrt

#endif // #ifndef RTRT_Hist2DDPY_H
