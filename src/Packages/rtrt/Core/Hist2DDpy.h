
#ifndef RTRT_Hist2DDPY_H
#define RTRT_Hist2DDPY_H 1

#include <X11/Xlib.h>
#include <Packages/rtrt/Core/DpyBase.h>
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

class Hist2DDpy : public DpyBase {
  Array1<VolumeVGBase*> vols;
  int** hist;
  int histmax;
  float vdatamin, vdatamax, gdatamin, gdatamax;
  void compute_hist(unsigned int fid);
  void draw_hist(unsigned int fid, XFontStruct* font_struct,
		 bool redraw_hist);

  // These handle mouse button events.  button indicates which button.
  // x and y are the location measured from the upper left corner of the
  // window.
  virtual void button_pressed(MouseButton button, const int x, const int y);
  virtual void button_released(MouseButton button, const int x, const int y);
  virtual void button_motion(MouseButton button, const int x, const int y);
  bool need_hist; // initialized to true
  bool redraw_isoval; // initialized to false

  ImplicitLine new_isoline;
  ImplicitLine clipline;
  ImplicitLine new_clipline;
  ImplicitLine perp_line, new_perp_line;

  // Computes the line perpendicular to l1 at (x,y) and stores it in l2.
  void compute_perp(ImplicitLine &l1, ImplicitLine &l2,
		    const float x, const float y, const float ratio = 1);

  // Using the bounds specified draws the line
  void draw_isoline(ImplicitLine &line, float xmin, float ymin,
		    float xmax, float ymax);
  
  bool clip;
  bool new_clip;
  bool use_perp, new_use_perp; // defaults to true
  
  bool have_line;
  float px0, py0, px1, py1;
  float cx, cy; // the center of the screen
  float ix, iy; // the intersection point along the diagonal
  void set_p();
  void set_lines();
  int whichp;
  bool have_p;
  
  void set_clip(float x0, float y0, float x1, float y1);

  // Called at the start of run.
  virtual void init();
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
