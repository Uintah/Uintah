
#ifndef __GRIDSPHERESDPY_H__
#define __GRIDSPHERESDPY_H__

#include <GL/glx.h>
#include <GL/glu.h>
#include "Array1.h"
#include <Core/Thread/Runnable.h>
#include <string>

namespace rtrt {

using SCIRun::Runnable;

class GridSpheres;

class GridSpheresDpy : public Runnable {
  
  int* hist;
  int* histmax;
  int xres, yres;
  friend class GridSpheres;
  Array1<GridSpheres*> grids;
  std::string *var_names;
  float* scales;
  float* min;
  float* max;
  float* range_begin;
  float* range_end;
  float* new_range_begin;
  float* new_range_end;
  int ndata;
  int colordata, newcolordata;
  
  void compute_hist(GLuint fid);
  void draw_hist(GLuint fid, XFontStruct* font_struct, int& redraw_range);
  void move(float* range, int x, int y, int& redraw_range);
  void changecolor(int y);

public:
  GridSpheresDpy(int colordata);
  //  GridSpheresDpy(GridSpheres* grid);
  virtual ~GridSpheresDpy();
  void attach(GridSpheres* g);
  virtual void run();
  void setup_vars();
  void animate(bool& changed);
};

} // end namespace rtrt

#endif
