
#ifndef __GRIDSPHERESDPY_H__
#define __GRIDSPHERESDPY_H__

#include <GL/glx.h>
#include <GL/glu.h>
#include <Packages/rtrt/Core/Array1.h>
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
  float* color_scales;
  float* original_min;
  float* original_max;
  float* min;
  float* max;
  float* range_begin;
  float* range_end;
  float* new_range_begin;
  float* new_range_end;
  float* color_begin;
  float* color_end;
  float* new_color_begin;
  float* new_color_end;
  int ndata;
  int colordata, newcolordata;
  char* in_file;
  
  void compute_hist(GLuint fid);
  void draw_hist(GLuint fid, XFontStruct* font_struct, int& redraw_range);
  void move(float* range, int x, int y, int& redraw_range);
  void move_min_max(float* range, int x, int y, int& redraw_range);
  void restore_min_max(int y, int &redraw_range);
  void compute_one_hist(int j);
  void changecolor(int y);

  inline float bound(float val, float min, float max)
    { return (val>min?(val<max?val:max):min); }
  inline int bound(int val, int min, int max)
    { return (val>min?(val<max?val:max):min); }

  void write_data_file(char *out_file);
  
public:
  GridSpheresDpy(int colordata, char *in_file=0);
  //  GridSpheresDpy(GridSpheres* grid);
  virtual ~GridSpheresDpy();
  void attach(GridSpheres* g);
  virtual void run();
  void setup_vars();
  void animate(bool& changed);
};

} // end namespace rtrt

#endif
