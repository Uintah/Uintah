
#ifndef __GRIDSPHERESDPY_H__
#define __GRIDSPHERESDPY_H__

#include <Packages/rtrt/Core/DpyBase.h>
#include <Packages/rtrt/Core/Array1.h>

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace rtrt {

using SCIRun::Runnable;

class GridSpheres;
class TextureGridSpheres;
class PCAGridSpheres;

class GridSpheresDpy : public DpyBase {
  
  int* hist;
  int* histmax;
  // This is the base size of the histogram.  Histograms are rescaled
  // to fit the window as needed.
  int nhist;
  // These are the scaled histograms
  int* scaled_hist;
  int nscaled_hist;
  friend class GridSpheres;
  friend class TextureGridSpheres;
  friend class PCAGridSpheres;
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
  int shade_method, new_shade_method;
  int radius_index, new_radius_index;
  char* in_file;
  
  void compute_hist(GLuint fid);
  void draw_hist(GLuint fid, XFontStruct* font_struct);
  // This scales the histograms to match the min and max of the range
  // as well as the horizontal resolution.
  void compute_scaled_hist();
  void compute_histmax();
  void move(float* range, int x, int y);
  void move_min_max(float* range, int x, int y);
  void restore_min_max(int y);
  void compute_one_hist(int j);
  void changecolor(int y);

  inline float bound(float val, float min, float max)
    { return (val>min?(val<max?val:max):min); }
  inline int bound(int val, int min, int max)
    { return (val>min?(val<max?val:max):min); }

  void write_data_file(char *out_file);

  // Inherited functions

  // Called at the start of run.
  virtual void init();
  virtual void display();
  virtual void resize(const int width, const int height);
  virtual void key_pressed(unsigned long key);
  virtual void button_pressed(MouseButton button, const int x, const int y);
  virtual void button_motion(MouseButton button, const int x, const int y);

  // Gui control variables
  bool need_hist;
public:
  GridSpheresDpy(int colordata, char *in_file=0);
  //  GridSpheresDpy(GridSpheres* grid);
  virtual ~GridSpheresDpy();
  void attach(GridSpheres* g);
  void set_var_names(std::string* in_var_names) { var_names = in_var_names; }
  void setup_vars();

  // This is frame safe.  If you call this function before any
  // GridSpheres are attached it will behave as expected.  If you call
  // this function after you attach a GridSpheres, but before
  // preprocess, then GridSpheres will pick up the change after the
  // renderer has started and animate has been called.  If you call
  // this after preprocess (i.e. during rendering) changes will be
  // noticed after the next frame.
  void set_radius_index(int new_ri);

  void animate(bool& changed);
};

} // end namespace rtrt

#endif
