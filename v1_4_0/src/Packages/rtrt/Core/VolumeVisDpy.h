
#ifndef __VOLUMEVISDPY_H__
#define __VOLUMEVISDPY_H__

#include <GL/glx.h>
#include <GL/glu.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>
#include <Core/Thread/Runnable.h>

namespace rtrt {

using SCIRun::Runnable;

class VolumeVis;

  class AlphaPos {
  public:
    AlphaPos() {}
    AlphaPos(float x, float val): x(x), val(val) {}
    
    float x; // between 0 and 1
    float val; // between 0 and 1
  };
  
class VolumeVisDpy : public Runnable {
  
  int* hist;
  int histmax;
  int xres, yres;
  friend class VolumeVis;
  Array1<VolumeVis*> volumes;
  char* in_file;
  float data_min, data_max;
  float scale;
  float original_t_inc, current_t_inc, t_inc;
  Array1<Color> colors_index;
  Array1<AlphaPos> alpha_list;
  int ncolors;
  int nalphas;

  ScalarTransform1D<float,Color*> color_transform;
  ScalarTransform1D<float,float> alpha_transform;
  ScalarTransform1D<int,float> alpha_stripes;

  void compute_hist(GLuint fid);
  void draw_hist(GLuint fid, XFontStruct* font_struct);
  //  void move(float* range, int x, int y, int& redraw_range);
  void draw_alpha_curve(GLuint fid, XFontStruct* font_struct);
  int min_x, max_x;
  int select_point(int x, int y);
  
  void write_data_file(char *out_file);
  
public:
  VolumeVisDpy(Array1<Color> &matls, Array1<AlphaPos> &alphas, int ncolors,
	       float t_inc, char *in_file=0);
  virtual ~VolumeVisDpy();
  void attach(VolumeVis* volume);
  virtual void run();
  void setup_vars();
  void animate(bool& changed);
  void rescale_alphas(float new_t_inc);
  void create_alpha_transfer();
  void create_color_transfer();
};

} // end namespace rtrt

#endif
