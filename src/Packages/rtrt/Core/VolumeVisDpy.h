
#ifndef __VOLUMEVISDPY_H__
#define __VOLUMEVISDPY_H__

//#include <GL/glx.h>
//#include <GL/glu.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>
#include <Packages/rtrt/Core/DpyBase.h>
#include <Core/Thread/Runnable.h>

namespace rtrt {

class VolumeVisBase;

  class AlphaPos {
  public:
    AlphaPos() {}
    AlphaPos(float x, float val): x(x), val(val) {}
    
    float x; // between 0 and 1
    float val; // between 0 and 1
  };
  
class VolumeVisDpy : public DpyBase {
  
  int* hist;
  int histmax;
  int* hist_log;
  int histmax_log;
  int nhist;
#if 0
  // For the bounding box changes
  double bbox_min_x, bbox_min_y, bbox_min_z;
  double bbox_max_x, bbox_max_y, bbox_max_z;
  double range_min_x, range_min_y, range_min_z;
  double range_max_x, range_max_y, range_max_z;
#endif
  
  Array1<VolumeVisBase*> volumes;
  char* in_file;
  float data_min, data_max;
  float scale;
  float original_t_inc, current_t_inc;
  Array1<Color> colors_index;
  Array1<AlphaPos> alpha_list;
  int ncolors;
  int nalphas;

  ScalarTransform1D<float,Color*> color_transform;
  ScalarTransform1D<float,float> alpha_transform;
  ScalarTransform1D<int,float> alpha_stripes;

  void compute_hist(GLuint fid);
  void draw_bbox(GLuint fid, XFontStruct* font_struct);
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
  void attach(VolumeVisBase* volume);
  virtual void run();
  void setup_vars();
  void animate(bool& changed);
  void rescale_alphas(float new_t_inc);
  void create_alpha_transfer();
  void create_color_transfer();
  inline float lookup_alpha(float val) {
    return alpha_transform.lookup(val);
  }
  inline Color* lookup_color(float val) {
    return color_transform.lookup(val);
  }

  float t_inc;
};

} // end namespace rtrt

#endif
