
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
  
  ScalarTransform1D<float,Color*> color_transform;
  ScalarTransform1D<float,float> alpha_transform;

  void compute_hist(GLuint fid);
  void draw_hist(GLuint fid, XFontStruct* font_struct);
  //  void move(float* range, int x, int y, int& redraw_range);

  void write_data_file(char *out_file);
  
public:
  VolumeVisDpy(Array1<Color*> *matls, Array1<float> *alphas, float t_inc,
	       char *in_file=0);
  virtual ~VolumeVisDpy();
  void attach(VolumeVis* volume);
  virtual void run();
  void setup_vars();
  void animate(bool& changed);
  void rescale_alphas(float new_t_inc);
};

} // end namespace rtrt

#endif
