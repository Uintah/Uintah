
#ifndef __VOLUMEVISDPY_H__
#define __VOLUMEVISDPY_H__

#include <GL/glx.h>
#include <GL/glu.h>
#include <Packages/rtrt/Core/Array1.h>
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
  
  void compute_hist(GLuint fid);
  void draw_hist(GLuint fid, XFontStruct* font_struct);
  //  void move(float* range, int x, int y, int& redraw_range);

  void write_data_file(char *out_file);
  
public:
  VolumeVisDpy(char *in_file=0);
  virtual ~VolumeVisDpy();
  void attach(VolumeVis* volume);
  virtual void run();
  void setup_vars();
  void animate(bool& changed);
};

} // end namespace rtrt

#endif
