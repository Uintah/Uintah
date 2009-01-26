/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



#ifndef __VOLUMEVISDPY_H__
#define __VOLUMEVISDPY_H__

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
  bool new_fast_render_mode;
  unsigned long long new_course_hash;

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
  void create_alpha_hash();
  inline float lookup_alpha(float val) {
    return alpha_transform.lookup_bound(val);
  }
  inline Color* lookup_color(float val) {
    return color_transform.lookup_bound(val);
  }

  float t_inc;
  bool fast_render_mode;
  // If this variable *ever* doesn't have 64 bits the code needs to be updated
  unsigned long long course_hash;
};

} // end namespace rtrt

#endif
