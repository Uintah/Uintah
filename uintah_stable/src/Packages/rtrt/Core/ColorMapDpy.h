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

  class ColorPos {
  public:
    ColorPos() {}
    ColorPos(float x, float val, const Color &c): x(x), val(val), c(c) {}
    
    float x; // between 0 and 1
    float val; // between 0 and 1
    Color c;
  };
  
class ColorMapDpy : public DpyBase {
  char* filebase;
  float data_min, data_max;
  float scale;
  Array1<ColorPos> colors_index;
  int num_bins;

  ScalarTransform1D<float,Color> color_transform;
  ScalarTransform1D<float,float> alpha_transform;

  
  //  ScalarTransform1D<int,Color> color_bins;
  //  ScalarTransform1D<int,float> alpha_bins;

  void draw_transform();
  int min_x, max_x;
  int select_point(int x, int y);
  
  void write_data_file(char *out_file=0);
  int xstart, xend, width, ystart, yend, height;
  void set_viewport_params();
  
  // Display variables
  int selected_point;
public:
  ColorMapDpy(Array1<ColorPos> &matls, int num_bins=256);
  ColorMapDpy(char *filebase, int num_bins=256);
  virtual ~ColorMapDpy();

  void attach(float min, float max);
  void get_transfer_pointers(ScalarTransform1D<float,Color> *& color_trans_out,
			     ScalarTransform1D<float,float> *& alpha_trans_out)
  {
    color_trans_out = &color_transform;
    alpha_trans_out = &alpha_transform;
  }
  ScalarTransform1D<float, Color>* get_color_transfer_pointer() {
    return &color_transform;
  }
  ScalarTransform1D<float, float>* get_alpha_transfer_pointer() {
    return &alpha_transform;
  }
  
  // event functions
  virtual void init();
  virtual void display();
  virtual void resize(const int width, const int height);
#if 0
  virtual void key_pressed(unsigned long key);
  virtual void button_pressed(MouseButton button, const int x, const int y);
  virtual void button_released(MouseButton button, const int x, const int y);
  virtual void button_motion(MouseButton button, const int x, const int y);
#endif

  // Helper functions
  void create_transfer();
};

} // end namespace rtrt

#endif
