
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
