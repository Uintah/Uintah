#ifndef RTRT_REGULAR_COLOR_MAP_H__
#define RTRT_REGULAR_COLOR_MAP_H__ 1

#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>

namespace rtrt {

  template<class T> class Array1;
  
  class RegularColorMap {
  protected:
    // We will maintain two sets of lookup tables.  The first is the
    // list of colors which will need to be interpolated to blend them
    // together.  The second is the blended result which has been
    // padded out to the size specified.
    ScalarTransform1D<Color> color_list_lookup;
    ScalarTransform1D<Color> blended_colors_lookup;

    // These pointers are used to update the colormap on animate
    // This pointer should be 0 if there is not pending copy.
    Array1<Color> *new_blended_colors;

    void fillColor(int type, Array1<Color> &colors);
  public:
    // We need a way to create one of these things.
    RegularColorMap(const char* filename, int size = 256);
    RegularColorMap(Array1<Color> &input_color_list, int size = 256);

    // This can be used to generate default values
    RegularColorMap(int type = 0, int size = 256);

    enum {
      InvRainbow = 0,
      Rainbow = 1,
      InvBlackBody = 2,
      BlackBody = 3,
      GrayScale = 4,
      InvGrayScale = 5
    }

    // These are used to change the colors in the colormap.  True is
    // returned if everything went OK.
    bool changeColorMap(int type);
    bool changeColorMap(const char* filename);

    // This allows one to change the number of bins used for the blended_colors
    bool changeSize(int new_size);

    // This allows a material to be animated at the frame change
    virtual void animate(double t, bool& changed);
  };

} // end namespace rtrt

#endif
