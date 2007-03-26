#ifndef COLORMAP_H
#define COLORMAP_H 1

#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

/*
ColorMaps are used by cutting planes to color the interiors of objects.
They can be dynamically added to, deleted from and changed and then saved to file.
A color map is a sorted list of color/value pairs.
Values are colored by linear interpolation.
*/

namespace rtrt {

class Color;

//each cell in the color map has a color and an idx between 0.0 and 1.0;
class ColorCell {
 public:
  Color c;
  float v;
  inline ColorCell( Color c, float v) : c(c), v(v) {}
  inline ColorCell() {}
  inline ~ColorCell(){}
};

class ColorMap {
  char filename[80];
public:
  ColorMap(const int num_bins = 256);
  ColorMap(Array1<ColorCell> &color_in, const int num_bins = 255);
  ColorMap(char* filebase, const int num_bins = 256, bool valuelast = true);
  ~ColorMap();
  
  ScalarTransform1D<float, Color> slices;
  Array1<ColorCell> ColorCells;

  Color indexColorMap(float v);

  // This should be called when ever you change ColorCells
  void create_slices();
  
  inline int size() { return ColorCells.size(); }
  friend ostream& operator<<(ostream& out, const ColorMap& c);
  void save();
};

} // end namespace rtrt

#endif
