#ifndef COLORMAP_H
#define COLORMAP_H 1

#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array1.h>
#include <iostream>
#include <Packages/rtrt/Core/ScalarTransform1D.h>

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
  inline ~ColorCell(){}
};

class ColorMap {
  char filename[80];
  ScalarTransform1D<float, Color> slices;
public:
  Array1<ColorCell*> ColorCells;
  ColorMap(const int num_bins = 256);
  ColorMap(char* filebase, const int num_bins = 256);

  Color indexColorMap(float v);

  // This should be called when ever you change ColorCells
  void create_slices();
  
  inline int size() { return ColorCells.size(); }
  friend ostream& operator<<(ostream& out, const ColorMap& c);
  void save();
};

} // end namespace rtrt

#endif
