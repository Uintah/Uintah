#ifndef COLORMAP_H
#define COLORMAP_H 1

#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array1.h>
#include <iostream>


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
 public:
  Array1<ColorCell*> ColorCells;
  int size;
  char filename[80];
  ColorMap();
  ColorMap(char* filebase);
  Color indexColorMap(float v);
  friend ostream& operator<<(ostream& out, const ColorMap& c);
  void save();
  bool valid;
};

} // end namespace rtrt

#endif
