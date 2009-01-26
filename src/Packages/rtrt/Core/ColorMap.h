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
