#include <Packages/rtrt/Core/RegularColorMap.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array1.h>

#include <iostream>

using namespace rtrt;
using namespace std;

///////////////////////////////////////////////////////////////////
// Constructors
///////////////////////////////////////////////////////////////////
RegularColorMap::RegularColorMap(const char* filename, int size) {
}

RegularColorMap::RegularColorMap(Array1<Color> &input_color_list, int size) {
}

RegularColorMap::RegularColorMap(int type = 0, int size) {
}

//////////////////////////////////////////////////////////////////
// protected functions
//////////////////////////////////////////////////////////////////

void RegularColorMap::fillColor(int type, Array1<Color> &colors) {
  // Make sure we are dealing with a clean array
  colors.remove_all();
  // Need to switch off based on the type
  switch (type) {
  case InvRainbow:
    colors.add(Color(0, 0, 1));
    colors.add(Color(0, 0.40000001, 1));
    colors.add(Color(0, 0.80000001, 1));
    colors.add(Color(0, 1, 0.80000001));
    colors.add(Color(0, 1, 0.40000001));
    colors.add(Color(0, 1, 0));
    colors.add(Color(0.40000001, 1, 0));
    colors.add(Color(0.80000001, 1, 0));
    colors.add(Color(1, 0.91764706, 0));
    colors.add(Color(1, 0.80000001, 0));
    colors.add(Color(1, 0.40000001, 0));
    colors.add(Color(1, 0, 0));
    break;
  case Rainbow:
    colors.add(Color(1, 0, 0));
    colors.add(Color(1, 0.40000001, 0));
    colors.add(Color(1, 0.80000001, 0));
    colors.add(Color(1, 0.91764706, 0));
    colors.add(Color(0.80000001, 1, 0));
    colors.add(Color(0.40000001, 1, 0));
    colors.add(Color(0, 1, 0));
    colors.add(Color(0, 1, 0.40000001));
    colors.add(Color(0, 1, 0.80000001));
    colors.add(Color(0, 0.80000001, 1));
    colors.add(Color(0, 0.40000001, 1));
    colors.add(Color(0, 0, 1));
    break;
  case InvBlackBody:
    colors.add(Color(1, 1, 1));
    colors.add(Color(1, 1, 0.70588237));
    colors.add(Color(1, 0.96862745, 0.47058824));
    colors.add(Color(1, 0.89411765, 0.3137255));
    colors.add(Color(1, 0.80000001, 0.21568628));
    colors.add(Color(1, 0.63921571, 0.078431375));
    colors.add(Color(1, 0.47058824, 0));
    colors.add(Color(0.90196079, 0.27843139, 0));
    colors.add(Color(0.78431374, 0.16078432, 0));
    colors.add(Color(0.60000002, 0.070588239, 0));
    colors.add(Color(0.40000001, 0.0078431377, 0));
    colors.add(Color(0.20392157, 0, 0));
    colors.add(Color(0, 0, 0));
    break;
  case BlackBody:
    colors.add(Color(0, 0, 0));
    colors.add(Color(0.20392157, 0, 0));
    colors.add(Color(0.40000001, 0.0078431377, 0));
    colors.add(Color(0.60000002, 0.070588239, 0));
    colors.add(Color(0.78431374, 0.16078432, 0));
    colors.add(Color(0.90196079, 0.27843139, 0));
    colors.add(Color(1, 0.47058824, 0));
    colors.add(Color(1, 0.63921571, 0.078431375));
    colors.add(Color(1, 0.80000001, 0.21568628));
    colors.add(Color(1, 0.89411765, 0.3137255));
    colors.add(Color(1, 0.96862745, 0.47058824));
    colors.add(Color(1, 1, 0.70588237));
    colors.add(Color(1, 1, 1));
    break;
  case GrayScale:
    colors.add(Color(0,0,0));
    colors.add(Color(1,1,1));
    break;
  case InvGrayScale:
    colors.add(Color(1,1,1));
    colors.add(Color(0,0,0));
    break;
  default:
    cerr << "Invalid type "<<type<<" using gray scale\n";
    colors.add(Color(1,1,1));
    colors.add(Color(0,0,0));
  }
}

//////////////////////////////////////////////////////////////////
// public functions
//////////////////////////////////////////////////////////////////

bool RegularColorMap::changeColorMap(int type) {
  if (new_blended_colors != 0) {
    delete new_blended_colors;
  }
}

bool RegularColorMap::changeColorMap(const char* filename) {
}

bool RegularColorMap::changeSize(int new_size) {
}

void RegularColorMap::animate(double t, bool& changed) {
}

