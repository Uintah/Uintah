#include <Packages/rtrt/Core/RegularColorMap.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array1.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <fstream>
#include <string>
#include <sgi_stl_warnings_on.h>

using namespace rtrt;
using namespace std;

///////////////////////////////////////////////////////////////////
// Constructors/Destructors
///////////////////////////////////////////////////////////////////
RegularColorMap::RegularColorMap(const char* filename, int size):
  lock("RegularColorMap lock"),
  update_blended_colors(false), new_blended_colors(size),
  blended_colors(size)
{
  if (!fillColor(filename, color_list)) {
    cerr << "RegularColorMap::RegularColorMap(file)"
	 << ": Unable to load colormap from file, using default\n";
    fillColor(RegCMap_GrayScale, color_list);
  }
  fillBlendedColors(blended_colors);
  blended_colors_lookup.set_results_ptr(&blended_colors);
}

RegularColorMap::RegularColorMap(Array1<Color> &input_color_list, int size):
  lock("RegularColorMap lock"),
  color_list(input_color_list), 
  update_blended_colors(false), new_blended_colors(size),
  blended_colors(size)  
{
  fillBlendedColors(blended_colors);
  blended_colors_lookup.set_results_ptr(&blended_colors);
}

RegularColorMap::RegularColorMap(int type, int size):
  lock("RegularColorMap lock"),
  update_blended_colors(false), new_blended_colors(size),
  blended_colors(size)
{
  fillColor(type, color_list);
  fillBlendedColors(blended_colors);
  blended_colors_lookup.set_results_ptr(&blended_colors);
}

RegularColorMap::~RegularColorMap() {
}

//////////////////////////////////////////////////////////////////
// protected functions
//////////////////////////////////////////////////////////////////

void RegularColorMap::fillColor(int type, Array1<Color> &colors) {
  // Make sure we are dealing with a clean array
  colors.remove_all();
  // Need to switch off based on the type
  switch (type) {
  case RegCMap_InvRainbow:
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
  case RegCMap_Rainbow:
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
  case RegCMap_InvBlackBody:
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
  case RegCMap_BlackBody:
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
  case RegCMap_GrayScale:
    colors.add(Color(0,0,0));
    colors.add(Color(1,1,1));
    break;
  case RegCMap_InvGrayScale:
    colors.add(Color(1,1,1));
    colors.add(Color(0,0,0));
    break;
  case RegCMap_InvRIsoLum:
    colors.add(Color(0.528, 0.528, 1.0));
    colors.add(Color(0.304, 0.5824, 1.0));
    colors.add(Color(0.0, 0.6656, 0.832));
    colors.add(Color(0.0, 0.712, 0.5696));
    colors.add(Color(0.0, 0.744, 0.2976));
    colors.add(Color(0.0, 0.76, 0.0));
    colors.add(Color(0.304, 0.76, 0.0));
    colors.add(Color(0.5504, 0.688, 0.0));
    colors.add(Color(0.68, 0.624, 0.0));
    colors.add(Color(0.752, 0.6016, 0.0));
    colors.add(Color(1.0, 0.5008, 0.168));
    colors.add(Color(1.0, 0.424, 0.424));
    break;
  default:
    cerr << "RegularColorMap::fillColor(type):Invalid type "<<type
	 <<" using gray scale\n";
    colors.add(Color(1,1,1));
    colors.add(Color(0,0,0));
  }
}

bool RegularColorMap::fillColor(const char* file, Array1<Color> &colors) {
  char *me = "RegularColorMap::fillColor(file)";
  Array1<Color> new_colors;
  ifstream infile(file);
  if (!infile) {
    cerr << me << ":Color map file, "<<file<<" cannot be opened for reading\n";
    return false;
  }

  float r = 0;
  infile >> r;
  float max = r;
  do {
    // slurp up the colors
    float g = 0;
    float b = 0;
    infile >> g >> b;
    if (r > max)
      max = r;
    if (g > max)
      max = g;
    if (b > max)
      max = b;
    new_colors.add(Color(r,g,b));
    //    cout << "Added: "<<new_colors[new_colors.size()-1]<<endl;
    infile >> r;
  } while(infile);

  if (max > 1) {
    cerr << me << ":Renormalizing colors for range of 0 to 255\n";
    float inv255 = 1.0f/255.0f;
    for(int i = 0; i < new_colors.size(); i++) {
      new_colors[i] = new_colors[i] * inv255;
      //      cout << "new_colors["<<i<<"] = "<<new_colors[i]<<endl;
    }
  }

  // Copy the contents of new_colors to colors
  colors = new_colors;

  return true;
}

void RegularColorMap::fillBlendedColors(Array1<Color> &blended) {
  ScalarTransform1D<int, Color> color_list_lookup;
  color_list_lookup.set_results_ptr(&color_list);
  color_list_lookup.scale(0, blended.size()-1);
  for(int i = 0; i < blended.size(); i++)
    blended[i] = color_list_lookup.interpolate(i);
}

//////////////////////////////////////////////////////////////////
// public functions
//////////////////////////////////////////////////////////////////

int RegularColorMap::parseType(const char* typeSin) {
  int type = RegCMap_Unknown;

  string typeS(typeSin);
  if (typeS == "InvRainbow" ||
      typeS == "invrainbow" ||
      typeS == "ir"         )
    type = RegCMap_InvRainbow;
  else if (typeS == "Rainbow" ||
           typeS == "rainbow" ||
           typeS == "r"         )
    type = RegCMap_Rainbow;
  else if (typeS == "InvBlackBody" ||
           typeS == "invblackbody" ||
           typeS == "ib"         )
    type = RegCMap_InvBlackBody;
  else if (typeS == "BlackBody" ||
           typeS == "blackbody" ||
           typeS == "b"         )
    type = RegCMap_BlackBody;
  else if (typeS == "InvGrayScale" ||
           typeS == "invgrayscale" ||
           typeS == "ig"         )
    type = RegCMap_InvGrayScale;
  else if (typeS == "GrayScale" ||
           typeS == "grayscale" ||
           typeS == "g"         )
    type = RegCMap_GrayScale;
  else if (typeS == "InvRIsoLum" ||
      typeS == "invrisolum" ||
      typeS == "iril"         )
    type = RegCMap_InvRIsoLum;
  
  return type;
}

void RegularColorMap::changeColorMap(int type) {
  lock.lock();
  // Load up the new color_list
  fillColor(type, color_list);

  // Build the interpolated values
  fillBlendedColors(new_blended_colors);
  
  update_blended_colors = true;
  lock.unlock();
}

bool RegularColorMap::changeColorMap(const char* filename) {
  lock.lock();
  if (fillColor(filename, color_list)) {
    // Only do this if color_list was successfully updated
    fillBlendedColors(new_blended_colors);
    update_blended_colors = true;
    lock.unlock();
    return true;
  } else {
    lock.unlock();
    return false;
  }
}

bool RegularColorMap::changeSize(int new_size) {
  if (new_size <= 0) {
    cerr << "RegularColorMap::changeSize: bad new_size "<<new_size<<"\n";
    return false;
  }
  lock.lock();

  new_blended_colors.resize(new_size);
  fillBlendedColors(new_blended_colors);
  update_blended_colors = true;

  lock.unlock();
  return true;
}

void RegularColorMap::animate(double /*t*/, bool& changed) {
  // We want to only copy over the values if update_blended_colors is true
  if (update_blended_colors) {
    lock.lock();
    update_blended_colors = false;
    blended_colors = new_blended_colors;
    lock.unlock();

    changed  = true;
  }
}

