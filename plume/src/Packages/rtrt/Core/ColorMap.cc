#include <Packages/rtrt/Core/ColorMap.h>

#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array1.h>
#include <Packages/rtrt/Core/ScalarTransform1D.h>

#include <sgi_stl_warnings_off.h>
#include <fstream>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <stdio.h>
#include <unistd.h>

using namespace rtrt;
using namespace std;

ColorMap::ColorMap(const int num_bins) {
  //default, black to white
  ColorCells.add(ColorCell(Color(0.0,0.0,0.0), 0.0));
  ColorCells.add(ColorCell(Color(1.0,1.0,1.0), 1.0));

  // Allocate the slices
  Array1<Color> *color_slices = new Array1<Color>(num_bins);
  slices.set_results_ptr(color_slices);
  create_slices();
}

ColorMap::ColorMap(Array1<ColorCell> &color_in, const int num_bins):
  ColorCells(color_in)
{
  // Allocate the slices
  Array1<Color> *color_slices = new Array1<Color>(num_bins);
  slices.set_results_ptr(color_slices);
  create_slices();
}

ColorMap::ColorMap(char * filebase, const int num_bins, bool valuelast) {
  
  char buf[200];
  sprintf(buf, "%s.cmp", filebase);
  sprintf(filename, "%s", filebase);
  ifstream in(buf);
  if(!in){
    cerr << "COLORMAP ERROR " << filename << ".cmp NO SUCH FILE!" << "\n";
    cerr << "Creating default color map.\n";
    ColorCells.add(ColorCell(Color(0.0,0.0,0.0), 0.0));
    ColorCells.add(ColorCell(Color(1.0,1.0,1.0), 1.0));
    return;
  } 
  float r,g,b,v;
  while (!in.eof()) {
    if (valuelast)
      in >> r >> g >> b >> v;
    else
      in >> v >> r >> g >> b;
    if (!in.eof()) {
      ColorCells.add(ColorCell(Color(r,g,b), v));
    }
  }
  ColorCells[0].v = 0.0;
  ColorCells[ColorCells.size()-1].v = 1.0;

  // Allocate the slices
  Array1<Color> *color_slices = new Array1<Color>(num_bins);
  slices.set_results_ptr(color_slices);
  create_slices();
}

ColorMap::~ColorMap() {
  // Since the slices were allocated by us, we should delete them
  Array1<Color> *color_slices = slices.get_results_ptr();
  if (color_slices)
    delete color_slices;
}

Color ColorMap::indexColorMap(float v) {
  return slices.lookup_bound(v);
}

// This assumes that all the values in ColorCells are ordered
void ColorMap::create_slices() {
  // loop over the colors and fill in the slices
  int start = 0;
  int end;
  int nindex = slices.size() - 1;
  // These next three variables are for cases when the placement range
  // is not [0..1].  This will renormalize the values.
  float value_min = ColorCells[0].v;
  float value_max = ColorCells[ColorCells.size()-1].v;
  float irange = 1/(value_max - value_min);
  for (int s_index = 0; s_index < (ColorCells.size()-1); s_index++) {
    // the start has already been computed, so you need to figure out
    // the end.
    end = (int)((ColorCells[s_index+1].v-value_min) * irange * nindex);
    Color val = ColorCells[s_index].c;
    Color val_inc = (ColorCells[s_index+1].c - val) * (1.0f/(end - start));
    for (int i = start; i <= end; i++) {
      //    cout << "val = "<<val<<", ";
      slices[i] = val;
      val += val_inc;
    }
    start = end;
  }
}

ostream& operator<<(ostream& out, const ColorMap& cmp)
{
  for (int i = 0; i < cmp.ColorCells.size(); i++) {
    out << cmp.ColorCells[i].c << " @ " << cmp.ColorCells[i].v << "\n";
  }
  return out;
}

void ColorMap::save() {
  
  char buf[200];
  //choose not to overwrite the existing map
  sprintf(buf, "%snew.cmp", filename);
  ofstream out(buf);
  if(!out){
    cerr << "COLORMAP ERROR " << filename << "new.cmp CAN NOT WRITE!" << "\n";
    return;
  } 
  for (int i = 0; i < ColorCells.size(); i++) {
    out << ColorCells[i].c.red() << " ";
    out << ColorCells[i].c.green() << " ";
    out << ColorCells[i].c.blue() << " ";
    out << ColorCells[i].v << endl;
  }
  out.close();
}
