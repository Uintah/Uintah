#include <Packages/rtrt/Core/ColorMap.h>
#include <Packages/rtrt/Core/Color.h>
#include <Packages/rtrt/Core/Array1.h>

#include <stdio.h>
#include <unistd.h>

#include <fstream>
#include <iostream>

using namespace rtrt;
using namespace std;

ColorMap::ColorMap() {
  //default, black to white
  ColorCells.add(new ColorCell(Color(0.0,0.0,0.0), 0.0));
  ColorCells.add(new ColorCell(Color(1.0,1.0,1.0), 1.0));
  size = 2;
}

ColorMap::ColorMap(char * filebase) {
  
    char buf[200];
    sprintf(buf, "%s.cmp", filebase);
    sprintf(filename, "%s", filebase);
    ifstream in(buf);
    if(!in){
      cerr << "COLORMAP ERROR " << filename << ".cmp NO SUCH FILE!" << "\n";
      valid = false;
      return;
    } 
    valid = true;
    size = 0;
    float r,g,b,v;
    while (!in.eof()) {
      in >> r >> g >> b >> v;
      if (!in.eof()) {
	size++;
	ColorCells.add(new ColorCell(Color(r,g,b), v));
      }
    }
    ColorCells[0]->v = 0.0;
    ColorCells[size-1]->v = 1.0;
}

Color ColorMap::indexColorMap(float v) {
  //look up a value and return a color
  if (v < 0.0f) v = 0.0f;
  if (v > 1.0f) v = 1.0f;
    
  if (valid) {

    int nearestlow = 0;
    int middle = ColorCells.size()/2;
    int nearesthigh = ColorCells.size();

    //    bool found = false;
    //    while (!found) {
    for (;;) {
      if (nearestlow+1 == nearesthigh) {
	break;
      }
      if ((ColorCells[middle]->v) == v) {
	nearestlow = middle;
	nearesthigh = middle;
	break;
      }
      if ((ColorCells[middle]->v) > v) {
	nearesthigh = middle;
	middle = nearestlow + (middle-nearestlow)/2;
      }
      if ((ColorCells[middle]->v) < v) {
	nearestlow = middle;
	middle = middle + (nearesthigh-middle)/2;
      }
      
    }

    float rng = ColorCells[nearesthigh]->v - ColorCells[nearestlow]->v;
    float dif;
    if (rng)
      dif = (v - ColorCells[nearestlow]->v) / rng;
    else 
      dif = 0.0;

    return Interpolate(ColorCells[nearesthigh]->c, 
		       ColorCells[nearestlow]->c, 
		       dif); 
  } else {
    return Interpolate(Color(1.0,1.0,1.0),
		       Color(0.0,0.0,0.0),
		       v); 
  }
}

ostream& operator<<(ostream& out, const ColorMap& cmp)
{
  if (cmp.valid) {
    for (int i = 0; i < cmp.ColorCells.size(); i++) {
      out << cmp.ColorCells[i]->c << " @ " << cmp.ColorCells[i]->v << "\n";
    }
  } else
    out << "COLORMAP ERROR " << cmp.filename << ".cmp NO SUCH FILE!" << "\n";

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
    out << ColorCells[i]->c.red() << " ";
    out << ColorCells[i]->c.green() << " ";
    out << ColorCells[i]->c.blue() << " ";
    out << ColorCells[i]->v << endl;
  }
  out.close();
}
