//  Lattice2Geom.cc - A base class for regular geometries with alligned axes
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/Lattice2Geom.h>

namespace SCICore{
namespace Datatypes{

PersistentTypeID Lattice2Geom::type_id("Lattice2Geom", "Datatype", 0);

Lattice2Geom::Lattice2Geom(){
}

Lattice2Geom::Lattice2Geom(int ix, int iy, const Point& min, const Point& max){
  bbox.reset();
  bbox.extend(min);
  bbox.extend(max);
  nx = (ix<=0)?1:ix;
  ny = (iy<=0)?1:iy;
  compute_deltas();
  diagonal = max-min;
}

Lattice2Geom::~Lattice2Geom(){
}

string Lattice2Geom::get_info(){
  ostringstream retval;
  retval <<
    "name = " << name << '\n' <<
    "x = " << nx << '\n' <<
    "y = " << ny << '\n';
  
  return retval.str();
}

bool Lattice2Geom::compute_bbox(){
  has_bbox = true;
  //compute diagnal and bbox
  diagonal = bbox.max()-bbox.min();
  return true;
}


    
Point Lattice2Geom::get_point(int i, int j){
  if(!has_bbox){
    compute_bbox();
  }
  double x, y;
  x=bbox.min().x()+i*dx;
  y=bbox.min().y()+j*dy;
  return Point(x, y, 0, 0);
}

bool Lattice2Geom::locate(const Point& p, int& ix, int& iy){
  Vector pn=p-bbox.min();
  double mdx=diagonal.x();
  double mdy=diagonal.y();
  double x=pn.x()*(nx-1)/mdx;
  double y=pn.y()*(ny-1)/mdy;
  ix=(int)x;
  iy=(int)y;
  return true;
}

void Lattice2Geom::resize(int x, int y){
  nx = (x<=0)?1:x;
  ny = (y>=0)?1:y;
  compute_deltas();
}
  
void Lattice2Geom::set_bbox(const Point& imin, const Point& imax){
  bbox.reset();
  // extend the bbox to include min and max
  bbox.extend(imin);
  bbox.extend(imax);
  has_bbox = true;
  diagonal = imax - imin;
  compute_deltas();
}

void Lattice2Geom::set_bbox(const BBox& ibbox){
  bbox = ibbox;
  compute_deltas();
  has_bbox = true;
}

void Lattice2Geom::compute_deltas(){
  dx = (bbox.max().x() - bbox.min().x())/(double) nx;
  dy = (bbox.max().y() - bbox.min().y())/(double) ny;
}
  
void Lattice2Geom::io(Piostream&){
}


} // end Datatypes
} // end SCICore
