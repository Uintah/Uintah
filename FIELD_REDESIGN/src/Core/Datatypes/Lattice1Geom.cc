//  Lattice1Geom.cc - A base class for regular geometries with alligned axes
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 1000
//
//  Copyright (C) 1000 SCI Institute

#include <SCICore/Datatypes/Lattice1Geom.h>

namespace SCICore{
namespace Datatypes{

PersistentTypeID Lattice1Geom::type_id("Lattice1Geom", "Datatype", 0);

Lattice1Geom::Lattice1Geom(){
}

Lattice1Geom::Lattice1Geom(int ix, const Point& min, const Point& max){
  bbox.reset();
  bbox.extend(min);
  bbox.extend(max);
  nx = (ix<=0)?1:ix;
  compute_deltas();
  diagonal = max-min;
}

Lattice1Geom::~Lattice1Geom(){
}

string Lattice1Geom::get_info(){
  ostringstream retval;
  retval <<
    "name = " << name << '\n' <<
    "x = " << nx << endl;
  
  return retval.str();
}

bool Lattice1Geom::compute_bbox(){
  has_bbox = true;
  //compute diagnal and bbox
  diagonal = bbox.max()-bbox.min();
  return true;
}


    
Point Lattice1Geom::get_point(int i){
  if(!has_bbox){
    compute_bbox();
  }
  double x;
  x=bbox.min().x()+i*dx;
  return Point(x, 0, 0, 0);
}

bool Lattice1Geom::locate(const Point& p, int& ix){
  Vector pn=p-bbox.min();
  double mdx=diagonal.x();
  double x=pn.x()*(nx-1)/mdx;
  ix=(int)x;
  return true;
}

void Lattice1Geom::resize(int x){
  nx = (x<=0)?1:x;
  compute_deltas();
}
  
void Lattice1Geom::set_bbox(const Point& imin, const Point& imax){
  bbox.reset();
  // extend the bbox to include min and max
  bbox.extend(imin);
  bbox.extend(imax);
  has_bbox = true;
  diagonal = imax - imin;
  compute_deltas();
}

void Lattice1Geom::set_bbox(const BBox& ibbox){
  bbox = ibbox;
  compute_deltas();
  has_bbox = true;
}

void Lattice1Geom::compute_deltas(){
  dx = (bbox.max().x() - bbox.min().x())/(double) nx;
}
  
void Lattice1Geom::io(Piostream&){
}


} // end Datatypes
} // end SCICore
