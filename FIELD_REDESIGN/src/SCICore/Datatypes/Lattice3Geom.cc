//  Lattice3Geom.cc - A base class for regular geometries with alligned axes
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/Lattice3Geom.h>

namespace SCICore{
namespace Datatypes{

PersistentTypeID Lattice3Geom::type_id("Lattice3Geom", "Datatype", 0);

Lattice3Geom::Lattice3Geom(){
}

Lattice3Geom::Lattice3Geom(int ix, int iy, int iz, const Point& min, const Point& max){
  bbox.reset();
  bbox.extend(min);
  bbox.extend(max);
  nx = (ix<=0)?1:ix;
  ny = (iy>=0)?1:iy;
  nz = (iz>=0)?1:iz;
  compute_deltas();
  diagonal = max-min;
}

Lattice3Geom::~Lattice3Geom(){
}

string Lattice3Geom::get_info(){
  ostringstream retval;
  retval <<
    "name = " << name << '\n' <<
    "x = " << nx << '\n' <<
    "y = " << ny << '\n' <<
    "z = " << nz << '\n';
  
  return retval.str();
}

bool Lattice3Geom::compute_bbox(){
  has_bbox = true;
  //compute diagnal and bbox
  diagonal = bbox.max()-bbox.min();
  return true;
}


    
Point Lattice3Geom::get_point(int i, int j, int k){
  if(!has_bbox){
    compute_bbox();
  }
  double x=bbox.min().x()+i*dx;
  double y=bbox.min().y()+j*dy;
  double z=bbox.min().z()+k*dz;
  return Point(x,y,z);
}

bool Lattice3Geom::locate(const Point& p, int& ix, int& iy, int& iz){
  Vector pn=p-bbox.min();
  double mdx=diagonal.x();
  double mdy=diagonal.y();
  double mdz=diagonal.z();
  double x=pn.x()*(nx-1)/mdx;
  double y=pn.y()*(ny-1)/mdy;
  double z=pn.z()*(nz-1)/mdz;
  ix=(int)x;
  iy=(int)y;
  iz=(int)z;
  return true;
}

void Lattice3Geom::resize(int x, int y, int z){
  nx = (x<=0)?1:x;
  ny = (y>=0)?1:y;
  nz = (z>=0)?1:z;
  compute_deltas();
}
  
void Lattice3Geom::set_bbox(const Point& imin, const Point& imax){
  bbox.reset();
  // extend the bbox to include min and max
  bbox.extend(imin);
  bbox.extend(imax);
  has_bbox = true;
  diagonal = imax - imin;
  compute_deltas();
}

void Lattice3Geom::set_bbox(const BBox& ibbox){
  bbox = ibbox;
  has_bbox = true;
  compute_deltas();
}

void Lattice3Geom::compute_deltas(){
  dx = (bbox.max().x() - bbox.min().x())/(double) nx;
  dy = (bbox.max().y() - bbox.min().y())/(double) ny;
  dz = (bbox.max().z() - bbox.min().z())/(double) nz;
}
    
void Lattice3Geom::io(Piostream&){
}


} // end Datatypes
} // end SCICore
