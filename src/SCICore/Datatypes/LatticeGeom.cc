//  LatticeGeom.cc - A base class for regular geometries with alligned axes
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/LatticeGeom.h>

namespace SCICore{
namespace Datatypes{

PersistentTypeID LatticeGeom::type_id("LatticeGeom", "Datatype", 0);

LatticeGeom::LatticeGeom(){
}

LatticeGeom::LatticeGeom(int ix, int iy, int iz, const Point& min, const Point& max){
  bbox.reset();
  bbox.extend(min);
  bbox.extend(max);
  nx = ix;
  ny = iy;
  nz = iz;
  // Compute the distance between grid lines
  dx = (max.x() - min.x())/(double) ix;
  dy = (max.y() - min.y())/(double) iy;
  dz = (max.z() - min.z())/(double) iz;
  dims_set = 3;
  diagonal = max-min;
}

LatticeGeom::~LatticeGeom(){
}

string LatticeGeom::get_info(){
  ostringstream retval;
  retval <<
    "name = " << name << '\n' <<
    "dims = " << dims_set << '\n' <<
    "x = " << nx << '\n' <<
    "y = " << ny << '\n' <<
    "z = " << nz << '\n';
  
  return retval.str();
}

bool LatticeGeom::compute_bbox(){
  has_bbox = true;
  //compute diagnal and bbox
  diagonal = bbox.max()-bbox.min();
  return true;
}


    
Point LatticeGeom::get_point(int i, int j, int k){
  if(!has_bbox){
    compute_bbox();
  }
  double x, y, z;
  // Make sure to avoid division by 0
  if(nx == 1){
    x= bbox.min().x();
    y=bbox.min().y()+diagonal.y()*(double)j/(double)(ny-1.0);
    z=bbox.min().z()+diagonal.z()*(double)k/(double)(nz-1.0);
  }
  else if(ny == 1){
    y = bbox.min().y();
    x=bbox.min().x()+diagonal.x()*(double)i/(double)(nx-1.0);
    z=bbox.min().z()+diagonal.z()*(double)k/(double)(nz-1.0); 
  }
  else if(nz == 1){
    z = bbox.min().z();
    x=bbox.min().x()+diagonal.x()*(double)i/(double)(nx-1.0);
    y=bbox.min().y()+diagonal.y()*(double)j/(double)(ny-1.0);
  }
  else{
    x=bbox.min().x()+diagonal.x()*(double)i/(double)(nx-1.0);
    y=bbox.min().y()+diagonal.y()*(double)j/(double)(ny-1.0);
    z=bbox.min().z()+diagonal.z()*(double)k/(double)(nz-1.0);
  }
  return Point(x,y,z);
}

bool LatticeGeom::locate(const Point& p, int& ix, int& iy, int& iz){
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
}

void LatticeGeom::resize(int x, int y, int z){
  nx = x;
  ny = y;
  nz = z;
  dims_set = 3;
}


void LatticeGeom::io(Piostream&){
}


} // end Datatypes
} // end SCICore
