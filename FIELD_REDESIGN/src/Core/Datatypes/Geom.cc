//  Geom.cc - Describes an entity in space -- abstract base class
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute
#include <SCICore/Datatypes/Geom.h>
#include <SCICore/Datatypes/LatticeGeom.h>

namespace SCICore{
  namespace Datatypes{

Geom::Geom(): has_bbox(0){
}

bool Geom::get_bbox(BBox& ibbox){
  if(has_bbox){
    ibbox = bbox;
    return 1;
  }
  else{
    if(compute_bbox()){
      ibbox = bbox;
      return 1;
    }
    else{
      return 0;
    }
  }
}

bool Geom::set_bbox(const Point& imin, const Point& imax){
  bbox.reset();
  // extend the bbox to include min and max
  bbox.extend(imin);
  bbox.extend(imax);
  has_bbox = true;
  diagonal = imax - imin;
  return true;
}

bool Geom::set_bbox(const BBox& ibbox){
  bbox = ibbox;
  has_bbox = true;
  return true;
}


bool Geom::longest_dimension(double& odouble){    
  if(!has_bbox){
    compute_bbox();
  }
  odouble = Max(diagonal.x(), diagonal.y(), diagonal.z());
  return true;
}

bool Geom::get_diagonal(Vector& ovec){
  if(~has_bbox){
    compute_bbox();
  }
  ovec = diagonal;
  return true;
}

LatticeGeom* Geom::get_latticegeom(){
  return dynamic_cast<LatticeGeom*>(this);
}

}  // end datatypes
} // end scicore
