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


namespace SCICore {
namespace Datatypes {

Geom::Geom()
{
}


bool
Geom::getBoundingBox(BBox& ibbox)
{
  if (d_bbox.valid() || computeBoundingBox())
    {
      ibbox = d_bbox;
      return true;
    }
  else
    {
      return false;
    }
}


bool
Geom::longestDimension(double& odouble)
{
  if (!d_bbox.valid())
    {
      computeBoundingBox();
    }
  odouble = d_bbox.longest_edge();
  return true;
}


bool
Geom::getDiagonal(Vector& ovec)
{
  if(!d_bbox.valid())
    {
      computeBoundingBox();
    }
  ovec = d_bbox.diagonal();
  return true;
}


}  // end datatypes
} // end scicore
