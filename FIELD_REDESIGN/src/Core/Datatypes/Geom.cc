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
#include <SCICore/Datatypes/Lattice3Geom.h>
#include <SCICore/Datatypes/MeshGeom.h>

namespace SCICore{
  namespace Datatypes{

Geom::Geom()
{
}


bool
Geom::get_bbox(BBox& ibbox)
{
  if (bbox.valid() || compute_bbox())
    {
      ibbox = bbox;
      return true;
    }
  else
    {
      return false;
    }
}


bool
Geom::longest_dimension(double& odouble)
{
  if (!bbox.valid())
    {
      compute_bbox();
    }
  odouble = bbox.longest_edge();
  return true;
}


bool
Geom::get_diagonal(Vector& ovec)
{
  if(!bbox.valid())
    {
      compute_bbox();
    }
  ovec = bbox.diagonal();
  return true;
}


}  // end datatypes
} // end scicore
