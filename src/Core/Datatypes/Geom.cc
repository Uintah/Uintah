//  Geom.cc - Describes an entity in space -- abstract base class
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Institute
#include <Core/Datatypes/Geom.h>


namespace SCIRun {

//////////
// PIO support

string Geom::typeName(int){
  static string className = "Geom";
  return className;
}

PersistentTypeID Geom::type_id(Geom::typeName(0), 
			       "Datatype", 
			       0);

using std::cout;
using std::endl;

#define GEOM_VERSION 1
void Geom::io(Piostream& stream){
  
  stream.begin_class(typeName(0).c_str(), GEOM_VERSION);
  
  Pio(stream, d_name);
  Pio(stream, d_bbox);

  stream.end_class();
}


Geom::Geom(){
}

Geom::~Geom(){
}

string Geom::getTypeName(int n){
  return typeName(n);
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


} // End namespace SCIRun
