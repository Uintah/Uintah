/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

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

PersistentTypeID Geom::type_id(Geom::typeName(0), "Datatype", 0);

using std::cout;
using std::endl;

#define GEOM_VERSION 1
void Geom::io(Piostream& stream){
  
  stream.begin_class(typeName(-1), GEOM_VERSION);
  
  Pio(stream, name_);
  Pio(stream, bbox_);

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
  if (bbox_.valid() || computeBoundingBox())
    {
      ibbox = bbox_;
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
  if (!bbox_.valid())
    {
      computeBoundingBox();
    }
  odouble = bbox_.longest_edge();
  return true;
}


bool
Geom::getDiagonal(Vector& ovec)
{
  if(!bbox_.valid())
    {
      computeBoundingBox();
    }
  ovec = bbox_.diagonal();
  return true;
}


} // End namespace SCIRun
