// Attrib.cc - the base attribute class.
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/Attrib.h>

namespace SCICore{
namespace Datatypes{

Attrib::Attrib(int ix, int iy, int iz)
  :nx(ix), ny(iy), nz(iz), dims_set(3){
}

Attrib::Attrib(int ix, int iy)
  :nx(ix), ny(iy), dims_set(2){
}

Attrib::Attrib(int ix)
  :nx(ix), dims_set(1){
}

Attrib::Attrib()
  :dims_set(0){
}

Attrib::Attrib(const Attrib& copy):
  name(copy.name), nx(copy.nx), ny(copy.ny),
  nz(copy.nz), dims_set(copy.dims_set){
}



}  // end datatypes
}  // end scicore
