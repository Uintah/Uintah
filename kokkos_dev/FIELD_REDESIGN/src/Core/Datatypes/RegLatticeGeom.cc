//  RegLatticeGeom.cc - A lattice with equally spaced axis in 1,
//  2, or 3 dimensions
//
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//
//  Copyright (C) 2000 SCI Institute

#include <SCICore/Datatypes/RegLatticeGeom.h>

namespace SCICore{
namespace Datatypes{

PersistentTypeID RegLatticeGeom::type_id("RegLatticeGeom", "Datatype", 0);

RegLatticeGeom::RegLatticeGeom(){
}

RegLatticeGeom::~RegLatticeGeom(){
}

string RegLatticeGeom::get_info(){
  string retval;
  return retval;
}

void RegLatticeGeom::io(Piostream&){
}


} // end Datatypes
} // end SCICore
