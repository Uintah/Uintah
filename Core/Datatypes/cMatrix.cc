
/*
 *  cMatrix.cc : ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <Core/Datatypes/cMatrix.h>
#include <iostream>
using std::cerr;

namespace SCIRun {

// Dd: Should this be here?
PersistentTypeID cMatrix::type_id("cMatrix", "Datatype", 0);

void cMatrix::io(Piostream&) {
  cerr << "cMatrix::io not finished\n";
}

} // End namespace SCIRun

