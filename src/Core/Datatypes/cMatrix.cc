//static char *id="@(#) $Id$";

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

#include <SCICore/CoreDatatypes/cMatrix.h>
#include <iostream.h>

namespace SCICore {
namespace CoreDatatypes {

// Dd: Should this be here?
PersistentTypeID cMatrix::type_id("cMatrix", "Datatype", 0);

void cMatrix::io(Piostream&) {
  cerr << "cMatrix::io not finished\n";
}

} // End namespace CoreDatatypes
} // End namespace SCICore

//
// $Log$
// Revision 1.2  1999/08/17 06:39:01  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:56:34  mcq
// Initial commit
//
// Revision 1.1  1999/04/25 04:07:24  dav
// Moved files into CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//
