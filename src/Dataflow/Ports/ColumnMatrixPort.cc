//static char *id="@(#) $Id$";

/*
 *  ColumnMatrixPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <PSECore/CommonDatatypes/ColumnMatrixPort.h>

namespace PSECore {
namespace CommonDatatypes {

using namespace SCICore::CoreDatatypes;

clString SimpleIPort<ColumnMatrixHandle>::port_type("ColumnMatrix");
clString SimpleIPort<ColumnMatrixHandle>::port_color("dodgerblue4");

} // End namespace CommonDatatypes
} // End namespace PSECore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:07  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:46  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:50  dav
// Import sources
//
//

