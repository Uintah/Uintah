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

#include <CommonDatatypes/ColumnMatrixPort.h>

namespace PSECommon {
namespace CommonDatatypes {

clString SimpleIPort<ColumnMatrixHandle>::port_type("ColumnMatrix");
clString SimpleIPort<ColumnMatrixHandle>::port_color("dodgerblue4");

} // End namespace CommonDatatypes
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:55:46  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:50  dav
// Import sources
//
//

