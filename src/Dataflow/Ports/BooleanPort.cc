//static char *id="@(#) $Id$";

/*
 *  sciBooleanPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <CommonDatatypes/BooleanPort.h>

namespace PSECommon {
namespace CommonDatatypes {

clString SimpleIPort<sciBooleanHandle>::port_type("Boolean");
clString SimpleIPort<sciBooleanHandle>::port_color("blue4");

} // End namespace CommonDatatypes
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:55:45  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:51  dav
// Import sources
//
//

