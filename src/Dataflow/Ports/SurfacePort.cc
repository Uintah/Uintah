//static char *id="@(#) $Id$";

/*
 *  SurfacePort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <CommonDatatypes/SurfacePort.h>

namespace PSECommon {
namespace CommonDatatypes {

clString SimpleIPort<SurfaceHandle>::port_type("Surface");
clString SimpleIPort<SurfaceHandle>::port_color("SteelBlue4");

} // End namespace CommonDatatypes
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:55:50  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//
