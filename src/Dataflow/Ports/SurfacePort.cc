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

#include <PSECore/CommonDatatypes/SurfacePort.h>

namespace PSECore {
namespace CommonDatatypes {

using namespace SCICore::CoreDatatypes;

clString SimpleIPort<SurfaceHandle>::port_type("Surface");
clString SimpleIPort<SurfaceHandle>::port_color("SteelBlue4");

} // End namespace CommonDatatypes
} // End namespace PSECore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:12  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:50  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//
