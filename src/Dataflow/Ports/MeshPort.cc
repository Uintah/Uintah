//static char *id="@(#) $Id$";

/*
 *  MeshPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <CommonDatatypes/MeshPort.h>

namespace PSECommon {
namespace CommonDatatypes {

clString SimpleIPort<MeshHandle>::port_type("Mesh");
clString SimpleIPort<MeshHandle>::port_color("orange red");

} // End namespace CommonDatatypes
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:55:48  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:51  dav
// Import sources
//
//
