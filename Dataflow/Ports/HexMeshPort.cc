//static char *id="@(#) $Id$";

/*
 *  HexMeshPort.cc
 *
 *  Written by:
 *   Peter Jensen
 *   Sourced from MeshPort.cc by David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <CommonDatatypes/HexMeshPort.h>

namespace PSECommon {
namespace CommonDatatypes {

clString SimpleIPort<HexMeshHandle>::port_type("HexMesh");
clString SimpleIPort<HexMeshHandle>::port_color("yellow green");

} // End namespace CommonDatatypes
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  1999/07/27 16:55:47  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:50  dav
// Import sources
//
//
