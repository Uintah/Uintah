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

#include <PSECore/CommonDatatypes/HexMeshPort.h>

namespace PSECore {
namespace CommonDatatypes {

using namespace SCICore::CoreDatatypes;

clString SimpleIPort<HexMeshHandle>::port_type("HexMesh");
clString SimpleIPort<HexMeshHandle>::port_color("yellow green");

} // End namespace CommonDatatypes
} // End namespace PSECore

//
// $Log$
// Revision 1.2  1999/08/17 06:38:08  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:47  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:50  dav
// Import sources
//
//
