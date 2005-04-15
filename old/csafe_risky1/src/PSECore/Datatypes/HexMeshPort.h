
/*
 *  HexMeshPort.h
 *
 *  Written by:
 *   Peter Jensen
 *   Sourced from MeshPort.h by David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCI_project_HexMeshPort_h
#define SCI_project_HexMeshPort_h 1

#include <PSECore/Datatypes/SimplePort.h>
#include <SCICore/Datatypes/HexMesh.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Datatypes;

typedef SimpleIPort<HexMeshHandle> HexMeshIPort;
typedef SimpleOPort<HexMeshHandle> HexMeshOPort;

} // End namespace Datatypes
} // End namespace PSECore

//
// $Log$
// Revision 1.3  1999/08/25 03:48:20  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:08  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:47  mcq
// Initial commit
//
// Revision 1.4  1999/05/06 20:17:01  dav
// added back PSECore .h files
//
// Revision 1.2  1999/04/27 23:18:35  dav
// looking for lost files to commit
//
// Revision 1.1.1.1  1999/04/24 23:12:47  dav
// Import sources
//
//

#endif
