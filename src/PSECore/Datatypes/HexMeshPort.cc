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

#include <PSECore/Datatypes/HexMeshPort.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Datatypes;

extern "C" {
PSECORESHARE IPort* make_HexMeshIPort(Module* module, const clString& name) {
  return new SimpleIPort<HexMeshHandle>(module,name);
}
PSECORESHARE OPort* make_HexMeshOPort(Module* module, const clString& name) {
  return new SimpleOPort<HexMeshHandle>(module,name);
}
}

template<> clString SimpleIPort<HexMeshHandle>::port_type("HexMesh");
template<> clString SimpleIPort<HexMeshHandle>::port_color("yellow green");

} // End namespace Datatypes
} // End namespace PSECore

//
// $Log$
// Revision 1.5  2000/11/22 17:14:41  moulding
// added extern "C" make functions for input and output ports (to be used
// by the auto-port facility).
//
// Revision 1.4  1999/08/30 20:19:23  sparker
// Updates to compile with -LANG:std on SGI
// Other linux/irix porting oscillations
//
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
// Revision 1.1.1.1  1999/04/24 23:12:50  dav
// Import sources
//
//
