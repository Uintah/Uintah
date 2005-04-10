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

#include <PSECore/Datatypes/SurfacePort.h>
#include <SCICore/Malloc/Allocator.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Datatypes;

extern "C" {
PSECORESHARE IPort* make_SurfaceIPort(Module* module, const clString& name) {
  return scinew SimpleIPort<SurfaceHandle>(module,name);
}
PSECORESHARE OPort* make_SurfaceOPort(Module* module, const clString& name) {
  return scinew SimpleOPort<SurfaceHandle>(module,name);
}
}

template<> clString SimpleIPort<SurfaceHandle>::port_type("Surface");
template<> clString SimpleIPort<SurfaceHandle>::port_color("SteelBlue4");

} // End namespace Datatypes
} // End namespace PSECore

//
// $Log$
// Revision 1.6  2000/11/29 09:49:37  moulding
// changed all instances of "new" to "scinew"
//
// Revision 1.5  2000/11/22 17:14:42  moulding
// added extern "C" make functions for input and output ports (to be used
// by the auto-port facility).
//
// Revision 1.4  1999/08/30 20:19:24  sparker
// Updates to compile with -LANG:std on SGI
// Other linux/irix porting oscillations
//
// Revision 1.3  1999/08/25 03:48:24  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
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
