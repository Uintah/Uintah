//static char *id="@(#) $Id$";

/*
 *  VoidStarPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <PSECore/Datatypes/VoidStarPort.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Datatypes;

extern "C" {
PSECORESHARE IPort* make_VoidStarIPort(Module* module, const clString& name) {
  return new SimpleIPort<VoidStarHandle>(module,name);
}
PSECORESHARE OPort* make_VoidStarOPort(Module* module, const clString& name) {
  return new SimpleOPort<VoidStarHandle>(module,name);
}
}

template<> clString SimpleIPort<VoidStarHandle>::port_type("VoidStar");
template<> clString SimpleIPort<VoidStarHandle>::port_color("gold1");

} // End namespace Datatypes
} // End namespace PSECore

//
// $Log$
// Revision 1.5  2000/11/22 17:14:42  moulding
// added extern "C" make functions for input and output ports (to be used
// by the auto-port facility).
//
// Revision 1.4  1999/08/30 20:19:24  sparker
// Updates to compile with -LANG:std on SGI
// Other linux/irix porting oscillations
//
// Revision 1.3  1999/08/25 03:48:25  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.2  1999/08/17 06:38:13  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:55:51  mcq
// Initial commit
//
// Revision 1.1.1.1  1999/04/24 23:12:48  dav
// Import sources
//
//
