//static char *id="@(#) $Id$";

/*
 *  PathPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   November 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <PSECore/Datatypes/PathPort.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Containers;

extern "C" {
PSECORESHARE IPort* make_PathIPort(Module* module, const clString& name) {
  return new SimpleIPort<PathHandle>(module,name);
}
PSECORESHARE OPort* make_PathOPort(Module* module, const clString& name) {
  return new SimpleOPort<PathHandle>(module,name);
}
}

template<> clString SimpleIPort<PathHandle>::port_type("Path");
template<> clString SimpleIPort<PathHandle>::port_color("chocolate4");

} // End namespace Datatypes
} // End namespace PSECore

//
// $Log$
// Revision 1.2  2000/11/22 17:14:41  moulding
// added extern "C" make functions for input and output ports (to be used
// by the auto-port facility).
//
// Revision 1.1  2000/07/19 06:35:50  samsonov
// PathPort datatype moved from DaveW
//
// Revision 1.1  1999/12/02 21:57:29  dmw
// new camera path datatypes and modules
//
//
