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
#include <SCICore/Malloc/Allocator.h>

namespace PSECore {
namespace Datatypes {

using namespace SCICore::Containers;

extern "C" {
PSECORESHARE IPort* make_PathIPort(Module* module, const clString& name) {
  return scinew SimpleIPort<PathHandle>(module,name);
}
PSECORESHARE OPort* make_PathOPort(Module* module, const clString& name) {
  return scinew SimpleOPort<PathHandle>(module,name);
}
}

template<> clString SimpleIPort<PathHandle>::port_type("Path");
template<> clString SimpleIPort<PathHandle>::port_color("chocolate4");

} // End namespace Datatypes
} // End namespace PSECore
