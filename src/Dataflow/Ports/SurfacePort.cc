
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

#include <Dataflow/Ports/SurfacePort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


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

} // End namespace SCIRun

