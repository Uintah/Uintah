
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

#include <Dataflow/Ports/VoidStarPort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


extern "C" {
PSECORESHARE IPort* make_VoidStarIPort(Module* module, const clString& name) {
  return scinew SimpleIPort<VoidStarHandle>(module,name);
}
PSECORESHARE OPort* make_VoidStarOPort(Module* module, const clString& name) {
  return scinew SimpleOPort<VoidStarHandle>(module,name);
}
}

template<> clString SimpleIPort<VoidStarHandle>::port_type("VoidStar");
template<> clString SimpleIPort<VoidStarHandle>::port_color("gold1");

} // End namespace SCIRun

