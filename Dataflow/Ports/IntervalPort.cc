
/*
 *  IntervalPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Ports/IntervalPort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


extern "C" {
PSECORESHARE IPort* make_IntervalIPort(Module* module, const clString& name) {
  return scinew SimpleIPort<IntervalHandle>(module,name);
}
PSECORESHARE OPort* make_IntervalOPort(Module* module, const clString& name) {
  return scinew SimpleOPort<IntervalHandle>(module,name);
}
}

template<> clString SimpleIPort<IntervalHandle>::port_type("Interval");
template<> clString SimpleIPort<IntervalHandle>::port_color("mediumseagreen");

} // End namespace SCIRun

