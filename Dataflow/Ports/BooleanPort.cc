
/*
 *  sciBooleanPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Ports/BooleanPort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


extern "C" {
PSECORESHARE IPort* make_sciBooleanIPort(Module* module, 
					 const clString& name) {
  return scinew SimpleIPort<sciBooleanHandle>(module,name);
}
PSECORESHARE OPort* make_sciBooleanOPort(Module* module, 
					 const clString& name) {
  return scinew SimpleOPort<sciBooleanHandle>(module,name);
}
}

template<> clString SimpleIPort<sciBooleanHandle>::port_type("Boolean");
template<> clString SimpleIPort<sciBooleanHandle>::port_color("blue4");

} // End namespace SCIRun


