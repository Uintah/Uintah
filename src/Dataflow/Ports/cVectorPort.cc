
/*
 *  cVectorPort.cc : ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <Dataflow/Ports/cVectorPort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


extern "C" {
PSECORESHARE IPort* make_cVectorIPort(Module* module, const clString& name) {
  return scinew SimpleIPort<cVectorHandle>(module,name);
}
PSECORESHARE OPort* make_cVectorOPort(Module* module, const clString& name) {
  return scinew SimpleOPort<cVectorHandle>(module,name);
}
}

template<> clString SimpleIPort<cVectorHandle>::port_type("cVector");
template<> clString SimpleIPort<cVectorHandle>::port_color("yellow");

} // End namespace SCIRun

