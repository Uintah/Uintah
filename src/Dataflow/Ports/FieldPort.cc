// FieldPort.cc
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Group


#include <Dataflow/Ports/FieldPort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {

extern "C" {
PSECORESHARE IPort* make_FieldIPort(Module* module, const clString& name) {
  return scinew SimpleIPort<FieldHandle>(module,name);
}
PSECORESHARE OPort* make_FieldOPort(Module* module, const clString& name) {
  return scinew SimpleOPort<FieldHandle>(module,name);
}
}

template<> clString SimpleIPort<FieldHandle>::port_type("Field");
template<> clString SimpleIPort<FieldHandle>::port_color("yellow");

} // End namespace SCIRun

