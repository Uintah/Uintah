// FieldSetPort.cc
//  Written by:
//   Michael Callahan
//   Department of Computer Science
//   University of Utah
//   February 2001
//  Copyright (C) 2001 SCI Group


#include <Dataflow/Ports/FieldSetPort.h>
#include <Core/Malloc/Allocator.h>


namespace SCIRun {

extern "C" {
PSECORESHARE IPort* make_FieldSetIPort(Module* module, const clString& name) {
  return scinew SimpleIPort<FieldSetHandle>(module,name);
}
PSECORESHARE OPort* make_FieldSetOPort(Module* module, const clString& name) {
  return scinew SimpleOPort<FieldSetHandle>(module,name);
}
}

template<> clString SimpleIPort<FieldSetHandle>::port_type("FieldSet");
template<> clString SimpleIPort<FieldSetHandle>::port_color("orange");

} // End namespace SCIRun

