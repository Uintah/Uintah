
/*
 *  VectorField.h: The Vector Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Ports/VectorFieldPort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


extern "C" {
PSECORESHARE IPort* make_VectorFieldIPort(Module* module,
					  const clString& name) {
  return scinew SimpleIPort<VectorFieldHandle>(module,name);
}
PSECORESHARE OPort* make_VectorFieldOPort(Module* module,
					  const clString& name) {
  return scinew SimpleOPort<VectorFieldHandle>(module,name);
}
}

template<> clString SimpleIPort<VectorFieldHandle>::port_type("VectorField");
template<> clString SimpleIPort<VectorFieldHandle>::port_color("orchid4");

} // End namespace SCIRun

