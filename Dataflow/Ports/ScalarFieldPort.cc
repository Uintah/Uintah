
/*
 *  ScalarField.h: The Scalar Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


extern "C" {
PSECORESHARE IPort* make_ScalarFieldIPort(Module* module,
					  const clString& name) {
  return scinew SimpleIPort<ScalarFieldHandle>(module,name);
}
PSECORESHARE OPort* make_ScalarFieldOPort(Module* module,
					  const clString& name) {
  return scinew SimpleOPort<ScalarFieldHandle>(module,name);
}
}

template<> clString SimpleIPort<ScalarFieldHandle>::port_type("ScalarField");
template<> clString SimpleIPort<ScalarFieldHandle>::port_color("VioletRed2");

} // End namespace SCIRun

