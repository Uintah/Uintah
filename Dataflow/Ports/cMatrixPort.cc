
/*
 *  cMatrixPort.h
 *
 *  Written by:
 *   Leonid Zhukov
 *   Department of Computer Science
 *   University of Utah
 *   August 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Dataflow/Ports/cMatrixPort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


extern "C" {
PSECORESHARE IPort* make_cMatrixIPort(Module* module, const clString& name) {
  return scinew SimpleIPort<cMatrixHandle>(module,name);
}
PSECORESHARE OPort* make_cMatrixOPort(Module* module, const clString& name) {
  return scinew SimpleOPort<cMatrixHandle>(module,name);
}
}

template<> clString SimpleIPort<cMatrixHandle>::port_type("cMatrix");
template<> clString SimpleIPort<cMatrixHandle>::port_color("red");

} // End namespace SCIRun

