
/*
 *  MatrixPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/share/share.h>

#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {

extern "C" {
PSECORESHARE IPort* make_MatrixIPort(Module* module, const clString& name) {
  return scinew SimpleIPort<MatrixHandle>(module,name);
}
PSECORESHARE OPort* make_MatrixOPort(Module* module, const clString& name) {
  return scinew SimpleOPort<MatrixHandle>(module,name);
}
}


template<> clString SimpleIPort<MatrixHandle>::port_type("Matrix");
template<> clString SimpleIPort<MatrixHandle>::port_color("dodger blue");

} // End namespace SCIRun

