
/*
 *  ColumnMatrixPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Ports/ColumnMatrixPort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


extern "C" {
PSECORESHARE IPort* make_ColumnMatrixIPort(Module* module,
					   const clString& name) {
  return scinew SimpleIPort<ColumnMatrixHandle>(module,name);
}
PSECORESHARE OPort* make_ColumnMatrixOPort(Module* module,
					   const clString& name) {
  return scinew SimpleOPort<ColumnMatrixHandle>(module,name);
}
}

template<> clString SimpleIPort<ColumnMatrixHandle>::port_type("ColumnMatrix");
template<> clString SimpleIPort<ColumnMatrixHandle>::port_color("dodgerblue4");

} // End namespace SCIRun


