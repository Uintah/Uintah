
/*
 *  MeshPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Ports/MeshPort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


extern "C" {
PSECORESHARE IPort* make_MeshIPort(Module* module, const clString& name) {
  return scinew SimpleIPort<MeshHandle>(module,name);
}
PSECORESHARE OPort* make_MeshOPort(Module* module, const clString& name) {
  return scinew SimpleOPort<MeshHandle>(module,name);
}
}

template<> clString SimpleIPort<MeshHandle>::port_type("Mesh");
template<> clString SimpleIPort<MeshHandle>::port_color("orange red");

} // End namespace SCIRun

