
/*
 *  HexMeshPort.cc
 *
 *  Written by:
 *   Peter Jensen
 *   Sourced from MeshPort.cc by David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Dataflow/Ports/HexMeshPort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


extern "C" {
PSECORESHARE IPort* make_HexMeshIPort(Module* module, const clString& name) {
  return scinew SimpleIPort<HexMeshHandle>(module,name);
}
PSECORESHARE OPort* make_HexMeshOPort(Module* module, const clString& name) {
  return scinew SimpleOPort<HexMeshHandle>(module,name);
}
}

template<> clString SimpleIPort<HexMeshHandle>::port_type("HexMesh");
template<> clString SimpleIPort<HexMeshHandle>::port_color("yellow green");

} // End namespace SCIRun

