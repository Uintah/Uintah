/*
 *  NrrdPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Nrrd/Dataflow/Ports/NrrdPort.h>
#include <Nrrd/share/share.h>
#include <Core/Malloc/Allocator.h>

namespace SCINrrd {

using namespace SCIRun;

extern "C" {
  NrrdSHARE IPort* make_NrrdIPort(Module* module, const clString& name) {
  return scinew SimpleIPort<NrrdDataHandle>(module,name);
}
  NrrdSHARE OPort* make_NrrdOPort(Module* module, const clString& name) {
  return scinew SimpleOPort<NrrdDataHandle>(module,name);
}
}
} // End namespace SCINrrd

namespace SCIRun {
template<> clString SimpleIPort<SCINrrd::NrrdDataHandle>::port_type("Nrrd");
template<> clString SimpleIPort<SCINrrd::NrrdDataHandle>::port_color("Purple4");
} // End namespace SCIRun

