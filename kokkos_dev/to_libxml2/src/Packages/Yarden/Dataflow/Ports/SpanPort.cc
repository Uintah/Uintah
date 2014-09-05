
/*
 *  SpanPort.cc
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Nov. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/Array1.h>
#include <Core/Datatypes/ScalarField.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Geometry/BBox.h>
#include <Packages/Yarden/Dataflow/Ports/SpanPort.h>
#include <Packages/Yarden/share/share.h>
#include <Core/Malloc/Allocator.h>

namespace Yarden {

using namespace SCIRun;

extern "C" {
  YardenSHARE IPort* make_SpanUniverseIPort(Module* module,
					   const clString& name) {
  return scinew SimpleIPort<SpanUniverseHandle>(module,name);
}
  YardenSHARE OPort* make_SpanUniverseOPort(Module* module,
					   const clString& name) {
  return scinew SimpleOPort<SpanUniverseHandle>(module,name);
}
}
} // End namespace Yarden

namespace SCIRun {
template<> clString SimpleIPort<Yarden::SpanUniverseHandle>::port_type("SpanUniverse");
template<> clString SimpleIPort<Yarden::SpanUniverseHandle>::port_color("SteelBlue4");

} // End namespace SCIRun
