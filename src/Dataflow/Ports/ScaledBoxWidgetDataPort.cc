
/*
 *  ScaledBoxWidgetDataPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Ports/ScaledBoxWidgetDataPort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


extern "C" {
PSECORESHARE IPort* make_ScaledBoxWidgetDataIPort(Module* module,
						  const clString& name) {
  return scinew SimpleIPort<ScaledBoxWidgetDataHandle>(module,name);
}
PSECORESHARE OPort* make_ScaledBoxWidgetDataOPort(Module* module,
						  const clString& name) {
  return scinew SimpleOPort<ScaledBoxWidgetDataHandle>(module,name);
}
}

template<> clString SimpleIPort<ScaledBoxWidgetDataHandle>::port_type("Data");
template<> clString SimpleIPort<ScaledBoxWidgetDataHandle>::port_color("black");

} // End namespace SCIRun

