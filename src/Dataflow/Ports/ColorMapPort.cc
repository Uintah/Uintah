
/*
 *  ColorMapPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Ports/ColorMapPort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


extern "C" {
PSECORESHARE IPort* make_ColorMapIPort(Module* module, const clString& name) {
  return scinew SimpleIPort<ColorMapHandle>(module,name);
}
PSECORESHARE OPort* make_ColorMapOPort(Module* module, const clString& name) {
  return scinew SimpleOPort<ColorMapHandle>(module,name);
}
}

template<> clString ColorMapIPort::port_type("ColorMap");
template<> clString ColorMapIPort::port_color("blueviolet");

} // End namespace SCIRun


