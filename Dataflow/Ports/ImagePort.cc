
/*
 *  ImagePort.cc
 *
 *  Written by:
 *   Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Oct 1996
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Ports/ImagePort.h>
#include <Dataflow/share/share.h>
#include <Core/Malloc/Allocator.h>


using namespace SCIRun;

extern "C" {
PSECORESHARE IPort* make_ImageIPort(Module* module, const clString& name) {
  return scinew SimpleIPort<ImageHandle>(module,name);
}
PSECORESHARE OPort* make_ImageOPort(Module* module, const clString& name) {
  return scinew SimpleOPort<ImageHandle>(module,name);
}
}

template<> clString SimpleIPort<ImageHandle>::port_type("Image");
template<> clString SimpleIPort<ImageHandle>::port_color("misty rose");


