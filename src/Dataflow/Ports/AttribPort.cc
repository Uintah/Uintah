// AttribPort.cc
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Group

#include <Dataflow/Ports/AttribPort.h>

namespace SCIRun {


template<> clString SimpleIPort<AttribHandle>::port_type("Attrib");
template<> clString SimpleIPort<AttribHandle>::port_color("Orange");

} // End namespace SCIRun

