// GeomPort.cc
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Group


#include <Dataflow/Ports/GeomPort.h>

namespace SCIRun {


template<> clString SimpleIPort<GeomHandle>::port_type("Geom");
template<> clString SimpleIPort<GeomHandle>::port_color("blue");

} // End namespace SCIRun

