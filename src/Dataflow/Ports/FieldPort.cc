// FieldPort.cc
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Group


#include <Dataflow/Ports/FieldPort.h>

namespace SCIRun {


template<> clString SimpleIPort<FieldHandle>::port_type("Field");
template<> clString SimpleIPort<FieldHandle>::port_color("yellow");

} // End namespace SCIRun

