// FieldWrapperPort.cc
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Group


#include <Dataflow/Ports/FieldWrapperPort.h>

namespace SCIRun {


template<> clString SimpleIPort<FieldWrapperHandle>::port_type("FieldWrapper");
template<> clString SimpleIPort<FieldWrapperHandle>::port_color("blue");

} // End namespace SCIRun

