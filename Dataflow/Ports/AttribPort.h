// AttribPort.h
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Group


#ifndef SCI_project_AttribPort_h
#define SCI_project_AttribPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Core/Datatypes/Attrib.h>

namespace SCIRun {


typedef SimpleIPort<AttribHandle> AttribIPort;
typedef SimpleOPort<AttribHandle> AttribOPort;

} // End namespace SCIRun

#endif
