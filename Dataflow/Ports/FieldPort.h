// FieldPort.h
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Group

#ifndef SCI_project_FieldPort_h
#define SCI_project_FieldPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Core/Datatypes/Field.h>

namespace SCIRun {


typedef SimpleIPort<FieldHandle> FieldIPort;
typedef SimpleOPort<FieldHandle> FieldOPort;

} // End namespace SCIRun

#endif
