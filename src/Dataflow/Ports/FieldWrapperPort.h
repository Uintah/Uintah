// FieldWrapperPort.h
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Group


#ifndef SCI_project_FieldWrapperPort_h
#define SCI_project_FieldWrapperPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Core/Datatypes/FieldWrapper.h>

namespace SCIRun {


typedef SimpleIPort<FieldWrapperHandle> FieldWrapperIPort;
typedef SimpleOPort<FieldWrapperHandle> FieldWrapperOPort;

} // End namespace SCIRun

#endif
