// FieldSetPort.h
//  Written by:
//   Michael Callahan
//   Department of Computer Science
//   University of Utah
//   February 2001
//  Copyright (C) 2001 SCI Group


#ifndef SCI_project_FieldSetPort_h
#define SCI_project_FieldSetPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Core/Datatypes/FieldSet.h>


namespace SCIRun {

typedef SimpleIPort<FieldSetHandle> FieldSetIPort;
typedef SimpleOPort<FieldSetHandle> FieldSetOPort;

} // End namespace SCIRun

#endif  // SCI_project_FieldSetPort_h
