// DomainPort.h
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Group

#ifndef SCI_project_DomainPort_h
#define SCI_project_DomainPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Core/Datatypes/Domain.h>

namespace SCIRun {


typedef SimpleIPort<DomainHandle> DomainIPort;
typedef SimpleOPort<DomainHandle> DomainOPort;

} // End namespace SCIRun

#endif
