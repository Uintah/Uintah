
// cVectorPort.h
//  Written by:
//   Leonid Zhukov
//   Department of Computer Science
//   University of Utah
//   August 1997
//  Copyright (C) 1997 SCI Group

#ifndef SCI_project_cVectorPort_h
#define SCI_project_cVectorPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Core/Datatypes/cVector.h>

namespace SCIRun {


typedef SimpleIPort<cVectorHandle> cVectorIPort;
typedef SimpleOPort<cVectorHandle> cVectorOPort;

} // End namespace SCIRun


#endif
