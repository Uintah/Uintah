// GeomPort.h
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Group


#ifndef SCI_project_GeomPort_h
#define SCI_project_GeomPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Core/Datatypes/Geom.h>

namespace SCIRun {


typedef SimpleIPort<GeomHandle> GeomIPort;
typedef SimpleOPort<GeomHandle> GeomOPort;

} // End namespace SCIRun

#endif
