
/*
 *  BooleanPort.h
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_BooleanPort_h
#define SCI_project_BooleanPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Core/Datatypes/Boolean.h>

namespace SCIRun {


typedef SimpleIPort<sciBooleanHandle> sciBooleanIPort;
typedef SimpleOPort<sciBooleanHandle> sciBooleanOPort;

} // End namespace SCIRun


#endif
