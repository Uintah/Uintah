/*
 *  NrrdPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_Nrrd_NrrdPort_h
#define SCI_Nrrd_NrrdPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Nrrd/Core/Datatypes/NrrdData.h>

namespace SCINrrd {

typedef SCIRun::SimpleIPort<NrrdDataHandle> NrrdIPort;
typedef SCIRun::SimpleOPort<NrrdDataHandle> NrrdOPort;

} // End namespace SCINrrd


#endif
