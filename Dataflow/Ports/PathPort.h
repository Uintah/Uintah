
/*
 *  PathPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   Novemeber 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef SCI_Dataflow_Datatypes_PathPort_h
#define SCI_Dataflow_Datatypes_PathPort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Core/Datatypes/Path.h>

namespace SCIRun {


typedef SimpleIPort<PathHandle> PathIPort;
typedef SimpleOPort<PathHandle> PathOPort;

} // End namespace SCIRun


#endif
