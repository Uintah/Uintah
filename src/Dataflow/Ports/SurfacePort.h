
/*
 *  SurfacePort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_SurfacePort_h
#define SCI_project_SurfacePort_h 1

#include <Dataflow/Ports/SimplePort.h>
#include <Core/Datatypes/Surface.h>

namespace SCIRun {


typedef SimpleIPort<SurfaceHandle> SurfaceIPort;
typedef SimpleOPort<SurfaceHandle> SurfaceOPort;

} // End namespace SCIRun


#endif
