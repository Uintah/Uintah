
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

#include <Datatypes/SimplePort.h>
#include <Datatypes/Surface.h>

typedef SimpleIPort<SurfaceHandle> SurfaceIPort;
typedef SimpleOPort<SurfaceHandle> SurfaceOPort;

#endif
