
/*
 *  OctreePort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_OctreePort_h
#define SCI_project_OctreePort_h 1

#include <Datatypes/SimplePort.h>
#include <Datatypes/Octree.h>

typedef SimpleIPort<OctreeTopHandle> OctreeIPort;
typedef SimpleOPort<OctreeTopHandle> OctreeOPort;

#endif
