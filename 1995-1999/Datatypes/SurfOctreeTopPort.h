
/*
 *  SurfOctreeTopPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#ifndef SCI_project_SurfOctreeTopPort_h
#define SCI_project_SurfOctreeTopPort_h 1

#include <Datatypes/SimplePort.h>
#include <Datatypes/SurfOctree.h>

typedef SimpleIPort<SurfOctreeTopHandle> SurfOctreeTopIPort;
typedef SimpleOPort<SurfOctreeTopHandle> SurfOctreeTopOPort;

#endif
