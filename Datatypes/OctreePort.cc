
/*
 *  OctreePort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/OctreePort.h>

clString SimpleIPort<OctreeTopHandle>::port_type("Octree");
clString SimpleIPort<OctreeTopHandle>::port_color("blue");


#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<OctreeTopHandle>;
template class SimpleOPort<OctreeTopHandle>;
template class SimplePortComm<OctreeTopHandle>;
template class Mailbox<SimplePortComm<OctreeTopHandle>*>;

#endif
