
/*
 *  SurfOctreeTopPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   May 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#include <Datatypes/SurfOctreeTopPort.h>

clString SimpleIPort<SurfOctreeTopHandle>::port_type("SurfOctreeTop");
clString SimpleIPort<SurfOctreeTopHandle>::port_color("grey70");


#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<SurfOctreeTopHandle>;
template class SimpleOPort<SurfOctreeTopHandle>;
template class SimplePortComm<SurfOctreeTopHandle>;
template class Mailbox<SimplePortComm<SurfOctreeTopHandle>*>;

#endif
