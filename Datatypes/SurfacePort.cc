
/*
 *  SurfacePort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/SurfacePort.h>

clString SimpleIPort<SurfaceHandle>::port_type("Surface");
clString SimpleIPort<SurfaceHandle>::port_color("SteelBlue4");


#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<SurfaceHandle>;
template class SimpleOPort<SurfaceHandle>;
template class SimplePortComm<SurfaceHandle>;
template class Mailbox<SimplePortComm<SurfaceHandle>*>;

#endif
