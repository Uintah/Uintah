
/*
 *  VoidStarPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Datatypes/VoidStarPort.h>

clString SimpleIPort<VoidStarHandle>::port_type("VoidStar");
clString SimpleIPort<VoidStarHandle>::port_color("gold1");

#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<VoidStarHandle>;
template class SimpleOPort<VoidStarHandle>;
template class SimplePortComm<VoidStarHandle>;
template class Mailbox<SimplePortComm<VoidStarHandle>*>;

#endif
