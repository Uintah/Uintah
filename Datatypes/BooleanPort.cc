
/*
 *  sciBooleanPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/BooleanPort.h>

clString SimpleIPort<sciBooleanHandle>::port_type("Boolean");
clString SimpleIPort<sciBooleanHandle>::port_color("blue4");

#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<sciBooleanHandle>;
template class SimpleOPort<sciBooleanHandle>;
template class SimplePortComm<sciBooleanHandle>;
template class Mailbox<SimplePortComm<sciBooleanHandle>*>;

#endif
