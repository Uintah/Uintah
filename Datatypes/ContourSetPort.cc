
/*
 *  ContourSetPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/ContourSetPort.h>

clString SimpleIPort<ContourSetHandle>::port_type("ContourSet");
clString SimpleIPort<ContourSetHandle>::port_color("#388e8e");

#ifdef __GNUG__
#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<ContourSetHandle>;
template class SimpleOPort<ContourSetHandle>;
template class SimplePortComm<ContourSetHandle>;
template class Mailbox<SimplePortComm<ContourSetHandle>*>;

#endif
