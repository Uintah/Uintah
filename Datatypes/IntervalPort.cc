
/*
 *  IntervalPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/IntervalPort.h>

clString SimpleIPort<IntervalHandle>::port_type("Interval");
clString SimpleIPort<IntervalHandle>::port_color("mediumseagreen");

#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<IntervalHandle>;
template class SimpleOPort<IntervalHandle>;
template class SimplePortComm<IntervalHandle>;
template class Mailbox<SimplePortComm<IntervalHandle>*>;

#endif
