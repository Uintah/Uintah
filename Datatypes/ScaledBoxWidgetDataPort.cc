
/*
 *  ScaledBoxWidgetDataPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/ScaledBoxWidgetDataPort.h>

clString SimpleIPort<ScaledBoxWidgetDataHandle>::port_type("Data");
clString SimpleIPort<ScaledBoxWidgetDataHandle>::port_color("black");

#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<ScaledBoxWidgetDataHandle>;
template class SimpleOPort<ScaledBoxWidgetDataHandle>;
template class SimplePortComm<ScaledBoxWidgetDataHandle>;
template class Mailbox<SimplePortComm<ScaledBoxWidgetDataHandle>*>;

#endif
