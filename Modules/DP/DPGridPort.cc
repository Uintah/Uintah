
/*
 *  DPGridPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Modules/DP/DPGridPort.h>

clString SimpleIPort<DPGridHandle>::port_type("DPGrid");
clString SimpleIPort<DPGridHandle>::port_color("dodger blue");

#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<DPGridHandle>;
template class SimpleOPort<DPGridHandle>;
template class SimplePortComm<DPGridHandle>;
template class Mailbox<SimplePortComm<DPGridHandle>*>;

#endif
