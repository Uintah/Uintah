
/*
 *  DPLinEqPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Modules/DP/DPLinEqPort.h>

clString SimpleIPort<DPLinEqHandle>::port_type("DPLinEq");
clString SimpleIPort<DPLinEqHandle>::port_color("purple");

#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<DPLinEqHandle>;
template class SimpleOPort<DPLinEqHandle>;
template class SimplePortComm<DPLinEqHandle>;
template class Mailbox<SimplePortComm<DPLinEqHandle>*>;

#endif
