
/*
 *  DPVecPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Modules/DP/DPVecPort.h>

clString SimpleIPort<DPVecHandle>::port_type("DPVec");
clString SimpleIPort<DPVecHandle>::port_color("green");

#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<DPVecHandle>;
template class SimpleOPort<DPVecHandle>;
template class SimplePortComm<DPVecHandle>;
template class Mailbox<SimplePortComm<DPVecHandle>*>;

#endif
