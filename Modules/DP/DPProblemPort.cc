
/*
 *  DPProblemPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Modules/DP/DPProblemPort.h>

clString SimpleIPort<DPProblemHandle>::port_type("DPProblem");
clString SimpleIPort<DPProblemHandle>::port_color("red");

#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<DPProblemHandle>;
template class SimpleOPort<DPProblemHandle>;
template class SimplePortComm<DPProblemHandle>;
template class Mailbox<SimplePortComm<DPProblemHandle>*>;

#endif
