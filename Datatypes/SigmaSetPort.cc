
/*
 *  SigmaSetPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#include <Datatypes/SigmaSetPort.h>

clString SimpleIPort<SigmaSetHandle>::port_type("SigmaSet");
clString SimpleIPort<SigmaSetHandle>::port_color("chocolate4");

#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<SigmaSetHandle>;
template class SimpleOPort<SigmaSetHandle>;
template class SimplePortComm<SigmaSetHandle>;
template class Mailbox<SimplePortComm<SigmaSetHandle>*>;

#endif
