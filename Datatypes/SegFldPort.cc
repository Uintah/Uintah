
/*
 *  SegFld.h: The Scalar Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/SegFldPort.h>

clString SimpleIPort<SegFldHandle>::port_type("SegFld");
clString SimpleIPort<SegFldHandle>::port_color("Green");


#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<SegFldHandle>;
template class SimpleOPort<SegFldHandle>;
template class SimplePortComm<SegFldHandle>;
template class Mailbox<SimplePortComm<SegFldHandle>*>;

#endif
