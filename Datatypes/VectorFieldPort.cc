
/*
 *  VectorField.h: The Vector Field Data type
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/VectorFieldPort.h>

clString SimpleIPort<VectorFieldHandle>::port_type("VectorField");
clString SimpleIPort<VectorFieldHandle>::port_color("orchid4");

#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>

template class SimpleIPort<VectorFieldHandle>;
template class SimpleOPort<VectorFieldHandle>;
template class SimplePortComm<VectorFieldHandle>;
template class Mailbox<SimplePortComm<VectorFieldHandle>*>;

#endif
