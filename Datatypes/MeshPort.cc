
/*
 *  MeshPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Datatypes/MeshPort.h>

using sci::MeshHandle;

clString SimpleIPort<MeshHandle>::port_type("Mesh");
clString SimpleIPort<MeshHandle>::port_color("orange red");

#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<MeshHandle>;
template class SimpleOPort<MeshHandle>;
template class SimplePortComm<MeshHandle>;
template class Mailbox<SimplePortComm<MeshHandle>*>;

#endif
