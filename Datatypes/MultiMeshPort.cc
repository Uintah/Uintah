
/*
 *  MultiMeshPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Datatypes/MultiMeshPort.h>

clString SimpleIPort<MultiMeshHandle>::port_type("MultiMesh");
clString SimpleIPort<MultiMeshHandle>::port_color("red");

#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<MultiMeshHandle>;
template class SimpleOPort<MultiMeshHandle>;
template class SimplePortComm<MultiMeshHandle>;
template class Mailbox<SimplePortComm<MultiMeshHandle>*>;

#endif
