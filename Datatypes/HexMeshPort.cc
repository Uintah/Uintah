
/*
 *  HexMeshPort.cc
 *
 *  Written by:
 *   Peter Jensen
 *   Sourced from MeshPort.cc by David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Datatypes/HexMeshPort.h>

clString SimpleIPort<HexMeshHandle>::port_type("HexMesh");
clString SimpleIPort<HexMeshHandle>::port_color("yellow green");

#ifdef __GNUG__

#include <Datatypes/SimplePort.cc>
#include <Multitask/Mailbox.cc>
template class SimpleIPort<HexMeshHandle>;
template class SimpleOPort<HexMeshHandle>;
template class SimplePortComm<HexMeshHandle>;
template class Mailbox<SimplePortComm<HexMeshHandle>*>;

#endif
