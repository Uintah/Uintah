
/*
 *  HexMeshPort.h
 *
 *  Written by:
 *   Peter Jensen
 *   Sourced from MeshPort.h by David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#ifndef SCI_project_HexMeshPort_h
#define SCI_project_HexMeshPort_h 1

#include <Datatypes/SimplePort.h>
#include <Datatypes/HexMesh.h>

typedef Mailbox<SimplePortComm<HexMeshHandle>*> _cfront_bug_HexMesh_;
typedef SimpleIPort<HexMeshHandle> HexMeshIPort;
typedef SimpleOPort<HexMeshHandle> HexMeshOPort;

#endif
