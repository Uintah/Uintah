
/*
 *  MultiMeshPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef SCI_project_MultiMeshPort_h
#define SCI_project_MultiMeshPort_h 1

#include <Datatypes/SimplePort.h>
#include <Datatypes/MultiMesh.h>

typedef Mailbox<SimplePortComm<MultiMeshHandle>*> _cfront_bug_MultiMesh_;
typedef SimpleIPort<MultiMeshHandle> MultiMeshIPort;
typedef SimpleOPort<MultiMeshHandle> MultiMeshOPort;

#endif
