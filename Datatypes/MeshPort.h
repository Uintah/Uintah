
/*
 *  MeshPort.h
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_MeshPort_h
#define SCI_project_MeshPort_h 1

#include <Datatypes/SimplePort.h>
#include <Datatypes/Mesh.h>

typedef Mailbox<SimplePortComm<sci::MeshHandle>*> _cfront_bug_Mesh_;
typedef SimpleIPort<sci::MeshHandle> MeshIPort;
typedef SimpleOPort<sci::MeshHandle> MeshOPort;

#endif
