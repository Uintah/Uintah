
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

#include <Dataflow/Ports/SimplePort.h>
#include <Core/Datatypes/HexMesh.h>

namespace SCIRun {


typedef SimpleIPort<HexMeshHandle> HexMeshIPort;
typedef SimpleOPort<HexMeshHandle> HexMeshOPort;

} // End namespace SCIRun


#endif
