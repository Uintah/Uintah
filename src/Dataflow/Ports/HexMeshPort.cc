/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Dataflow/Ports/HexMeshPort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


extern "C" {
PSECORESHARE IPort* make_HexMeshIPort(Module* module, const clString& name) {
  return scinew SimpleIPort<HexMeshHandle>(module,name);
}
PSECORESHARE OPort* make_HexMeshOPort(Module* module, const clString& name) {
  return scinew SimpleOPort<HexMeshHandle>(module,name);
}
}

template<> clString SimpleIPort<HexMeshHandle>::port_type("HexMesh");
template<> clString SimpleIPort<HexMeshHandle>::port_color("yellow green");

} // End namespace SCIRun

