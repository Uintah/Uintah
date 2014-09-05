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
 *  PathPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   November 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Dataflow/Ports/PathPort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


extern "C" {
PSECORESHARE IPort* make_PathIPort(Module* module, const string& name) {
  return scinew SimpleIPort<PathHandle>(module,name);
}
PSECORESHARE OPort* make_PathOPort(Module* module, const string& name) {
  return scinew SimpleOPort<PathHandle>(module,name);
}
}

template<> string SimpleIPort<PathHandle>::port_type("Path");
template<> string SimpleIPort<PathHandle>::port_color("chocolate4");

} // End namespace SCIRun
