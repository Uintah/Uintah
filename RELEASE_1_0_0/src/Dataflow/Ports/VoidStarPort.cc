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
 *  VoidStarPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Dataflow/Ports/VoidStarPort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


extern "C" {
PSECORESHARE IPort* make_VoidStarIPort(Module* module, const clString& name) {
  return scinew SimpleIPort<VoidStarHandle>(module,name);
}
PSECORESHARE OPort* make_VoidStarOPort(Module* module, const clString& name) {
  return scinew SimpleOPort<VoidStarHandle>(module,name);
}
}

template<> clString SimpleIPort<VoidStarHandle>::port_type("VoidStar");
template<> clString SimpleIPort<VoidStarHandle>::port_color("gold1");

} // End namespace SCIRun

