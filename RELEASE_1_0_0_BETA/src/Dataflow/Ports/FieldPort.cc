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

// FieldPort.cc
//  Written by:
//   Eric Kuehne
//   Department of Computer Science
//   University of Utah
//   April 2000
//  Copyright (C) 2000 SCI Group


#include <Dataflow/Ports/FieldPort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {

extern "C" {
PSECORESHARE IPort* make_FieldIPort(Module* module, const clString& name) {
  return scinew SimpleIPort<FieldHandle>(module,name);
}
PSECORESHARE OPort* make_FieldOPort(Module* module, const clString& name) {
  return scinew SimpleOPort<FieldHandle>(module,name);
}
}

template<> clString SimpleIPort<FieldHandle>::port_type("Field");
template<> clString SimpleIPort<FieldHandle>::port_color("yellow");

} // End namespace SCIRun

