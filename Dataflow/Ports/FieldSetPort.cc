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

// FieldSetPort.cc
//  Written by:
//   Michael Callahan
//   Department of Computer Science
//   University of Utah
//   February 2001
//  Copyright (C) 2001 SCI Group


#include <Dataflow/Ports/FieldSetPort.h>
#include <Core/Malloc/Allocator.h>


namespace SCIRun {

extern "C" {
PSECORESHARE IPort* make_FieldSetIPort(Module* module, const string& name) {
  return scinew SimpleIPort<FieldSetHandle>(module,name);
}
PSECORESHARE OPort* make_FieldSetOPort(Module* module, const string& name) {
  return scinew SimpleOPort<FieldSetHandle>(module,name);
}
}

template<> string SimpleIPort<FieldSetHandle>::port_type_("FieldSet");
template<> string SimpleIPort<FieldSetHandle>::port_color_("orange");

} // End namespace SCIRun

