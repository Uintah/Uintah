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
PSECORESHARE IPort* make_FieldIPort(Module* module, const string& name) {
  return scinew SimpleIPort<FieldHandle>(module,name);
}
PSECORESHARE OPort* make_FieldOPort(Module* module, const string& name) {
  return scinew SimpleOPort<FieldHandle>(module,name);
}
}

template<> string SimpleIPort<FieldHandle>::port_type_("Field");
template<> string SimpleIPort<FieldHandle>::port_color_("yellow");


//! Specialization for field ports.
//! Field ports must only send const fields i.e. frozen fields.
template<>
void SimpleOPort<FieldHandle>::send(const FieldHandle& data)
{
  if (data.get_rep() && (! data->is_frozen())) {
    data->freeze();
  }
  do_send(data);
}

template<>
void SimpleOPort<FieldHandle>::send_intermediate(const FieldHandle& data)
{
  if (data.get_rep() && (! data->is_frozen())) {
    data->freeze();
  }
  do_send_intermediate(data);
}

} // End namespace SCIRun

