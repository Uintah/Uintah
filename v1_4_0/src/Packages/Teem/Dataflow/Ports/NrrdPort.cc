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
 *  NrrdPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <Teem/share/share.h>
#include <Core/Malloc/Allocator.h>

namespace SCITeem {

using namespace SCIRun;

extern "C" {
  TeemSHARE IPort* make_NrrdIPort(Module* module, const string& name) {
  return scinew SimpleIPort<NrrdDataHandle>(module,name);
}
  TeemSHARE OPort* make_NrrdOPort(Module* module, const string& name) {
  return scinew SimpleOPort<NrrdDataHandle>(module,name);
}
}
} // End namespace SCITeem

namespace SCIRun {
template<> string SimpleIPort<SCITeem::NrrdDataHandle>::port_type_("Nrrd");
template<> string SimpleIPort<SCITeem::NrrdDataHandle>::port_color_("Purple4");
} // End namespace SCIRun

