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
 *  ImagePort.cc
 *
 *  Written by:
 *   Steve Parker
 *   Department of Computer Science
 *   University of Utah
 *   Oct 1996
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Ports/ImagePort.h>
#include <Dataflow/share/share.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun { // Namespace {} necessary for xlC AIX compilation.

extern "C" {
PSECORESHARE IPort* make_ImageIPort(Module* module, const string& name) {
  return scinew SimpleIPort<ImageHandle>(module,name);
}
PSECORESHARE OPort* make_ImageOPort(Module* module, const string& name) {
  return scinew SimpleOPort<ImageHandle>(module,name);
}
}

template<> string SimpleIPort<ImageHandle>::port_type_("Image");
template<> string SimpleIPort<ImageHandle>::port_color_("misty rose");

} // End namespace SCIRun
