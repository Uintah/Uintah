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


#include <Packages/Volume/Dataflow/Ports/TexturePort.h>
#include <Dataflow/share/share.h>
#include <Core/Malloc/Allocator.h>

namespace Volume {

using SCIRun::SimpleIPort;
using SCIRun::SimpleOPort;

extern "C" {
PSECORESHARE IPort* make_TextureIPort(Module* module,
					 const string& name) {
  return scinew SimpleIPort<TextureHandle>(module,name);
}
PSECORESHARE OPort* make_TextureOPort(Module* module,
					 const string& name) {
  return scinew SimpleOPort<TextureHandle>(module,name);
}
}

template<> string SimpleIPort<TextureHandle>::port_type_("Texture");
template<> string SimpleIPort<TextureHandle>::port_color_("wheat");


} // End namespace Volume

