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


#include <Dataflow/Ports/GLTexture3DPort.h>
#include <Dataflow/share/share.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {



extern "C" {
PSECORESHARE IPort* make_GLTexture3DIPort(Module* module,
					 const clString& name) {
  return scinew SimpleIPort<GLTexture3DHandle>(module,name);
}
PSECORESHARE OPort* make_GLTexture3DOPort(Module* module,
					 const clString& name) {
  return scinew SimpleOPort<GLTexture3DHandle>(module,name);
}
}

template<> clString SimpleIPort<GLTexture3DHandle>::port_type("GLTexture3D");
template<> clString SimpleIPort<GLTexture3DHandle>::port_color("gray40");


} // End namespace SCIRun

