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
 *  ColorMapPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   November 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/Ports/ColorMapPort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


extern "C" {
PSECORESHARE IPort* make_ColorMapIPort(Module* module, const string& name) {
  return scinew SimpleIPort<ColorMapHandle>(module,name);
}
PSECORESHARE OPort* make_ColorMapOPort(Module* module, const string& name) {
  return scinew SimpleOPort<ColorMapHandle>(module,name);
}
}

template<> string ColorMapIPort::port_type_("ColorMap");
template<> string ColorMapIPort::port_color_("blueviolet");

} // End namespace SCIRun


