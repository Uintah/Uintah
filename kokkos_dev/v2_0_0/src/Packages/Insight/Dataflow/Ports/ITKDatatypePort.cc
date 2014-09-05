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
 *  ITKDatatypePort.cc
 *
 *  Written by:
 *   Darby J Brown
 *   Department of Computer Science
 *   University of Utah
 *   January 2003
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Insight/Dataflow/Ports/ITKDatatypePort.h>
#include <Insight/share/share.h>
#include <Core/Malloc/Allocator.h>

namespace Insight {

using namespace SCIRun;

extern "C" {
  InsightSHARE IPort* make_ITKDatatypeIPort(Module* module, const string& name) {
  return scinew SimpleIPort<ITKDatatypeHandle>(module,name);
}
  InsightSHARE OPort* make_ITKDatatypeOPort(Module* module, const string& name) {
  return scinew SimpleOPort<ITKDatatypeHandle>(module,name);
}
}
} // End namespace Insight

namespace SCIRun {
template<> string SimpleIPort<Insight::ITKDatatypeHandle>::port_type_("ITKDatatype");
template<> string SimpleIPort<Insight::ITKDatatypeHandle>::port_color_("pink");
} // End namespace SCIRun

