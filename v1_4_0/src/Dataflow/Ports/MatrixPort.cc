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
 *  MatrixPort.cc
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Dataflow/share/share.h>

#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {

extern "C" {
PSECORESHARE IPort* make_MatrixIPort(Module* module, const string& name) {
  return scinew SimpleIPort<MatrixHandle>(module,name);
}
PSECORESHARE OPort* make_MatrixOPort(Module* module, const string& name) {
  return scinew SimpleOPort<MatrixHandle>(module,name);
}
}


template<> string SimpleIPort<MatrixHandle>::port_type_("Matrix");
template<> string SimpleIPort<MatrixHandle>::port_color_("dodgerblue");

} // End namespace SCIRun

