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
 *  GuiVar.cc: Interface to TCL variables
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   September 1994
 *
 *  Changes for distributed Dataflow:
 *   Michelle Miller 
 *   Thu May 14 01:24:12 MDT 1998
 * FIX: error cases and GuiVar* get()
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Core/Parts/GuiVar.h>
#include <Core/Parts/Part.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>

namespace SCIRun {

GuiVar::GuiVar(const string& name, Part *part)
  : name_(name), part_(part)
{
  part_->add_gui_var( this );
}

GuiVar::~GuiVar()
{
  part_->rem_gui_var( this );
}

void
GuiVar::update() 
{
  part_->var_set( this ); 
}

template class GuiTriple<Point>;
template class GuiTriple<Vector>;

} // End namespace SCIRun


















