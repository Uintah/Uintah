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
 *  BuildInterpolant.cc:  Build an interpolant field -- a field that says
 *         how to project the data from one field onto the data of a second
 *         field.
 *
 *  Written by:
 *   David Weinstein
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
using std::cerr;
#include <stdio.h>

namespace SCIRun {

class BuildInterpolant : public Module
{
  GuiString   interp_op_gui_;

public:
  BuildInterpolant(const clString& id);
  virtual ~BuildInterpolant();
  virtual void execute();
};

extern "C" Module* make_BuildInterpolant(const clString& id)
{
  return new BuildInterpolant(id);
}

BuildInterpolant::BuildInterpolant(const clString& id) : 
  Module("BuildInterpolant", id, Filter, "Fields", "SCIRun"),
  interp_op_gui_("interp_op_gui", id, this)
{
}

BuildInterpolant::~BuildInterpolant()
{
}

void
BuildInterpolant::execute()
{
  
}

} // End namespace SCIRun
