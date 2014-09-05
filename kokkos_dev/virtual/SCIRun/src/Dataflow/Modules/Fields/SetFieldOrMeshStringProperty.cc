/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

/*
 *  SetFieldOrMeshStringProperty: Set a property for a Field (or its Mesh)
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   June 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <iostream>
#include <sstream>

namespace SCIRun {

class SetFieldOrMeshStringProperty : public Module {
  GuiString prop_;
  GuiString val_;
  GuiInt mesh_prop_; // is this property for the mesh or the Field?
public:
  SetFieldOrMeshStringProperty(GuiContext* ctx);
  virtual ~SetFieldOrMeshStringProperty();
  virtual void execute();
};

DECLARE_MAKER(SetFieldOrMeshStringProperty)
SetFieldOrMeshStringProperty::SetFieldOrMeshStringProperty(GuiContext* ctx)
: Module("SetFieldOrMeshStringProperty", ctx, Filter,"MiscField", "SCIRun"),
  prop_(get_ctx()->subVar("prop"), "units"),
  val_(get_ctx()->subVar("val"), "cm"),
  mesh_prop_(get_ctx()->subVar("meshprop"), 1)
{
}

SetFieldOrMeshStringProperty::~SetFieldOrMeshStringProperty()
{
}

void
SetFieldOrMeshStringProperty::execute()
{
  FieldHandle fldH;
  if (!get_input_handle("Input", fldH)) return;

  fldH->generation = fldH->compute_new_generation();

  // set this new property
  if (mesh_prop_.get())
  {
    fldH->mesh()->generation = fldH->mesh()->compute_new_generation();
    fldH->mesh()->set_property(prop_.get(), val_.get(), false);
  }
  else
  { 
    fldH->set_property(prop_.get(), val_.get(), false);
  }
  
  send_output_handle("Output", fldH);
}

} // End namespace SCIRun
