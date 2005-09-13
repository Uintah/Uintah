/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
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
 *  SetProperty: Set a property for a Field (or its Mesh)
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   June 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <Dataflow/Ports/FieldPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include <sstream>

namespace SCIRun {

class SetProperty : public Module {
  GuiString prop_;
  GuiString val_;
  GuiInt mesh_prop_; // is this property for the mesh or the Field?
public:
  SetProperty(GuiContext* ctx);
  virtual ~SetProperty();
  virtual void execute();
};

DECLARE_MAKER(SetProperty)
SetProperty::SetProperty(GuiContext* ctx)
: Module("SetProperty", ctx, Filter,"FieldsOther", "SCIRun"),
  prop_(ctx->subVar("prop")), val_(ctx->subVar("val")),
  mesh_prop_(ctx->subVar("meshprop"))
{
}

SetProperty::~SetProperty()
{
}

void SetProperty::execute() {
  FieldIPort *ifield = (FieldIPort *)get_iport("Input");
  FieldOPort *ofield = (FieldOPort *)get_oport("Output");

  FieldHandle fldH;
  if (!ifield->get(fldH))
    return;
  if (!fldH.get_rep()) {
    warning("Empty input field.");
    return;
  }

  fldH->generation++;

  // set this new property
  if (mesh_prop_.get()) {
    fldH->mesh()->generation++;
    fldH->mesh()->set_property(prop_.get(), val_.get(), false);
  } else 
    fldH->set_property(prop_.get(), val_.get(), false);
  
  ofield->send(fldH);
}
} // End namespace SCIRun
